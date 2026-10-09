"""Exact window checkpoints for explicit, source-compatible prefix locking."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import uuid

CACHE_VERSION = 1
MAX_SEED = 0xffffffffffffffff
SEED_SCHEME = "h3-window-seed-increment-v2"


def window_seed(master_seed, index):
    """First window uses the master; later windows increment with uint64 wrap."""
    master_seed, index = int(master_seed), int(index)
    if not 0 <= master_seed <= MAX_SEED or index < 0:
        raise ValueError("Master seed must be unsigned 64-bit and window index nonnegative")
    return (master_seed + index) & MAX_SEED


def root():
    import folder_paths
    directory = Path(folder_paths.get_output_directory()) / "__h3_removal_cache"
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def scope_key(node_id, extra_pnginfo):
    workflow_id = str((extra_pnginfo or {}).get("workflow", {}).get("id", ""))
    return hashlib.sha256(f"{workflow_id}\0{node_id}".encode()).hexdigest()


def tensor_hash(tensor):
    import torch
    digest = hashlib.sha256()
    digest.update(str((tuple(tensor.shape), str(tensor.dtype))).encode())
    # Avoid a second full-video byte allocation, including for CUDA inputs.
    for frame in tensor:
        digest.update(frame.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def configuration_key(source, anchor, settings, execution_prompt, node_id):
    """Hash actual pixels plus the model/encoder/VAE ancestry and file versions."""
    import folder_paths
    graph = execution_prompt or {}
    fields = {"unet_name": "diffusion_models", "clip_name": "text_encoders",
              "vae_name": "vae", "lora_name": "loras", "ckpt_name": "checkpoints"}

    def ancestry(link, visited=()):
        if not isinstance(link, list) or len(link) != 2 or link[0] not in graph:
            return link
        key = link[0]
        if key in visited:
            raise ValueError("Model graph contains a cycle")
        node = graph[key]
        inputs = {k: ancestry(v, visited + (key,)) for k, v in sorted(node["inputs"].items())}
        versions = {}
        for field, category in fields.items():
            name = node["inputs"].get(field)
            if isinstance(name, str):
                path = folder_paths.get_full_path(category, name)
                if path:
                    stat = Path(path).stat()
                    versions[field] = [stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns]
        return {"class": node["class_type"], "inputs": inputs, "files": versions, "output": link[1]}

    inputs = graph.get(str(node_id), {}).get("inputs", {})
    models = {k: ancestry(inputs.get(k)) for k in ("model", "clip", "video_vae", "audio_vae")}
    data = {"version": CACHE_VERSION, "source": tensor_hash(source), "anchor": tensor_hash(anchor[:1]),
            "settings": settings, "models": models}
    return hashlib.sha256(json.dumps(data, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _directory(scope):
    if not re.fullmatch(r"[0-9a-f]{64}", scope):
        raise ValueError("Invalid window cache scope")
    directory = root() / "windows" / scope
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def _record_path(scope, record_id, suffix):
    if not re.fullmatch(r"[0-9a-f]{32}", record_id):
        raise ValueError("Invalid window checkpoint ID")
    return _directory(scope) / (record_id + suffix)


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def metadata(scope, record_id):
    path = _record_path(scope, record_id, ".json")
    if not path.is_file():
        raise ValueError("Saved window is unavailable on this server. Unlock it and render again.")
    data = json.loads(path.read_text())
    if data.get("record_id") != record_id or data.get("scope") != scope:
        raise ValueError("Window checkpoint ownership does not match")
    return data


def save(scope, config_key, index, start, window_frames, source_count, seed, previous_records, frames, audio, window_metadata=None):
    import torch
    record_id = uuid.uuid4().hex
    path = _record_path(scope, record_id, ".pt")
    temporary = path.with_suffix(".part")
    frames = frames.detach().cpu().contiguous().clone()
    waveform = audio["waveform"].detach().cpu().contiguous().clone()
    torch.save({"frames": frames, "waveform": waveform, "sample_rate": int(audio["sample_rate"])}, temporary)
    temporary.replace(path)
    data = {"record_id": record_id, "scope": scope, "config_key": config_key,
            "index": index, "start": start, "window_frames": window_frames, "source_count": source_count,
            "seed": str(seed), "previous_records": list(previous_records), "sha256": file_hash(path),
            "frames_sha256": tensor_hash(frames)}
    if window_metadata is not None:
        data["window_metadata"] = window_metadata
    path = _record_path(scope, record_id, ".json")
    temporary = path.with_suffix(".part")
    temporary.write_text(json.dumps(data), encoding="utf-8")
    temporary.replace(path)
    return data


def load(scope, record_id, config_key):
    import torch
    data = metadata(scope, record_id)
    if data["config_key"] != config_key:
        raise ValueError("Locked windows use different inputs or settings. Unlock them before rendering.")
    path = _record_path(scope, record_id, ".pt")
    if not path.is_file() or file_hash(path) != data["sha256"]:
        raise ValueError("Saved window is missing or damaged. Unlock it and render again.")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    return payload["frames"], {"waveform": payload["waveform"], "sample_rate": payload["sample_rate"]}


def controls_for_configuration(controls, scope, config_key, previous_config_key=None):
    """Start a fresh set on changed inputs; keep strict checkpoint validation.

    The config hashes actual prepared pixels/audio, references and model/settings.
    Legacy workflows have no binding, so inspect their locked checkpoint metadata.
    Unchanged missing/damaged/foreign checkpoints still fail rather than resample.
    """
    if len(controls) > 131072:
        raise ValueError("Too many window controls")
    data = json.loads(controls or "{}")
    if not isinstance(data, dict) or not isinstance(data.get("windows", {}), dict):
        raise ValueError("Invalid window controls")
    bound_key = data.get("config_key") or previous_config_key
    changed = bool(bound_key and bound_key != config_key)
    for item in data.get("windows", {}).values():
        if not isinstance(item, dict):
            raise ValueError("Invalid window control entry")
        if not changed and item.get("locked") and item.get("record_id"):
            record = metadata(scope, item["record_id"])
            changed |= record["config_key"] != config_key
    if changed:
        data = {"windows": {}}
    data["config_key"] = config_key
    return json.dumps(data), changed


def plan(controls, scope, config_key, starts, window_frames, source_count, base_seed,
         derive_seeds=False, window_plan=None):
    """Only a contiguous prefix can be locked: later windows depend on earlier ones."""
    if len(controls) > 131072:
        raise ValueError("Too many window controls")
    data = json.loads(controls or "{}")
    if not isinstance(data, dict) or not isinstance(data.get("windows", {}), dict):
        raise ValueError("Invalid window controls")
    if window_plan is not None and (len(window_plan) != len(starts) or
            any(entry["source_start"] != start for entry, start in zip(window_plan, starts))):
        raise ValueError("Window plan does not match its source starts")
    windows = data.get("windows", {})
    result, previous_records, unlocked = [], [], False
    for index, start in enumerate(starts):
        item = windows.get(str(index), {})
        if not isinstance(item, dict):
            raise ValueError("Invalid window control entry")
        default_seed = window_seed(base_seed, index) if derive_seeds else base_seed
        seed = int(item.get("seed", default_seed))
        if not 0 <= seed <= MAX_SEED:
            raise ValueError("Window seed must be an unsigned 64-bit integer")
        record_id = item.get("record_id", "") if item.get("locked") else ""
        if record_id:
            if unlocked:
                raise ValueError("Lock earlier windows first. A changed window invalidates all later windows.")
            record = metadata(scope, record_id)
            expected = {"config_key": config_key, "index": index, "start": start,
                        "window_frames": window_plan[index]["window_frames"] if window_plan is not None else window_frames,
                        "source_count": source_count, "previous_records": previous_records}
            if window_plan is not None:
                expected["window_metadata"] = window_plan[index]
            if any(record.get(k) != v for k, v in expected.items()):
                raise ValueError("Locked windows use different inputs, settings or history. Unlock them before rendering.")
            if "seed" in item and str(seed) != record["seed"]:
                raise ValueError("Unlock a window before changing its seed")
            seed = int(record["seed"])
            previous_records = previous_records + [record_id]
        else:
            if item.get("locked"):
                raise ValueError("This window has no saved checkpoint. Render it before locking.")
            unlocked = True
        result.append({"seed": seed, "record_id": record_id})
    return result
