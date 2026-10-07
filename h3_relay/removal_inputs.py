"""Persistent prepared masks; lazy inputs keep SAM out of unchanged rerolls."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from . import removal_cache

VERSION = 1
# Unknown/stochastic upstream nodes fall back to ordinary evaluation. Do not
# guess whether arbitrary external data or custom-node outputs are unchanged.
SUPPORTED = frozenset({
    "LoadVideo", "GetVideoComponents", "LoadImage", "ImageFromBatch",
    "CheckpointLoaderSimple", "UNETLoader", "CLIPLoader", "DualCLIPLoader",
    "CLIPTextEncode", "SAM3_Detect", "SAM3_VideoTrack", "SAM3_TrackToMask",
    "ImageToMask", "MaskToImage", "InvertMask", "GrowMask", "SolidMask",
})
MODEL_FIELDS = {"ckpt_name": "checkpoints", "unet_name": "diffusion_models",
                "clip_name": "text_encoders", "clip_name1": "text_encoders",
                "clip_name2": "text_encoders"}


def signature(execution_prompt, node_id, expand_pixels):
    """Use upstream settings and real source bytes, before resolving tensors."""
    import folder_paths
    graph = execution_prompt or {}
    own = graph.get(str(node_id))
    if not own:
        return None
    memo = {}

    def ancestry(link, visited=()):
        if not isinstance(link, list) or len(link) != 2 or link[0] not in graph:
            return link
        key = link[0]
        if key in visited:
            raise ValueError("Mask input graph contains a cycle")
        if key not in memo:
            node = graph[key]
            kind = node["class_type"]
            if kind not in SUPPORTED:
                raise LookupError("Unsupported prepared-mask upstream node")
            inputs = {k: ancestry(v, visited + (key,)) for k, v in sorted(node["inputs"].items())
                      if k not in {"video-preview", "image-preview"}}
            files = {}
            if kind in {"LoadVideo", "LoadImage"}:
                field = "file" if kind == "LoadVideo" else "image"
                path = Path(folder_paths.get_annotated_filepath(node["inputs"][field]))
                files[field] = {"sha256": removal_cache.file_hash(path)}
            for field, category in MODEL_FIELDS.items():
                name = node["inputs"].get(field)
                if isinstance(name, str):
                    path = folder_paths.get_full_path(category, name)
                    if not path:
                        raise LookupError("Model file is unavailable")
                    stat = Path(path).stat()
                    files[field] = [stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns]
            memo[key] = {"class": kind, "inputs": inputs, "files": files}
        return {"node": memo[key], "output": link[1]}

    try:
        data = {"version": VERSION, "expand_pixels": int(expand_pixels),
                "images": ancestry(own["inputs"].get("images")),
                "mask": ancestry(own["inputs"].get("mask"))}
        # Literal IMAGE/MASK inputs are not a verifiable persisted graph.
        if not all(isinstance(own["inputs"].get(k), list) for k in ("images", "mask")):
            return None
        return hashlib.sha256(json.dumps(data, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    except (LookupError, OSError, TypeError):
        return None


def paths(scope, key):
    import re
    if not re.fullmatch(r"[0-9a-f]{64}", scope) or not re.fullmatch(r"[0-9a-f]{64}", key):
        raise ValueError("Invalid prepared-mask cache key")
    directory = removal_cache.root() / "prepared" / scope
    directory.mkdir(parents=True, exist_ok=True)
    return directory / (key + ".json"), directory / (key + ".pt")


def available(scope, key):
    if not key:
        return False
    metadata, tensor = paths(scope, key)
    return metadata.is_file() and tensor.is_file()


def save(scope, key, frames):
    import torch
    metadata, tensor = paths(scope, key)
    temporary = tensor.with_suffix(".part")
    frames = frames.detach().cpu().contiguous()
    torch.save({"frames": frames}, temporary)
    temporary.replace(tensor)
    data = {"version": VERSION, "scope": scope, "key": key,
            "sha256": removal_cache.file_hash(tensor),
            "frames_sha256": removal_cache.tensor_hash(frames), "shape": list(frames.shape)}
    temporary = metadata.with_suffix(".part")
    temporary.write_text(json.dumps(data), encoding="utf-8")
    temporary.replace(metadata)
    return data


def load(scope, key):
    import torch
    metadata, tensor = paths(scope, key)
    data = json.loads(metadata.read_text())
    if (data.get("version"), data.get("scope"), data.get("key")) != (VERSION, scope, key):
        raise ValueError("Prepared-mask cache ownership does not match")
    if removal_cache.file_hash(tensor) != data["sha256"]:
        raise ValueError("Prepared-mask cache is damaged. Remove this prepared cache entry and run again.")
    return torch.load(tensor, map_location="cpu", weights_only=True)["frames"]
