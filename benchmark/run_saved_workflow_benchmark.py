#!/usr/bin/env python3
"""Queue a saved H3 Relay workflow unchanged and record host resource use.

The source workflow is read-only. Only the in-memory run namespace and final
filename are changed so a benchmark cannot reuse an earlier run's artifacts.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import hashlib
import json
import os
import pathlib
import shutil
import subprocess
import time
import uuid
from typing import Any

try:
    import aiohttp
except ModuleNotFoundError:  # Parser-only tests do not need runtime monitoring.
    aiohttp = None

try:
    import psutil
except ModuleNotFoundError:  # Parser-only tests do not need runtime monitoring.
    psutil = None


WIDGET_INPUTS = {
    "LoadImage": ("image",),
    "PrimitiveStringMultiline": ("value",),
    "H3RelayH3HybridModelLoader": (
        "base_model", "overlay_model", "overlay_preset",
        "manual_cache_revision", "block_range_start", "block_range_end",
        "final_adaln_from_overlay", "custom_overlays", "custom_base",
        "weight_dtype",
    ),
    "H3RelayH3ModelLoader": (
        "model_name", "text_encoder_name", "video_vae_name",
        "audio_vae_name", "weight_dtype", "manual_cache_revision",
    ),
    "H3RelayModelLoRA": ("lora_name", "strength"),
    "H3RelayAttention": ("attention",),
    "H3RelaySequenceStart": (
        "run_name", "global_prompt", "width", "height",
        "h3_overlap_frames", "sampler", "scheduler", "spectrum_enabled",
    ),
    "H3RelayGenerateShot": (
        "seed", "control_after_generate", "duration_seconds", "h3_steps",
        "output_crf", "ref_image_size", "shot_id",
    ),
    "H3RelayLTXModelLoader": (
        "model_name", "vae_name", "latent_2x_model_name",
        "text_encoder_name", "distilled_lora", "distilled_strength",
        "pixel_upscale_ic_lora", "pixel_upscale_ic_strength", "weight_dtype",
        "manual_cache_revision",
    ),
    "H3RelayEnhanceShot": (
        "output_crf", "context_window_frames", "context_overlap_frames",
        "vae_temporal_tile_frames", "vae_temporal_overlap_frames",
    ),
    "H3RelayInterpolationModelLoader": (
        "model_name", "manual_cache_revision",
    ),
    "H3RelayInterpolateShot": ("multiplier", "output_crf", "chunk_frames"),
    "H3RelayAssemble": ("output_stage", "filename", "audio_bitrate"),
    "H3RelayCacheManager": (
        "action", "keep_revisions_per_shot", "budget_gb",
    ),
}

SKIP_SERVER_INPUTS = {"control_after_generate"}


def sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_text(command: list[str]) -> str:
    return subprocess.check_output(command, text=True).strip()


def directory_size(path: pathlib.Path) -> int:
    if not path.exists():
        return 0
    total = 0
    for root, _directories, files in os.walk(path):
        for name in files:
            try:
                total += (pathlib.Path(root) / name).stat().st_size
            except FileNotFoundError:
                pass
    return total


def resolve_origin(
    workflow: dict[str, Any], link_id: int, nodes: dict[int, dict[str, Any]],
    links: dict[int, list[Any]],
) -> tuple[int, int]:
    link = links[int(link_id)]
    origin_id, origin_slot = int(link[1]), int(link[2])
    origin = nodes[origin_id]
    if int(origin.get("mode", 0)) != 4:
        return origin_id, origin_slot

    outputs = origin.get("outputs") or []
    output_type = outputs[origin_slot].get("type") if origin_slot < len(outputs) else None
    candidates = [
        item for item in (origin.get("inputs") or [])
        if item.get("link") is not None and item.get("type") == output_type
    ]
    if not candidates:
        candidates = [
            item for item in (origin.get("inputs") or [])
            if item.get("link") is not None
        ]
    if not candidates:
        raise ValueError("Bypassed node %s has no connected passthrough input" % origin_id)
    return resolve_origin(workflow, int(candidates[0]["link"]), nodes, links)


def api_prompt_from_workflow(
    workflow: dict[str, Any], benchmark_run_name: str,
) -> dict[str, Any]:
    nodes = {int(node["id"]): node for node in workflow.get("nodes", [])}
    links = {int(link[0]): link for link in workflow.get("links", [])}
    prompt: dict[str, Any] = {}

    for node_id, node in nodes.items():
        node_type = str(node.get("type") or "")
        mode = int(node.get("mode", 0))
        if node_type in {"MarkdownNote", "Note"} or mode in {2, 4}:
            continue
        widget_names = WIDGET_INPUTS.get(node_type)
        if widget_names is None:
            raise ValueError("No benchmark widget mapping for %s (node %s)" % (
                node_type, node_id))
        values = list(node.get("widgets_values") or [])
        named_values = node.get("widgets_values_named") or {}
        if node_type == "H3RelayInterpolateShot" and len(values) == 2:
            values.append(48)
        offset = (
            1 if not named_values
            and len(values) == len(widget_names) + 1
            and values[0] == ""
            else 0)
        if len(values) - offset < len(widget_names):
            raise ValueError(
                "%s node %s has %d widgets; expected %d" %
                (node_type, node_id, len(values), len(widget_names)))
        widget_values = {
            name: named_values.get(name, values[index + offset])
            for index, name in enumerate(widget_names)
        }
        inputs: dict[str, Any] = {}
        for item in node.get("inputs") or []:
            name = str(item["name"])
            if item.get("link") is not None:
                origin_id, origin_slot = resolve_origin(
                    workflow, int(item["link"]), nodes, links)
                inputs[name] = [str(origin_id), origin_slot]
            elif name in widget_values and name not in SKIP_SERVER_INPUTS:
                inputs[name] = widget_values[name]
        if node_type == "H3RelayInterpolateShot":
            inputs.setdefault("chunk_frames", int(widget_values["chunk_frames"]))

        if node_type == "H3RelaySequenceStart":
            inputs["run_name"] = benchmark_run_name
        elif node_type == "H3RelayAssemble":
            inputs["filename"] = benchmark_run_name + "_assembled"

        prompt[str(node_id)] = {
            "class_type": node_type,
            "inputs": inputs,
            "_meta": {"title": node.get("title") or node_type},
        }
    return prompt


def service_details() -> tuple[int, pathlib.Path | None]:
    pid = int(run_text([
        "systemctl", "--user", "show", "comfyui-h3.service",
        "-p", "MainPID", "--value",
    ]))
    control_group = run_text([
        "systemctl", "--user", "show", "comfyui-h3.service",
        "-p", "ControlGroup", "--value",
    ])
    memory_file = pathlib.Path("/sys/fs/cgroup" + control_group) / "memory.current"
    return pid, memory_file if memory_file.exists() else None


def resource_sample(
    started: float, service_pid: int, cgroup_memory: pathlib.Path | None,
    active_node: str,
) -> dict[str, Any]:
    gpu = run_text([
        "nvidia-smi",
        "--query-gpu=memory.used,memory.total,utilization.gpu,power.draw,temperature.gpu",
        "--format=csv,noheader,nounits",
    ]).split(",")
    virtual = psutil.virtual_memory()
    swap = psutil.swap_memory()
    rss = 0
    try:
        process = psutil.Process(service_pid)
        rss = process.memory_info().rss
        for child in process.children(recursive=True):
            try:
                rss += child.memory_info().rss
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        pass
    cgroup_bytes = 0
    if cgroup_memory is not None:
        try:
            cgroup_bytes = int(cgroup_memory.read_text().strip())
        except (FileNotFoundError, ValueError, OSError):
            pass
    return {
        "timestamp": time.time(),
        "elapsed_seconds": time.monotonic() - started,
        "active_node": active_node,
        "gpu_memory_used_mib": int(float(gpu[0].strip())),
        "gpu_memory_total_mib": int(float(gpu[1].strip())),
        "gpu_utilization_percent": float(gpu[2].strip()),
        "gpu_power_watts": float(gpu[3].strip()),
        "gpu_temperature_c": float(gpu[4].strip()),
        "system_memory_used_bytes": int(virtual.total - virtual.available),
        "system_memory_available_bytes": int(virtual.available),
        "swap_used_bytes": int(swap.used),
        "service_rss_bytes": int(rss),
        "service_cgroup_bytes": int(cgroup_bytes),
    }


def gib(value: float) -> float:
    return value / (1024 ** 3)


def mib(value: float) -> float:
    return value / (1024 ** 2)


def percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * fraction)))
    return float(ordered[index])


def ffprobe(path: pathlib.Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    raw = subprocess.check_output([
        "ffprobe", "-v", "error", "-show_streams", "-show_format",
        "-of", "json", str(path),
    ], text=True)
    return json.loads(raw)


def final_output_path(comfy_root: pathlib.Path, run_name: str) -> pathlib.Path | None:
    directory = comfy_root / "output" / "h3_chains" / run_name / "enhanced" / "final"
    if not directory.exists():
        return None
    candidates = sorted(directory.glob("*.mp4"), key=lambda path: path.stat().st_mtime)
    return candidates[-1] if candidates else None


async def main() -> int:
    if aiohttp is None or psutil is None:
        raise RuntimeError(
            "full benchmark execution requires the aiohttp and psutil packages")
    parser = argparse.ArgumentParser()
    parser.add_argument("--workflow", required=True, type=pathlib.Path)
    parser.add_argument("--expected-sha256", required=True)
    parser.add_argument("--endpoint", default="http://100.71.50.55:8189")
    parser.add_argument("--comfy-root", required=True, type=pathlib.Path)
    parser.add_argument("--results-root", required=True, type=pathlib.Path)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()

    source_hash = sha256(args.workflow)
    if source_hash != args.expected_sha256:
        raise SystemExit(
            "Saved workflow changed after audit: expected %s, found %s. "
            "Refusing to queue it." % (args.expected_sha256, source_hash))
    workflow = json.loads(args.workflow.read_text(encoding="utf-8"))
    stamp = time.strftime("%Y%m%d_%H%M%S")
    original_run = next(
        node for node in workflow["nodes"]
        if node.get("type") == "H3RelaySequenceStart"
    )["widgets_values"][0]
    run_name = "%s_fullspec_%s" % (original_run, stamp)
    result_dir = args.results_root / run_name
    result_dir.mkdir(parents=True, exist_ok=False)
    shutil.copy2(args.workflow, result_dir / "source_workflow.json")
    prompt = api_prompt_from_workflow(workflow, run_name)
    (result_dir / "submitted_api_prompt.json").write_text(
        json.dumps(prompt, indent=2) + "\n", encoding="utf-8")
    manifest = {
        "source_workflow": str(args.workflow),
        "source_sha256": source_hash,
        "source_modified": args.workflow.stat().st_mtime,
        "original_run_name": original_run,
        "benchmark_run_name": run_name,
        "endpoint": args.endpoint,
        "created_at": time.time(),
    }
    (result_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print("RESULT_DIR=%s" % result_dir, flush=True)
    print("RUN_NAME=%s" % run_name, flush=True)
    if args.prepare_only:
        return 0

    # A final read-only guard immediately before submission.
    final_hash = sha256(args.workflow)
    if final_hash != source_hash:
        raise SystemExit("Saved workflow changed during prompt preparation; refusing to queue.")

    service_pid, cgroup_memory = service_details()
    cache_root = args.comfy_root / "user" / "__h3_relay_cache"
    cache_before = directory_size(cache_root)
    disk_before = shutil.disk_usage(args.comfy_root)
    client_id = str(uuid.uuid4())
    requested_prompt_id = str(uuid.uuid4())
    payload = {
        "prompt": prompt,
        "client_id": client_id,
        "prompt_id": requested_prompt_id,
        "partial_execution_targets": ["50"],
        "extra_data": {
            "extra_pnginfo": {"workflow": workflow},
            "benchmark_run_name": run_name,
            "source_workflow_sha256": source_hash,
        },
    }

    samples: list[dict[str, Any]] = []
    events: list[dict[str, Any]] = []
    active_node = "preflight"
    started = time.monotonic()
    completed = False
    terminal_type = ""
    history: dict[str, Any] = {}
    last_print = -30.0
    last_sample = -1.0
    last_history_poll = -10.0

    async with aiohttp.ClientSession() as session:
        ws_url = args.endpoint.replace("http://", "ws://").replace("https://", "wss://")
        async with session.ws_connect(ws_url + "/ws?clientId=" + client_id, heartbeat=30) as ws:
            async with session.post(args.endpoint + "/prompt", json=payload) as response:
                body = await response.json()
                if response.status >= 400:
                    raise RuntimeError("Prompt validation failed: %s" % json.dumps(body))
                prompt_id = str(body["prompt_id"])
            print("PROMPT_ID=%s" % prompt_id, flush=True)

            while not completed:
                try:
                    message = await asyncio.wait_for(ws.receive(), timeout=1.0)
                except asyncio.TimeoutError:
                    message = None
                if message is not None and message.type == aiohttp.WSMsgType.TEXT:
                    packet = json.loads(message.data)
                    data = packet.get("data") or {}
                    if data.get("prompt_id") in {None, prompt_id}:
                        event = {
                            "timestamp": time.time(),
                            "elapsed_seconds": time.monotonic() - started,
                            "type": packet.get("type"),
                            "data": data,
                        }
                        events.append(event)
                        if packet.get("type") == "executing" and data.get("node"):
                            node = str(data["node"])
                            top = node.split(".", 1)[0]
                            if top != active_node:
                                active_node = top
                                title = prompt.get(top, {}).get("_meta", {}).get("title", top)
                                print(
                                    "STAGE elapsed=%.1fs node=%s title=%s" %
                                    (event["elapsed_seconds"], top, title), flush=True)
                        if packet.get("type") in {
                            "execution_success", "execution_error",
                            "execution_interrupted",
                        } and data.get("prompt_id") == prompt_id:
                            completed = True
                            terminal_type = str(packet.get("type"))

                elapsed = time.monotonic() - started
                if elapsed - last_sample < 1.0 and not completed:
                    continue
                last_sample = elapsed
                sample = resource_sample(
                    started, service_pid, cgroup_memory, active_node)
                samples.append(sample)
                if sample["elapsed_seconds"] - last_print >= 30:
                    last_print = sample["elapsed_seconds"]
                    print(
                        "METRIC elapsed=%.0fs stage=%s vram=%d/%dMiB gpu=%.0f%% "
                        "ram=%.1fGiB service=%.1fGiB swap=%.1fGiB" % (
                            sample["elapsed_seconds"], active_node,
                            sample["gpu_memory_used_mib"],
                            sample["gpu_memory_total_mib"],
                            sample["gpu_utilization_percent"],
                            gib(sample["system_memory_used_bytes"]),
                            gib(sample["service_cgroup_bytes"]),
                            gib(sample["swap_used_bytes"]),
                        ), flush=True)

                if (not completed and
                        sample["elapsed_seconds"] - last_history_poll >= 10):
                    last_history_poll = sample["elapsed_seconds"]
                    async with session.get(args.endpoint + "/history/" + prompt_id) as response:
                        maybe = await response.json()
                    if prompt_id in maybe and maybe[prompt_id].get("status", {}).get("completed"):
                        history = maybe[prompt_id]
                        completed = True
                        terminal_type = history.get("status", {}).get("status_str", "completed")

            if not history:
                async with session.get(args.endpoint + "/history/" + prompt_id) as response:
                    history_payload = await response.json()
                history = history_payload.get(prompt_id, {})

    finished = time.monotonic()
    final_sample = resource_sample(started, service_pid, cgroup_memory, "complete")
    samples.append(final_sample)
    cache_after = directory_size(cache_root)
    disk_after = shutil.disk_usage(args.comfy_root)
    output_path = final_output_path(args.comfy_root, run_name)
    probe = ffprobe(output_path) if output_path else {}

    with (result_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(samples[0]))
        writer.writeheader()
        writer.writerows(samples)
    with (result_dir / "events.jsonl").open("w", encoding="utf-8") as handle:
        for event in events:
            handle.write(json.dumps(event, separators=(",", ":")) + "\n")
    (result_dir / "history.json").write_text(
        json.dumps(history, indent=2) + "\n", encoding="utf-8")
    (result_dir / "ffprobe.json").write_text(
        json.dumps(probe, indent=2) + "\n", encoding="utf-8")

    gpu_values = [float(item["gpu_memory_used_mib"]) for item in samples]
    ram_values = [float(item["system_memory_used_bytes"]) for item in samples]
    service_values = [float(item["service_cgroup_bytes"]) for item in samples]
    swap_values = [float(item["swap_used_bytes"]) for item in samples]
    report = {
        **manifest,
        "prompt_id": prompt_id,
        "terminal_type": terminal_type,
        "history_status": history.get("status", {}),
        "duration_seconds": finished - started,
        "sample_count": len(samples),
        "gpu": {
            "baseline_mib": gpu_values[0],
            "peak_mib": max(gpu_values),
            "p95_mib": percentile(gpu_values, 0.95),
            "total_mib": samples[0]["gpu_memory_total_mib"],
        },
        "system_ram": {
            "baseline_gib": gib(ram_values[0]),
            "peak_gib": gib(max(ram_values)),
            "p95_gib": gib(percentile(ram_values, 0.95)),
        },
        "service_memory": {
            "baseline_gib": gib(service_values[0]),
            "peak_gib": gib(max(service_values)),
            "p95_gib": gib(percentile(service_values, 0.95)),
        },
        "swap": {
            "baseline_gib": gib(swap_values[0]),
            "peak_gib": gib(max(swap_values)),
        },
        "cache_growth_gib": gib(cache_after - cache_before),
        "disk_free_before_gib": gib(disk_before.free),
        "disk_free_after_gib": gib(disk_after.free),
        "output_path": str(output_path) if output_path else None,
        "output_size_bytes": output_path.stat().st_size if output_path else None,
    }
    (result_dir / "report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8")
    markdown = """# H3 Relay full-workflow resource report

- Status: **{status}**
- Runtime: **{runtime:.1f} minutes**
- Source workflow SHA-256: `{source_hash}`
- Benchmark namespace: `{run_name}`
- GPU VRAM: baseline **{gpu_base:.0f} MiB**, peak **{gpu_peak:.0f} MiB**, p95 **{gpu_p95:.0f} MiB**
- System RAM used: baseline **{ram_base:.1f} GiB**, peak **{ram_peak:.1f} GiB**, p95 **{ram_p95:.1f} GiB**
- ComfyUI service memory: baseline **{svc_base:.1f} GiB**, peak **{svc_peak:.1f} GiB**, p95 **{svc_p95:.1f} GiB**
- Swap used: baseline **{swap_base:.1f} GiB**, peak **{swap_peak:.1f} GiB**
- Managed-cache growth: **{cache_growth:.1f} GiB**
- Final output: `{output}`

The observed peaks are measurements, not minimum recommendations. Hardware
headroom and stage-level interpretation should be added after reviewing the
metrics and execution event timeline.
""".format(
        status=terminal_type,
        runtime=report["duration_seconds"] / 60,
        source_hash=source_hash,
        run_name=run_name,
        gpu_base=report["gpu"]["baseline_mib"],
        gpu_peak=report["gpu"]["peak_mib"],
        gpu_p95=report["gpu"]["p95_mib"],
        ram_base=report["system_ram"]["baseline_gib"],
        ram_peak=report["system_ram"]["peak_gib"],
        ram_p95=report["system_ram"]["p95_gib"],
        svc_base=report["service_memory"]["baseline_gib"],
        svc_peak=report["service_memory"]["peak_gib"],
        svc_p95=report["service_memory"]["p95_gib"],
        swap_base=report["swap"]["baseline_gib"],
        swap_peak=report["swap"]["peak_gib"],
        cache_growth=report["cache_growth_gib"],
        output=report["output_path"] or "not produced",
    )
    (result_dir / "REPORT.md").write_text(markdown, encoding="utf-8")
    print("REPORT=%s" % (result_dir / "REPORT.md"), flush=True)
    print("OUTPUT=%s" % (output_path or ""), flush=True)
    print("TERMINAL=%s" % terminal_type, flush=True)
    return 0 if terminal_type in {"execution_success", "success"} else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
