"""Small, temporary video previews published before the next removal window."""
from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
import re
import threading
import uuid

_LOCK = threading.RLock()
_RUNS = {}
_REGISTERED = False
_LOG = logging.getLogger(__name__)
EVENT = "h3_relay.removal.preview"


def _root():
    from .removal_cache import root as cache_root
    root = cache_root() / "previews"
    root.mkdir(parents=True, exist_ok=True)
    return root


def _key(workflow_id, node_id):
    return hashlib.sha256(f"{workflow_id}\0{node_id}".encode()).hexdigest()


def _write_json(path, data):
    temporary = path.with_suffix(".part")
    temporary.write_text(json.dumps(data), encoding="utf-8")
    temporary.replace(path)


def _publish(run):
    _write_json(_root() / run["run_id"] / "manifest.json", run)
    from server import PromptServer
    # Broadcast intentionally: another browser may be watching the same workflow.
    try:
        PromptServer.instance.send_sync(EVENT, json.loads(json.dumps(run)))
    except Exception:
        _LOG.warning("Could not send removal preview; it remains available on disk", exc_info=True)


def begin_run(node_id, extra_pnginfo, starts, window_frames, source_frames, seed,
              config_key=None, controls_reset=False, window_plan=None, window_policy=None):
    from comfy_execution.utils import get_executing_context
    context = get_executing_context()
    workflow = (extra_pnginfo or {}).get("workflow", {})
    run_id = uuid.uuid4().hex
    run = {"run_id": run_id, "node_id": str(node_id or ""),
           "workflow_id": str(workflow.get("id", "")),
           "prompt_id": str(context.prompt_id if context else ""),
           "window_frames": window_frames, "source_frames": source_frames,
           "seed": str(seed), "total": len(starts), "segments": [], "status": "rendering"}
    if window_plan is not None:
        run.update(window_plan=window_plan, window_policy=window_policy)
    if config_key is not None:
        run.update(config_key=config_key, controls_reset=controls_reset)
    with _LOCK:
        (_root() / run_id).mkdir()
        _RUNS[run_id] = run
        while len(_RUNS) > 64:
            del _RUNS[next(iter(_RUNS))]
        _write_json(_root() / (_key(run["workflow_id"], run["node_id"]) + ".json"),
                    {"run_id": run_id})
        _publish(run)
    return run_id


def encode_preview(frames, directory, index):
    """Encode at most 512px wide; never keep another full-resolution batch."""
    import av
    import numpy as np
    from PIL import Image
    from fractions import Fraction

    directory = Path(directory)
    name = f"window-{index + 1:03d}"
    video, poster = directory / (name + ".mp4"), directory / (name + ".jpg")
    temporary = directory / (name + ".part.mp4")
    height, width = frames.shape[1:3]
    scale = min(1., 512 / width, 512 / height)
    size = (max(2, int(width * scale) // 2 * 2), max(2, int(height * scale) // 2 * 2))
    try:
        with av.open(str(temporary), mode="w") as container:
            stream = container.add_stream("libx264", rate=Fraction(24, 1))
            stream.width, stream.height = size
            stream.pix_fmt = "yuv420p"
            stream.options = {"crf": "23", "preset": "veryfast"}
            for index, frame in enumerate(frames):
                pixels = (frame.detach().cpu().numpy().clip(0, 1) * 255).astype(np.uint8)
                image = Image.fromarray(pixels).resize(size, Image.Resampling.LANCZOS)
                if index == 0:
                    image.save(poster, quality=85)
                container.mux(stream.encode(av.VideoFrame.from_ndarray(np.asarray(image), format="rgb24")))
            container.mux(stream.encode(None))
        temporary.replace(video)
    finally:
        temporary.unlink(missing_ok=True)
    return video.name, poster.name


def publish_window(run_id, index, frames, start, discarded_overlap, record=None, reused=False):
    if not re.fullmatch(r"[0-9a-f]{32}", run_id) or not (_root() / run_id / "manifest.json").is_file():
        raise ValueError("Unknown removal preview run")
    segment = {"index": index, "start": start, "frames": len(frames),
               "discarded_overlap": discarded_overlap, "fps": 24}
    if record:
        segment.update(record_id=record["record_id"], seed=record["seed"],
                       frames_sha256=record["frames_sha256"], reused=reused)
    try:
        video, poster = encode_preview(frames, _root() / run_id, index)
        subfolder = "__h3_removal_cache/previews/" + run_id
        segment.update(video={"filename": video, "subfolder": subfolder, "type": "output"},
                       poster={"filename": poster, "subfolder": subfolder, "type": "output"})
    except Exception as error:
        # A display/encoder failure must not discard expensive generated frames.
        _LOG.warning("Window preview failed", exc_info=True)
        segment["error"] = f"Preview unavailable: {type(error).__name__}"
    with _LOCK:
        run = _RUNS.get(run_id)
        if run is None:
            run = json.loads((_root() / run_id / "manifest.json").read_text())
        if run.get("window_plan") is not None:
            segment["window_metadata"] = run["window_plan"][index]
            segment["window_frames"] = run["window_plan"][index]["window_frames"]
        run["segments"] = sorted([s for s in run["segments"] if s["index"] != index] + [segment],
                                 key=lambda s: s["index"])
        if len(run["segments"]) == run["total"]:
            run["status"] = "complete"
        _publish(run)


def latest_run(workflow_id, node_id, run_id=None):
    with _LOCK:
        if run_id is None:
            pointer = _root() / (_key(workflow_id, node_id) + ".json")
            if not pointer.exists():
                return None
            run_id = json.loads(pointer.read_text())["run_id"]
        if not re.fullmatch(r"[0-9a-f]{32}", run_id):
            return None
        manifest = _root() / run_id / "manifest.json"
        if not manifest.exists():
            return None
        run = json.loads(manifest.read_text())
        return run if (run["workflow_id"], run["node_id"]) == (workflow_id, node_id) else None


def register_routes():
    global _REGISTERED
    if _REGISTERED:
        return
    from aiohttp import web
    from server import PromptServer
    if PromptServer.instance is None:
        return

    @PromptServer.instance.routes.get("/h3_relay/removal/previews")
    async def previews(request):
        workflow_id = request.query.get("workflow_id", "")
        node_id = request.query.get("node_id", "")
        if len(workflow_id) > 512 or len(node_id) > 128:
            raise web.HTTPBadRequest()
        run = latest_run(workflow_id, node_id, request.query.get("run_id"))
        if run and run["status"] != "complete":
            running, pending = PromptServer.instance.prompt_queue.get_current_queue()
            if not any(str(item[1]) == run["prompt_id"] for item in running + pending):
                run["status"] = "stopped"
        return web.json_response(run, headers={"Cache-Control": "no-store"})

    _REGISTERED = True
