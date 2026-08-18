"""Bounded-memory staged queue orchestration for H3 Relay graphs."""

from __future__ import annotations

import asyncio
import copy
import logging
import time
import uuid
from typing import Any

try:
    import execution
    from aiohttp import web
    from server import PromptServer
except ImportError:  # Runtime-contract imports outside a live ComfyUI server.
    execution = None
    web = None
    PromptServer = None


_LOG = logging.getLogger("h3_relay.staged")
_RUNS: dict[str, dict[str, Any]] = {}
_TASKS: dict[str, asyncio.Task] = {}
_ROUTES_REGISTERED = False

STAGED_CLASSES = {
    "H3RelayGenerateShot": "h3",
    "H3RelayEnhanceShot": "ltx",
    "H3RelayInterpolateShot": "rife",
}


def _link_source(value: Any) -> str | None:
    if (isinstance(value, (list, tuple)) and len(value) == 2
            and isinstance(value[0], (str, int))):
        return str(value[0])
    return None


def _ancestor_order(prompt: dict[str, Any], target: str) -> list[str]:
    visited: set[str] = set()
    visiting: set[str] = set()
    ordered: list[str] = []

    def visit(node_id: str) -> None:
        node_id = str(node_id)
        if node_id in visited:
            return
        if node_id in visiting:
            raise ValueError("H3 Relay staged graph contains a dependency cycle.")
        node = prompt.get(node_id)
        if not isinstance(node, dict):
            raise ValueError("H3 Relay staged graph is missing node %s." % node_id)
        visiting.add(node_id)
        for value in (node.get("inputs") or {}).values():
            source = _link_source(value)
            if source is not None:
                visit(source)
        visiting.remove(node_id)
        visited.add(node_id)
        ordered.append(node_id)

    visit(str(target))
    return ordered


def build_stage_plan(prompt: dict[str, Any], assemble_node_id: str) -> list[dict[str, str]]:
    assemble_node_id = str(assemble_node_id)
    assemble = prompt.get(assemble_node_id)
    if not isinstance(assemble, dict) or assemble.get("class_type") != "H3RelayAssemble":
        raise ValueError("Select an H3 Relay Assemble node for staged execution.")
    ordered = _ancestor_order(prompt, assemble_node_id)
    h3 = [node_id for node_id in ordered
          if prompt[node_id].get("class_type") == "H3RelayGenerateShot"]
    finishing = [node_id for node_id in ordered
                 if prompt[node_id].get("class_type") in {
                     "H3RelayEnhanceShot", "H3RelayInterpolateShot"}]
    if not h3:
        raise ValueError("The selected Assemble node has no H3 shot ancestors.")
    if not finishing:
        raise ValueError("The selected Assemble node has no finishing-stage ancestors.")
    targets = h3 + finishing + [assemble_node_id]
    h3_indices = {node_id: index for index, node_id in enumerate(h3, 1)}
    ltx_nodes = [node_id for node_id in finishing
                 if prompt[node_id].get("class_type") == "H3RelayEnhanceShot"]
    rife_nodes = [node_id for node_id in finishing
                  if prompt[node_id].get("class_type") == "H3RelayInterpolateShot"]
    ltx_indices = {node_id: index for index, node_id in enumerate(ltx_nodes, 1)}
    rife_indices = {node_id: index for index, node_id in enumerate(rife_nodes, 1)}
    plan = [{
        "node_id": node_id,
        "kind": (STAGED_CLASSES.get(prompt[node_id].get("class_type"), "assemble")),
        "title": str((prompt[node_id].get("_meta") or {}).get("title")
                     or prompt[node_id].get("class_type") or node_id),
        "shot_index": str(
            h3_indices.get(node_id)
            or ltx_indices.get(node_id)
            or rife_indices.get(node_id)
            or len(h3)
        ),
    } for node_id in targets]
    ordered_positions = {node_id: index for index, node_id in enumerate(ordered)}
    for stage in plan:
        node = prompt[stage["node_id"]]
        position = ordered_positions[stage["node_id"]]
        stage["delivery_count"] = str(sum(
            1 for candidate in ordered[:position]
            if prompt[candidate].get("class_type") == "H3RelayInterpolateShot"
        ))
        if stage["kind"] == "ltx" and int(stage["shot_index"]) > 1:
            source = _link_source((node.get("inputs") or {}).get("previous_enhanced"))
            source_type = (prompt.get(source, {}).get("class_type") if source else None)
            stage["previous_stage"] = (
                "interpolated" if source_type == "H3RelayInterpolateShot" else "ltx"
            )
        if stage["kind"] == "assemble":
            stage["delivery_count"] = str(sum(
                1 for candidate in ordered
                if prompt[candidate].get("class_type") == "H3RelayInterpolateShot"
            ))
            source = _link_source((node.get("inputs") or {}).get("enhanced"))
            source_type = (prompt.get(source, {}).get("class_type") if source else None)
            stage["source_stage"] = (
                "interpolated" if source_type == "H3RelayInterpolateShot" else "ltx"
            )
    return plan


def _run_name(prompt: dict[str, Any], target: str) -> str | None:
    for node_id in _ancestor_order(prompt, target):
        node = prompt[node_id]
        if node.get("class_type") != "H3RelaySequenceStart":
            continue
        value = (node.get("inputs") or {}).get("run_name")
        return str(value) if isinstance(value, str) and value.strip() else None
    return None


def _restore_node_id(kind: str, shot_index: int) -> str:
    return "h3_relay_restore_%s_%04d" % (kind, int(shot_index))


def rewrite_stage_with_disk_restores(
    prompt: dict[str, Any], stage: dict[str, str], run_name: str,
) -> dict[str, Any]:
    prompt = copy.deepcopy(prompt)
    node = prompt[stage["node_id"]]
    shot_index = int(stage["shot_index"])

    def raw_restore(index: int) -> list[Any]:
        node_id = _restore_node_id("raw", index)
        prompt[node_id] = {
            "class_type": "H3RelayInternalRestoreRawSequence",
            "inputs": {"run_name": run_name, "shot_index": int(index)},
            "_meta": {"title": "Restore raw H3 sequence through shot %d" % index},
        }
        return [node_id, 0]

    def enhanced_restore(
        index: int, restore_stage: str, delivery_count: int,
    ) -> list[Any]:
        node_id = _restore_node_id(restore_stage, index)
        prompt[node_id] = {
            "class_type": "H3RelayInternalRestoreEnhanced",
            "inputs": {
                "run_name": run_name,
                "shot_index": int(index),
                "stage": restore_stage,
                "delivery_count": int(delivery_count),
            },
            "_meta": {
                "title": "Restore %s sequence through shot %d"
                         % (restore_stage, index),
            },
        }
        return [node_id, 0]

    inputs = node.setdefault("inputs", {})
    if stage["kind"] == "h3" and shot_index > 1:
        inputs["sequence"] = raw_restore(shot_index - 1)
    elif stage["kind"] == "ltx":
        inputs["sequence"] = raw_restore(shot_index)
        if shot_index > 1:
            inputs["previous_enhanced"] = enhanced_restore(
                shot_index - 1,
                str(stage.get("previous_stage") or "interpolated"),
                int(stage.get("delivery_count") or 0),
            )
        else:
            inputs.pop("previous_enhanced", None)
    elif stage["kind"] == "rife":
        inputs["enhanced"] = enhanced_restore(
            shot_index, "ltx", int(stage.get("delivery_count") or 0)
        )
    elif stage["kind"] == "assemble":
        inputs["enhanced"] = enhanced_restore(
            shot_index,
            str(stage.get("source_stage") or "interpolated"),
            int(stage.get("delivery_count") or 0),
        )
    return prompt


def _public_run(run: dict[str, Any]) -> dict[str, Any]:
    return {
        key: copy.deepcopy(value)
        for key, value in run.items()
        if key not in {"prompt", "workflow", "client_id"}
    }


async def _queue_partial(
    run: dict[str, Any], stage: dict[str, str], stage_index: int,
) -> str:
    server = PromptServer.instance
    prompt = rewrite_stage_with_disk_restores(
        run["prompt"], stage, run.get("run_name") or ""
    )
    if getattr(server, "node_replace_manager", None) is not None:
        server.node_replace_manager.apply_replacements(prompt)
    prompt_id = str(uuid.uuid4())
    valid = await execution.validate_prompt(prompt_id, prompt, [stage["node_id"]])
    if not valid[0]:
        raise ValueError(
            "Staged validation failed for %s: %s" % (stage["title"], valid[1]))
    number = server.number
    server.number += 1
    extra_data = {
        "client_id": run.get("client_id"),
        "extra_pnginfo": {"workflow": run.get("workflow")},
        "h3_relay_staged_run_id": run["run_id"],
        "h3_relay_stage_index": stage_index,
        "h3_relay_stage_count": len(run["stages"]),
        "h3_relay_stage_kind": stage["kind"],
    }
    server.prompt_queue.put((
        number, prompt_id, prompt, extra_data, valid[2], {},
    ))
    return prompt_id


async def _wait_for_history(run: dict[str, Any], prompt_id: str) -> dict[str, Any]:
    queue = PromptServer.instance.prompt_queue
    while True:
        if run.get("cancel_requested"):
            raise asyncio.CancelledError
        history = queue.get_history(prompt_id=prompt_id)
        if prompt_id in history:
            return history[prompt_id]
        await asyncio.sleep(0.25)


async def _release_memory(run: dict[str, Any], stage: dict[str, str]) -> None:
    mode = str(run.get("memory_mode") or "minimum_ram")
    if mode == "keep_models":
        return
    if mode == "balanced" and stage["kind"] not in {"rife", "assemble"}:
        return
    server = PromptServer.instance
    prompt_id = str(uuid.uuid4())
    prompt = {
        "release": {
            "class_type": "H3RelayInternalMemoryRelease",
            "inputs": {},
            "_meta": {"title": "H3 Relay staged memory release"},
        }
    }
    valid = await execution.validate_prompt(prompt_id, prompt, ["release"])
    if not valid[0]:
        raise RuntimeError("H3 Relay memory-release validation failed: %s" % valid[1])
    number = server.number
    server.number += 1
    server.prompt_queue.put((
        number,
        prompt_id,
        prompt,
        {"h3_relay_staged_run_id": run["run_id"]},
        valid[2],
        {},
    ))
    history = await _wait_for_history(run, prompt_id)
    if str((history.get("status") or {}).get("status_str") or "error") != "success":
        raise RuntimeError("H3 Relay staged memory release failed.")
    # The worker handles free_memory immediately after recording history.
    await asyncio.sleep(0.25)


async def _run_stages(run_id: str) -> None:
    run = _RUNS[run_id]
    run["status"] = "running"
    run["started_at"] = time.time()
    try:
        for index, stage in enumerate(run["stages"], 1):
            if run.get("cancel_requested"):
                raise asyncio.CancelledError
            run["current_stage"] = index
            run["stage"] = copy.deepcopy(stage)
            run["updated_at"] = time.time()
            prompt_id = await _queue_partial(run, stage, index)
            run["current_prompt_id"] = prompt_id
            run["jobs"].append({
                "index": index,
                "node_id": stage["node_id"],
                "kind": stage["kind"],
                "title": stage["title"],
                "prompt_id": prompt_id,
                "started_at": time.time(),
                "status": "queued",
            })
            history = await _wait_for_history(run, prompt_id)
            job = run["jobs"][-1]
            job["finished_at"] = time.time()
            job_status = str((history.get("status") or {}).get("status_str") or "error")
            job["status"] = job_status
            if job_status != "success":
                raise RuntimeError("%s failed; staged execution stopped." % stage["title"])
            await _release_memory(run, stage)
            run["updated_at"] = time.time()
        run["status"] = "success"
        run["finished_at"] = time.time()
        run["current_prompt_id"] = None
    except asyncio.CancelledError:
        run["status"] = "cancelled"
        run["finished_at"] = time.time()
        run["current_prompt_id"] = None
    except Exception as exc:
        _LOG.exception("H3 Relay staged run %s failed", run_id)
        run["status"] = "error"
        run["error"] = str(exc)
        run["finished_at"] = time.time()
        run["current_prompt_id"] = None
    finally:
        run["updated_at"] = time.time()
        _TASKS.pop(run_id, None)


async def _submit(request):
    try:
        body = await request.json()
        prompt = body.get("prompt")
        workflow = body.get("workflow")
        assemble_node_id = str(body.get("assemble_node_id") or "")
        if not isinstance(prompt, dict) or not prompt:
            raise ValueError("Staged execution requires the serialized API prompt.")
        stages = build_stage_plan(prompt, assemble_node_id)
        run_name = _run_name(prompt, assemble_node_id)
        if not run_name:
            raise ValueError(
                "Staged execution requires a literal Sequence Start run_name."
            )
        memory_mode = str(body.get("memory_mode") or "minimum_ram")
        if memory_mode not in {"minimum_ram", "balanced", "keep_models"}:
            raise ValueError("Unknown staged memory mode: %s" % memory_mode)
    except (TypeError, ValueError) as exc:
        return web.json_response({"error": str(exc)}, status=400)
    run_id = str(uuid.uuid4())
    run = {
        "run_id": run_id,
        "status": "queued",
        "created_at": time.time(),
        "updated_at": time.time(),
        "started_at": None,
        "finished_at": None,
        "current_stage": 0,
        "current_prompt_id": None,
        "stage": None,
        "stages": stages,
        "jobs": [],
        "memory_mode": memory_mode,
        "run_name": run_name,
        "cancel_requested": False,
        "error": None,
        "prompt": copy.deepcopy(prompt),
        "workflow": copy.deepcopy(workflow),
        "client_id": body.get("client_id"),
    }
    _RUNS[run_id] = run
    task = asyncio.create_task(_run_stages(run_id))
    _TASKS[run_id] = task
    return web.json_response(_public_run(run), status=202)


async def _status(request):
    run_id = str(request.match_info.get("run_id") or "")
    run = _RUNS.get(run_id)
    if run is None:
        return web.json_response({"error": "Unknown staged run."}, status=404)
    return web.json_response(_public_run(run))


async def _cancel(request):
    run_id = str(request.match_info.get("run_id") or "")
    run = _RUNS.get(run_id)
    if run is None:
        return web.json_response({"error": "Unknown staged run."}, status=404)
    run["cancel_requested"] = True
    run["updated_at"] = time.time()
    return web.json_response(_public_run(run))


def register_routes() -> bool:
    global _ROUTES_REGISTERED
    if _ROUTES_REGISTERED:
        return True
    if (PromptServer is None or web is None or execution is None
            or getattr(PromptServer, "instance", None) is None):
        return False
    routes = PromptServer.instance.routes
    routes.post("/h3_relay/staged")(_submit)
    routes.get("/h3_relay/staged/{run_id}")(_status)
    routes.post("/h3_relay/staged/{run_id}/cancel")(_cancel)
    _ROUTES_REGISTERED = True
    return True


__all__ = [
    "build_stage_plan", "register_routes", "rewrite_stage_with_disk_restores",
]
