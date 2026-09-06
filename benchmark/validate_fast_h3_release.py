#!/usr/bin/env python3
"""Validate a saved FastH3 workflow through fresh, staged ComfyUI execution.

Runs only against an already prepared, exclusively owned runtime. The source
workflow stays unchanged; all overrides, prompts and histories are retained.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import time
import urllib.request
import urllib.error
import uuid

from run_saved_workflow_benchmark import api_prompt_from_workflow


def request(endpoint, path, body=None):
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(
        endpoint + path, data=data,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as response:
            return json.load(response)
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"HTTP {exc.code}: {exc.read().decode()}") from exc


def save(root, name, value):
    with (root / name).open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2)
        handle.write("\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--workflow", type=pathlib.Path, required=True)
    parser.add_argument("--result-dir", type=pathlib.Path, required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--width", type=int)
    parser.add_argument("--height", type=int)
    parser.add_argument("--duration", type=float)
    parser.add_argument("--raw-first-shot", action="store_true", help="Prune a larger saved graph to its first raw shot for a standard-model regression")
    parser.add_argument("--resume", action="store_true", help="Record deliberate reuse of the named run's accepted checkpoints")
    args = parser.parse_args()
    root = args.result_dir
    root.mkdir(parents=True, exist_ok=False)
    queue = request(args.endpoint, "/queue")
    if queue["queue_running"] or queue["queue_pending"]:
        raise RuntimeError("Validation requires an empty, exclusively owned queue")
    schema = request(args.endpoint, "/object_info")
    if "SolAttnMiniMax" in schema:
        raise RuntimeError("Temporary SolAttnMiniMax node must be absent")
    if "H3RelayInternalFastH3VSA" not in schema:
        raise RuntimeError("Relay-owned VSA node is missing")
    save(root, "system-stats.json", request(args.endpoint, "/system_stats"))
    source = args.workflow.read_bytes()
    workflow = json.loads(source)
    prompt = api_prompt_from_workflow(workflow, args.run_name)
    if args.raw_first_shot:
        first = next(key for key, node in prompt.items()
                     if node["class_type"] == "H3RelayGenerateShot")
        needed = set()
        def visit(key):
            if key in needed:
                return
            needed.add(key)
            for value in prompt[key]["inputs"].values():
                if isinstance(value, list) and len(value) == 2 and str(value[0]) in prompt:
                    visit(str(value[0]))
        visit(first)
        prompt = {key: node for key, node in prompt.items() if key in needed}
    for node in prompt.values():
        if node["class_type"] == "H3RelaySequenceStart":
            for name in ("width", "height"):
                value = getattr(args, name)
                if value is not None:
                    node["inputs"][name] = value
        elif node["class_type"] == "H3RelayGenerateShot" and args.duration is not None:
            node["inputs"]["duration_seconds"] = args.duration
    targets = [key for key, node in prompt.items()
               if node["class_type"] in {"H3RelayAssemble", "H3RelayAssembleRaw"}]
    enhanced_targets = [key for key in targets
                        if prompt[key]["class_type"] == "H3RelayAssemble"]
    if enhanced_targets:
        targets = enhanced_targets
    if not targets:
        shots = [key for key, node in prompt.items()
                 if node["class_type"] == "H3RelayGenerateShot"]
        if len(shots) != 1:
            raise RuntimeError("A workflow with multiple shots needs an assembler")
        targets = ["release_validation_assemble"]
        prompt[targets[0]] = {
            "class_type": "H3RelayAssembleRaw",
            "inputs": {"sequence": [shots[0], 0],
                       "filename": args.run_name + "_raw", "audio_bitrate": 256},
        }
    if len(targets) != 1:
        raise RuntimeError("Select a workflow with exactly one final assembler")
    missing = sorted({n["class_type"] for n in prompt.values()} - schema.keys())
    if missing:
        raise RuntimeError("Missing node classes: " + ", ".join(missing))
    save(root, "source-workflow.json", workflow)
    save(root, "prompt.json", prompt)
    save(root, "manifest.json", {
        "run_name": args.run_name, "workflow_sha256": hashlib.sha256(source).hexdigest(),
        "prompt_sha256": hashlib.sha256(json.dumps(prompt, sort_keys=True).encode()).hexdigest(),
        "overrides": {k: getattr(args, k) for k in ("width", "height", "duration")},
        "temporary_attention_node_present": False,
        "raw_first_shot": args.raw_first_shot,
        "execution": "resume_accepted_checkpoints" if args.resume else "fresh_unique_namespace_with_disk_restored_stages",
    })
    if prompt[targets[0]]["class_type"] == "H3RelayAssembleRaw":
        submitted = request(args.endpoint, "/prompt", {
            "prompt": prompt, "client_id": str(uuid.uuid4()),
            "extra_data": {"extra_pnginfo": {"workflow": workflow}},
        })
        save(root, "submission.json", submitted)
        pid = submitted["prompt_id"]
        print(json.dumps({"prompt_id": pid}), flush=True)
        deadline = time.monotonic() + 7200
        while time.monotonic() < deadline:
            history = request(args.endpoint, "/history/" + pid)
            if pid in history:
                save(root, "history-" + pid + ".json", history[pid])
                save(root, "result.json", history[pid]["status"])
                return 0 if history[pid]["status"]["status_str"] == "success" else 1
            time.sleep(2)
        raise TimeoutError("Raw validation exceeded two hours; inspect owned prompt")
    submitted = request(args.endpoint, "/h3_relay/staged", {
        "prompt": prompt, "workflow": workflow, "assemble_node_id": targets[0],
        "client_id": str(uuid.uuid4()), "memory_mode": "balanced",
    })
    save(root, "submission.json", submitted)
    print(json.dumps({"staged_run_id": submitted["run_id"]}), flush=True)
    previous = None
    histories = set()
    deadline = time.monotonic() + 7200
    with (root / "events.jsonl").open("x", encoding="utf-8") as events:
        while time.monotonic() < deadline:
            state = request(args.endpoint, "/h3_relay/staged/" + submitted["run_id"])
            signature = (state["status"], state["current_stage"], state["current_prompt_id"])
            if signature != previous:
                events.write(json.dumps(state) + "\n")
                events.flush()
                print(json.dumps({"status": state["status"], "stage": state["current_stage"],
                                  "prompt_id": state["current_prompt_id"], "error": state["error"]}), flush=True)
                previous = signature
            for job in state["jobs"]:
                pid = job["prompt_id"]
                if pid not in histories:
                    history = request(args.endpoint, "/history/" + pid)
                    if pid in history:
                        save(root, "history-" + pid + ".json", history[pid])
                        histories.add(pid)
            if state["status"] in {"success", "error", "cancelled"}:
                save(root, "result.json", state)
                return 0 if state["status"] == "success" else 1
            time.sleep(2)
    raise TimeoutError("Staged validation exceeded two hours; inspect its owned run before recovery")


if __name__ == "__main__":
    raise SystemExit(main())
