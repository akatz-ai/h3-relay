"""Discover and restore H3 Chain Plan archives without loading prompt history."""

from __future__ import annotations

import json
import os
import re
from datetime import datetime, timezone
from typing import Any

try:
    from .asset_store import RunAssetStore
except ImportError:  # Standalone unit tests import this module without a package.
    from asset_store import RunAssetStore


PLAN_NODE_TYPE = "MiniMaxH3ChainPlan"
PLAN_WIDGET_NAMES = (
    "plan_json",
    "run_name",
    "generation_fingerprint",
    "width",
    "height",
    "context_length",
    "encode_mode",
    "anchor_mode",
    "crop",
    "audio_mode",
    "audio_context_length",
    "default_duration_seconds",
    "default_steps",
    "base_seed",
    "segment_crf",
    "video_blend_frames",
    "continuation_mode",
)

H3_CONTEXT_LENGTHS = (
    1, 5, 22, 39, 56, 73, 90, 107, 124,
    141, 158, 175, 192, 209, 226, 243,
)


def _safe_name(value: Any, fallback: str = "") -> str:
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value or "").strip())
    text = text.strip("._-")
    return (text or fallback)[:96]


def _read_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _json_string(value: Any) -> str | None:
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return None
        if not isinstance(parsed, (dict, list)):
            return None
        return json.dumps(parsed, ensure_ascii=False, indent=2)
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, indent=2)
    return None


def _restorable_widget_value(name: str, value: Any) -> Any:
    if name == "plan_json":
        return _json_string(value)
    if name == "base_seed" and isinstance(value, int):
        # aiohttp JSON would otherwise turn uint64 values above 2^53 into an
        # inexact JavaScript Number before the Plan widget sees them.
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    # API-prompt links are lists such as [node_id, output_slot]. Restoring a
    # widget must never replace or invent graph connections.
    return None


def _api_prompt_inputs(document: Any, run_name: str) -> dict[str, Any]:
    if not isinstance(document, dict):
        return {}
    candidates = []
    exact = []
    for node in document.values():
        if not isinstance(node, dict) or node.get("class_type") != PLAN_NODE_TYPE:
            continue
        inputs = node.get("inputs")
        if not isinstance(inputs, dict):
            continue
        candidates.append(inputs)
        if _safe_name(inputs.get("run_name")) == run_name:
            exact.append(inputs)
    selected = exact[0] if exact else (candidates[0] if len(candidates) == 1 else None)
    if selected is None:
        return {}
    restored = {}
    for name in PLAN_WIDGET_NAMES:
        if name not in selected:
            continue
        value = _restorable_widget_value(name, selected[name])
        if value is not None:
            restored[name] = value
    return restored


def _workflow_inputs(document: Any, run_name: str) -> dict[str, Any]:
    if not isinstance(document, dict) or not isinstance(document.get("nodes"), list):
        return {}
    candidates = []
    exact = []
    for node in document["nodes"]:
        if not isinstance(node, dict) or node.get("type") != PLAN_NODE_TYPE:
            continue
        widgets = node.get("widgets_values")
        if not isinstance(widgets, list):
            continue
        candidates.append(widgets)
        if len(widgets) > 1 and _safe_name(widgets[1]) == run_name:
            exact.append(widgets)
    selected = exact[0] if exact else (candidates[0] if len(candidates) == 1 else None)
    if selected is None or len(selected) < 15:
        return {}
    # widgets_values is positional and old workflows can predate fields added
    # near the front of the Plan. Refuse a shifted layout rather than applying
    # plausible-looking values to the wrong widgets; plan.json still restores
    # the effective scene configuration.
    if (selected[5] not in H3_CONTEXT_LENGTHS
            or selected[6] not in ("video", "frames")
            or selected[7] not in ("head", "before")
            or selected[8] not in ("disabled", "center")
            or selected[9] not in (
                "source_track", "generated_audio", "source_plus_timeline")):
        return {}
    if len(selected) > 16 and selected[16] not in ("guide", "masked_av"):
        return {}
    restored = {}
    for name, value in zip(PLAN_WIDGET_NAMES, selected):
        value = _restorable_widget_value(name, value)
        if value is not None:
            restored[name] = value
    # 0.3 workflows end at segment_crf. New widgets are appended, so old
    # positional values remain exact and receive their compatibility defaults.
    restored.setdefault("video_blend_frames", 0)
    restored.setdefault("continuation_mode", "guide")
    return restored


def _editor_plan(archive: dict[str, Any]) -> dict[str, Any]:
    existing = archive.get("editor_plan")
    if isinstance(existing, dict) and isinstance(existing.get("shots"), list):
        return existing
    shots = []
    for offset, shot in enumerate(archive.get("shots") or []):
        if not isinstance(shot, dict):
            continue
        value = {
            "id": shot.get("id", "clip_%04d" % (offset + 1)),
            "prompt": shot.get("scene_prompt", shot.get("prompt", "")),
        }
        if shot.get("raw_frames") is not None:
            value["length"] = int(shot["raw_frames"])
        if shot.get("steps") is not None:
            value["steps"] = int(shot["steps"])
        if shot.get("seed") is not None:
            value["seed"] = str(shot["seed"])
        if shot.get("continuation_mode") in ("guide", "masked_av"):
            value["continuation_mode"] = shot["continuation_mode"]
        shots.append(value)
    return {
        "prompt_prefix": archive.get("prompt_prefix", ""),
        "shots": shots,
    }


def _archive_inputs(archive: Any, run_name: str) -> dict[str, Any]:
    if not isinstance(archive, dict):
        return {}
    editor = _editor_plan(archive)
    if not editor.get("shots"):
        return {}
    restored: dict[str, Any] = {
        "plan_json": json.dumps(editor, ensure_ascii=False, indent=2),
        "run_name": run_name,
    }
    compatibility = archive.get("compatibility")
    if isinstance(compatibility, dict):
        for name in (
            "generation_fingerprint", "width", "height", "context_length",
            "encode_mode", "anchor_mode", "crop", "audio_mode",
            "audio_context_length", "segment_crf", "video_blend_frames",
            "continuation_mode",
        ):
            if name in compatibility:
                restored[name] = compatibility[name]
    if "segment_crf" in archive:
        restored["segment_crf"] = archive["segment_crf"]
    restored.setdefault("video_blend_frames", 0)
    restored.setdefault("continuation_mode", "guide")
    return restored


def _scene_count_from_plan(path: str) -> int | None:
    if not os.path.isfile(path):
        return None
    try:
        archive = _read_json(path)
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(archive, dict):
        return None
    editor = archive.get("editor_plan")
    shots = editor.get("shots") if isinstance(editor, dict) else archive.get("shots")
    return len(shots) if isinstance(shots, list) else None


def _iso_mtime(path: str) -> str:
    timestamp = os.path.getmtime(path)
    return datetime.fromtimestamp(timestamp, timezone.utc).isoformat(
        timespec="seconds").replace("+00:00", "Z")


class RunArchiveManager:
    def __init__(self, output_root: str, input_root: str | None = None):
        self.output_root = os.path.abspath(output_root)
        self.chains_root = os.path.join(self.output_root, "h3_chains")
        self.assets = RunAssetStore(self.output_root, input_root)

    def _run_dir(self, run_name: Any) -> tuple[str, str]:
        run = _safe_name(run_name)
        if not run:
            raise ValueError("A non-empty H3 chain run_name is required.")
        path = os.path.realpath(os.path.join(self.chains_root, run))
        root = os.path.realpath(self.output_root)
        if os.path.commonpath([root, path]) != root:
            raise ValueError("H3 run path escapes the output directory.")
        return path, run

    def list_runs(self) -> list[dict[str, Any]]:
        if not os.path.isdir(self.chains_root):
            return []
        runs = []
        for entry in os.scandir(self.chains_root):
            if not entry.is_dir(follow_symlinks=False):
                continue
            run_name = _safe_name(entry.name)
            if not run_name or run_name != entry.name:
                continue
            directory = entry.path
            plan_path = os.path.join(directory, "plan.json")
            api_path = os.path.join(directory, "api_prompt.json")
            workflow_path = os.path.join(directory, "workflow.json")
            archive_paths = [path for path in (plan_path, api_path, workflow_path)
                             if os.path.isfile(path)]
            checkpoint_dir = os.path.join(directory, "checkpoints")
            asset_manifest = os.path.join(directory, "references", "manifest.json")
            checkpoints = 0
            if os.path.isdir(checkpoint_dir):
                checkpoints = sum(
                    1 for name in os.listdir(checkpoint_dir)
                    if re.fullmatch(r"clip_\d{4}\.json", name))
            modified_paths = [directory]
            if os.path.isdir(checkpoint_dir):
                modified_paths.append(checkpoint_dir)
            if os.path.isfile(asset_manifest):
                modified_paths.append(asset_manifest)
            modified_paths.extend(archive_paths)
            newest = max(modified_paths, key=os.path.getmtime)
            try:
                asset_summary = self.assets.summary(run_name)
            except (OSError, ValueError, json.JSONDecodeError):
                asset_summary = {"asset_count": 0, "asset_bytes": 0}
            runs.append({
                "run_name": run_name,
                "modified_at": _iso_mtime(newest),
                "scene_count": _scene_count_from_plan(plan_path),
                "checkpoint_count": checkpoints,
                "restorable": bool(archive_paths),
                "archive_bytes": (
                    sum(os.path.getsize(path) for path in archive_paths)
                    + int(asset_summary.get("asset_bytes") or 0)),
                **asset_summary,
                "sources": {
                    "api_prompt": os.path.isfile(api_path),
                    "workflow": os.path.isfile(workflow_path),
                    "plan": os.path.isfile(plan_path),
                },
            })
        runs.sort(key=lambda item: item["modified_at"], reverse=True)
        return runs

    def load_run(self, run_name: Any) -> dict[str, Any]:
        directory, run = self._run_dir(run_name)
        if not os.path.isdir(directory):
            raise ValueError("H3 run %r does not exist." % run)

        restored: dict[str, Any] = {}
        sources = []
        warnings = []
        plan_path = os.path.join(directory, "plan.json")
        if os.path.isfile(plan_path):
            try:
                restored.update(_archive_inputs(_read_json(plan_path), run))
                if restored:
                    sources.append("plan.json")
            except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
                warnings.append("plan.json: %s" % exc)

        workflow_path = os.path.join(directory, "workflow.json")
        if os.path.isfile(workflow_path):
            try:
                values = _workflow_inputs(_read_json(workflow_path), run)
                if values:
                    restored.update(values)
                    sources.append("workflow.json")
            except (OSError, json.JSONDecodeError) as exc:
                warnings.append("workflow.json: %s" % exc)

        api_path = os.path.join(directory, "api_prompt.json")
        if os.path.isfile(api_path):
            try:
                values = _api_prompt_inputs(_read_json(api_path), run)
                if values:
                    restored.update(values)
                    sources.append("api_prompt.json")
            except (OSError, json.JSONDecodeError) as exc:
                warnings.append("api_prompt.json: %s" % exc)

        restored["run_name"] = run
        if "plan_json" not in restored:
            detail = "; ".join(warnings) if warnings else "no usable Plan archive was found"
            raise ValueError("Could not restore H3 run %r: %s." % (run, detail))
        parsed = json.loads(restored["plan_json"])
        shots = parsed.get("shots") if isinstance(parsed, dict) else parsed
        try:
            assets = self.assets.prepare_restore(run)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            assets = {"run_name": run, "bindings": [], "warnings": [str(exc)]}
        warnings.extend(assets.get("warnings") or [])
        return {
            "run_name": run,
            "scene_count": len(shots) if isinstance(shots, list) else None,
            "plan_inputs": restored,
            "sources": sources,
            "warnings": warnings,
            "assets": assets,
        }
