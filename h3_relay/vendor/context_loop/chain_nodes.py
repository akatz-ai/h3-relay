"""Disk-backed recursive clip chains specialized for MiniMax H3.

The visible graph contains one H3 sampling body.  Chain Start and Chain End
recursively clone that body with ComfyUI's GraphBuilder, carrying only the
previous clip's context tail and compact AV latent into the next iteration.
Each iteration is persisted before recursion, so a long chain can resume from
the first unfinished clip instead of starting over.

The recursive graph traversal is adapted from Ethanfel's SxCP loop nodes in
ComfyUI-Prompt-Builder, using the same ComfyUI expansion pattern with a single,
typed MiniMax chain state rather than arbitrary carry sockets.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import math
import os
import re
import secrets
import shutil
import subprocess
import sys
import time
import uuid
import wave
from collections import deque
from datetime import datetime
from fractions import Fraction
from typing import Any

import folder_paths

from ... import cache as relay_cache

try:
    import av
except ImportError:  # ComfyUI normally ships PyAV.
    av = None

try:
    import torch
except ImportError:  # ComfyUI always ships torch; keeps local imports clear.
    torch = None

try:
    import numpy as np
except ImportError:  # NumPy ships with ComfyUI and PyAV.
    np = None

try:
    from PIL import Image, PngImagePlugin
except ImportError:  # Pillow ships with ComfyUI.
    Image = PngImagePlugin = None

try:
    from safetensors.torch import load_file as _st_load, save_file as _st_save
except ImportError:
    _st_load = _st_save = None

try:
    from comfy_execution.graph_utils import GraphBuilder, ExecutionBlocker, is_link
except ImportError:
    GraphBuilder = None
    ExecutionBlocker = None

    def is_link(value):
        return isinstance(value, list) and len(value) == 2

try:
    from comfy_api.latest import InputImpl
except ImportError:
    InputImpl = None

try:
    from aiohttp import web
    from server import PromptServer
except ImportError:
    web = None
    PromptServer = None

from .nodes import (
    MiniMaxH3MotionContext,
    _claim_inline_patch_ownership,
    _prepare_native_guide_conditioning,
    _resize,
    _streams_from_latent,
)
from .prompt_history import PromptHistoryStore
from .prompt_optimizer import optimize_prompt_payload
from .run_manager import RunArchiveManager
from .asset_store import MAX_ASSET_BINDINGS, RunAssetStore


_LOG = logging.getLogger("minimax_h3_context_loop.chain")

FPS = 24
PLAN_VERSION = 2
MAX_SHOTS = 128
MAX_SEED = 0xFFFFFFFFFFFFFFFF
MAX_H3_FRAMES = 3592  # largest 17k+5 value accepted by H3's 3600-frame socket
GUIDE_CONTEXT_LENGTHS = (
    1, 5, 22, 39, 56, 73, 90, 107, 124,
    141, 158, 175, 192, 209, 226, 243,
)
SLIDING_HISTORY_CONTEXT_LENGTHS = tuple(range(18, 244, 17))
H3_CONTEXT_LENGTHS = tuple(sorted(set(
    GUIDE_CONTEXT_LENGTHS + SLIDING_HISTORY_CONTEXT_LENGTHS)))
AUDIO_MODES = ("source_track", "generated_audio", "source_plus_timeline")
CONTINUATION_MODES = ("guide", "sliding_history", "masked_av")
REFERENCE_AUDIO_TIMELINE_MODES = ("standalone", "source_timeline")

PLAN_TYPE = "H3_CHAIN_PLAN"
STATE_TYPE = "H3_CHAIN_STATE"
FLOW_TYPE = "H3_CHAIN_FLOW"
SEGMENT_TYPE = "H3_CHAIN_SEGMENT"
MANIFEST_TYPE = "H3_CHAIN_MANIFEST"
EXTERNAL_CONTEXT_TYPE = "H3_CHAIN_EXTERNAL_CONTEXT"
REFERENCE_SCHEDULE_TYPE = "H3_REFERENCE_SCHEDULE"
TAGGED_REFERENCE_TYPE = "H3_TAGGED_REFERENCES"
REFERENCE_SCHEDULE_VERSION = 1

_PENDING_REVIEWS: dict[str, dict[str, Any]] = {}
_PENDING_FINAL_REVIEW_PREVIEWS: dict[
    tuple[str, str], dict[str, Any]
] = {}


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"))


def _fingerprint(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _safe_name(value: str, fallback: str = "chain") -> str:
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value or "").strip())
    text = text.strip("._-")
    return (text or fallback)[:96]


def _expand_filename_date(value: str, now: datetime | None = None) -> str:
    """Expand ComfyUI-style date tokens before filename sanitization."""
    current = now or datetime.now()
    replacements = {
        "yyyy": "%Y", "yy": "%y", "MM": "%m", "dd": "%d",
        "HH": "%H", "hh": "%I", "mm": "%M", "ss": "%S",
    }

    def replace_date(match: re.Match[str]) -> str:
        pattern = match.group(1)
        strftime_pattern = re.sub(
            r"yyyy|yy|MM|dd|HH|hh|mm|ss",
            lambda token: replacements[token.group(0)],
            pattern,
        )
        return current.strftime(strftime_pattern)

    text = re.sub(r"%date:([^%]+)%", replace_date, str(value or ""))
    simple_tokens = {
        "%year%": "%Y", "%month%": "%m", "%day%": "%d",
        "%hour%": "%H", "%minute%": "%M", "%second%": "%S",
    }
    for token, pattern in simple_tokens.items():
        text = text.replace(token, current.strftime(pattern))
    return text


def _available_versioned_path(path: str) -> str:
    """Return path unchanged when free, otherwise add a numeric version."""
    if not os.path.exists(path):
        return path
    root, extension = os.path.splitext(path)
    version = 1
    while True:
        candidate = "%s_%03d%s" % (root, version, extension)
        if not os.path.exists(candidate):
            return candidate
        version += 1


def _prompt_text(value: Any, label: str) -> str:
    """Normalize a prompt string or a human-editable JSON array of lines."""
    if isinstance(value, list):
        if not all(isinstance(line, str) for line in value):
            raise ValueError("%s line arrays may contain only strings." % label)
        return "\n".join(value).strip()
    return str(value or "").strip()


def _h3_frame_length(seconds: float) -> int:
    """Round a duration up to H3's valid 17k+5 frame grid."""
    seconds = float(seconds)
    if not math.isfinite(seconds) or seconds <= 0:
        raise ValueError("H3 shot duration must be a finite positive number.")
    # Subtract a tiny tolerance so an exactly frame-aligned decimal does not
    # jump a frame because of binary floating-point representation.
    requested = max(5, int(math.ceil(seconds * FPS - 1e-9)))
    length = requested + (5 - requested % 17) % 17
    if length > MAX_H3_FRAMES:
        raise ValueError(
            "H3 shot duration %.6fs rounds to %d frames; the largest valid "
            "17k+5 length is %d frames (%.6fs)." %
            (seconds, length, MAX_H3_FRAMES, MAX_H3_FRAMES / float(FPS)))
    return length


def _validate_h3_length(length: Any, label: str) -> int:
    length = int(length)
    if length < 5 or length > MAX_H3_FRAMES or length % 17 != 5:
        raise ValueError(
            "%s must be an H3-valid frame length between 5 and %d "
            "with length %% 17 == 5; got %d." %
            (label, MAX_H3_FRAMES, length))
    return length


def _parse_scene_range(value: Any, total: int,
                       fallback_start: int) -> tuple[int, int]:
    """Parse one inclusive, contiguous scene selection.

    Disjoint selections are deliberately rejected: every H3 scene depends on
    its immediate predecessor, so skipping a scene inside a render selection
    would either break continuity or silently reuse an invalid checkpoint.
    """
    total = int(total)
    fallback_start = int(fallback_start)
    text = str(value or "").strip()
    if not text:
        start, end = fallback_start, total
    else:
        compact = re.sub(r"\s+", "", text)
        if "," in compact:
            raise ValueError(
                "scene_range supports one contiguous inclusive range only, "
                "such as '3' or '3:8'. Comma selections are not safe for a "
                "seamless chain.")
        match = re.fullmatch(r"(\d+)(?::(\d+))?", compact)
        if match is None:
            raise ValueError(
                "scene_range must be blank, one scene like '3', or one "
                "inclusive range like '3:8'.")
        start = int(match.group(1))
        end = int(match.group(2) or start)
    if start < 1 or start > total:
        raise ValueError("scene_range start must be between 1 and %d." % total)
    if end < start:
        raise ValueError("scene_range end must be greater than or equal to start.")
    if end > total:
        raise ValueError("scene_range end must be between %d and %d." %
                         (start, total))
    return start, end


_REFERENCE_TAG_RE = re.compile(r"[A-Za-z][A-Za-z0-9_-]{0,63}")
_REFERENCE_ALIAS_RE = re.compile(
    r"(?<![A-Za-z0-9_])@([A-Za-z][A-Za-z0-9_-]{0,63})")
REFERENCE_COMPLIANCE_MODES = ("strict", "soft", "disabled")
REFERENCE_VIDEO_TIMELINE_MODES = (
    "restart_each_scene",
    "sequential",
)


def _reference_compliance_mode(value: Any) -> str:
    # Compatibility with the short-lived BOOLEAN form: True was strict and
    # False was the warning-only behavior now named soft.
    if isinstance(value, bool):
        return "strict" if value else "soft"
    mode = str(value or "strict").strip().lower()
    if mode not in REFERENCE_COMPLIANCE_MODES:
        raise ValueError(
            "Reference prompt compliance must be strict, soft, or disabled; "
            "got %r." % value)
    return mode


def _downstream_reference_compliance(
        dynprompt: Any, unique_id: Any) -> str:
    """Find the strictest Scheduled Ref2VA policy consuming this schedule.

    Schedule-builder nodes execute before the wrapper. Looking downstream is
    therefore the only way an upstream missing source_audio_slice can honor a
    disabled policy without changing ComfyUI's execution semantics.
    """
    if dynprompt is None or unique_id is None:
        return "strict"
    try:
        node_ids = list(dynprompt.all_node_ids())
    except (AttributeError, TypeError):
        return "strict"
    queue = [unique_id]
    visited = {str(unique_id)}
    modes = []
    while queue:
        parent = queue.pop(0)
        for node_id in node_ids:
            if str(node_id) in visited:
                continue
            try:
                node = dynprompt.get_node(node_id)
            except Exception:
                continue
            inputs = node.get("inputs") if isinstance(node, dict) else None
            if not isinstance(inputs, dict) or not any(
                    isinstance(value, list) and len(value) == 2
                    and str(value[0]) == str(parent)
                    for value in inputs.values()):
                continue
            visited.add(str(node_id))
            if node.get("class_type") in (
                    "MiniMaxH3ScheduledReferenceToVideo",
                    "MiniMaxH3TaggedReferenceToVideo"):
                try:
                    modes.append(_reference_compliance_mode(
                        inputs.get(
                            "prompt_compliance",
                            inputs.get("reference_policy", "strict"))))
                except ValueError:
                    modes.append("strict")
            else:
                queue.append(node_id)
    if not modes or "strict" in modes:
        return "strict"
    return "soft" if "soft" in modes else "disabled"


def _skipped_reference_result(
        previous: Any, label: str, reason: Any) -> tuple[Any, str, str]:
    try:
        schedule = _make_reference_schedule(
            _reference_schedule_entries(previous))
    except (TypeError, ValueError):
        schedule = _make_reference_schedule([])
    message = "%s skipped because compliance is disabled: %s" % (
        label, str(reason))
    _LOG.warning("H3 scheduled-reference warning: %s", message)
    return schedule, schedule["fingerprint"], message


def _skipped_tagged_reference_result(
        previous: Any, label: str, reason: Any) -> tuple[Any, str, str]:
    try:
        references = _make_tagged_references(
            _tagged_reference_entries(previous))
    except (TypeError, ValueError):
        references = _make_tagged_references([])
    message = "%s skipped because reference policy is disabled: %s" % (
        label, str(reason))
    _LOG.warning("H3 tagged-reference warning: %s", message)
    return references, references["fingerprint"], message


def _normalize_reference_tag(value: Any, label: str) -> str:
    tag = str(value or "").strip()
    if tag.startswith("@"):
        tag = tag[1:]
    if _REFERENCE_TAG_RE.fullmatch(tag) is None:
        raise ValueError(
            "%s must be a stable tag such as 'hero_face' or '@hero-face'." %
            label)
    return tag


def _parse_reference_selector(
        value: Any, total: int | None = None) -> tuple[tuple[int, int], ...]:
    """Parse a disjoint, one-based scene selector and merge overlaps."""
    text = re.sub(r"\s+", "", str(value or "")).lower()
    if text in ("", "*", "all"):
        return ()
    ranges = []
    for token in text.split(","):
        match = re.fullmatch(r"(\d+)(?::(\d+))?", token)
        if match is None:
            raise ValueError(
                "Reference scenes must be blank/all, one scene like '3', "
                "or comma-separated inclusive ranges like '1,3,5:8'.")
        start = int(match.group(1))
        end = int(match.group(2) or start)
        if start < 1:
            raise ValueError("Reference scene numbers start at 1.")
        if end < start:
            raise ValueError(
                "Reference range %s ends before it starts." % token)
        if total is not None and end > int(total):
            raise ValueError(
                "Reference range %s exceeds this plan's %d scenes." %
                (token, int(total)))
        ranges.append((start, end))
    merged = []
    for start, end in sorted(ranges):
        if merged and start <= merged[-1][1] + 1:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return tuple(merged)


def _reference_selector_text(ranges: Any) -> str:
    if not ranges:
        return "all"
    return ",".join(
        str(start) if int(start) == int(end) else "%d:%d" % (start, end)
        for start, end in ranges)


def _reference_is_active(entry: dict[str, Any], scene: int) -> bool:
    ranges = entry.get("ranges") or ()
    return not ranges or any(
        int(start) <= int(scene) <= int(end) for start, end in ranges)


def _reference_entry_contract(entry: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "kind", "tag", "scenes", "content_hash", "audio_tag",
        "audio_hash",
    )
    contract = {key: entry[key] for key in keys if key in entry}
    # Keep old schedules bit-identical. Timeline metadata enters the resume
    # contract only when a reference opts into non-default playback.
    default_timeline = (
        "standalone" if entry.get("kind") == "audio"
        else "restart_each_scene")
    timeline_mode = str(entry.get("timeline_mode") or default_timeline)
    if timeline_mode != default_timeline:
        contract["timeline_mode"] = timeline_mode
    if bool(entry.get("align_audio_reference")):
        contract["align_audio_reference"] = True
    if str(entry.get("activation") or "schedule") != "schedule":
        contract["activation"] = str(entry["activation"])
    return contract


def _reference_schedule_entries(value: Any) -> list[dict[str, Any]]:
    if value is None:
        return []
    if (not isinstance(value, dict) or
            int(value.get("version", -1)) != REFERENCE_SCHEDULE_VERSION or
            not isinstance(value.get("entries"), list)):
        raise ValueError(
            "Scheduled references must come from this pack's Picture, Video, "
            "or Audio Schedule nodes.")
    return list(value["entries"])


def _reference_entry_tags(entry: dict[str, Any]) -> tuple[str, ...]:
    tags = [str(entry["tag"])]
    if entry.get("audio_tag"):
        tags.append(str(entry["audio_tag"]))
    return tuple(tags)


def _make_reference_schedule(
        entries: list[dict[str, Any]]) -> dict[str, Any]:
    contracts = [_reference_entry_contract(entry) for entry in entries]
    return {
        "version": REFERENCE_SCHEDULE_VERSION,
        "entries": entries,
        "fingerprint": _fingerprint({
            "version": REFERENCE_SCHEDULE_VERSION,
            "entries": contracts,
        }),
    }


def _tagged_reference_entries(value: Any) -> list[dict[str, Any]]:
    entries = _reference_schedule_entries(value)
    if not isinstance(value, dict) or value.get("activation") != "prompt":
        raise ValueError(
            "Tagged references must come from this pack's Tagged Picture, "
            "Video, or Audio Ref nodes.")
    if any(entry.get("activation") != "prompt" for entry in entries):
        raise ValueError(
            "A tagged-reference chain cannot contain legacy scheduled "
            "entries. Use one node family or the other.")
    return entries


def _make_tagged_references(
        entries: list[dict[str, Any]]) -> dict[str, Any]:
    result = _make_reference_schedule(entries)
    result["activation"] = "prompt"
    return result


def _append_scheduled_reference(
        previous: Any, *, kind: str, tag: Any, scenes: Any,
        value: Any, content_hash: str, audio: Any = None,
        audio_tag: Any = "", audio_hash: str = "",
        compliance_mode: str = "strict",
        timeline_mode: Any = None,
        align_audio_reference: Any = False) -> dict[str, Any]:
    compliance = _reference_compliance_mode(compliance_mode)
    entries = _reference_schedule_entries(previous)
    try:
        normalized_tag = _normalize_reference_tag(tag, "Reference tag")
    except ValueError as exc:
        if compliance != "disabled":
            raise
        normalized_tag = "reference_%s" % str(content_hash)[:12]
        _LOG.warning(
            "H3 scheduled-reference warning: %s Using internal tag @%s.",
            exc, normalized_tag)
    try:
        ranges = _parse_reference_selector(scenes)
    except ValueError as exc:
        if compliance != "disabled":
            raise
        ranges = ()
        _LOG.warning(
            "H3 scheduled-reference warning: %s Treating the reference as "
            "active in every scene.", exc)
    entry = {
        "kind": str(kind),
        "tag": normalized_tag,
        "scenes": _reference_selector_text(ranges),
        "ranges": ranges,
        "value": value,
        "content_hash": str(content_hash),
    }
    if kind == "video":
        normalized_timeline = str(
            timeline_mode or "restart_each_scene").strip().lower()
        if normalized_timeline not in REFERENCE_VIDEO_TIMELINE_MODES:
            raise ValueError(
                "Scheduled video timeline_mode must be one of %s." %
                (REFERENCE_VIDEO_TIMELINE_MODES,))
        entry["timeline_mode"] = normalized_timeline
    elif kind == "audio":
        normalized_timeline = str(
            timeline_mode or "standalone").strip().lower()
        if normalized_timeline not in REFERENCE_AUDIO_TIMELINE_MODES:
            raise ValueError(
                "Scheduled audio timeline_mode must be one of %s." %
                (REFERENCE_AUDIO_TIMELINE_MODES,))
        entry["timeline_mode"] = normalized_timeline
        entry["align_audio_reference"] = bool(align_audio_reference)
    if audio is not None:
        try:
            normalized_audio_tag = _normalize_reference_tag(
                audio_tag or (normalized_tag + "_audio"),
                "Paired video-audio tag")
        except ValueError as exc:
            if compliance != "disabled":
                raise
            normalized_audio_tag = "%s_audio" % normalized_tag
            _LOG.warning(
                "H3 scheduled-reference warning: %s Using internal tag @%s.",
                exc, normalized_audio_tag)
        entry.update({
            "audio": audio,
            "audio_tag": normalized_audio_tag,
            "audio_hash": str(audio_hash),
        })

    existing_tags = {
        alias for existing in entries
        for alias in _reference_entry_tags(existing)
    }
    new_tags = _reference_entry_tags(entry)
    if compliance != "disabled" and len(set(new_tags)) != len(new_tags):
        raise ValueError(
            "A scheduled video's @tag and paired @audio_tag must be "
            "different.")
    duplicates = existing_tags.intersection(new_tags)
    if compliance != "disabled" and duplicates:
        duplicate = sorted(duplicates)[0]
        raise ValueError(
            "Scheduled reference tag @%s is already in this chain." %
            duplicate)
    return _make_reference_schedule(entries + [entry])


def _append_tagged_reference(
        previous: Any, *, kind: str, tag: Any, value: Any,
        content_hash: str, audio: Any = None, audio_tag: Any = "",
        audio_hash: str = "", compliance_mode: str = "strict",
        timeline_mode: Any = None,
        align_audio_reference: Any = False) -> dict[str, Any]:
    if previous is None:
        seed = None
    else:
        seed = _make_reference_schedule(_tagged_reference_entries(previous))
    collection = _append_scheduled_reference(
        seed, kind=kind, tag=tag, scenes="all", value=value,
        content_hash=content_hash, audio=audio, audio_tag=audio_tag,
        audio_hash=audio_hash, compliance_mode=compliance_mode,
        timeline_mode=timeline_mode,
        align_audio_reference=align_audio_reference)
    entries = collection["entries"]
    entries[-1]["activation"] = "prompt"
    return _make_tagged_references(entries)


def _scheduled_video_reference_slice(
        entry: dict[str, Any], state: Any, scene: int, scene_count: int,
        length: int) -> tuple[Any, Any, str]:
    """Resolve one active video ref without silently restarting its timeline."""
    timeline_mode = str(
        entry.get("timeline_mode") or "restart_each_scene")
    video = entry["value"]
    audio = entry.get("audio")
    if timeline_mode == "restart_each_scene":
        return video, audio, ""
    if timeline_mode != "sequential":
        raise ValueError(
            "Scheduled video @%s has unknown timeline mode %r." %
            (entry.get("tag", "video"), timeline_mode))
    if not isinstance(state, dict) or not isinstance(state.get("plan"), dict):
        raise ValueError(
            "Sequential scheduled video @%s requires Current Shot state on "
            "Scheduled Ref2VA's state input." % entry.get("tag", "video"))
    if int(state.get("index", -1)) != int(scene):
        raise ValueError(
            "Scheduled Ref2VA received scene %d but Current Shot state is at "
            "scene %s." % (int(scene), state.get("index")))
    shots = state["plan"].get("shots")
    if not isinstance(shots, list) or len(shots) != int(scene_count):
        raise ValueError(
            "Sequential scheduled video requires state from the same %d-scene "
            "Plan connected to Scheduled Ref2VA." % int(scene_count))
    current = shots[int(scene) - 1]
    if int(current.get("raw_frames", -1)) != int(length):
        raise ValueError(
            "Scheduled Ref2VA length %d does not match scene %d's %s raw "
            "frames in Current Shot state." %
            (int(length), int(scene), current.get("raw_frames")))

    ranges = entry.get("ranges") or ()
    if entry.get("activation") == "prompt":
        origin_scene = next((
            index for index, shot in enumerate(shots, 1)
            if set(_REFERENCE_ALIAS_RE.findall(_prompt_text(
                shot.get("prompt", ""), "Scene %d prompt" % index
            ))).intersection(_reference_entry_tags(entry))
        ), int(scene))
    else:
        origin_scene = int(ranges[0][0]) if ranges else 1
    if origin_scene < 1 or origin_scene > len(shots):
        raise ValueError(
            "Sequential scheduled video @%s has an invalid first active "
            "scene %d." % (entry.get("tag", "video"), origin_scene))
    origin_start = int(shots[origin_scene - 1]["generation_start_frame"])
    current_start = int(current["generation_start_frame"])
    source_start = current_start - origin_start
    source_end = source_start + int(length)
    if source_start < 0:
        raise ValueError(
            "Sequential scheduled video @%s resolved a negative source "
            "window for scene %d." %
            (entry.get("tag", "video"), int(scene)))
    available = int(video.shape[0])
    if source_end > available:
        raise ValueError(
            "Sequential scheduled video @%s needs source frames %d:%d for "
            "scene %d, but the 24 fps reference contains only %d frames. "
            "Supply a longer reference, shorten the Plan, or use "
            "restart_each_scene." %
            (entry.get("tag", "video"), source_start, source_end,
             int(scene), available))
    sliced_video = video[source_start:source_end]
    sliced_audio = None
    if audio is not None:
        waveform, sample_rate = _validate_audio(
            audio, "Sequential scheduled video @%s audio" % entry.get(
                "tag", "video"))
        sample_start = int(round(source_start / float(FPS) * sample_rate))
        sample_end = int(round(source_end / float(FPS) * sample_rate))
        available_samples = int(waveform.shape[-1])
        if sample_end > available_samples:
            raise ValueError(
                "Sequential scheduled video @%s needs paired-audio samples "
                "%d:%d for scene %d, but the soundtrack contains only %d "
                "samples at %d Hz." %
                (entry.get("tag", "video"), sample_start, sample_end,
                 int(scene), available_samples, sample_rate))
        sliced_audio = {
            "waveform": waveform[..., sample_start:sample_end],
            "sample_rate": sample_rate,
        }
    detail = "@%s sequential frames %d:%d (origin scene %d)" % (
        entry.get("tag", "video"), source_start, source_end, origin_scene)
    return sliced_video, sliced_audio, detail


def _tagged_audio_reference_value(
        entry: dict[str, Any], state: Any, scene: int, scene_count: int,
        length: int) -> tuple[Any, str]:
    """Resolve a tagged standalone clip or an exact Plan-timeline slice."""
    timeline_mode = str(entry.get("timeline_mode") or "standalone")
    audio = entry["value"]
    if timeline_mode == "standalone":
        return audio, ""
    if timeline_mode != "source_timeline":
        raise ValueError(
            "Tagged audio @%s has unknown timeline mode %r." %
            (entry.get("tag", "audio"), timeline_mode))
    if not isinstance(state, dict) or not isinstance(state.get("plan"), dict):
        raise ValueError(
            "Tagged audio @%s uses source_timeline and requires Current Shot "
            "state on Tagged Ref2VA's state input. Keep the full source track "
            "connected to Tagged Audio Ref; do not connect "
            "source_audio_slice there because its fingerprint-to-Plan link "
            "would create a cycle." % entry.get("tag", "audio"))
    if int(state.get("index", -1)) != int(scene):
        raise ValueError(
            "Tagged Ref2VA received scene %d but Current Shot state is at "
            "scene %s." % (int(scene), state.get("index")))
    plan = state["plan"]
    shots = plan.get("shots")
    if not isinstance(shots, list) or len(shots) != int(scene_count):
        raise ValueError(
            "Tagged audio @%s source_timeline requires state from the same "
            "%d-scene Plan connected to Tagged Ref2VA." %
            (entry.get("tag", "audio"), int(scene_count)))
    current = shots[int(scene) - 1]
    if int(current.get("raw_frames", -1)) != int(length):
        raise ValueError(
            "Tagged Ref2VA length %d does not match scene %d's %s raw frames "
            "in Current Shot state." %
            (int(length), int(scene), current.get("raw_frames")))
    compatibility = plan.get("compatibility")
    if not isinstance(compatibility, dict) or compatibility.get(
            "audio_mode") not in ("source_track", "source_plus_timeline"):
        raise ValueError(
            "Tagged audio @%s source_timeline requires the Plan audio mode "
            "source_track or source_plus_timeline." %
            entry.get("tag", "audio"))
    expected_hash = str(compatibility.get("source_audio_hash") or "")
    entry_hash = str(entry.get("content_hash") or "")
    if not expected_hash or expected_hash == "none":
        raise ValueError(
            "Tagged audio @%s source_timeline has no Loop Start source-audio "
            "fingerprint to validate." % entry.get("tag", "audio"))
    if not entry_hash or entry_hash != expected_hash:
        raise ValueError(
            "Tagged audio @%s source_timeline received a different full "
            "source track than H3 Chain Loop Start. Wire the same Load Audio "
            "output to both nodes." % entry.get("tag", "audio"))
    external_lead = int(current.get("external_context_frames", 0))
    if int(scene) == 1 and external_lead > 0:
        sliced = _slice_audio_after_external_context(
            audio, state.get("previous_audio"), int(current["raw_frames"]),
            external_lead, pad_silence=bool(compatibility.get(
                "source_audio_silent_padding")))
    else:
        sliced = _slice_audio(
            audio, current["audio_start_seconds"],
            current["audio_duration_seconds"],
            pad_silence=bool(compatibility.get(
                "source_audio_silent_padding")))
    if bool(entry.get("align_audio_reference")):
        sliced, alignment = _align_audio_reference_to_h3_grid(
            sliced, int(length))
    else:
        alignment = "frame-exact"
    start = float(current["audio_start_seconds"])
    end = start + float(current["audio_duration_seconds"])
    detail = "@%s source timeline %.3f..%.3fs; %s" % (
        entry.get("tag", "audio"), start, end, alignment)
    return sliced, detail


def _active_reference_bindings(
        schedule: Any, scene: int, scene_count: int,
        compliance_mode: str = "strict",
        warnings: list[str] | None = None,
        activation_tags: set[str] | None = None) -> dict[str, Any]:
    scene, scene_count = int(scene), int(scene_count)
    mode = _reference_compliance_mode(compliance_mode)
    warnings = warnings if warnings is not None else []
    if scene_count < 1 or scene < 1 or scene > scene_count:
        raise ValueError(
            "Scheduled Ref2VA scene index must be between 1 and %d; got %d." %
            (scene_count, scene))
    if activation_tags is None:
        entries = _reference_schedule_entries(schedule)
        for entry in entries:
            try:
                _parse_reference_selector(
                    entry.get("scenes", "all"), scene_count)
            except ValueError as exc:
                if mode != "disabled":
                    raise
                warnings.append(str(exc))
                _LOG.warning("H3 scheduled-reference warning: %s", exc)
        active = [
            entry for entry in entries if _reference_is_active(entry, scene)]
    else:
        entries = _tagged_reference_entries(schedule)
        active = [
            entry for entry in entries
            if activation_tags.intersection(_reference_entry_tags(entry))]
    pictures = [entry for entry in active if entry.get("kind") == "picture"]
    videos = [entry for entry in active if entry.get("kind") == "video"]
    audios = [entry for entry in active if entry.get("kind") == "audio"]
    unknown = [entry.get("kind") for entry in active
               if entry.get("kind") not in ("picture", "video", "audio")]
    if unknown:
        message = "Unknown scheduled reference kind %r." % unknown[0]
        if mode != "disabled":
            raise ValueError(message)
        warnings.append(message)
        _LOG.warning("H3 scheduled-reference warning: %s", message)
    if len(pictures) > 9:
        message = (
            "Scene %d activates %d pictures; only the first 9 were kept." %
            (scene, len(pictures)))
        if mode != "disabled":
            raise ValueError(
                "Scene %d activates %d pictures; stock H3 Ref2VA supports 9." %
                (scene, len(pictures)))
        warnings.append(message)
        _LOG.warning("H3 scheduled-reference warning: %s", message)
        pictures = pictures[:9]
    if len(videos) > 3:
        message = (
            "Scene %d activates %d videos; only the first 3 were kept." %
            (scene, len(videos)))
        if mode != "disabled":
            raise ValueError(
                "Scene %d activates %d videos; stock H3 Ref2VA supports 3." %
                (scene, len(videos)))
        warnings.append(message)
        _LOG.warning("H3 scheduled-reference warning: %s", message)
        videos = videos[:3]
    if len(audios) > 3:
        message = (
            "Scene %d activates %d standalone audios; only the first 3 "
            "were kept." % (scene, len(audios)))
        if mode != "disabled":
            raise ValueError(
                "Scene %d activates %d standalone audios; stock H3 Ref2VA "
                "supports 3." % (scene, len(audios)))
        warnings.append(message)
        _LOG.warning("H3 scheduled-reference warning: %s", message)
        audios = audios[:3]

    aliases: dict[str, str] = {}
    presentation: list[dict[str, Any]] = []
    for ordinal, entry in enumerate(pictures, 1):
        label = "<Picture %d>" % ordinal
        aliases[entry["tag"]] = label
        presentation.append({
            "entry": entry, "role": "picture", "tag": entry["tag"],
            "label": label,
        })

    audio_ordinal = 0
    for ordinal, entry in enumerate(videos, 1):
        if entry.get("audio") is not None:
            audio_ordinal += 1
            audio_label = "<Audio %d>" % audio_ordinal
            aliases[entry["audio_tag"]] = audio_label
            presentation.append({
                "entry": entry, "role": "audio",
                "tag": entry["audio_tag"], "label": audio_label,
            })
        video_label = "<Video %d>" % ordinal
        aliases[entry["tag"]] = video_label
        presentation.append({
            "entry": entry, "role": "video", "tag": entry["tag"],
            "label": video_label,
        })
    for entry in audios:
        audio_ordinal += 1
        label = "<Audio %d>" % audio_ordinal
        aliases[entry["tag"]] = label
        presentation.append({
            "entry": entry, "role": "audio", "tag": entry["tag"],
            "label": label,
        })
    return {
        "pictures": pictures,
        "videos": videos,
        "audios": audios,
        "aliases": aliases,
        "presentation": presentation,
        "all_tags": {
            alias for entry in entries for alias in _reference_entry_tags(entry)
        },
    }


def _replace_reference_aliases(
        text: str, bindings: dict[str, Any], scene: int,
        compliance_mode: str = "strict",
        warnings: list[str] | None = None) -> str:
    aliases = bindings["aliases"]
    all_tags = bindings["all_tags"]
    warnings = warnings if warnings is not None else []
    mode = _reference_compliance_mode(compliance_mode)

    if mode == "disabled":
        return str(text)

    def compliance_error(message: str, original: str) -> str:
        if mode == "strict":
            raise ValueError(message)
        if message not in warnings:
            warnings.append(message)
            _LOG.warning(
                "H3 scheduled-reference prompt compliance warning: %s "
                "The unresolved tag is being passed to H3 unchanged.",
                message)
        return original

    def replace(match):
        tag = match.group(1)
        if tag in aliases:
            return aliases[tag]
        if tag in all_tags:
            return compliance_error(
                "Scheduled reference @%s is not active in scene %d." %
                (tag, int(scene)), match.group(0))
        return compliance_error(
            "Prompt uses unknown scheduled reference tag @%s." % tag,
            match.group(0))

    return _REFERENCE_ALIAS_RE.sub(replace, str(text))


def _compile_scheduled_reference_prompt(
        schedule: Any, scene: int, scene_count: int,
        prompt: Any,
        compliance_mode: str = "strict") -> tuple[str, str, dict[str, Any]]:
    mode = _reference_compliance_mode(compliance_mode)
    warnings: list[str] = []
    try:
        bindings = _active_reference_bindings(
            schedule, scene, scene_count, mode, warnings)
    except (TypeError, ValueError) as exc:
        if mode != "disabled":
            raise
        message = "Reference schedule ignored: %s" % exc
        warnings.append(message)
        _LOG.warning("H3 scheduled-reference warning: %s", message)
        bindings = {
            "pictures": [], "videos": [], "audios": [], "aliases": {},
            "presentation": [], "all_tags": set(),
        }
    normalized_prompt = str(prompt or "").replace(
        "\r\n", "\n").replace("\r", "\n").strip()
    compiled_body = _replace_reference_aliases(
        normalized_prompt, bindings, scene, mode, warnings)
    mapping_lines = []
    for item in bindings["presentation"]:
        mapping_lines.append("@%s -> %s" % (
            item["tag"], item["label"]))
    summary = "scene %d/%d: %s" % (
        int(scene), int(scene_count),
        "; ".join(mapping_lines) if mapping_lines
        else "no scheduled references")
    if warnings:
        summary += "; warning-only: %s" % " ".join(warnings)
    elif mode == "disabled":
        summary += "; prompt compliance disabled; @tags passed unchanged"
    bindings["compliance_mode"] = mode
    bindings["compliance_warnings"] = warnings
    return compiled_body, summary, bindings


def _compile_tagged_reference_prompt(
        references: Any, scene: int, scene_count: int, prompt: Any,
        compliance_mode: str = "strict") -> tuple[str, str, dict[str, Any]]:
    """Activate only registered references mentioned by this scene prompt."""
    mode = _reference_compliance_mode(compliance_mode)
    warnings: list[str] = []
    normalized_prompt = str(prompt or "").replace(
        "\r\n", "\n").replace("\r", "\n").strip()
    prompt_tags = set(_REFERENCE_ALIAS_RE.findall(normalized_prompt))
    try:
        bindings = _active_reference_bindings(
            references, scene, scene_count, mode, warnings,
            activation_tags=prompt_tags)
    except (TypeError, ValueError) as exc:
        if mode != "disabled":
            raise
        message = "Tagged references ignored: %s" % exc
        warnings.append(message)
        _LOG.warning("H3 tagged-reference warning: %s", message)
        bindings = {
            "pictures": [], "videos": [], "audios": [], "aliases": {},
            "presentation": [], "all_tags": set(),
        }

    aliases = bindings["aliases"]

    def replace_registered(match: re.Match[str]) -> str:
        return aliases.get(match.group(1), match.group(0))

    compiled = (_REFERENCE_ALIAS_RE.sub(replace_registered, normalized_prompt)
                if mode != "disabled" else normalized_prompt)
    mapping_lines = [
        "@%s -> %s" % (item["tag"], item["label"])
        for item in bindings["presentation"]
    ]
    summary = "scene %d/%d: %s" % (
        int(scene), int(scene_count),
        "; ".join(mapping_lines) if mapping_lines
        else "no tagged references used by prompt")
    if warnings:
        summary += "; warning-only: %s" % " ".join(warnings)
    elif mode == "disabled":
        summary += "; reference policy disabled; @tags passed unchanged"
    bindings["compliance_mode"] = mode
    bindings["compliance_warnings"] = warnings
    return compiled, summary, bindings


def _derived_seed(base_seed: int, index: int, shot_id: str) -> int:
    payload = "%d:%d:%s" % (int(base_seed), int(index), shot_id)
    return int.from_bytes(hashlib.sha256(payload.encode("utf-8")).digest()[:8],
                          "big")


def _history_contract(plan: dict[str, Any], through_index: int) -> dict[str, Any]:
    shots = []
    for shot in plan["shots"][:int(through_index)]:
        contract = {
            "id": shot["id"],
            "prompt_hash": shot["prompt_hash"],
            "seed": shot["seed"],
            "steps": shot["steps"],
            "raw_frames": shot["raw_frames"],
            "delivered_frames": shot["delivered_frames"],
            "generation_start_frame": shot["generation_start_frame"],
        }
        # Keep legacy/default-guide history hashes stable. An explicit scene
        # override is generation-significant and must invalidate only that
        # scene and the continuation history after it.
        if "continuation_mode" in shot:
            contract["continuation_mode"] = shot["continuation_mode"]
        shots.append(contract)
    return {
        "version": PLAN_VERSION,
        "compatibility": plan["compatibility"],
        "shots": shots,
    }


def _history_hash(plan: dict[str, Any], through_index: int) -> str:
    return _fingerprint(_history_contract(plan, through_index))


def _audio_fingerprint(audio: Any) -> str:
    if torch is None:
        raise RuntimeError("Source-audio checkpoint validation requires torch.")
    waveform = audio["waveform"].detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(int(audio["sample_rate"])).encode("ascii"))
    digest.update(str(tuple(int(part) for part in waveform.shape)).encode("ascii"))
    digest.update(str(waveform.dtype).encode("ascii"))
    digest.update(memoryview(waveform.numpy()).cast("B"))
    return digest.hexdigest()


def _tensor_fingerprint(value: Any) -> str:
    """Hash a tensor without materializing one giant Python bytes object."""
    if torch is None or not torch.is_tensor(value):
        raise ValueError("H3 external video fingerprinting requires a tensor.")
    digest = hashlib.sha256()
    digest.update(str(tuple(int(part) for part in value.shape)).encode("ascii"))
    digest.update(str(value.dtype).encode("ascii"))
    chunks = value.detach().split(8, dim=0) if value.ndim else (value.detach(),)
    for chunk in chunks:
        cpu = chunk.to(device="cpu").contiguous()
        digest.update(memoryview(cpu.numpy()).cast("B"))
    return digest.hexdigest()


def _validate_audio(audio: Any, label: str,
                    expected_frames: int | None = None) -> tuple[Any, int]:
    if torch is None:
        raise RuntimeError("H3 chain audio validation requires torch.")
    # ComfyUI AUDIO producers may return a dict, a lazy mapping, or another
    # proxy implementing the same two-key protocol. Validate the actual audio
    # fields instead of enforcing a particular Python container class.
    try:
        waveform = audio["waveform"]
        sample_rate = int(audio["sample_rate"])
    except (KeyError, TypeError, AttributeError, ValueError, OverflowError) as exc:
        raise ValueError(
            "%s must provide ComfyUI AUDIO waveform and sample_rate fields; "
            "got %s." % (label, type(audio).__name__)) from exc
    if not torch.is_tensor(waveform) or waveform.ndim not in (1, 2, 3):
        raise ValueError(
            "%s waveform must be a 1D, 2D, or 3D tensor; got %r." %
            (label, getattr(waveform, "shape", None)))
    if sample_rate <= 0:
        raise ValueError("%s sample rate must be positive." % label)
    samples = int(waveform.shape[-1])
    if samples < 1:
        raise ValueError("%s waveform is empty." % label)
    if expected_frames is not None:
        expected = int(round(int(expected_frames) / float(FPS) * sample_rate))
        if samples != expected:
            raise ValueError(
                "%s contains %d samples at %d Hz; expected exactly %d samples "
                "for %d delivered frames at %d fps. Wire decoded audio through "
                "MiniMax H3 Contex Loop Trim with match_tail enabled." %
                (label, samples, sample_rate, expected, int(expected_frames), FPS))
    return waveform, sample_rate


def _audio_is_silent(waveform: Any) -> bool:
    if torch is None:
        return False
    return float(waveform.detach().abs().max().item()) <= 1e-6


def _pad_audio_to_samples(audio: dict[str, Any], samples: int,
                          label: str) -> dict[str, Any]:
    waveform, sample_rate = _validate_audio(audio, label)
    target = int(samples)
    current = int(waveform.shape[-1])
    if current >= target:
        return {"waveform": waveform[..., :target], "sample_rate": sample_rate}
    shape = list(waveform.shape)
    shape[-1] = target - current
    padding = torch.zeros(shape, dtype=waveform.dtype, device=waveform.device)
    return {
        "waveform": torch.cat((waveform, padding), dim=-1),
        "sample_rate": sample_rate,
    }


def _audio_waveform_3d(audio: dict[str, Any], label: str) -> tuple[Any, int]:
    """Return the first Comfy audio batch as [1, channels, samples]."""
    waveform, sample_rate = _validate_audio(audio, label)
    if waveform.ndim == 1:
        waveform = waveform.unsqueeze(0).unsqueeze(0)
    elif waveform.ndim == 2:
        waveform = waveform.unsqueeze(0)
    else:
        waveform = waveform[:1]
    channels = int(waveform.shape[1])
    if channels not in (1, 2):
        raise ValueError("%s must be mono or stereo; got %d channels." %
                         (label, channels))
    return waveform, sample_rate


def _resample_audio_exact(audio: dict[str, Any], sample_rate: int,
                          samples: int, channels: int,
                          label: str) -> dict[str, Any]:
    """Resample/channel-match audio to one exact frame-locked tensor."""
    waveform, source_rate = _audio_waveform_3d(audio, label)
    sample_rate = int(sample_rate)
    samples = int(samples)
    channels = int(channels)
    if sample_rate <= 0 or samples < 1 or channels not in (1, 2):
        raise ValueError("Invalid target audio format for %s." % label)
    waveform = waveform.to(dtype=torch.float32)
    if int(waveform.shape[1]) != channels:
        if int(waveform.shape[1]) == 1 and channels == 2:
            waveform = waveform.expand(-1, 2, -1)
        elif int(waveform.shape[1]) == 2 and channels == 1:
            waveform = waveform.mean(dim=1, keepdim=True)
    current = int(waveform.shape[-1])
    rate_adjusted = int(round(current * sample_rate / float(source_rate)))
    if source_rate != sample_rate and rate_adjusted > 0:
        waveform = torch.nn.functional.interpolate(
            waveform.reshape(-1, 1, current), size=rate_adjusted,
            mode="linear", align_corners=False).reshape(
                1, channels, rate_adjusted)
    current = int(waveform.shape[-1])
    if current < samples:
        padding = torch.zeros(
            (1, channels, samples - current), dtype=waveform.dtype,
            device=waveform.device)
        waveform = torch.cat((waveform, padding), dim=-1)
    else:
        waveform = waveform[..., :samples]
    return {
        "waveform": waveform.detach().cpu().contiguous(),
        "sample_rate": sample_rate,
    }


def _resample_audio_tail_exact(audio: dict[str, Any], sample_rate: int,
                               samples: int, channels: int,
                               label: str) -> dict[str, Any]:
    """Resample and end-align an exact tail, left-padding when necessary."""
    waveform, source_rate = _audio_waveform_3d(audio, label)
    sample_rate = int(sample_rate)
    samples = int(samples)
    channels = int(channels)
    if sample_rate <= 0 or samples < 1 or channels not in (1, 2):
        raise ValueError("Invalid target audio tail format for %s." % label)
    waveform = waveform.to(dtype=torch.float32)
    if int(waveform.shape[1]) != channels:
        if int(waveform.shape[1]) == 1 and channels == 2:
            waveform = waveform.expand(-1, 2, -1)
        elif int(waveform.shape[1]) == 2 and channels == 1:
            waveform = waveform.mean(dim=1, keepdim=True)
    current = int(waveform.shape[-1])
    rate_adjusted = int(round(current * sample_rate / float(source_rate)))
    if source_rate != sample_rate and rate_adjusted > 0:
        waveform = torch.nn.functional.interpolate(
            waveform.reshape(-1, 1, current), size=rate_adjusted,
            mode="linear", align_corners=False).reshape(
                1, channels, rate_adjusted)
    current = int(waveform.shape[-1])
    if current < samples:
        padding = torch.zeros(
            (1, channels, samples - current), dtype=waveform.dtype,
            device=waveform.device)
        waveform = torch.cat((padding, waveform), dim=-1)
    else:
        waveform = waveform[..., current - samples:]
    return {
        "waveform": waveform.detach().cpu().contiguous(),
        "sample_rate": sample_rate,
    }


def _validate_source_audio_hash(compatibility: dict[str, Any],
                                source_audio: dict[str, Any] | None,
                                usage: str) -> None:
    if source_audio is None:
        raise ValueError("%s requires source_audio." % usage)
    _validate_audio(source_audio, "%s source audio" % usage)
    expected = str(compatibility.get("source_audio_hash") or "")
    if not expected or expected == "none":
        raise ValueError("%s has no source-audio fingerprint to validate." % usage)
    actual = _audio_fingerprint(source_audio)
    if actual != expected:
        raise ValueError(
            "%s received a different source waveform than H3 Chain Loop Start. "
            "Wire the same AUDIO value to Start, Current Shot, and Assemble." % usage)


def _external_context_contract(external_context: dict[str, Any]) -> dict[str, Any]:
    frames = external_context.get("context_frames")
    audio = external_context.get("context_audio")
    return {
        "version": int(external_context.get("version", 0)),
        "base_plan_hash": str(external_context.get("base_plan_hash") or ""),
        "context_frames": int(getattr(frames, "shape", (0,))[0]),
        "context_frames_sha256": _tensor_fingerprint(frames),
        "context_audio_sha256": (
            _audio_fingerprint(audio) if audio is not None else "none"),
    }


def _plan_with_external_context(
    plan: dict[str, Any],
    external_context: dict[str, Any] | None,
) -> dict[str, Any]:
    """Make scene 1 a real continuation from an imported video tail."""
    if external_context is None:
        return plan
    if not isinstance(external_context, dict):
        raise ValueError(
            "H3 Chain Loop Start external_context must come from MiniMax H3 "
            "Existing Video Context.")
    expected_base = str(plan.get("base_plan_hash") or plan["plan_hash"])
    if str(external_context.get("base_plan_hash") or "") != expected_base:
        raise ValueError(
            "H3 existing-video context was prepared for a different Chain Plan. "
            "Reconnect the current Plan to the adapter and queue again.")
    contract = _external_context_contract(external_context)
    context_hash = _fingerprint(contract)
    if context_hash != str(external_context.get("context_hash") or ""):
        raise ValueError(
            "H3 existing-video context changed after it was prepared; refusing "
            "to use an unverifiable video tail.")

    span = int(contract["context_frames"])
    configured = int(plan["compatibility"]["context_length"])
    if span != configured:
        raise ValueError(
            "H3 existing-video context contains %d frames; this plan requires "
            "exactly %d." % (span, configured))

    prepared = dict(plan)
    prepared["base_plan_hash"] = expected_base
    prepared["shots"] = [dict(shot) for shot in plan["shots"]]
    prepared["compatibility"] = dict(plan["compatibility"])
    prepared["compatibility"].update({
        "external_context_hash": context_hash,
        "external_context_frames": span,
    })
    prelude = external_context.get("prelude")
    prepared["prelude"] = (_json_document(prelude)
                           if isinstance(prelude, dict) else None)

    stitched_frames = 0
    anchor_mode = prepared["compatibility"]["anchor_mode"]
    for offset, shot in enumerate(prepared["shots"]):
        raw_frames = int(shot["raw_frames"])
        if offset == 0:
            if anchor_mode == "head":
                if raw_frames <= span:
                    raise ValueError(
                        "H3 scene 1 has %d raw frames, not enough for the "
                        "%d-frame imported-video overlap." % (raw_frames, span))
                generation_start = -span
                delivered_frames = raw_frames - span
            else:
                generation_start = 0
                delivered_frames = raw_frames
            shot["external_context_frames"] = span
        elif anchor_mode == "head":
            generation_start = stitched_frames - configured
            delivered_frames = raw_frames - configured
        else:
            generation_start = stitched_frames
            delivered_frames = raw_frames
        shot["generation_start_frame"] = generation_start
        shot["delivered_frames"] = delivered_frames
        # Scene 1's negative pre-roll comes from the imported video/audio, not
        # from the extension soundtrack. Current Shot builds that composite
        # explicitly and begins the new source track at frame zero.
        shot["audio_start_seconds"] = max(0, generation_start) / float(FPS)
        shot["audio_duration_seconds"] = raw_frames / float(FPS)
        stitched_frames += delivered_frames

    for shot in prepared["shots"][:-1]:
        if int(shot["delivered_frames"]) < configured:
            raise ValueError(
                "Shot %d (%s) delivers only %d frames, but the next clip "
                "requires %d context frames." %
                (shot["index"], shot["id"], shot["delivered_frames"],
                 configured))

    prepared["total_delivered_frames"] = stitched_frames
    prepared["plan_hash"] = _fingerprint({
        "base_plan_hash": expected_base,
        "external_context_hash": context_hash,
    })
    cfg = prepared["compatibility"]
    prepared["summary"] = (
        "%d clips; %d delivered frames (%.3fs) at %dx%d; context=%d; "
        "blend=%d; audio=%s; imported video; run=%s" %
        (len(prepared["shots"]), stitched_frames,
         stitched_frames / float(FPS), cfg["width"], cfg["height"],
         configured, int(cfg.get("video_blend_frames", 0)),
         cfg["audio_mode"], prepared["run_name"]))
    return prepared


def _plan_with_source_audio(plan: dict[str, Any],
                            source_audio: dict[str, Any] | None) -> dict[str, Any]:
    mode = plan["compatibility"]["audio_mode"]
    if mode in ("source_track", "source_plus_timeline"):
        if source_audio is None:
            raise ValueError("H3 chain audio mode %s requires source_audio on "
                             "Loop Start." % mode)
        waveform, sample_rate = _validate_audio(
            source_audio, "H3 Chain Loop Start source audio")
        required_samples = int(round(
            int(plan["total_delivered_frames"]) / float(FPS) * sample_rate))
        silent_padding = False
        if int(waveform.shape[-1]) < required_samples:
            if _audio_is_silent(waveform):
                silent_padding = True
            else:
                raise ValueError(
                    "H3 Chain Loop Start source audio is too short: it contains %d "
                    "samples at %d Hz, but this plan requires at least %d samples "
                    "for %d delivered frames. Only silent placeholder audio is "
                    "automatically padded." %
                    (int(waveform.shape[-1]), sample_rate, required_samples,
                     int(plan["total_delivered_frames"])))
        source_hash = _audio_fingerprint(source_audio)
    else:
        source_hash = "none"
        silent_padding = False
    prepared = dict(plan)
    prepared["base_plan_hash"] = str(
        plan.get("base_plan_hash") or plan["plan_hash"])
    prepared["compatibility"] = dict(plan["compatibility"])
    prepared["compatibility"]["source_audio_hash"] = source_hash
    prepared["compatibility"]["source_audio_silent_padding"] = silent_padding
    if plan["compatibility"].get("external_context_hash"):
        prepared["plan_hash"] = _fingerprint({
            "prepared_plan_hash": plan["plan_hash"],
            "source_audio_hash": source_hash,
        })
    else:
        # Preserve the exact pre-v0.3.6 hash contract for ordinary chains so
        # every existing checkpoint remains resumable.
        prepared["plan_hash"] = _fingerprint({
            "base_plan_hash": plan["plan_hash"],
            "source_audio_hash": source_hash,
        })
    return prepared


def _retime_review_plan(plan: dict[str, Any]) -> None:
    """Rebuild every derived timeline field after a review length edit."""
    context_length = int(plan["compatibility"]["context_length"])
    anchor_mode = str(plan["compatibility"]["anchor_mode"])
    external_span = int(
        plan["compatibility"].get("external_context_frames", 0))
    stitched_frames = 0
    for offset, shot in enumerate(plan["shots"]):
        raw_frames = _validate_h3_length(
            shot["raw_frames"], "Shot %d length" % (offset + 1))
        if offset == 0:
            if external_span and anchor_mode == "head":
                if raw_frames <= external_span:
                    raise ValueError(
                        "H3 scene 1 has %d raw frames, not enough for the "
                        "%d-frame imported-video overlap." %
                        (raw_frames, external_span))
                generation_start = -external_span
                delivered_frames = raw_frames - external_span
            else:
                generation_start = 0
                delivered_frames = raw_frames
            if external_span:
                shot["external_context_frames"] = external_span
        else:
            if raw_frames <= context_length:
                raise ValueError(
                    "Shot %d has %d raw frames, not enough for a %d-frame "
                    "continuation overlap." %
                    (offset + 1, raw_frames, context_length))
            if anchor_mode == "head":
                generation_start = stitched_frames - context_length
                delivered_frames = raw_frames - context_length
            else:
                generation_start = stitched_frames
                delivered_frames = raw_frames
        shot["generation_start_frame"] = generation_start
        shot["delivered_frames"] = delivered_frames
        shot["audio_start_seconds"] = max(0, generation_start) / float(FPS)
        shot["audio_duration_seconds"] = raw_frames / float(FPS)
        stitched_frames += delivered_frames

    for shot in plan["shots"][:-1]:
        if int(shot["delivered_frames"]) < context_length:
            raise ValueError(
                "Shot %d (%s) delivers only %d frames, but the next clip "
                "requires %d context frames." %
                (shot["index"], shot["id"], shot["delivered_frames"],
                 context_length))
    plan["total_delivered_frames"] = stitched_frames
    cfg = plan["compatibility"]
    imported = "; imported video" if external_span else ""
    plan["summary"] = (
        "%d clips; %d delivered frames (%.3fs) at %dx%d; context=%d; "
        "blend=%d; audio=%s%s; run=%s" %
        (len(plan["shots"]), stitched_frames,
         stitched_frames / float(FPS), cfg["width"], cfg["height"],
         context_length, int(cfg.get("video_blend_frames", 0)),
         cfg["audio_mode"], imported, plan["run_name"]))


def _plan_with_review_revision(plan: dict[str, Any], index: int,
                               scene_prompt: str, seed: int,
                               raw_frames: int | None = None) -> dict[str, Any]:
    """Revise the current scene while preserving the accepted history contract."""
    index = int(index)
    if index < 1 or index > len(plan["shots"]):
        raise ValueError("H3 review revision index is outside the plan.")
    scene_prompt = str(scene_prompt or "").replace("\r\n", "\n").replace("\r", "\n").strip()
    prefix = str(plan.get("prompt_prefix") or "").strip()
    if not scene_prompt and not prefix:
        raise ValueError(
            "H3 review retry requires a scene prompt or shared prompt.")
    seed = int(seed)
    if seed < 0 or seed > MAX_SEED:
        raise ValueError("H3 review retry seed is outside the uint64 range.")

    revised = dict(plan)
    revised["shots"] = [dict(shot) for shot in plan["shots"]]
    shot = revised["shots"][index - 1]
    full_prompt = "\n\n".join(part for part in (prefix, scene_prompt) if part)
    shot["scene_prompt"] = scene_prompt
    shot["prompt"] = full_prompt
    shot["prompt_hash"] = hashlib.sha256(
        full_prompt.encode("utf-8")).hexdigest()
    shot["seed"] = seed
    if raw_frames is not None:
        shot["raw_frames"] = _validate_h3_length(
            raw_frames, "H3 review retry length")
    _retime_review_plan(revised)

    overrides = dict(revised.get("review_overrides") or {})
    overrides[str(index)] = {
        "scene_prompt": scene_prompt,
        "prompt_hash": shot["prompt_hash"],
        "seed": seed,
        "raw_frames": int(shot["raw_frames"]),
    }
    revised["review_overrides"] = overrides
    base_plan_hash = str(revised.get("base_plan_hash") or revised["plan_hash"])
    source_hash = str(
        revised.get("compatibility", {}).get("source_audio_hash") or "none")
    external_hash = str(
        revised.get("compatibility", {}).get("external_context_hash") or "none")
    revision_contract = {
        "base_plan_hash": base_plan_hash,
        "source_audio_hash": source_hash,
        "review_overrides": overrides,
    }
    if external_hash != "none":
        revision_contract["external_context_hash"] = external_hash
    revised["plan_hash"] = _fingerprint(revision_contract)
    return revised


def _normalize_plan(
    plan_json: str,
    run_name: str,
    width: int,
    height: int,
    context_length: int,
    encode_mode: str,
    anchor_mode: str,
    crop: str,
    audio_mode: str,
    audio_context_length: int,
    default_duration_seconds: float,
    default_steps: int,
    base_seed: int,
    segment_crf: int,
    generation_fingerprint: str = "",
    video_blend_frames: int = 0,
    continuation_mode: str = "guide",
) -> dict[str, Any]:
    try:
        raw = json.loads(str(plan_json or ""))
    except json.JSONDecodeError as exc:
        raise ValueError("H3 Chain Plan JSON is invalid: %s" % exc) from exc
    if isinstance(raw, list):
        raw = {"shots": raw}
    if not isinstance(raw, dict):
        raise ValueError("H3 Chain Plan must be a JSON object or a list of shots.")

    raw_shots = raw.get("shots")
    if not isinstance(raw_shots, list) or not raw_shots:
        raise ValueError("H3 Chain Plan requires a non-empty 'shots' list.")
    if len(raw_shots) > MAX_SHOTS:
        raise ValueError("H3 Chain Plan supports at most %d shots." % MAX_SHOTS)

    width, height = int(width), int(height)
    if width < 32 or height < 32 or width % 32 or height % 32:
        raise ValueError("H3 chain width and height must be positive multiples of 32.")
    context_length = int(context_length)
    if context_length not in H3_CONTEXT_LENGTHS:
        raise ValueError("H3 context length must be one of %s." % (H3_CONTEXT_LENGTHS,))
    if encode_mode not in ("video", "frames"):
        raise ValueError("Unknown H3 context encode mode %r." % encode_mode)
    if anchor_mode not in ("head", "before"):
        raise ValueError("Unknown H3 context anchor mode %r." % anchor_mode)
    if continuation_mode not in CONTINUATION_MODES:
        raise ValueError(
            "Unknown H3 continuation mode %r." % continuation_mode)
    if crop not in ("disabled", "center"):
        raise ValueError("Unknown H3 context crop mode %r." % crop)
    if audio_mode not in AUDIO_MODES:
        raise ValueError("Unknown H3 chain audio mode %r." % audio_mode)
    default_steps = max(1, min(10000, int(default_steps)))
    base_seed = max(0, min(MAX_SEED, int(base_seed)))
    segment_crf = max(0, min(51, int(segment_crf)))
    video_blend_frames = int(video_blend_frames)
    if video_blend_frames < 0 or video_blend_frames > context_length:
        raise ValueError(
            "H3 video blend length must be between 0 and context_length (%d)." %
            context_length)
    if anchor_mode != "head" and video_blend_frames:
        raise ValueError(
            "H3 video blending requires anchor_mode=head because before mode "
            "does not reproduce a leading overlap to blend.")

    defaults = raw.get("defaults") if isinstance(raw.get("defaults"), dict) else {}
    default_duration = float(defaults.get(
        "duration_seconds", raw.get("duration_seconds", default_duration_seconds)))
    default_steps = int(defaults.get("steps", raw.get("steps", default_steps)))
    if not math.isfinite(default_duration) or default_duration <= 0:
        raise ValueError("Default shot duration must be a finite positive number.")
    if default_steps < 1:
        raise ValueError("Default sampler steps must be at least 1.")

    prompt_prefix = _prompt_text(
        raw.get("prompt_prefix", raw.get("global_prompt", "")),
        "H3 Chain prompt_prefix",
    )
    seen_ids: set[str] = set()
    shots: list[dict[str, Any]] = []
    resolved_continuation_modes: list[str] = []
    stitched_frames = 0
    for offset, item in enumerate(raw_shots):
        index = offset + 1
        if isinstance(item, str):
            item = {"prompt": item}
        if not isinstance(item, dict):
            raise ValueError("Shot %d must be an object or prompt string." % index)

        shot_continuation_mode = item.get(
            "continuation_mode", continuation_mode)
        if shot_continuation_mode not in CONTINUATION_MODES:
            raise ValueError(
                "Shot %d has unknown H3 continuation mode %r." %
                (index, shot_continuation_mode))
        if (shot_continuation_mode in ("guide", "masked_av")
                and context_length not in GUIDE_CONTEXT_LENGTHS):
            raise ValueError(
                "H3 %s continuation requires a 17k+5 context length from %s "
                "(shot %d)." %
                (shot_continuation_mode, GUIDE_CONTEXT_LENGTHS, index))
        if shot_continuation_mode == "sliding_history":
            if context_length not in SLIDING_HISTORY_CONTEXT_LENGTHS:
                raise ValueError(
                    "H3 sliding_history continuation requires a 17k+1 context "
                    "length from %s (shot %d)." %
                    (SLIDING_HISTORY_CONTEXT_LENGTHS, index))
            if encode_mode != "video":
                raise ValueError(
                    "H3 sliding_history continuation requires encode_mode=video "
                    "(shot %d)." % index)
            if anchor_mode != "head":
                raise ValueError(
                    "H3 sliding_history continuation requires anchor_mode=head "
                    "(shot %d)." % index)
            if video_blend_frames:
                raise ValueError(
                    "H3 sliding_history continuation already uses an exact "
                    "history/boundary join and requires video_blend_frames=0 "
                    "(shot %d)." % index)
        if shot_continuation_mode == "masked_av":
            if context_length < 5:
                raise ValueError(
                    "H3 masked AV continuation requires context_length of at "
                    "least 5 frames (shot %d)." % index)
            if encode_mode != "video":
                raise ValueError(
                    "H3 masked AV continuation requires encode_mode=video "
                    "(shot %d)." % index)
            if anchor_mode != "head":
                raise ValueError(
                    "H3 masked AV continuation requires anchor_mode=head "
                    "because it preserves a real target-latent prefix that "
                    "Loop Trim must remove (shot %d)." % index)
        resolved_continuation_modes.append(shot_continuation_mode)

        shot_id = _safe_name(item.get("id", "clip_%04d" % index),
                             "clip_%04d" % index)
        if shot_id in seen_ids:
            raise ValueError("Duplicate H3 shot id %r." % shot_id)
        seen_ids.add(shot_id)

        prompt = _prompt_text(item.get("prompt", ""),
                              "Shot %d (%s) prompt" % (index, shot_id))
        if not prompt and not prompt_prefix:
            raise ValueError(
                "Shot %d (%s) requires a scene prompt or shared prompt." %
                (index, shot_id))
        scene_prompt = prompt
        prompt = "\n\n".join(
            part for part in (prompt_prefix, scene_prompt) if part)

        explicit_length = item.get("length", item.get("frames"))
        if explicit_length is None:
            duration = float(item.get("duration_seconds", default_duration))
            if not math.isfinite(duration) or duration <= 0:
                raise ValueError(
                    "Shot %d duration must be a finite positive number." % index)
            raw_frames = _h3_frame_length(duration)
        else:
            raw_frames = _validate_h3_length(explicit_length,
                                                   "Shot %d length" % index)

        if index == 1:
            generation_start_frame = 0
            delivered_frames = raw_frames
        else:
            if raw_frames <= context_length:
                raise ValueError(
                    "Shot %d has %d raw frames, not enough for a %d-frame "
                    "continuation overlap." % (index, raw_frames, context_length))
            if anchor_mode == "head":
                generation_start_frame = stitched_frames - context_length
                delivered_frames = raw_frames - context_length
            else:
                # `before` places context at negative coordinates, so no
                # repeated head is delivered or trimmed from the new clip.
                generation_start_frame = stitched_frames
                delivered_frames = raw_frames

        steps = int(item.get("steps", default_steps))
        if steps < 1 or steps > 10000:
            raise ValueError("Shot %d steps must be between 1 and 10000." % index)
        seed_value = item.get("seed")
        seed = (_derived_seed(base_seed, index, shot_id)
                if seed_value is None else int(seed_value))
        if seed < 0 or seed > MAX_SEED:
            raise ValueError("Shot %d seed is outside the uint64 range." % index)

        shot = {
            "index": index,
            "id": shot_id,
            # Kept separately so the review gate can edit only this scene
            # without duplicating the shared prompt prefix.
            "scene_prompt": scene_prompt,
            "prompt": prompt,
            "prompt_hash": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
            "seed": seed,
            "steps": steps,
            "raw_frames": raw_frames,
            "delivered_frames": delivered_frames,
            "generation_start_frame": generation_start_frame,
            "audio_start_seconds": generation_start_frame / float(FPS),
            "audio_duration_seconds": raw_frames / float(FPS),
        }
        # Absence means inherit the Plan node default. Omitting inherited
        # values preserves old guide plan and checkpoint hashes exactly.
        if "continuation_mode" in item:
            shot["continuation_mode"] = shot_continuation_mode
        shots.append(shot)
        stitched_frames += delivered_frames

    for shot in shots[:-1]:
        if shot["delivered_frames"] < context_length:
            raise ValueError(
                "Shot %d (%s) delivers only %d frames, but the next clip "
                "requires %d context frames. Increase its length or reduce "
                "context_length." %
                (shot["index"], shot["id"], shot["delivered_frames"],
                 context_length))

    compatibility = {
        "fps": FPS,
        "width": width,
        "height": height,
        "context_length": context_length,
        "encode_mode": encode_mode,
        "anchor_mode": anchor_mode,
        "crop": crop,
        "audio_mode": audio_mode,
        "audio_context_length": max(0, int(audio_context_length)),
        "segment_crf": segment_crf,
        "video_blend_frames": video_blend_frames,
        # Model, VAE, references, CFG, and scheduler live outside this node's
        # inputs. This caller-supplied tag lets a workflow make those external
        # generation dependencies part of the resume contract.
        "generation_fingerprint": str(generation_fingerprint or "").strip(),
    }
    # Preserve the exact compatibility/history hashes of every pre-feature
    # guide plan. A missing key is the stable serialized spelling of `guide`;
    # only the behavior-changing experimental mode extends the contract.
    if continuation_mode != "guide":
        compatibility["continuation_mode"] = continuation_mode
    plan = {
        "version": PLAN_VERSION,
        "run_name": _safe_name(run_name, "h3_chain"),
        "prompt_prefix": prompt_prefix,
        "shots": shots,
        "compatibility": compatibility,
        "segment_crf": segment_crf,
        "total_delivered_frames": stitched_frames,
    }
    plan["plan_hash"] = _fingerprint({
        "compatibility": compatibility,
        "shots": [{k: v for k, v in shot.items()
                   if k not in ("prompt", "scene_prompt")}
                  for shot in shots],
    })
    continuation_summary = (
        resolved_continuation_modes[0]
        if len(set(resolved_continuation_modes)) == 1 else "mixed"
    )
    plan["summary"] = (
        "%d clips; %d delivered frames (%.3fs) at %dx%d; context=%d/%s; "
        "blend=%d; audio=%s; run=%s" %
        (len(shots), stitched_frames, stitched_frames / float(FPS), width,
         height, context_length, continuation_summary, video_blend_frames,
         audio_mode, plan["run_name"]))
    return plan


def _output_root() -> str:
    return os.path.abspath(folder_paths.get_output_directory())


def _input_root() -> str:
    return os.path.abspath(folder_paths.get_input_directory())


def _run_dir(plan: dict[str, Any]) -> str:
    run_name = _safe_name(plan["run_name"], "h3_chain")
    cached = relay_cache.cache_path("h3_chains", run_name)
    legacy = os.path.abspath(os.path.join(
        _output_root(), "h3_chains", run_name
    ))
    if os.path.isdir(cached) or not os.path.isdir(legacy):
        return cached
    return legacy


def _launch_directory(path: str) -> tuple[bool, str | None]:
    """Ask the host desktop to reveal a directory without invoking a shell."""
    try:
        if os.name == "nt":
            startfile = getattr(os, "startfile", None)
            if startfile is None:
                return False, "This Python build does not provide os.startfile."
            startfile(path)
            return True, None
        if sys.platform == "darwin":
            commands = [["open", path]]
        else:
            commands = []
            xdg_open = shutil.which("xdg-open")
            gio = shutil.which("gio")
            if xdg_open:
                commands.append([xdg_open, path])
            if gio:
                commands.append([gio, "open", path])
        if not commands:
            return False, "No supported host folder opener was found."

        errors = []
        for command in commands:
            try:
                result = subprocess.run(
                    command, stdout=subprocess.DEVNULL,
                    stderr=subprocess.PIPE, timeout=5, check=False)
            except subprocess.TimeoutExpired:
                errors.append("%s timed out" % os.path.basename(command[0]))
                continue
            if result.returncode == 0:
                return True, None
            detail = result.stderr.decode("utf-8", errors="replace").strip()
            errors.append(detail or "%s exited with status %d" % (
                os.path.basename(command[0]), result.returncode))
        return False, "; ".join(errors)
    except OSError as exc:
        return False, str(exc)


def _open_run_output_directory(run_name: Any) -> dict[str, Any]:
    normalized = _safe_name(run_name, "")
    if not normalized:
        raise ValueError("A non-empty H3 chain run_name is required.")
    path = _run_dir({"run_name": normalized})
    os.makedirs(path, exist_ok=True)
    opened, error = _launch_directory(path)
    return {
        "ok": True,
        "opened": bool(opened),
        "run_name": normalized,
        "path": path,
        "error": str(error or ""),
    }


def _relative_output_path(path: str) -> str:
    return relay_cache.artifact_uri(path)


def _absolute_output_path(path: str) -> str:
    return relay_cache.resolve_artifact(path)


def _video_output_item(path: str) -> dict[str, str]:
    return relay_cache.video_output_item(path)


def _final_review_preview_key(document: dict[str, Any]) -> tuple[str, str]:
    return (
        _safe_name(document.get("run_name"), "h3_chain"),
        str(document.get("plan_hash") or ""),
    )


def _publish_final_review_preview(
    manifest: dict[str, Any], final_path: str, status: str
) -> None:
    """Return the completed final assembly to the gate that approved it."""
    if manifest.get("format") != "h3_chain_manifest_v3":
        return
    pending = _PENDING_FINAL_REVIEW_PREVIEWS.pop(
        _final_review_preview_key(manifest), None)
    if pending is None or PromptServer is None or PromptServer.instance is None:
        return
    payload = {
        "token": pending["token"],
        "node_id": pending["node_id"],
        "action": "final",
        "status": status,
        "final_video": _video_output_item(final_path),
    }
    try:
        PromptServer.instance.send_sync(
            "minimax_h3_context_loop_review_resolved", payload,
            pending.get("client_id"))
    except Exception as exc:
        # Assembly is already complete. A disconnected browser must not turn a
        # successful render into a failed ComfyUI execution.
        _LOG.warning(
            "H3 Chain could not publish the final preview to Review Gate: %s",
            exc)


def _artifact_paths(plan: dict[str, Any], index: int) -> dict[str, str]:
    run_dir = _run_dir(plan)
    return {
        "run_dir": run_dir,
        "segment": os.path.join(run_dir, "segments", "clip_%04d.mp4" % index),
        "blend_segment": os.path.join(
            run_dir, "blend_segments", "clip_%04d.mp4" % index),
        "generated_audio": os.path.join(
            run_dir, "generated_audio", "clip_%04d.wav" % index),
        "checkpoint": os.path.join(run_dir, "checkpoints",
                                   "clip_%04d.safetensors" % index),
        "metadata": os.path.join(run_dir, "checkpoints", "clip_%04d.json" % index),
    }


def _run_archive_paths(plan: dict[str, Any]) -> dict[str, str]:
    run_dir = _run_dir(plan)
    return {
        "plan": os.path.join(run_dir, "plan.json"),
        "workflow": os.path.join(run_dir, "workflow.json"),
        "api_prompt": os.path.join(run_dir, "api_prompt.json"),
    }


def _versioned_path(path: str, transaction: str) -> str:
    stem, extension = os.path.splitext(path)
    return "%s.%s%s" % (stem, transaction, extension)


def _file_sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            block = handle.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _safe_unlink(path: str) -> None:
    try:
        os.unlink(path)
    except FileNotFoundError:
        pass


def _atomic_text(path: str, value: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    temporary = "%s.%s.tmp" % (path, uuid.uuid4().hex)
    try:
        # Write exact UTF-8 bytes so Windows does not silently translate LF to
        # CRLF. Prompt hashes are defined over the normalized UTF-8 text.
        with open(temporary, "wb") as handle:
            handle.write(str(value).encode("utf-8"))
        os.replace(temporary, path)
    finally:
        _safe_unlink(temporary)


def _preserve_previous_revision(plan: dict[str, Any], index: int,
                                previous_metadata: Any) -> str | None:
    """Keep the superseded scene metadata beside its immutable artifacts."""
    if not isinstance(previous_metadata, dict):
        return None
    previous = previous_metadata.get("segment")
    if not isinstance(previous, dict):
        return None
    canonical = _artifact_paths(plan, index)
    existing = previous.get("revision_metadata")
    if isinstance(existing, str):
        try:
            path = _absolute_output_path(existing)
        except (ValueError, OSError):
            path = ""
        if (path and os.path.isfile(path)):
            return _relative_output_path(path)

    revision = str(previous.get("revision") or "")
    if re.fullmatch(r"[0-9a-f]{32}", revision) is None:
        name = os.path.basename(str(previous.get("segment") or ""))
        match = re.fullmatch(
            r"clip_%04d\.([0-9a-f]{32})\.mp4" % index, name)
        revision = match.group(1) if match is not None else uuid.uuid4().hex
    snapshot_path = _versioned_path(canonical["metadata"], revision)
    snapshot = dict(previous_metadata)
    snapshot_segment = dict(previous)
    snapshot_segment["revision"] = revision
    snapshot_segment["revision_metadata"] = _relative_output_path(snapshot_path)
    snapshot["segment"] = snapshot_segment
    _atomic_json(snapshot_path, snapshot)
    return _relative_output_path(snapshot_path)


def _atomic_json(path: str, value: Any) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    temporary = "%s.%s.tmp" % (path, uuid.uuid4().hex)
    try:
        with open(temporary, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(temporary, path)
    finally:
        _safe_unlink(temporary)


def _read_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _effective_editor_plan(plan: dict[str, Any]) -> dict[str, Any]:
    """Return an exact, editable plan source for this execution revision."""
    return {
        "prompt_prefix": str(plan.get("prompt_prefix") or ""),
        "shots": [dict({
            "id": shot["id"],
            "prompt": shot.get("scene_prompt", ""),
            "length": int(shot["raw_frames"]),
            "steps": int(shot["steps"]),
            # A decimal string remains exact when the workflow passes through
            # JavaScript, including uint64 values above Number.MAX_SAFE_INTEGER.
            "seed": str(int(shot["seed"])),
        }, **({"continuation_mode": shot["continuation_mode"]}
               if "continuation_mode" in shot else {}))
            for shot in plan["shots"]],
    }


def _json_document(value: Any) -> Any:
    """Clone one JSON document, accepting ComfyUI's occasional string form."""
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return None
    if not isinstance(value, (dict, list)):
        return None
    try:
        return json.loads(json.dumps(value, ensure_ascii=False))
    except (TypeError, ValueError):
        return None


def _matching_plan_node_ids(api_prompt: Any,
                            plan: dict[str, Any]) -> tuple[Any, set[str]]:
    document = _json_document(api_prompt)
    if not isinstance(document, dict):
        return None, set()
    effective_json = json.dumps(
        _effective_editor_plan(plan), ensure_ascii=False, indent=2)
    candidates: list[tuple[str, dict[str, Any]]] = []
    exact: list[tuple[str, dict[str, Any]]] = []
    for node_id, node in document.items():
        if not isinstance(node, dict) or node.get("class_type") != "MiniMaxH3ChainPlan":
            continue
        inputs = node.get("inputs")
        if not isinstance(inputs, dict):
            continue
        candidate = (str(node_id), inputs)
        candidates.append(candidate)
        run_name = inputs.get("run_name")
        if (isinstance(run_name, str) and
                _safe_name(run_name, "h3_chain") == plan["run_name"]):
            exact.append(candidate)
    selected = exact or (candidates if len(candidates) == 1 else [])
    for _node_id, inputs in selected:
        inputs["plan_json"] = effective_json
    return document, {node_id for node_id, _inputs in selected}


def _patched_workflow(workflow: Any, plan: dict[str, Any],
                      plan_node_ids: set[str]) -> Any:
    document = _json_document(workflow)
    if not isinstance(document, dict):
        return None
    nodes = document.get("nodes")
    if not isinstance(nodes, list):
        return document
    effective_json = json.dumps(
        _effective_editor_plan(plan), ensure_ascii=False, indent=2)
    candidates = [node for node in nodes if isinstance(node, dict) and
                  node.get("type") == "MiniMaxH3ChainPlan"]
    selected = []
    for node in candidates:
        widgets = node.get("widgets_values")
        node_id = str(node.get("id"))
        run_name = (widgets[1] if isinstance(widgets, list) and len(widgets) > 1
                    else None)
        if (node_id in plan_node_ids or
                (isinstance(run_name, str) and
                 _safe_name(run_name, "h3_chain") == plan["run_name"])):
            selected.append(node)
    if not selected and len(candidates) == 1:
        selected = candidates
    for node in selected:
        widgets = node.get("widgets_values")
        if isinstance(widgets, list) and widgets:
            widgets[0] = effective_json
    return document


def _write_run_archives(plan: dict[str, Any], api_prompt: Any = None,
                        extra_pnginfo: Any = None) -> dict[str, str]:
    """Persist recovery documents and return output-relative paths.

    `plan.json` is always written and represents the exact effective revision,
    including review-gate prompt/seed changes. The frontend workflow and API
    prompt are written when ComfyUI supplies their standard hidden metadata.
    Existing workflow archives are retained if a non-Comfy caller later saves
    another segment without hidden metadata.
    """
    paths = _run_archive_paths(plan)
    archived_plan = dict(plan)
    archived_plan["format"] = "h3_chain_plan_archive_v1"
    archived_plan["editor_plan"] = _effective_editor_plan(plan)
    _atomic_json(paths["plan"], archived_plan)

    patched_prompt, plan_node_ids = _matching_plan_node_ids(api_prompt, plan)
    if patched_prompt is not None:
        _atomic_json(paths["api_prompt"], patched_prompt)

    workflow = None
    if isinstance(extra_pnginfo, dict):
        workflow = extra_pnginfo.get("workflow")
    patched_workflow = _patched_workflow(workflow, plan, plan_node_ids)
    if patched_workflow is not None:
        _atomic_json(paths["workflow"], patched_workflow)

    return _available_run_archives(plan)


def _available_run_archives(plan: dict[str, Any]) -> dict[str, str]:
    paths = _run_archive_paths(plan)
    return {key: _relative_output_path(path) for key, path in paths.items()
            if os.path.isfile(path)}


def _archive_media_metadata(archives: Any) -> dict[str, str]:
    """Load ComfyUI-compatible video tags from persisted run archives."""
    if not isinstance(archives, dict):
        return {}
    metadata = {}
    for archive_key, tag in (("api_prompt", "prompt"),
                             ("workflow", "workflow"),
                             ("plan", "h3_plan")):
        value = archives.get(archive_key)
        if not isinstance(value, str):
            continue
        try:
            document = _read_json(_absolute_output_path(value))
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            _LOG.warning("H3 Chain could not embed %s metadata: %s",
                         archive_key, exc)
            continue
        metadata[tag] = json.dumps(document, ensure_ascii=False,
                                   separators=(",", ":"))
    return metadata


def _prompt_fields(plan: dict[str, Any], index: int) -> dict[str, Any]:
    shot = plan["shots"][int(index) - 1]
    return {
        "prompt_prefix": str(plan.get("prompt_prefix") or ""),
        "scene_prompt": str(shot.get("scene_prompt") or ""),
        "prompt": str(shot.get("prompt") or ""),
        "prompt_hash": str(shot["prompt_hash"]),
    }


def _tensor_cpu_clone(value: Any) -> Any:
    if torch is not None and torch.is_tensor(value):
        return value.detach().cpu().contiguous().clone()
    return value


def _compact_latent(latent: dict[str, Any]) -> dict[str, Any]:
    parts = _streams_from_latent(latent)
    if len(parts) < 2:
        raise ValueError("H3 Chain requires a sampled MiniMax AV latent.")
    return {"samples": [_tensor_cpu_clone(parts[0]),
                        _tensor_cpu_clone(parts[1])]}


def _public_segment(value: dict[str, Any]) -> dict[str, Any]:
    return {key: value[key] for key in (
        "index", "id", "segment", "checkpoint", "metadata",
        "blend_segment", "blend_segment_sha256", "blend_frames",
        "revision", "revision_metadata", "supersedes", "prompt_file",
        "generated_audio", "generated_audio_sha256",
        "raw_frames", "delivered_frames", "history_hash",
        "prompt_prefix", "scene_prompt", "prompt", "prompt_hash", "archives",
        "seed", "steps", "sample_rate", "segment_sha256",
        "checkpoint_sha256", "prompt_file_sha256", "predecessor_revision",
        "predecessor_checkpoint_sha256") if key in value}


def _verify_segment_artifacts(segment: dict[str, Any], index: int) -> None:
    if int(segment.get("index", -1)) != int(index):
        raise ValueError(
            "H3 chain metadata slot %d points to segment index %r." %
            (index, segment.get("index")))
    for key, hash_key in (("segment", "segment_sha256"),
                          ("checkpoint", "checkpoint_sha256")):
        value = segment.get(key)
        expected_hash = str(segment.get(hash_key) or "")
        if not isinstance(value, str) or not expected_hash:
            raise ValueError(
                "H3 chain clip %d metadata has no verified %s artifact." %
                (index, key))
        artifact = _absolute_output_path(value)
        if not os.path.isfile(artifact):
            raise FileNotFoundError(
                "H3 chain clip %d %s is missing: %s" %
                (index, key, artifact))
        actual_hash = _file_sha256(artifact)
        if actual_hash != expected_hash:
            raise ValueError(
                "H3 chain clip %d %s failed its SHA-256 integrity check." %
                (index, key))
    blend_frames = int(segment.get("blend_frames", 0))
    if blend_frames:
        value = segment.get("blend_segment")
        expected_hash = str(segment.get("blend_segment_sha256") or "")
        if not isinstance(value, str) or not expected_hash:
            raise ValueError(
                "H3 chain clip %d metadata has no verified blend segment." %
                index)
        artifact = _absolute_output_path(value)
        if not os.path.isfile(artifact):
            raise FileNotFoundError(
                "H3 chain clip %d blend segment is missing: %s" %
                (index, artifact))
        if _file_sha256(artifact) != expected_hash:
            raise ValueError(
                "H3 chain clip %d blend segment failed its SHA-256 integrity "
                "check." % index)
    generated_audio = segment.get("generated_audio")
    if generated_audio is not None:
        expected_hash = str(segment.get("generated_audio_sha256") or "")
        if not isinstance(generated_audio, str) or not expected_hash:
            raise ValueError(
                "H3 chain clip %d metadata has no verified generated-audio "
                "sidecar." % index)
        audio_path = _absolute_output_path(generated_audio)
        if not os.path.isfile(audio_path):
            raise FileNotFoundError(
                "H3 chain clip %d generated-audio sidecar is missing: %s" %
                (index, audio_path))
        if _file_sha256(audio_path) != expected_hash:
            raise ValueError(
                "H3 chain clip %d generated-audio sidecar failed its SHA-256 "
                "integrity check." % index)
    prompt_file = segment.get("prompt_file")
    if isinstance(prompt_file, str):
        prompt_path = _absolute_output_path(prompt_file)
        if not os.path.isfile(prompt_path):
            raise FileNotFoundError(
                "H3 chain clip %d prompt sidecar is missing: %s" %
                (index, prompt_path))
        artifact_hash = str(segment.get("prompt_file_sha256") or "")
        if artifact_hash:
            actual_hash = _file_sha256(prompt_path)
        else:
            # Records saved before prompt_file_sha256 used prompt_hash for this
            # check. Windows text-mode writes converted LF to CRLF, so compare
            # their normalized text while retaining strict raw-byte checks for
            # all newly saved sidecars.
            with open(prompt_path, "r", encoding="utf-8", newline=None) as handle:
                prompt_text = handle.read()
            prompt_text = prompt_text.replace("\r\n", "\n").replace("\r", "\n")
            actual_hash = hashlib.sha256(prompt_text.encode("utf-8")).hexdigest()
            artifact_hash = str(segment.get("prompt_hash") or "")
        if not artifact_hash or actual_hash != artifact_hash:
            raise ValueError(
                "H3 chain clip %d prompt sidecar failed its SHA-256 integrity "
                "check." % index)


def _load_resume_state(plan: dict[str, Any], start_clip: int) -> dict[str, Any]:
    if _st_load is None:
        raise RuntimeError("safetensors is required to resume H3 chains.")
    previous_index = start_clip - 1
    segments = []
    previous_meta = None
    for index in range(1, previous_index + 1):
        paths = _artifact_paths(plan, index)
        if not os.path.isfile(paths["metadata"]):
            raise FileNotFoundError(
                "Cannot resume clip %d: metadata for predecessor clip %d is "
                "missing: %s" % (start_clip, index, paths["metadata"]))
        metadata = _read_json(paths["metadata"])
        expected = _history_hash(plan, index)
        if metadata.get("history_hash") != expected:
            raise ValueError(
                "Cannot resume clip %d: clip %d was generated from different "
                "settings, prompts, seeds, or durations." % (start_clip, index))
        segment = metadata.get("segment")
        if not isinstance(segment, dict):
            raise ValueError("Checkpoint metadata for clip %d has no segment." % index)
        if segment.get("history_hash") != expected:
            raise ValueError(
                "Checkpoint segment record for clip %d has a mismatched history."
                % index)
        _verify_segment_artifacts(segment, index)
        restored = _public_segment(segment)
        for key, value in _prompt_fields(plan, index).items():
            restored.setdefault(key, value)
        segments.append(restored)
        previous_meta = metadata

    if previous_meta is None:
        raise RuntimeError("Internal resume error: predecessor metadata unavailable.")
    checkpoint = _absolute_output_path(previous_meta["segment"]["checkpoint"])
    tensors = _st_load(checkpoint)
    required = {"context_frames", "video", "audio"}
    missing = sorted(required - set(tensors))
    if missing:
        raise ValueError("H3 chain checkpoint is missing tensors: %s" % missing)
    expected_context = min(
        int(plan["compatibility"]["context_length"]),
        int(plan["shots"][previous_index - 1]["delivered_frames"]))
    if int(tensors["context_frames"].shape[0]) != expected_context:
        raise ValueError(
            "H3 chain predecessor checkpoint contains %d context frames; "
            "expected %d." %
            (int(tensors["context_frames"].shape[0]), expected_context))
    return {
        "plan": plan,
        "index": start_clip,
        "previous_frames": tensors["context_frames"],
        "previous_latent": {"samples": [tensors["video"], tensors["audio"]]},
        "segments": segments,
        "resumed_from": previous_index,
    }


def _initial_state(plan: dict[str, Any], start_clip: int,
                   end_clip: int | None = None,
                   external_context: dict[str, Any] | None = None) -> dict[str, Any]:
    total = len(plan["shots"])
    start_clip = int(start_clip)
    if start_clip < 1 or start_clip > total:
        raise ValueError("start_clip must be between 1 and %d." % total)
    end_clip = total if end_clip is None else int(end_clip)
    if end_clip < start_clip or end_clip > total:
        raise ValueError(
            "end_clip must be between start_clip %d and %d." %
            (start_clip, total))
    if start_clip > 1:
        state = _load_resume_state(plan, start_clip)
    else:
        state = {
            "plan": plan,
            "index": 1,
            "previous_frames": (
                external_context.get("context_frames")
                if isinstance(external_context, dict) else None),
            "previous_latent": None,
            "previous_audio": (
                external_context.get("context_audio")
                if isinstance(external_context, dict) else None),
            "external_context": bool(external_context is not None),
            "segments": [],
            "resumed_from": 0,
        }
    state["range_start"] = start_clip
    state["end_clip"] = end_clip
    return state


def _slice_audio(audio: dict[str, Any], start_seconds: float,
                 duration_seconds: float,
                 pad_silence: bool = False) -> dict[str, Any]:
    waveform, sample_rate = _validate_audio(audio, "H3 source audio")
    total = int(waveform.shape[-1])
    start = max(0, int(round(float(start_seconds) * sample_rate)))
    end = max(start + 1, int(round(
        (float(start_seconds) + float(duration_seconds)) * sample_rate)))
    wanted = end - start
    if pad_silence and end > total:
        padded = _pad_audio_to_samples(
            audio, end, "H3 silent placeholder audio")
        return {
            "waveform": padded["waveform"][..., start:end],
            "sample_rate": sample_rate,
        }
    if start >= total:
        raise ValueError(
            "H3 source audio ends at %.3fs, before this clip's %.3fs start." %
            (total / float(sample_rate), start_seconds))
    if end > total:
        raise ValueError(
            "H3 source audio is too short for this chain: clip window "
            "%.3f..%.3fs requires %d samples, but the waveform ends at %.3fs. "
            "Short audio would truncate the final video." %
            (start_seconds, start_seconds + duration_seconds, wanted,
             total / float(sample_rate)))
    return {"waveform": waveform[..., start:end], "sample_rate": sample_rate}


def _align_audio_reference_to_h3_grid(
        audio: Any, frame_count: int) -> tuple[Any, str]:
    waveform, sample_rate = _validate_audio(
        audio, "H3 aligned source-audio reference")
    # Stock H3 builds the generated audio stream with round(seconds * 40). Its
    # audio VAE then consumes 800 samples per latent at 32 kHz. Use floor at
    # other input rates so the later resample cannot spill into one additional
    # reference latent.
    target_steps = int(round(int(frame_count) / float(FPS) * 40.0))
    # A reference ending exactly on the target boundary (15.075s at 362
    # frames) still showed the same visual duplication seen with an overlong
    # reference, while a 5ms undercut (15.070s) did not. Keep the same number
    # of reference latents but leave a short zero-padded tail in the last one.
    safety_samples_32k = 160
    target_32k_samples = max(1, target_steps * 800 - safety_samples_32k)
    target_samples = max(1, int(math.floor(
        target_32k_samples * sample_rate / 32000.0)))
    current_samples = int(waveform.shape[-1])
    if current_samples <= target_samples:
        return audio, (
            "audio ref unchanged at %d samples (target %d steps, safe %.6fs)" %
            (current_samples, target_steps,
             target_32k_samples / 32000.0))
    return {
        "waveform": waveform[..., :target_samples],
        "sample_rate": sample_rate,
    }, (
        "audio ref aligned %d->%d samples (target %d steps, safe %.6fs)" %
        (current_samples, target_samples, target_steps,
         target_32k_samples / 32000.0))


def _slice_audio_after_external_context(
    source_audio: dict[str, Any],
    external_audio: dict[str, Any] | None,
    raw_frames: int,
    lead_frames: int,
    pad_silence: bool,
) -> dict[str, Any]:
    """Build scene 1 audio as imported tail + extension soundtrack start."""
    waveform, sample_rate = _audio_waveform_3d(
        source_audio, "H3 source audio")
    channels = int(waveform.shape[1])
    total_samples = int(round(int(raw_frames) / float(FPS) * sample_rate))
    lead_samples = int(round(int(lead_frames) / float(FPS) * sample_rate))
    lead_samples = min(lead_samples, total_samples)
    extension_samples = total_samples - lead_samples
    if int(waveform.shape[-1]) < extension_samples:
        if pad_silence and _audio_is_silent(waveform):
            source = _pad_audio_to_samples(
                {"waveform": waveform, "sample_rate": sample_rate},
                extension_samples, "H3 silent extension soundtrack")
            extension = source["waveform"]
        else:
            raise ValueError(
                "H3 extension soundtrack has %d samples; scene 1 requires %d "
                "after its imported-video audio lead." %
                (int(waveform.shape[-1]), extension_samples))
    else:
        extension = waveform[..., :extension_samples]
    if external_audio is None:
        lead = torch.zeros(
            (1, channels, lead_samples), dtype=extension.dtype,
            device=extension.device)
    else:
        lead = _resample_audio_tail_exact(
            external_audio, sample_rate, lead_samples, channels,
            "H3 existing-video context audio")["waveform"].to(
                device=extension.device, dtype=extension.dtype)
    return {
        "waveform": torch.cat((lead, extension), dim=-1),
        "sample_rate": sample_rate,
    }


def _write_segment_video(images: Any, path: str, fps: int, crf: int,
                         metadata: dict[str, Any] | None = None) -> None:
    if av is None or torch is None:
        raise RuntimeError("H3 segment saving requires PyAV and torch.")
    if len(images.shape) != 4 or int(images.shape[0]) < 1:
        raise ValueError("H3 segment images must be [frames,height,width,channels].")
    height, width = int(images.shape[1]), int(images.shape[2])
    if width % 2 or height % 2:
        raise ValueError("H.264 segment dimensions must be even.")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    temporary = path[:-4] + ".tmp.mp4"
    if os.path.exists(temporary):
        os.unlink(temporary)
    container = None
    try:
        container = av.open(
            temporary, mode="w",
            options={"movflags": "use_metadata_tags+faststart"})
        if metadata:
            for key, value in metadata.items():
                if value is not None:
                    container.metadata[str(key)] = str(value)
        stream = container.add_stream("libx264", rate=Fraction(int(fps), 1))
        stream.width = width
        stream.height = height
        stream.pix_fmt = "yuv420p"
        stream.options = {"crf": str(int(crf)), "preset": "medium"}
        for image in images:
            array = (torch.clamp(image[..., :3] * 255.0, 0, 255)
                     .to(device="cpu", dtype=torch.uint8).numpy())
            frame = av.VideoFrame.from_ndarray(array, format="rgb24")
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
        container.close()
        container = None
        os.replace(temporary, path)
    except Exception:
        if container is not None:
            container.close()
        if os.path.exists(temporary):
            os.unlink(temporary)
        raise


def _write_wav(audio: dict[str, Any], path: str) -> None:
    if torch is None:
        raise RuntimeError("H3 chain audio assembly requires torch.")
    waveform = audio["waveform"]
    if len(waveform.shape) == 3:
        waveform = waveform[0]
    elif len(waveform.shape) == 1:
        waveform = waveform.unsqueeze(0)
    if len(waveform.shape) != 2:
        raise ValueError("H3 chain audio must be [batch,channels,samples].")
    pcm = (torch.clamp(waveform, -1.0, 1.0).movedim(0, 1) * 32767.0)
    pcm = pcm.round().to(device="cpu", dtype=torch.int16).contiguous().numpy()
    with wave.open(path, "wb") as handle:
        handle.setnchannels(int(pcm.shape[1]))
        handle.setsampwidth(2)
        handle.setframerate(int(audio["sample_rate"]))
        handle.writeframes(pcm.tobytes())


def _atomic_wav(audio: dict[str, Any], path: str) -> None:
    """Publish a WAV without exposing a partial file to resume or the user."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    temporary = "%s.%s.tmp.wav" % (path, uuid.uuid4().hex)
    try:
        _write_wav(audio, temporary)
        os.replace(temporary, path)
    finally:
        _safe_unlink(temporary)


def _external_video_frame_indices(frame_count: int, source_fps: float) -> Any:
    frame_count = int(frame_count)
    source_fps = float(source_fps)
    if frame_count < 1:
        raise ValueError("H3 existing-video source contains no frames.")
    if not math.isfinite(source_fps) or source_fps <= 0:
        raise ValueError("H3 existing-video source_fps must be positive.")
    target_count = max(1, int(round(frame_count * FPS / source_fps)))
    # CFR sample at each 24 fps target timestamp. floor() avoids looking ahead
    # across the join; the final selected frame remains the latest available
    # source frame at that instant.
    return (torch.arange(target_count, dtype=torch.float64) *
            (source_fps / float(FPS))).floor().to(dtype=torch.long).clamp(
                min=0, max=frame_count - 1)


def _resolve_video_inputs(source_video: Any, source_frames: Any,
                          source_audio: Any, source_fps: float,
                          label: str) -> tuple[Any, Any, float, str]:
    """Resolve native VIDEO or decoded IMAGE/AUDIO without hiding provenance."""
    if source_video is not None and source_frames is not None:
        raise ValueError(
            "%s received both source_video and source_frames. Connect one "
            "video input route only." % label)
    input_route = "decoded IMAGE/AUDIO"
    if source_video is not None:
        get_components = getattr(source_video, "get_components", None)
        if not callable(get_components):
            raise ValueError(
                "%s source_video must be a native ComfyUI VIDEO value with "
                "get_components()." % label)
        try:
            components = get_components()
        except Exception as exc:
            raise ValueError(
                "%s source_video could not be decoded: %s" %
                (label, exc)) from exc
        source_frames = getattr(components, "images", None)
        try:
            source_fps = float(getattr(components, "frame_rate"))
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "%s source_video has no valid frame rate." % label) from exc
        if source_audio is None:
            source_audio = getattr(components, "audio", None)
        input_route = "native VIDEO"
    elif source_frames is None:
        raise ValueError(
            "%s requires source_video or source_frames." % label)
    return source_frames, source_audio, float(source_fps), input_route


class MiniMaxH3ReferenceVideoPrepare:
    """Prepare a synchronized source performance for one-pass H3 Ref2VA."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "length": ("INT", {
                    "default": 209, "min": 5, "max": MAX_H3_FRAMES,
                    "step": 17,
                    "tooltip": "Exact H3 output/reference length. It must "
                               "satisfy length % 17 == 5. The source video "
                               "and copied soundtrack must both cover it."}),
                "source_fps": ("FLOAT", {
                    "default": 24.0, "min": 0.001, "max": 1000.0,
                    "step": 0.001,
                    "tooltip": "Actual frame rate represented by "
                               "source_frames. It is ignored for native "
                               "VIDEO, which carries its own exact FPS."}),
            },
            "optional": {
                "source_video": ("VIDEO", {
                    "tooltip": "Native ComfyUI VIDEO from core Load Video or "
                               "another VIDEO loader. Its frames, embedded "
                               "audio, and exact FPS are decoded directly."}),
                "source_frames": ("IMAGE", {
                    "tooltip": "Decoded IMAGE batch from VHS or another "
                               "loader. Connect this instead of source_video "
                               "and provide its actual source_fps."}),
                "source_audio": ("AUDIO", {
                    "tooltip": "Soundtrack paired with source_frames, or an "
                               "override for native VIDEO audio. The node "
                               "copies its opening samples exactly; it never "
                               "time-stretches or silently pads them."}),
            },
        }

    RETURN_TYPES = ("IMAGE", "AUDIO", "INT", "STRING")
    RETURN_NAMES = ("ref_video", "source_audio", "length", "status")
    OUTPUT_TOOLTIPS = (
        "The source performance sampled at H3's 24 fps for Ref2VA.",
        "The original source waveform cut exactly to the selected duration.",
        "Validated H3 frame length for the stock Ref2VA length input.",
        "Input route, source timing, selected frame count, and copied audio.",
    )
    FUNCTION = "prepare"
    CATEGORY = "conditioning/minimax/contex_loop"
    DESCRIPTION = ("Normalize a native VIDEO or IMAGE/AUDIO source to an "
                   "exact one-pass H3 Ref2VA performance reference while "
                   "copying, not regenerating, its synchronized soundtrack.")

    def prepare(self, length=209, source_fps=24.0, source_video=None,
                source_frames=None, source_audio=None):
        length = _validate_h3_length(length, "H3 reference-video length")
        source_frames, source_audio, source_fps, input_route = (
            _resolve_video_inputs(
                source_video, source_frames, source_audio, source_fps,
                "H3 reference-video prep"))
        if torch is None or not torch.is_tensor(source_frames):
            raise ValueError(
                "H3 reference-video source_frames must be an IMAGE tensor.")
        if source_frames.ndim != 4 or int(source_frames.shape[-1]) < 3:
            raise ValueError(
                "H3 reference-video source_frames must be "
                "[frames,height,width,channels]; got %r." %
                (getattr(source_frames, "shape", None),))

        indices = _external_video_frame_indices(
            int(source_frames.shape[0]), source_fps)
        available = int(indices.numel())
        if available < length:
            raise ValueError(
                "H3 reference-video source becomes %d frames at 24 fps, but "
                "length is %d. Choose a shorter H3-valid length or supply a "
                "longer video." % (available, length))
        selected = source_frames.index_select(
            0, indices[:length].to(device=source_frames.device))

        if source_audio is None:
            raise ValueError(
                "H3 reference-video prep requires source audio so the final "
                "soundtrack can be copied unchanged.")
        waveform, sample_rate = _audio_waveform_3d(
            source_audio, "H3 reference-video source audio")
        required_samples = int(round(length / float(FPS) * sample_rate))
        available_samples = int(waveform.shape[-1])
        if available_samples < required_samples:
            raise ValueError(
                "H3 reference-video source audio contains %d samples at %d "
                "Hz, but %d frames require %d. Choose a shorter H3-valid "
                "length; this node will not pad or stretch the soundtrack." %
                (available_samples, sample_rate, length, required_samples))
        copied_audio = {
            "waveform": waveform[..., :required_samples].clone(),
            "sample_rate": sample_rate,
        }
        status = (
            "%s: %d frames at %.6g fps -> %d frames at %d fps; copied "
            "%d audio samples at %d Hz (%.3fs)" %
            (input_route, int(source_frames.shape[0]), source_fps, length, FPS,
             required_samples, sample_rate, length / float(FPS)))
        return selected, copied_audio, length, status


class MiniMaxH3ScheduledPictureReference:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {
                    "tooltip": "One reference picture. Ref2VA uses only the "
                               "first image when a batch is connected."}),
                "tag": ("STRING", {
                    "default": "hero_face",
                    "tooltip": "Stable alias used as @tag in prompts, for "
                               "example tag hero_face becomes @hero_face. "
                               "The tag is NOT a native Picture number. "
                               "Active pictures are renumbered from "
                               "<Picture 1> in every scene: if an earlier "
                               "picture is removed or inactive, @picture_2 "
                               "can correctly compile to <Picture 1>."}),
                "scenes": ("STRING", {
                    "default": "",
                    "tooltip": "Scenes where this picture is active. Leave "
                               "blank for all scenes; use 1, 1:4, or "
                               "1,3,5:8 for selected scenes. Only active "
                               "pictures consume <Picture N> numbers, so the "
                               "same @tag may receive a different native "
                               "number in different scenes."}),
            },
            "optional": {
                "previous": (REFERENCE_SCHEDULE_TYPE, {
                    "tooltip": "Optional schedule from another Picture, "
                               "Video, or Audio Schedule node. Chain nodes in "
                               "the stable priority order you want. Native "
                               "numbers are assigned only after inactive "
                               "entries are removed for the current scene."}),
            },
            "hidden": {
                "dynprompt": "DYNPROMPT",
                "unique_id": "UNIQUE_ID",
            },
        }

    RETURN_TYPES = (REFERENCE_SCHEDULE_TYPE, "STRING", "STRING")
    RETURN_NAMES = ("schedule", "schedule_fingerprint", "status")
    OUTPUT_TOOLTIPS = (
        "Reference schedule to chain into another entry or Scheduled Ref2VA.",
        "SHA-256 of every scheduled source, tag, and selector. Connect it to "
        "the Plan generation_fingerprint to protect checkpoint resume.",
        "Normalized tag, scene selector, entry count, and fingerprint.",
    )
    FUNCTION = "add"
    CATEGORY = "conditioning/minimax/contex_loop/references/legacy_schedule"
    DESCRIPTION = ("Add one scene-scheduled picture using a stable @tag. "
                   "Tags identify assets; they do not reserve native H3 "
                   "numbers. The final wrapper keeps only pictures active "
                   "in the current scene and numbers them compactly from "
                   "<Picture 1>. For example, if @picture_1 is removed or "
                   "inactive, @picture_2 automatically becomes <Picture 1>. "
                   "Write @picture_2 in the Plan prompt; the scheduler only "
                   "resolves aliases and never inserts prompt text.")

    def add(self, image, tag, scenes, previous=None,
            dynprompt=None, unique_id=None):
        mode = _downstream_reference_compliance(dynprompt, unique_id)
        try:
            if (torch is None or not torch.is_tensor(image) or image.ndim != 4 or
                    int(image.shape[0]) < 1 or int(image.shape[-1]) < 3):
                raise ValueError(
                    "Scheduled H3 picture must be an IMAGE tensor with shape "
                    "[batch,height,width,channels].")
            picture = image[:1]
            schedule = _append_scheduled_reference(
                previous, kind="picture", tag=tag, scenes=scenes,
                value=picture, content_hash=_tensor_fingerprint(picture),
                compliance_mode=mode)
        except (TypeError, ValueError) as exc:
            if mode == "disabled":
                return _skipped_reference_result(previous, "Picture reference", exc)
            raise
        entry = schedule["entries"][-1]
        status = "@%s picture on %s; %d sources; %s" % (
            entry["tag"], entry["scenes"], len(schedule["entries"]),
            schedule["fingerprint"][:12])
        return schedule, schedule["fingerprint"], status


class MiniMaxH3ScheduledVideoReference:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video": ("IMAGE", {
                    "tooltip": "Reference video frames at 24 fps. Use "
                               "Reference Video Prep when the loader source "
                               "has another frame rate."}),
                "tag": ("STRING", {
                    "default": "performance",
                    "tooltip": "Stable alias such as @performance. It is "
                               "NOT a native Video number. Active videos are "
                               "renumbered from <Video 1> per scene, so this "
                               "@tag remains valid if an earlier entry is "
                               "removed or inactive."}),
                "scenes": ("STRING", {
                    "default": "",
                    "tooltip": "Scenes where this video and its optional "
                               "paired soundtrack are active. Blank means all; "
                               "1, 1:4, and 1,3,5:8 are supported. Only "
                               "active videos consume <Video N> numbers."}),
                "audio_tag": ("STRING", {
                    "default": "",
                    "tooltip": "Alias for the paired soundtrack when audio "
                               "is connected. Blank derives @<video_tag>_audio. "
                               "This is also a stable alias, not a reserved "
                               "<Audio N> number."}),
                "timeline_mode": (list(REFERENCE_VIDEO_TIMELINE_MODES), {
                    "default": "restart_each_scene",
                    "tooltip": "restart_each_scene preserves the original "
                               "behavior: every active scene receives the "
                               "reference from frame 0. sequential advances "
                               "the 24 fps source along the Plan timeline, "
                               "repeating the same overlap as Motion Context. "
                               "Sequential mode requires Current Shot state "
                               "connected to Scheduled Ref2VA."}),
            },
            "optional": {
                "audio": ("AUDIO", {
                    "tooltip": "Optional soundtrack of this same reference "
                               "video. It stays index-paired with the video in "
                               "stock Ref2VA and receives its own audio tag."}),
                "previous": (REFERENCE_SCHEDULE_TYPE, {
                    "tooltip": "Optional preceding scheduled reference chain. "
                               "It sets stable priority order, not permanent "
                               "native label numbers."}),
            },
            "hidden": {
                "dynprompt": "DYNPROMPT",
                "unique_id": "UNIQUE_ID",
            },
        }

    RETURN_TYPES = (REFERENCE_SCHEDULE_TYPE, "STRING", "STRING")
    RETURN_NAMES = ("schedule", "schedule_fingerprint", "status")
    OUTPUT_TOOLTIPS = (
        "Reference schedule to chain into another entry or Scheduled Ref2VA.",
        "SHA-256 of all sources, tags, and selectors for checkpoint safety.",
        "Normalized video/audio tags, selector, entry count, and fingerprint.",
    )
    FUNCTION = "add"
    CATEGORY = "conditioning/minimax/contex_loop/references/legacy_schedule"
    DESCRIPTION = ("Add one scene-scheduled 24 fps video and an optional "
                   "index-paired soundtrack using stable @tags. Tags identify "
                   "assets while the wrapper assigns compact <Video N> and "
                   "<Audio N> labels from the entries active in each scene. "
                   "You may use @tags in Plan prompts when automatic renumbering "
                   "is useful; they are optional authoring aliases and this node "
                   "never inserts prompt text. Do not treat a tag suffix as a "
                   "fixed native number.")

    def add(self, video, tag, scenes, audio_tag,
            timeline_mode="restart_each_scene", audio=None, previous=None,
            dynprompt=None, unique_id=None):
        mode = _downstream_reference_compliance(dynprompt, unique_id)
        try:
            if (torch is None or not torch.is_tensor(video) or video.ndim != 4 or
                    int(video.shape[0]) < 5 or int(video.shape[-1]) < 3):
                raise ValueError(
                    "Scheduled H3 video must be an IMAGE batch containing at "
                    "least 5 frames.")
            paired_hash = ""
            if audio is not None:
                _validate_audio(audio, "Scheduled H3 reference-video audio")
                paired_hash = _audio_fingerprint(audio)
            schedule = _append_scheduled_reference(
                previous, kind="video", tag=tag, scenes=scenes,
                value=video, content_hash=_tensor_fingerprint(video), audio=audio,
                audio_tag=audio_tag, audio_hash=paired_hash,
                compliance_mode=mode, timeline_mode=timeline_mode)
        except (TypeError, ValueError) as exc:
            if mode == "disabled":
                return _skipped_reference_result(previous, "Video reference", exc)
            raise
        entry = schedule["entries"][-1]
        paired = (" + @%s" % entry["audio_tag"]
                  if entry.get("audio_tag") else "")
        status = "@%s%s video on %s; %s; %d sources; %s" % (
            entry["tag"], paired, entry["scenes"], entry["timeline_mode"],
            len(schedule["entries"]), schedule["fingerprint"][:12])
        return schedule, schedule["fingerprint"], status


class MiniMaxH3ScheduledAudioReference:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO", {
                    "tooltip": "Standalone reference audio. For a video's "
                               "synchronized soundtrack, use the paired audio "
                               "socket on Video Schedule instead."}),
                "tag": ("STRING", {
                    "default": "voice",
                    "tooltip": "Stable alias such as @voice. It is NOT a "
                               "native Audio number. Active audio references "
                               "are renumbered from <Audio 1> per scene, so "
                               "the @tag survives earlier entries being "
                               "removed or inactive."}),
                "scenes": ("STRING", {
                    "default": "",
                    "tooltip": "Scenes where this audio reference is active. "
                               "Blank means all; use 1, 1:4, or 1,3,5:8. "
                               "Only active audio references consume "
                               "<Audio N> numbers."}),
            },
            "optional": {
                "previous": (REFERENCE_SCHEDULE_TYPE, {
                    "tooltip": "Optional preceding scheduled reference chain. "
                               "It sets stable priority order, not permanent "
                               "native label numbers."}),
            },
            "hidden": {
                "dynprompt": "DYNPROMPT",
                "unique_id": "UNIQUE_ID",
            },
        }

    RETURN_TYPES = (REFERENCE_SCHEDULE_TYPE, "STRING", "STRING")
    RETURN_NAMES = ("schedule", "schedule_fingerprint", "status")
    OUTPUT_TOOLTIPS = (
        "Reference schedule to chain into another entry or Scheduled Ref2VA.",
        "SHA-256 of all sources, tags, and selectors for checkpoint safety.",
        "Normalized tag, scene selector, entry count, and fingerprint.",
    )
    FUNCTION = "add"
    CATEGORY = "conditioning/minimax/contex_loop/references/legacy_schedule"
    DESCRIPTION = ("Add one scene-scheduled standalone audio reference using "
                   "a stable @tag. The wrapper compactly renumbers active "
                   "audio as <Audio N> in each scene. Write the @tag and its "
                   "definition in the Plan prompt if you use the optional alias; "
                   "this node inserts no text.")

    def add(self, audio, tag, scenes, previous=None,
            dynprompt=None, unique_id=None):
        mode = _downstream_reference_compliance(dynprompt, unique_id)
        try:
            if audio is None:
                raise ValueError(
                    "Scheduled H3 standalone audio received no audio (None). "
                    "Most likely, this input is connected to Current Shot's "
                    "source_audio_slice while the Plan uses generated_audio; that "
                    "output is intentionally empty in generated_audio mode. For a "
                    "short voice/timbre reference, connect Load Audio directly to "
                    "Scheduled Audio Ref. For frame-exact source slices plus "
                    "generated-audio continuity, use source_plus_timeline and set "
                    "Assemble audio_source to generated if that is the final track "
                    "you want. Otherwise check that the upstream audio node is not "
                    "muted or bypassed, reconnect the AUDIO link, and queue again. "
                    "A playable browser preview does not guarantee that the socket "
                    "emitted AUDIO during this execution.")
            _validate_audio(audio, "Scheduled H3 standalone audio")
            schedule = _append_scheduled_reference(
                previous, kind="audio", tag=tag, scenes=scenes,
                value=audio, content_hash=_audio_fingerprint(audio),
                compliance_mode=mode)
        except (TypeError, ValueError) as exc:
            if mode == "disabled":
                return _skipped_reference_result(previous, "Audio reference", exc)
            raise
        entry = schedule["entries"][-1]
        status = "@%s audio on %s; %d sources; %s" % (
            entry["tag"], entry["scenes"], len(schedule["entries"]),
            schedule["fingerprint"][:12])
        return schedule, schedule["fingerprint"], status


class MiniMaxH3TaggedPictureReference:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {
                    "tooltip": "One reference picture. Ref2VA uses only the "
                               "first image when a batch is connected."}),
                "tag": ("STRING", {
                    "default": "hero_face",
                    "tooltip": "Stable alias used as @tag in scene prompts. "
                               "This picture is sent to H3 only in scenes "
                               "whose resolved prompt contains that tag."}),
            },
            "optional": {
                "previous": (TAGGED_REFERENCE_TYPE, {
                    "tooltip": "Optional preceding Tagged Ref chain. Chain "
                               "references in the priority order used for "
                               "scene-local native numbering."}),
            },
            "hidden": {"dynprompt": "DYNPROMPT", "unique_id": "UNIQUE_ID"},
        }

    RETURN_TYPES = (TAGGED_REFERENCE_TYPE, "STRING", "STRING")
    RETURN_NAMES = ("references", "reference_fingerprint", "status")
    FUNCTION = "add"
    CATEGORY = "conditioning/minimax/contex_loop/references/prompt_driven"
    DESCRIPTION = ("Register one picture under a stable @tag. No numeric "
                   "schedule is needed: Tagged Ref2VA activates it only when "
                   "the current scene prompt contains that tag, then assigns "
                   "the compact native <Picture N> label.")

    def add(self, image, tag, previous=None, dynprompt=None, unique_id=None):
        mode = _downstream_reference_compliance(dynprompt, unique_id)
        try:
            if (torch is None or not torch.is_tensor(image) or image.ndim != 4 or
                    int(image.shape[0]) < 1 or int(image.shape[-1]) < 3):
                raise ValueError(
                    "Tagged H3 picture must be an IMAGE tensor with shape "
                    "[batch,height,width,channels].")
            picture = image[:1]
            references = _append_tagged_reference(
                previous, kind="picture", tag=tag, value=picture,
                content_hash=_tensor_fingerprint(picture),
                compliance_mode=mode)
        except (TypeError, ValueError) as exc:
            if mode == "disabled":
                return _skipped_tagged_reference_result(
                    previous, "Tagged picture reference", exc)
            raise
        entry = references["entries"][-1]
        status = "@%s picture; prompt activated; %d sources; %s" % (
            entry["tag"], len(references["entries"]),
            references["fingerprint"][:12])
        return references, references["fingerprint"], status


class MiniMaxH3TaggedVideoReference:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video": ("IMAGE", {
                    "tooltip": "Reference video frames at 24 fps. Use "
                               "Reference Video Prep for other frame rates."}),
                "tag": ("STRING", {
                    "default": "performance",
                    "tooltip": "Stable video @tag. Mention this tag or its "
                               "paired audio tag in a scene prompt to activate "
                               "the reference block for that scene."}),
                "audio_tag": ("STRING", {
                    "default": "",
                    "tooltip": "Alias for connected paired audio. Blank "
                               "derives @<video_tag>_audio. Mentioning either "
                               "tag activates the paired video/audio block."}),
                "timeline_mode": (list(REFERENCE_VIDEO_TIMELINE_MODES), {
                    "default": "restart_each_scene",
                    "tooltip": "restart_each_scene begins the reference at "
                               "frame 0 whenever its tag is used. sequential "
                               "advances from the first Plan scene that uses "
                               "either paired tag and requires Current Shot "
                               "state on Tagged Ref2VA."}),
            },
            "optional": {
                "audio": ("AUDIO", {
                    "tooltip": "Optional synchronized soundtrack from the "
                               "same reference video."}),
                "previous": (TAGGED_REFERENCE_TYPE, {
                    "tooltip": "Optional preceding Tagged Ref chain."}),
            },
            "hidden": {"dynprompt": "DYNPROMPT", "unique_id": "UNIQUE_ID"},
        }

    RETURN_TYPES = (TAGGED_REFERENCE_TYPE, "STRING", "STRING")
    RETURN_NAMES = ("references", "reference_fingerprint", "status")
    FUNCTION = "add"
    CATEGORY = "conditioning/minimax/contex_loop/references/prompt_driven"
    DESCRIPTION = ("Register one 24 fps video and optional paired soundtrack "
                   "under stable @tags. Tagged Ref2VA activates the pair only "
                   "when either registered tag occurs in the current prompt.")

    def add(self, video, tag, audio_tag, timeline_mode="restart_each_scene",
            audio=None, previous=None, dynprompt=None, unique_id=None):
        mode = _downstream_reference_compliance(dynprompt, unique_id)
        try:
            if (torch is None or not torch.is_tensor(video) or video.ndim != 4 or
                    int(video.shape[0]) < 5 or int(video.shape[-1]) < 3):
                raise ValueError(
                    "Tagged H3 video must be an IMAGE batch containing at "
                    "least 5 frames.")
            paired_hash = ""
            if audio is not None:
                _validate_audio(audio, "Tagged H3 reference-video audio")
                paired_hash = _audio_fingerprint(audio)
            references = _append_tagged_reference(
                previous, kind="video", tag=tag, value=video,
                content_hash=_tensor_fingerprint(video), audio=audio,
                audio_tag=audio_tag, audio_hash=paired_hash,
                compliance_mode=mode, timeline_mode=timeline_mode)
        except (TypeError, ValueError) as exc:
            if mode == "disabled":
                return _skipped_tagged_reference_result(
                    previous, "Tagged video reference", exc)
            raise
        entry = references["entries"][-1]
        paired = " + @%s" % entry["audio_tag"] if entry.get("audio_tag") else ""
        status = "@%s%s video; prompt activated; %s; %d sources; %s" % (
            entry["tag"], paired, entry["timeline_mode"],
            len(references["entries"]), references["fingerprint"][:12])
        return references, references["fingerprint"], status


class MiniMaxH3TaggedAudioReference:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO", {
                    "tooltip": "Standalone voice, music, or sound reference."}),
                "tag": ("STRING", {
                    "default": "voice",
                    "tooltip": "Stable audio @tag. This reference is sent to "
                               "H3 only when the current prompt contains it."}),
                "timeline_mode": (list(REFERENCE_AUDIO_TIMELINE_MODES), {
                    "default": "standalone",
                    "tooltip": "standalone sends this AUDIO value unchanged "
                               "whenever @tag is active. source_timeline treats "
                               "it as the same full source track used by Loop "
                               "Start and derives the exact current-scene slice "
                               "inside Tagged Ref2VA. This preserves a static "
                               "fingerprint-to-Plan connection without a "
                               "circular Current Shot link."}),
                "align_audio_reference": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "source_timeline only. Apply the same optional "
                               "15.070s-safe H3 audio-grid cap as Current Shot. "
                               "The full source track and final assembled audio "
                               "are not modified."}),
            },
            "optional": {
                "previous": (TAGGED_REFERENCE_TYPE, {
                    "tooltip": "Optional preceding Tagged Ref chain."}),
            },
            "hidden": {"dynprompt": "DYNPROMPT", "unique_id": "UNIQUE_ID"},
        }

    RETURN_TYPES = (TAGGED_REFERENCE_TYPE, "STRING", "STRING")
    RETURN_NAMES = ("references", "reference_fingerprint", "status")
    FUNCTION = "add"
    CATEGORY = "conditioning/minimax/contex_loop/references/prompt_driven"
    DESCRIPTION = ("Register audio under a stable @tag. It can remain a fixed "
                   "standalone reference or hold the full Loop source track "
                   "while Tagged Ref2VA derives an exact per-scene timeline "
                   "slice without creating a fingerprint cycle.")

    def add(self, audio, tag, timeline_mode="standalone",
            align_audio_reference=False, previous=None, dynprompt=None,
            unique_id=None):
        mode = _downstream_reference_compliance(dynprompt, unique_id)
        try:
            if audio is None:
                raise ValueError(
                    "Tagged H3 audio received no AUDIO value. If Current Shot "
                    "uses generated_audio, source_audio_slice is intentionally "
                    "empty; connect a voice/music loader directly instead.")
            _validate_audio(audio, "Tagged H3 standalone audio")
            references = _append_tagged_reference(
                previous, kind="audio", tag=tag, value=audio,
                content_hash=_audio_fingerprint(audio), compliance_mode=mode,
                timeline_mode=timeline_mode,
                align_audio_reference=align_audio_reference)
        except (TypeError, ValueError) as exc:
            if mode == "disabled":
                return _skipped_tagged_reference_result(
                    previous, "Tagged audio reference", exc)
            raise
        entry = references["entries"][-1]
        aligned = "; H3-grid aligned" if entry.get(
            "align_audio_reference") else ""
        status = "@%s audio; prompt activated; %s%s; %d sources; %s" % (
            entry["tag"], entry["timeline_mode"], aligned,
            len(references["entries"]),
            references["fingerprint"][:12])
        return references, references["fingerprint"], status


class MiniMaxH3ScheduledReferenceToVideo:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip": ("CLIP", {
                    "tooltip": "MiniMax H3 text encoder used by stock Ref2VA."}),
                "vae": ("VAE", {
                    "tooltip": "MiniMax H3 video VAE used to encode active "
                               "pictures and videos."}),
                "audio_vae": ("VAE", {
                    "tooltip": "MiniMax H3 audio VAE used to encode active "
                               "standalone or video-paired audio references."}),
                "reference_schedule": (REFERENCE_SCHEDULE_TYPE, {
                    "tooltip": "Final chain from the scheduled Picture, "
                               "Video, and Audio reference nodes. For each "
                               "scene it removes inactive entries, compactly "
                               "assigns native labels by type, then resolves "
                               "stable @tags used in the Plan prompt."}),
                "clip_index": ("INT", {
                    "default": 1, "min": 1, "max": MAX_SHOTS,
                    "tooltip": "Current one-based scene. Connect Current "
                               "Shot clip_index so the active refs change on "
                               "each recursive iteration."}),
                "clip_count": ("INT", {
                    "default": 1, "min": 1, "max": MAX_SHOTS,
                    "tooltip": "Total scenes. Connect Current Shot clip_count "
                               "to validate schedule bounds."}),
                "prompt": ("STRING", {
                    "default": "", "multiline": True,
                    "dynamicPrompts": True,
                    "tooltip": "Scene prompt may use optional stable aliases such as "
                               "@hero_face and @performance. The wrapper "
                               "replaces them with native H3 labels for the "
                               "current scene. Example: @picture_2 becomes "
                               "<Picture 1> if it is the only active picture. "
                               "Aliases are a scheduler convenience, not required "
                               "H3 syntax. Native labels remain user-managed. All "
                               "reference definitions remain visible and "
                               "editable in the Plan or Prompt Editor."}),
                "width": ("INT", {
                    "default": 960, "min": 32, "max": 4096, "step": 32,
                    "tooltip": "Generation width forwarded unchanged to "
                               "stock MiniMax H3 Reference to Video."}),
                "height": ("INT", {
                    "default": 544, "min": 32, "max": 4096, "step": 32,
                    "tooltip": "Generation height forwarded unchanged to "
                               "stock MiniMax H3 Reference to Video."}),
                "length": ("INT", {
                    "default": 124, "min": 5, "max": 3600, "step": 17,
                    "tooltip": "H3-valid raw frame count from Current Shot."}),
                "ref_image_size": (["match", "max"], {
                    "default": "match",
                    "tooltip": "Stock Ref2VA picture sizing: match limits "
                               "each picture to generation pixel area; max "
                               "uses its high-fidelity 2048px-short-edge path."}),
            },
            "optional": {
                "state": (STATE_TYPE, {
                    "tooltip": "Current Shot state. Required only when an "
                               "active Scheduled Video Ref uses sequential "
                               "timeline mode; it supplies exact scene starts "
                               "and Motion Context overlap timing."}),
                "prompt_compliance": (list(REFERENCE_COMPLIANCE_MODES), {
                    "default": "strict",
                    "tooltip": "strict: compile active @tags and block unknown "
                               "or inactive tags. soft: compile active tags but "
                               "warn and preserve unresolved tags. disabled: "
                               "make every scheduler-authored check non-blocking, "
                               "pass the prompt unchanged, omit missing/invalid "
                               "scheduled media (including an empty generated-"
                               "audio source slice), and keep only stock H3's "
                               "supported reference capacity. Failures in CLIP, "
                               "VAE, sampling, or checkpoint execution remain "
                               "real execution errors."}),
            },
        }

    @classmethod
    def VALIDATE_INPUTS(cls, prompt_compliance="strict"):
        try:
            _reference_compliance_mode(prompt_compliance)
        except ValueError as exc:
            return str(exc)
        return True

    RETURN_TYPES = ("CONDITIONING", "LATENT", "STRING", "STRING", "STRING")
    RETURN_NAMES = (
        "positive", "latent", "compiled_prompt", "active_references",
        "schedule_fingerprint")
    OUTPUT_TOOLTIPS = (
        "Positive conditioning produced by stock MiniMax H3 Ref2VA.",
        "Empty MiniMax H3 AV latent produced by stock Ref2VA.",
        "Exact prompt sent to H3 after stable aliases compile to native labels.",
        "Human-readable mapping for this scene, for example "
        "@picture_2 -> <Picture 1>. Use it to verify renumbering.",
        "Full schedule fingerprint. Connect the schedule node's matching "
        "fingerprint to Plan generation_fingerprint when all scheduled "
        "sources are static.",
    )
    FUNCTION = "apply"
    CATEGORY = "conditioning/minimax/contex_loop/references/legacy_schedule"
    DESCRIPTION = ("Select scheduled references for the current scene, "
                   "remove inactive entries, and compactly number each media "
                   "type from 1. Stable @tags in the Plan prompt are compiled "
                   "to those scene-local native labels before core MiniMax H3 "
                   "Ref2VA runs; the scheduler inserts no prompt text. A tag "
                   "named @picture_2 may therefore map to <Picture 1>; inspect "
                   "the active_references output for the exact mapping.")

    def apply(self, clip, vae, audio_vae, reference_schedule, clip_index,
              clip_count, prompt, width, height, length,
              ref_image_size="match", state=None,
              prompt_compliance="strict"):
        if GraphBuilder is None:
            raise RuntimeError(
                "Scheduled H3 Ref2VA requires ComfyUI GraphBuilder.")
        compiled, summary, bindings = _compile_scheduled_reference_prompt(
            reference_schedule, clip_index, clip_count, prompt,
            prompt_compliance)
        graph = GraphBuilder()
        ref2va = graph.node("MiniMaxH3ReferenceToVideo", "ScheduledRef2VA")
        for key, value in (
                ("clip", clip), ("vae", vae), ("audio_vae", audio_vae),
                ("prompt", compiled), ("width", int(width)),
                ("height", int(height)), ("length", int(length)),
                ("ref_image_size", ref_image_size)):
            ref2va.set_input(key, value)
        for index, entry in enumerate(bindings["pictures"]):
            ref2va.set_input(
                "ref_images.ref_image_%d" % index, entry["value"])
        slice_details = []
        for index, entry in enumerate(bindings["videos"]):
            video, paired_audio, detail = _scheduled_video_reference_slice(
                entry, state, clip_index, clip_count, length)
            ref2va.set_input(
                "ref_videos.ref_video_%d" % index, video)
            if paired_audio is not None:
                ref2va.set_input(
                    "ref_video_audios.ref_video_audio_%d" % index,
                    paired_audio)
            if detail:
                slice_details.append(detail)
        for index, entry in enumerate(bindings["audios"]):
            ref2va.set_input(
                "ref_audios.ref_audio_%d" % index, entry["value"])
        if isinstance(reference_schedule, dict) and reference_schedule.get(
                "fingerprint"):
            fingerprint = str(reference_schedule["fingerprint"])
        elif _reference_compliance_mode(prompt_compliance) == "disabled":
            fingerprint = _fingerprint({
                "reference_schedule": "ignored",
                "prompt_compliance": "disabled",
            })
        else:
            raise ValueError(
                "Scheduled references have no valid schedule fingerprint.")
        if slice_details:
            summary += "; " + "; ".join(slice_details)
        return {
            "result": (
                ref2va.out(0), ref2va.out(1), compiled, summary, fingerprint),
            "expand": graph.finalize(),
        }


class MiniMaxH3TaggedReferenceToVideo:
    @classmethod
    def INPUT_TYPES(cls):
        scheduled = MiniMaxH3ScheduledReferenceToVideo.INPUT_TYPES()
        required = dict(scheduled["required"])
        required.pop("reference_schedule")
        required["prompt"] = ("STRING", {
            "default": "", "multiline": True, "dynamicPrompts": True,
            "tooltip": "Resolved current-scene prompt. Mention a registered "
                       "@tag to activate that asset for this scene. Only "
                       "registered reference tags are replaced with native "
                       "H3 labels; unrelated @syntax remains unchanged. The "
                       "node never inserts reference definitions or other "
                       "semantic prompt text."})
        required["references"] = (TAGGED_REFERENCE_TYPE, {
            "tooltip": "Final Tagged Picture/Video/Audio Ref chain. A source "
                       "is active only when its registered @tag occurs in the "
                       "resolved prompt for the current scene."})
        # Preserve the natural graph order: model inputs, references, current
        # scene metadata, prompt, and generation settings.
        ordered = {}
        for name in (
                "clip", "vae", "audio_vae", "references", "clip_index",
                "clip_count", "prompt", "width", "height", "length",
                "ref_image_size"):
            ordered[name] = required[name]
        optional = {
            "state": (STATE_TYPE, {
                "tooltip": "Current Shot state. Required by tagged video "
                           "sequential mode and tagged audio source_timeline. "
                           "It supplies the exact scene and overlap-aware "
                           "source offsets without routing dynamic media back "
                           "through the Plan fingerprint."}),
            "reference_policy": (list(REFERENCE_COMPLIANCE_MODES), {
                "default": "strict",
                "tooltip": "strict validates sources and stock H3 reference "
                           "capacity. soft retains those structural checks. "
                           "disabled makes pack-authored validation warning-only, "
                           "skips missing/invalid tagged media, and passes @tags "
                           "unchanged. Unregistered @syntax is always preserved "
                           "because it may represent a subject or dialogue tag."}),
        }
        return {"required": ordered, "optional": optional}

    @classmethod
    def VALIDATE_INPUTS(cls, reference_policy="strict"):
        try:
            _reference_compliance_mode(reference_policy)
        except ValueError as exc:
            return str(exc)
        return True

    RETURN_TYPES = ("CONDITIONING", "LATENT", "STRING", "STRING", "STRING")
    RETURN_NAMES = (
        "positive", "latent", "compiled_prompt", "active_references",
        "reference_fingerprint")
    OUTPUT_TOOLTIPS = (
        "Positive conditioning produced by stock MiniMax H3 Ref2VA.",
        "Empty MiniMax H3 AV latent produced by stock Ref2VA.",
        "Exact prompt sent to H3 after used registered tags become native labels.",
        "Scene-local mapping of prompt-used tags to native reference labels.",
        "Fingerprint of the complete registered source set for Plan checkpoint safety.",
    )
    FUNCTION = "apply"
    CATEGORY = "conditioning/minimax/contex_loop/references/prompt_driven"
    DESCRIPTION = ("Prompt-driven Ref2VA with no numeric reference schedule. "
                   "Each scene activates only registered @tags present in its "
                   "resolved prompt, compactly renumbers those assets to native "
                   "H3 labels, and leaves unrelated @syntax untouched.")

    def apply(self, clip, vae, audio_vae, references, clip_index,
              clip_count, prompt, width, height, length,
              ref_image_size="match", state=None,
              reference_policy="strict"):
        if GraphBuilder is None:
            raise RuntimeError(
                "Tagged H3 Ref2VA requires ComfyUI GraphBuilder.")
        compiled, summary, bindings = _compile_tagged_reference_prompt(
            references, clip_index, clip_count, prompt, reference_policy)
        graph = GraphBuilder()
        ref2va = graph.node("MiniMaxH3ReferenceToVideo", "TaggedRef2VA")
        for key, value in (
                ("clip", clip), ("vae", vae), ("audio_vae", audio_vae),
                ("prompt", compiled), ("width", int(width)),
                ("height", int(height)), ("length", int(length)),
                ("ref_image_size", ref_image_size)):
            ref2va.set_input(key, value)
        for index, entry in enumerate(bindings["pictures"]):
            ref2va.set_input(
                "ref_images.ref_image_%d" % index, entry["value"])
        slice_details = []
        for index, entry in enumerate(bindings["videos"]):
            video, paired_audio, detail = _scheduled_video_reference_slice(
                entry, state, clip_index, clip_count, length)
            ref2va.set_input("ref_videos.ref_video_%d" % index, video)
            if paired_audio is not None:
                ref2va.set_input(
                    "ref_video_audios.ref_video_audio_%d" % index,
                    paired_audio)
            if detail:
                slice_details.append(detail)
        for index, entry in enumerate(bindings["audios"]):
            audio, detail = _tagged_audio_reference_value(
                entry, state, clip_index, clip_count, length)
            ref2va.set_input(
                "ref_audios.ref_audio_%d" % index, audio)
            if detail:
                slice_details.append(detail)
        if isinstance(references, dict) and references.get("fingerprint"):
            fingerprint = str(references["fingerprint"])
        elif _reference_compliance_mode(reference_policy) == "disabled":
            fingerprint = _fingerprint({
                "tagged_references": "ignored",
                "reference_policy": "disabled",
            })
        else:
            raise ValueError(
                "Tagged references have no valid reference fingerprint.")
        if slice_details:
            summary += "; " + "; ".join(slice_details)
        return {
            "result": (
                ref2va.out(0), ref2va.out(1), compiled, summary, fingerprint),
            "expand": graph.finalize(),
        }


def _external_prelude_paths(plan: dict[str, Any], fingerprint: str) -> dict[str, str]:
    directory = os.path.join(_run_dir(plan), "source")
    stem = "existing_video_%s" % str(fingerprint)[:20]
    return {
        "video": os.path.join(directory, stem + ".mp4"),
        "audio": os.path.join(directory, stem + ".safetensors"),
        "metadata": os.path.join(directory, stem + ".json"),
    }


def _save_external_audio(audio: dict[str, Any], path: str) -> None:
    if _st_save is None:
        raise RuntimeError(
            "safetensors is required to preserve existing-video audio.")
    waveform, sample_rate = _audio_waveform_3d(
        audio, "H3 existing-video prelude audio")
    temporary = "%s.%s.tmp" % (path, uuid.uuid4().hex)
    try:
        _st_save({"waveform": waveform.detach().cpu().contiguous()}, temporary,
                 metadata={
                     "format": "h3_existing_video_audio_v1",
                     "sample_rate": str(sample_rate),
                 })
        os.replace(temporary, path)
    finally:
        _safe_unlink(temporary)


class MiniMaxH3ChainExternalVideo:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "plan": (PLAN_TYPE, {
                    "tooltip": "The active H3 Chain Plan. Its canvas, crop, "
                               "context length, quality, and run folder are "
                               "used to prepare the imported video tail."}),
                "source_fps": ("FLOAT", {
                    "default": 24.0, "min": 0.001, "max": 1000.0,
                    "step": 0.001,
                    "tooltip": "Actual frame rate represented by source_frames "
                               "when using the separate IMAGE/AUDIO route. It "
                               "is ignored when source_video is connected, "
                               "because native VIDEO carries its own exact FPS."}),
                "prepend_original": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Persist a normalized copy of the complete "
                               "existing video and place it before generated "
                               "scenes during partial/final assembly. Disable "
                               "to output only the extension."}),
            },
            "optional": {
                "source_video": ("VIDEO", {
                    "tooltip": "Native ComfyUI VIDEO from core Load Video or "
                               "another VIDEO-producing loader. Its frames, "
                               "embedded audio, and exact FPS are decoded "
                               "directly. Connect either source_video or "
                               "source_frames, not both."}),
                "source_frames": ("IMAGE", {
                    "tooltip": "Decoded IMAGE batch from VHS or another video "
                               "loader. Set source_fps to the loader's actual "
                               "output rate. Connect either source_frames or "
                               "source_video, not both."}),
                "source_audio": ("AUDIO", {
                    "tooltip": "Optional soundtrack decoded from the existing "
                               "video. Use it with source_frames, or connect it "
                               "to override a native VIDEO's embedded audio. "
                               "Its tail can seed scene 1 audio; when prepend "
                               "is enabled it is preserved before the extension."}),
            },
        }

    RETURN_TYPES = (EXTERNAL_CONTEXT_TYPE, "STRING")
    RETURN_NAMES = ("external_context", "status")
    OUTPUT_TOOLTIPS = (
        "Typed imported-video tail for Loop Start. It contains only the small "
        "recursive context plus verified prelude artifact paths.",
        "Source/normalized frame counts, context duration, audio availability, "
        "and prepend status.",
    )
    FUNCTION = "prepare"
    CATEGORY = "conditioning/minimax/contex_loop"
    DESCRIPTION = ("Turn a native VIDEO or separately decoded IMAGE/AUDIO "
                   "video into scene 1's visual/audio predecessor, with "
                   "optional original-video prepend during assembly.")

    def prepare(self, plan, source_frames=None, source_fps=24.0,
                prepend_original=True, source_audio=None, source_video=None):
        source_frames, source_audio, source_fps, input_route = (
            _resolve_video_inputs(
                source_video, source_frames, source_audio, source_fps,
                "H3 existing-video adapter"))
        if torch is None or not torch.is_tensor(source_frames):
            raise ValueError("H3 existing-video source_frames must be an IMAGE tensor.")
        if source_frames.ndim != 4 or int(source_frames.shape[-1]) < 3:
            raise ValueError(
                "H3 existing-video source_frames must be "
                "[frames,height,width,channels]; got %r." %
                (getattr(source_frames, "shape", None),))
        cfg = plan["compatibility"]
        context_length = int(cfg["context_length"])
        indices = _external_video_frame_indices(
            int(source_frames.shape[0]), float(source_fps))
        normalized_count = int(indices.numel())
        if normalized_count < context_length:
            raise ValueError(
                "H3 existing video becomes %d frames at 24 fps, but this plan "
                "needs at least %d context frames. Supply a longer video or "
                "reduce context_length." % (normalized_count, context_length))

        selected_indices = indices if bool(prepend_original) else indices[-context_length:]
        selected = source_frames.index_select(
            0, selected_indices.to(device=source_frames.device))
        normalized = _resize(
            selected, int(cfg["width"]), int(cfg["height"]), cfg["crop"])
        context_frames = _tensor_cpu_clone(normalized[-context_length:])

        normalized_audio = None
        context_audio = None
        if source_audio is not None:
            _source_waveform, source_rate = _audio_waveform_3d(
                source_audio, "H3 existing-video source audio")
            normalized_samples = int(round(
                normalized_count / float(FPS) * source_rate))
            normalized_audio = _resample_audio_exact(
                source_audio, source_rate, normalized_samples,
                int(_source_waveform.shape[1]),
                "H3 existing-video source audio")
            configured_audio_frames = int(cfg["audio_context_length"])
            first_continuation_mode = plan["shots"][0].get(
                "continuation_mode", cfg.get("continuation_mode", "guide"))
            if first_continuation_mode == "masked_av":
                # A clean target AV prefix is one physical interval. Unlike
                # guide mode, masked continuation cannot use an independently
                # sized audio-reference window.
                audio_context_frames = min(normalized_count, context_length)
            else:
                audio_context_frames = min(
                    normalized_count,
                    configured_audio_frames or context_length)
            context_samples = int(round(
                audio_context_frames / float(FPS) * source_rate))
            context_audio = {
                "waveform": _tensor_cpu_clone(
                    normalized_audio["waveform"][..., -context_samples:]),
                "sample_rate": source_rate,
            }

        external_context = {
            "version": 1,
            "base_plan_hash": str(plan.get("base_plan_hash") or plan["plan_hash"]),
            "context_frames": context_frames,
            "context_audio": context_audio,
            "prelude": None,
        }
        contract = _external_context_contract(external_context)
        external_context["context_hash"] = _fingerprint(contract)

        if bool(prepend_original):
            # The complete normalized source is needed only long enough to
            # persist an immutable stream-copy-compatible prelude. Recursive
            # state receives the short tail above, never this full tensor.
            if int(normalized.shape[0]) != normalized_count:
                raise RuntimeError(
                    "H3 existing-video normalization produced an unexpected "
                    "frame count.")
            content_fingerprint = _fingerprint({
                "frames": _tensor_fingerprint(normalized),
                "audio": (_audio_fingerprint(normalized_audio)
                          if normalized_audio is not None else "none"),
                "fps": FPS,
                "width": int(cfg["width"]),
                "height": int(cfg["height"]),
                "crop": cfg["crop"],
                "crf": int(plan["segment_crf"]),
            })
            paths = _external_prelude_paths(plan, content_fingerprint)
            os.makedirs(os.path.dirname(paths["video"]), exist_ok=True)
            cached = None
            if os.path.isfile(paths["metadata"]):
                try:
                    cached = _read_json(paths["metadata"])
                except (OSError, ValueError, json.JSONDecodeError):
                    cached = None
            video_reusable = bool(
                isinstance(cached, dict) and
                cached.get("source_fingerprint") == content_fingerprint and
                os.path.isfile(paths["video"]) and
                str(cached.get("video_sha256") or "") ==
                _file_sha256(paths["video"]))
            if not video_reusable:
                _write_segment_video(
                    normalized, paths["video"], FPS, int(plan["segment_crf"]),
                    metadata={
                        "title": "Existing video before H3 extension",
                        "comment": "Normalized 24 fps prelude for %s" %
                                   plan["run_name"],
                    })
            if normalized_audio is not None:
                audio_reusable = bool(
                    isinstance(cached, dict) and
                    cached.get("source_fingerprint") == content_fingerprint and
                    os.path.isfile(paths["audio"]) and
                    str(cached.get("audio_sha256") or "") ==
                    _file_sha256(paths["audio"]))
                if not audio_reusable:
                    _save_external_audio(normalized_audio, paths["audio"])
            prelude = {
                "format": "h3_existing_video_prelude_v1",
                "prepend": True,
                "source_fingerprint": content_fingerprint,
                "frame_count": normalized_count,
                "fps": FPS,
                "width": int(cfg["width"]),
                "height": int(cfg["height"]),
                "duration_seconds": normalized_count / float(FPS),
                "video": _relative_output_path(paths["video"]),
                "video_sha256": _file_sha256(paths["video"]),
                "source_fps": float(source_fps),
            }
            if normalized_audio is not None:
                prelude.update({
                    "audio": _relative_output_path(paths["audio"]),
                    "audio_sha256": _file_sha256(paths["audio"]),
                    "audio_sample_rate": int(normalized_audio["sample_rate"]),
                })
            _atomic_json(paths["metadata"], prelude)
            prelude["metadata"] = _relative_output_path(paths["metadata"])
            external_context["prelude"] = prelude

        status = (
            "%s: %d source frames at %.3f fps -> %d frames at %d fps; "
            "%d-frame (%.3fs) context; audio %s; original %s" %
            (input_route, int(source_frames.shape[0]), float(source_fps),
             normalized_count, FPS, context_length,
             context_length / float(FPS),
             "ready" if context_audio is not None else "not supplied",
             "will be prepended" if bool(prepend_original)
             else "will not be prepended"))
        return (external_context, status)


class MiniMaxH3ChainPlan:
    @classmethod
    def INPUT_TYPES(cls):
        sample = json.dumps({
            "shots": [
                {"id": "intro", "prompt": "Describe the opening shot."},
                {"id": "continuation", "prompt": "Continue the same take."},
            ]
        }, indent=2)
        return {
            "required": {
                "plan_json": ("STRING", {
                    "default": sample, "multiline": True,
                    "dynamicPrompts": False,
                    "tooltip": "The editable production plan behind the large "
                               "Scene Plan interface: shared prompt, ordered "
                               "scene prompts, optional lengths, sampler steps, "
                               "and per-scene seed overrides. Use the visual "
                               "editor for normal work and Raw JSON only for "
                               "import, export, or advanced editing. Reference "
                               "media is connected elsewhere; this JSON only "
                               "mentions its @tags or native <Picture/Video/Audio "
                               "N> labels."}),
                "run_name": ("STRING", {
                    "default": "h3_chain",
                    "tooltip": "Identity of one render history and its folder "
                               "under ComfyUI output/h3_chains. Keep it unchanged "
                               "to resume or regenerate scenes from that same "
                               "production. Use a new name for a separate render; "
                               "reusing a name intentionally exposes that run's "
                               "existing checkpoints to Review Gate and resume."}),
                "generation_fingerprint": ("STRING", {
                    "default": "",
                    "tooltip": "Checkpoint compatibility tag for generation "
                               "inputs not stored in plan_json. Connect Scheduled "
                               "Ref2VA's schedule_fingerprint when using scheduled "
                               "references. Otherwise enter/change a stable tag "
                               "whenever the model, VAE, LoRA, global references, "
                               "CFG, sampler, or scheduler changes. Resume rejects "
                               "a mismatched fingerprint instead of mixing runs."}),
                "width": ("INT", {
                    "default": 960, "min": 32, "max": 4096, "step": 32,
                    "tooltip": "Generation width for every scene. Connect the "
                               "Plan width output to the stock Ref2VA/I2V node "
                               "so its latent always matches the plan."}),
                "height": ("INT", {
                    "default": 544, "min": 32, "max": 4096, "step": 32,
                    "tooltip": "Generation height for every scene. Connect the "
                               "Plan height output to the stock Ref2VA/I2V node "
                               "so its latent always matches the plan."}),
                "context_length": (list(H3_CONTEXT_LENGTHS), {
                    "default": 22,
                    "tooltip": "Number of previous-scene video frames used to "
                               "continue motion. Use 22 for guide mode, 18 for "
                               "sliding_history, and 39 for masked_av so the AV "
                               "clocks meet exactly. "
                               "With head "
                               "anchors, those frames are regenerated at the "
                               "start and Loop Trim removes them, so later scenes "
                               "deliver raw scene frames minus context_length. "
                               "Larger values strengthen motion continuity but "
                               "produce fewer new frames per scene. This does not "
                               "control reference-audio duration."}),
                "encode_mode": (["video", "frames"], {
                    "default": "video",
                    "tooltip": "How the carried visual overlap is encoded. Use "
                               "video (recommended) to preserve the previous "
                               "frames as one motion-bearing latent clip. frames "
                               "creates separate still-image anchors, costs more "
                               "conditioning space, and is mainly for diagnosing "
                               "or experimenting with anchor behavior."}),
                "anchor_mode": (["head", "before"], {
                    "default": "head",
                    "tooltip": "Where previous frames sit on the next scene's "
                               "timeline. head is the tested default: it repeats "
                               "the overlap at the beginning, and Loop Trim must "
                               "remove exactly trim_frames. before places context "
                               "at negative time and returns no repeated head; use "
                               "it only for workflows deliberately built around "
                               "that experimental timing."}),
                "crop": (["disabled", "center"], {
                    "default": "disabled",
                    "tooltip": "How saved context frames are fitted when their "
                               "shape differs from the Plan canvas. disabled "
                               "resizes directly to width x height and may change "
                               "aspect ratio. center preserves aspect ratio, then "
                               "center-crops overflow. It does not crop Ref2VA "
                               "picture/video reference inputs."}),
                "audio_mode": (list(AUDIO_MODES), {
                    "default": "source_track",
                    "tooltip": "Controls timeline continuity and final audio; it "
                               "does NOT enable or disable @voice/<Audio N> "
                               "references. For a finished prerecorded voice, "
                               "dialogue, or song that must remain exact, choose "
                               "source_track: wire the full track to Loop Start "
                               "and Assemble, and feed Current Shot's exact slice "
                               "to Ref2VA/Scheduled Audio. For a short @voice "
                               "identity/timbre reference while H3 generates new "
                               "speech and sound, choose generated_audio: no full "
                               "source track is required, connect the audio VAE to "
                               "Loop Context, and save trimmed generated audio. "
                               "source_plus_timeline provides both an exact source "
                               "slice and previous generated-audio context; it is "
                               "experimental and usually not the first choice."}),
                "audio_context_length": ("INT", {
                    "default": 22, "min": 0, "max": 240,
                    "tooltip": "Amount of previous generated sound carried into "
                               "the next scene, measured in 24-fps video frames. "
                               "0 means use context_length; 22 is the tested "
                               "explicit value. Only generated_audio and "
                               "source_plus_timeline use it. source_track ignores "
                               "it because each scene receives a fresh exact "
                               "slice from the external track. masked_av also "
                               "ignores it and always preserves audio for the "
                               "same duration as context_length."}),
                "default_duration_seconds": ("FLOAT", {
                    "default": 15.0, "min": 0.1,
                    "max": MAX_H3_FRAMES / FPS, "step": 0.01,
                    "tooltip": "Fallback duration only when the scene and JSON "
                               "defaults both omit a duration/length. H3 cannot "
                               "generate every frame count, so seconds round UP "
                               "to the next valid 17k+5 raw length. In head mode, "
                               "continuation scenes then lose context_length "
                               "repeated frames from their delivered duration."}),
                "default_steps": ("INT", {
                    "default": 20, "min": 1, "max": 10000,
                    "tooltip": "Fallback sampler steps only when a scene and the "
                               "JSON defaults both omit steps. A value set under "
                               "a scene's Show advanced section overrides this."}),
                "base_seed": ("INT", {
                    "default": 0, "min": 0, "max": MAX_SEED,
                    "tooltip": "Base used to derive a stable different seed for "
                               "each scene that has no explicit seed. Review "
                               "Gate's Reroll seed does NOT change base_seed; it "
                               "writes an explicit override into that scene's "
                               "always-visible Scene seed field, leaving every other "
                               "scene reproducible and checkpoint-compatible."}),
                "segment_crf": ("INT", {
                    "default": 18, "min": 0, "max": 51,
                    "tooltip": "H.264 quality for each saved scene MP4 (and "
                               "normalized imported prelude): lower means higher "
                               "quality and larger files. 18 is visually high "
                               "quality; 0 is lossless and 51 is lowest quality. "
                               "This does not change model sampling or the saved "
                               "safetensors continuation checkpoint."}),
                "video_blend_frames": ("INT", {
                    "default": 0, "min": 0, "max": 243,
                    "tooltip": "Optional visual blend at each scene boundary, "
                               "in frames. 0 preserves the current hard-cut "
                               "assembly. A positive value must not exceed "
                               "context_length and requires head anchors. "
                               "Connect this output to Loop Trim's "
                               "retain_overlap_frames and its "
                               "images_with_overlap output to Segment Save. "
                               "Final and partial videos are re-encoded with a "
                               "linear cumulative blend; audio timing and the "
                               "delivered duration remain unchanged."}),
                "continuation_mode": (list(CONTINUATION_MODES), {
                    "default": "guide",
                    "tooltip": "Inherited default for scenes without a "
                               "per-scene continuation override. guide keeps "
                               "the established Motion Context "
                               "path: previous AV is supplied as fixed guide "
                               "rows while the overlap is regenerated. "
                               "sliding_history mirrors WanGP: 17 frames are "
                               "history and the 18th is a frame-zero boundary; "
                               "only that boundary is repeated and trimmed. "
                               "masked_av (experimental) VAE-encodes the "
                               "previous video tail into the current target "
                               "latent, copies its sampled audio tail, and "
                               "protects both with per-stream denoise masks. "
                               "masked_av requires video/head, context >= 5, "
                               "the Chain Context latent output wired to the "
                               "sampler, and native or compatible H3 AV-mask "
                               "support."}),
            },
            "optional": {
                "plan_json_input": ("STRING", {
                    "forceInput": True,
                    "tooltip": "Optional complete scene-plan JSON supplied by "
                               "another node, such as an LLM story director or "
                               "reusable STRING source. A non-empty connected "
                               "value overrides the visual editor's internal "
                               "plan_json for this execution and passes through "
                               "the same normalization and validation. Empty or "
                               "disconnected input uses the internal plan_json "
                               "unchanged."}),
            },
        }

    RETURN_TYPES = (PLAN_TYPE, "STRING", "INT", "INT", "INT", "INT")
    RETURN_NAMES = ("plan", "summary", "clip_count", "width", "height",
                    "video_blend_frames")
    OUTPUT_TOOLTIPS = (
        "Validated chain plan. Connect it to Loop Start and, for recovery, "
        "Manifest Load.",
        "Human-readable scene count, delivered duration, and compatibility "
        "summary.",
        "Number of scenes in the plan.",
        "Validated generation width; connect to the stock H3 conditioning node.",
        "Validated generation height; connect to the stock H3 conditioning node.",
        "Configured boundary blend length. Connect it to Loop Trim's "
        "retain_overlap_frames input.",
    )
    FUNCTION = "build"
    CATEGORY = "conditioning/minimax/contex_loop"
    DESCRIPTION = ("Parse and validate a frame-exact MiniMax H3 shot plan. "
                   "The plan computes valid lengths, overlaps, audio windows, "
                   "seeds, and checkpoint compatibility hashes.")

    def build(self, plan_json, run_name, generation_fingerprint, width, height,
              context_length,
              encode_mode, anchor_mode, crop, audio_mode,
              audio_context_length, default_duration_seconds, default_steps,
              base_seed, segment_crf, video_blend_frames=0,
              continuation_mode="guide",
              plan_json_input=None):
        effective_plan_json = (
            plan_json_input
            if isinstance(plan_json_input, str) and plan_json_input.strip()
            else plan_json
        )
        plan = _normalize_plan(
            effective_plan_json, run_name, width, height, context_length,
            encode_mode,
            anchor_mode, crop, audio_mode, audio_context_length,
            default_duration_seconds, default_steps, base_seed, segment_crf,
            generation_fingerprint, video_blend_frames, continuation_mode)
        return (plan, plan["summary"], len(plan["shots"]),
                plan["compatibility"]["width"],
                plan["compatibility"]["height"],
                plan["compatibility"]["video_blend_frames"])


class MiniMaxH3ChainScenePromptEditor:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "plan": (PLAN_TYPE, {
                    "tooltip": "Connect the H3 Chain Plan output. The companion "
                               "editor modifies that Plan node's active scene "
                               "prompt directly; this socket passes the "
                               "validated plan through unchanged."}),
            }
        }

    RETURN_TYPES = (PLAN_TYPE,)
    RETURN_NAMES = ("plan",)
    OUTPUT_TOOLTIPS = (
        "The connected validated plan, unchanged at execution time. You may "
        "insert this companion between Plan and Loop Start or connect it as "
        "an editor-only branch.",
    )
    FUNCTION = "passthrough"
    CATEGORY = "conditioning/minimax/contex_loop"
    DESCRIPTION = ("Large, keyboard-friendly companion editor synchronized "
                   "bidirectionally with each scene prompt in the connected "
                   "H3 Chain Plan.")

    def passthrough(self, plan):
        return (plan,)


class MiniMaxH3ChainRichScenePromptEditor:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "plan": (PLAN_TYPE, {
                    "tooltip": "Connect the H3 Chain Plan output. This "
                               "experimental rich editor changes only the "
                               "selected scene's prompt in the upstream Plan. "
                               "All other Plan fields pass through unchanged."}),
            }
        }

    RETURN_TYPES = (PLAN_TYPE,)
    RETURN_NAMES = ("plan",)
    OUTPUT_TOOLTIPS = (
        "The connected validated Plan, unchanged at execution time. The rich "
        "editor is an authoring companion and may be inline or on an "
        "editor-only branch.",
    )
    FUNCTION = "passthrough"
    CATEGORY = "conditioning/minimax/contex_loop"
    DESCRIPTION = ("Experimental prompt-only scene editor with color-coded "
                   "references, media previews, prompt guides, revision "
                   "history, and optional one-click Direct API or MCP agent "
                   "rewriting configured in ComfyUI Settings. "
                   "It does not edit Plan settings, schedules, or seeds.")

    def passthrough(self, plan):
        return (plan,)


class MiniMaxH3ChainPlanStudio:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "plan": (PLAN_TYPE, {
                    "tooltip": "Connect the H3 Chain Plan output. Plan Studio "
                               "provides an optional timeline-based authoring "
                               "interface and writes edits back to the connected "
                               "Plan; this socket passes the validated Plan "
                               "through unchanged."}),
            }
        }

    RETURN_TYPES = (PLAN_TYPE,)
    RETURN_NAMES = ("plan",)
    OUTPUT_TOOLTIPS = (
        "The connected validated Plan, unchanged at execution time. Plan "
        "Studio may be inline before Loop Start or connected as an editor-only "
        "branch.",
    )
    FUNCTION = "passthrough"
    CATEGORY = "conditioning/minimax/contex_loop"
    DESCRIPTION = ("Optional timeline-oriented H3 Plan authoring studio with "
                   "scene navigation, prompt revisions, saved-segment status, "
                   "and preview playback. The original Plan node is unchanged.")

    def passthrough(self, plan):
        return (plan,)


class MiniMaxH3ChainRunManager:
    @classmethod
    def INPUT_TYPES(cls):
        inputs = {
            "required": {
                "plan": (PLAN_TYPE, {
                    "tooltip": "Connect the active H3 Chain Plan. The Run "
                               "Manager can replace that Plan's prompts and "
                               "settings with a saved run after confirmation; "
                               "execution passes the current Plan through."}),
                "archive_images": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Keep content-addressed fallback copies of "
                               "connected picture assets in the run folder."}),
                "archive_audio": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Keep fallback copies of connected audio "
                               "references and source tracks."}),
                "archive_video": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Keep fallback copies of connected video "
                               "assets. Disabled by default because videos "
                               "can make a run archive very large."}),
                "asset_bindings_json": ("STRING", {
                    "default": "[]", "multiline": False,
                    "dynamicPrompts": False,
                    "tooltip": "Internal loader-binding manifest maintained "
                               "by the Run Manager interface."}),
            },
            "optional": {},
        }
        for index in range(MAX_ASSET_BINDINGS):
            inputs["optional"]["asset_%d" % index] = ("*", {
                "rawLink": True,
                "lazy": True,
                "tooltip": "Connect a loader output to register its source "
                           "file for this run. The manager reads the graph "
                           "link without decoding or retaining the media."})
        return inputs

    RETURN_TYPES = (PLAN_TYPE,)
    RETURN_NAMES = ("plan",)
    OUTPUT_TOOLTIPS = (
        "The connected Plan, unchanged. Put the manager inline when connected "
        "asset bindings should be archived automatically on execution.",
    )
    FUNCTION = "passthrough"
    CATEGORY = "conditioning/minimax/contex_loop"
    DESCRIPTION = ("Browse output/h3_chains projects; restore a saved run's "
                   "Plan and loader-backed reference assets; optionally keep "
                   "content-addressed image, audio, and video fallbacks.")

    def passthrough(self, plan, archive_images, archive_audio, archive_video,
                    asset_bindings_json, **_assets):
        try:
            bindings = json.loads(str(asset_bindings_json or "[]"))
            if not isinstance(bindings, list):
                raise ValueError("asset_bindings_json must contain a list")
            if bindings:
                result = RunAssetStore(_output_root(), _input_root()).save(
                    plan["run_name"], bindings, {
                        "images": bool(archive_images),
                        "audio": bool(archive_audio),
                        "video": bool(archive_video),
                    })
                if result.get("warnings"):
                    _LOG.warning(
                        "H3 Run Manager asset archive warnings for %s: %s",
                        plan["run_name"], "; ".join(result["warnings"]))
        except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
            # Recovery metadata is supplementary and must never waste a long
            # H3 generation that already reached this pass-through.
            _LOG.warning("H3 Run Manager could not archive assets: %s", exc)
        return (plan,)


class MiniMaxH3ChainFirstSceneImage:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "state": (STATE_TYPE, {
                    "tooltip": "Current state from H3 Chain Current Shot. The "
                               "scene number decides whether the image is "
                               "passed through."}),
                "image": ("IMAGE", {
                    "tooltip": "Opening image for scene 1. It is returned only "
                               "for the first scene in the plan and omitted for "
                               "every continuation scene."}),
            },
            "optional": {
                "last_frame": ("IMAGE", {
                    "tooltip": "Optional end-frame target for the current "
                               "loop scene. It is passed through unchanged on "
                               "every scene where the upstream socket supplies "
                               "an image. To alternate targets, drive an image "
                               "index switch with Current Shot's clip_index and "
                               "connect the selected image here."}),
            }
        }

    # Append last_frame rather than inserting it beside first_frame: existing
    # workflows may already use the boolean/status outputs by slot number.
    RETURN_TYPES = ("IMAGE", "BOOLEAN", "STRING", "IMAGE")
    RETURN_NAMES = ("first_frame", "is_first_scene", "status", "last_frame")
    OUTPUT_TOOLTIPS = (
        "Connect to the stock MiniMax H3 Image to Video first_frame input. "
        "Scene 1 receives the image; later scenes receive no first-frame "
        "keyframe and continue only from H3 Motion Context.",
        "True only while scene 1 is being generated.",
        "Reports whether the opening image and current last-frame target were "
        "supplied or omitted.",
        "Connect to the stock MiniMax H3 Image to Video last_frame input. The "
        "currently selected optional target passes through on every loop; use "
        "clip_index plus an upstream index switch for per-scene targets.",
    )
    FUNCTION = "select"
    CATEGORY = "conditioning/minimax/contex_loop"
    DESCRIPTION = ("Use one opening image only for scene 1 of a recursive "
                   "I2VA chain, and optionally pass a selected last-frame "
                   "target into each loop scene for FL2VA/L2VA conditioning.")

    def select(self, state, image, last_frame=None):
        index = int(state["index"])
        last_status = ("last-frame target supplied" if last_frame is not None
                       else "last-frame target omitted")
        if index == 1:
            return (image, True,
                    "scene 1: opening image supplied; %s" % last_status,
                    last_frame)
        return (None, False,
                "scene %d: opening image omitted for continuation; %s" % (
                    index, last_status),
                last_frame)


class MiniMaxH3ChainFrameIndexSwitch:
    @classmethod
    def INPUT_TYPES(cls):
        optional = {}
        for index in range(2, 9):
            optional["frame_%d" % index] = ("IMAGE", {
                "tooltip": "Optional last-frame target %d. Connected targets "
                           "are selected in order and wrap when clip_index "
                           "exceeds their count." % index})
        return {
            "required": {
                "clip_index": ("INT", {
                    "default": 1, "min": 1, "max": MAX_SHOTS,
                    "tooltip": "One-based scene index from H3 Chain Current "
                               "Shot. Scene 1 selects frame_1, scene 2 selects "
                               "frame_2, then selection wraps."}),
                "frame_1": ("IMAGE", {
                    "tooltip": "Last-frame target selected for scene 1. For "
                               "an A to B to A chain, connect frame B here "
                               "and the opening frame A to frame_2."}),
            },
            "optional": optional,
        }

    RETURN_TYPES = ("IMAGE", "INT", "STRING")
    RETURN_NAMES = ("image", "selected_index", "status")
    OUTPUT_TOOLTIPS = (
        "Selected image for the current scene. Connect this to Frame Gate's "
        "last_frame input.",
        "One-based target slot selected after wrapping.",
        "Reports the scene index, selected target, and connected target count.",
    )
    FUNCTION = "select"
    CATEGORY = "conditioning/minimax/contex_loop"
    DESCRIPTION = ("Select and wrap one last-frame target per loop scene. "
                   "Useful for alternating A/B endpoints in FL2VA chains.")

    def select(self, clip_index, frame_1, **kwargs):
        frames = [frame_1]
        for index in range(2, 9):
            frame = kwargs.get("frame_%d" % index)
            if frame is not None:
                frames.append(frame)
        scene_index = max(1, int(clip_index))
        selected = (scene_index - 1) % len(frames)
        return (
            frames[selected],
            selected + 1,
            "scene %d: selected frame_%d of %d" % (
                scene_index, selected + 1, len(frames)),
        )


class MiniMaxH3ChainLoopStart:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "plan": (PLAN_TYPE, {
                    "tooltip": "Validated output from H3 Chain Plan."}),
                "start_clip": ("INT", {
                    "default": 1, "min": 1, "max": MAX_SHOTS,
                    "tooltip": "Legacy/resume scene to render next. Use 1 for "
                               "a new chain. A non-empty scene_range overrides "
                               "this value. "
                               "A value above 1 loads and validates the saved "
                               "checkpoint for the preceding scene before "
                               "resuming."}),
            },
            "optional": {
                "scene_range": ("STRING", {
                    "default": "",
                    "tooltip": "Inclusive contiguous scenes to generate. "
                               "Leave blank to run from start_clip through the "
                               "end; use 3 for only scene 3 or 3:8 for scenes "
                               "3 through 8. A start above 1 requires the "
                               "preceding checkpoint. Disjoint comma selections "
                               "are rejected because they break continuity."}),
                "source_audio": ("AUDIO", {
                    "tooltip": "Full external soundtrack. Required by "
                               "source_track and source_plus_timeline. Current "
                               "Shot slices the exact window for each scene. A "
                               "short, completely silent placeholder is padded."}),
                "external_context": (EXTERNAL_CONTEXT_TYPE, {
                    "tooltip": "Optional output from MiniMax H3 Existing Video "
                               "Context. When connected, scene 1 continues from "
                               "that video's tail and its repeated head is "
                               "trimmed exactly like every later scene."}),
            },
            "hidden": {
                "initial_state": (STATE_TYPE,),
            },
        }

    RETURN_TYPES = (FLOW_TYPE, STATE_TYPE, "STRING")
    RETURN_NAMES = ("flow", "state", "status")
    OUTPUT_TOOLTIPS = (
        "Recursion control link. Connect directly to H3 Chain Loop End's flow "
        "input; do not route it through other nodes.",
        "Current chain state for Current Shot and the recursive body.",
        "Starting scene, total scene count, and resume/padding status.",
    )
    FUNCTION = "start"
    CATEGORY = "conditioning/minimax/contex_loop"
    DESCRIPTION = ("Start or resume a contiguous range of a sequential H3 "
                   "chain. Ranges beginning above 1 load and validate the "
                   "preceding segment checkpoint.")

    @classmethod
    def IS_CHANGED(cls, *args, **kwargs):
        return float("NaN")

    def start(self, plan, start_clip, source_audio=None, scene_range="",
              external_context=None, initial_state=None):
        if initial_state is None:
            prepared_plan = _plan_with_external_context(plan, external_context)
            prepared_plan = _plan_with_source_audio(prepared_plan, source_audio)
            range_start, range_end = _parse_scene_range(
                scene_range, len(prepared_plan["shots"]), start_clip)
            state = _initial_state(
                prepared_plan, range_start, range_end,
                external_context=external_context if range_start == 1 else None)
        else:
            state = dict(initial_state)
            prepared_plan = state["plan"]
            if prepared_plan.get("base_plan_hash") != plan.get("plan_hash"):
                raise ValueError("H3 chain plan changed during recursive execution.")
            state["plan"] = prepared_plan
        end_clip = int(state.get("end_clip", len(prepared_plan["shots"])))
        status = "clip %d/%d; selected range %d:%d" % (
            state["index"], len(prepared_plan["shots"]),
            int(state.get("range_start", state["index"])), end_clip)
        if state.get("resumed_from"):
            status += "; resumed from clip %d" % state["resumed_from"]
        if prepared_plan["compatibility"].get("source_audio_silent_padding"):
            status += "; silent source audio will be padded to the plan duration"
        if prepared_plan["compatibility"].get("external_context_hash"):
            status += "; scene 1 extends imported video"
            if isinstance(prepared_plan.get("prelude"), dict):
                status += "; original video will be prepended"
        return ("h3_chain", state, status)


class MiniMaxH3ChainCurrent:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "state": (STATE_TYPE, {
                    "tooltip": "Current state from H3 Chain Loop Start."}),
            },
            "optional": {
                "source_audio": ("AUDIO", {
                    "tooltip": "The same full source track connected to Loop "
                               "Start. It is sliced frame-exactly for the current "
                               "scene in source-track modes."}),
                "align_audio_reference": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Experimental. Cap only source_audio_slice 5ms "
                               "below H3's rounded 40 Hz target-audio boundary. "
                               "For 362 frames this changes 15.083333s to "
                               "15.070s while retaining 603 reference steps and "
                               "a short padded tail. Shorter slices are unchanged. "
                               "The full source track used by Assemble is never "
                               "modified."}),
            },
        }

    RETURN_TYPES = (STATE_TYPE, "INT", "INT", "STRING", "STRING", "INT",
                    "INT", "INT", "INT", "INT", "FLOAT", "FLOAT",
                    "AUDIO", "STRING")
    RETURN_NAMES = ("state", "clip_index", "clip_count", "shot_id", "prompt",
                    "noise_seed", "length", "steps", "width", "height",
                    "audio_start", "audio_duration", "source_audio_slice",
                    "status")
    OUTPUT_TOOLTIPS = (
        "Unchanged current state for Chain Context, Segment Save, Review, and "
        "Loop End.",
        "One-based scene number currently being generated.",
        "Total scenes in the plan.",
        "Stable scene identifier used in checkpoints and status messages.",
        "Shared prompt followed by the current scene prompt. Connect to the "
        "stock H3 conditioning node's prompt input.",
        "Resolved unsigned 64-bit seed for the current scene. Connect to the "
        "sampler noise seed.",
        "H3-valid RAW frame count, including the repeated head overlap on "
        "continuations. Connect to the stock H3 conditioning node's length.",
        "Resolved sampler steps for this scene.",
        "Plan generation width.",
        "Plan generation height.",
        "Start time in seconds of this scene's extension-track window. For an "
        "imported-video scene 1, its separate context lead precedes this time.",
        "Raw conditioning-audio duration in seconds, including any imported "
        "scene 1 context lead.",
        "Current source-audio window for Ref2VA. It is frame-exact normally, "
        "or capped to the target H3 audio grid when alignment is enabled. It is "
        "empty in generated_audio mode.",
        "Current scene timing, delivered frames, source window, and seed.",
    )
    FUNCTION = "current"
    CATEGORY = "conditioning/minimax/contex_loop"
    DESCRIPTION = ("Expose the current shot's prompt, seed, dimensions, valid "
                   "length, steps, and source-audio reference window.")

    def current(self, state, source_audio=None, align_audio_reference=False):
        plan = state["plan"]
        index = int(state["index"])
        shot = plan["shots"][index - 1]
        mode = plan["compatibility"]["audio_mode"]
        audio_slice = None
        alignment_status = "audio ref unavailable"
        if mode in ("source_track", "source_plus_timeline"):
            _validate_source_audio_hash(
                plan["compatibility"], source_audio, "H3 Chain Current Shot")
            external_lead = int(shot.get("external_context_frames", 0))
            if index == 1 and external_lead > 0:
                audio_slice = _slice_audio_after_external_context(
                    source_audio, state.get("previous_audio"),
                    int(shot["raw_frames"]), external_lead,
                    pad_silence=bool(plan["compatibility"].get(
                        "source_audio_silent_padding")))
            else:
                audio_slice = _slice_audio(
                    source_audio, shot["audio_start_seconds"],
                    shot["audio_duration_seconds"],
                    pad_silence=bool(plan["compatibility"].get(
                        "source_audio_silent_padding")))
            if bool(align_audio_reference):
                audio_slice, alignment_status = (
                    _align_audio_reference_to_h3_grid(
                        audio_slice, int(shot["raw_frames"])))
            else:
                alignment_status = "audio ref frame-exact"
        external_lead = int(shot.get("external_context_frames", 0))
        if index == 1 and external_lead > 0:
            audio_status = "imported lead %.3fs + song 0..%.3fs" % (
                external_lead / float(FPS),
                int(shot["delivered_frames"]) / float(FPS))
        else:
            audio_status = "song %.3f..%.3fs" % (
                shot["audio_start_seconds"],
                shot["audio_start_seconds"] + shot["audio_duration_seconds"])
        status = ("clip %d/%d %s; raw=%df delivered=%df; %s; %s; seed=%d" %
                  (index, len(plan["shots"]), shot["id"], shot["raw_frames"],
                   shot["delivered_frames"], audio_status, alignment_status,
                   shot["seed"]))
        cfg = plan["compatibility"]
        result = (
            state, index, len(plan["shots"]), shot["id"], shot["prompt"],
            shot["seed"], shot["raw_frames"], shot["steps"], cfg["width"],
            cfg["height"], shot["audio_start_seconds"],
            shot["audio_duration_seconds"], audio_slice, status,
        )
        # ComfyUI adds prompt_id and display_node to the resulting `executed`
        # event. The frontend therefore receives an authoritative loop index
        # without this pack reaching into ComfyUI's executor or changing its
        # queue semantics.
        active_scene = {
            "run_name": str(plan["run_name"]),
            "clip_index": index,
            "clip_count": len(plan["shots"]),
            "end_clip": int(state.get("end_clip", len(plan["shots"]))),
            "shot_id": str(shot["id"]),
            "seed": str(shot["seed"]),
        }
        # Prompt history is supplementary recovery data and must never block a
        # generation. Mark the exact scene prompt immutable as soon as this
        # execution reaches Current Shot; subsequent editor changes branch
        # from it while the Plan JSON remains compact.
        try:
            PromptHistoryStore(_output_root()).mark_executed(
                plan["run_name"], shot["id"], shot.get("scene_prompt", ""))
        except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
            _LOG.warning("H3 prompt history could not mark scene %s executed: %s",
                         shot["id"], exc)
        return {
            "ui": {"h3_chain_active_scene": [active_scene]},
            "result": result,
        }


class MiniMaxH3PatchPriority:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "conditioning": ("CONDITIONING", {
                    "tooltip": "Conditioning pass-through. Wire this directly "
                               "between Ref2VA/I2V and Contex Loop Context so "
                               "the node executes before continuation guides "
                               "are added."}),
            },
        }

    RETURN_TYPES = ("CONDITIONING", "STRING")
    RETURN_NAMES = ("conditioning", "status")
    OUTPUT_TOOLTIPS = (
        "The exact input conditioning, unchanged. Connect it to Contex Loop "
        "Context.",
        "Core-owned native guide status, or the legacy patch ownership result.",
    )
    FUNCTION = "claim"
    CATEGORY = "conditioning/minimax/contex_loop"
    DESCRIPTION = (
        "Pass conditioning through unchanged. Updated ComfyUI remains "
        "core-owned and needs no patch. On the warned legacy fallback, this "
        "may replace only an older compatible H3 Motion Context copy, retains "
        "recognised H3-Multishot/SolAttn behavior, and refuses unknown wrappers. "
        "Legacy ownership is process-global after execution.")

    def claim(self, conditioning):
        status = _claim_inline_patch_ownership()
        return (conditioning, status)


class MiniMaxH3ChainContext:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "state": (STATE_TYPE, {
                    "tooltip": "Current state from H3 Chain Current Shot."}),
                "conditioning": ("CONDITIONING", {
                    "tooltip": "Conditioning from the stock MiniMax H3 "
                               "Ref2VA/I2V node. Scene 1 passes through without "
                               "motion context; later scenes receive the saved "
                               "continuation context."}),
                "vae": ("VAE", {
                    "tooltip": "MiniMax H3 video VAE used to encode saved "
                               "context frames for continuation scenes."}),
                "latent": ("LATENT", {
                    "tooltip": "The CURRENT scene's empty AV latent from the "
                               "stock H3 conditioning node. Chain Context "
                               "passes it through in guide mode or returns a "
                               "masked preserved-prefix copy in masked_av mode."}),
            },
            "optional": {
                "audio_vae": ("VAE", {
                    "tooltip": "H3 audio VAE used only when scene 1 continues "
                               "from imported video audio. Later loop scenes "
                               "reuse their saved AV latent directly. It may be "
                               "left disconnected for visual-only context or "
                               "source_track mode."}),
            }
        }

    RETURN_TYPES = ("CONDITIONING", "INT", "BOOLEAN", "LATENT")
    RETURN_NAMES = ("conditioning", "trim_frames", "is_continuation",
                    "latent")
    OUTPUT_TOOLTIPS = (
        "Conditioning ready for the H3 guider/sampler: scene 1 passes through "
        "unless Existing Video Context seeds it; later scenes always continue.",
        "Repeated leading frames to remove after decoding. Connect to "
        "MiniMax H3 Contex Loop Trim.",
        "True for resumed/continued scenes, false for the first scene.",
        "Sampler-ready target latent. In guide mode this is the input latent "
        "unchanged. In masked_av mode its preserved AV prefix and nested "
        "denoise mask carry the previous scene into the current target. Wire "
        "this output to the sampler for both modes so Plan can switch safely.",
    )
    FUNCTION = "apply"
    CATEGORY = "conditioning/minimax/contex_loop"
    DESCRIPTION = ("Apply each scene's inherited or overridden guide, sliding "
                   "history, or masked AV continuation, including scene 1 when "
                   "Existing Video Context is connected.")

    def apply(self, state, conditioning, vae, latent, audio_vae=None):
        index = int(state["index"])
        plan = state["plan"]
        cfg = plan["compatibility"]
        shot = plan["shots"][index - 1]
        continuation_mode = shot.get(
            "continuation_mode", cfg.get("continuation_mode", "guide"))
        external_first = index == 1 and bool(state.get("external_context"))
        if index == 1 and not external_first:
            if any(
                    candidate.get(
                        "continuation_mode",
                        cfg.get("continuation_mode", "guide")) ==
                    "sliding_history"
                    for candidate in plan["shots"]):
                from .sliding_context import require_sliding_history_support

                require_sliding_history_support()
            if any(
                    candidate.get(
                        "continuation_mode",
                        cfg.get("continuation_mode", "guide")) == "masked_av"
                    for candidate in plan["shots"]):
                # Fail before spending minutes on scene 1 if this ComfyUI
                # cannot run a masked continuation required by a later scene.
                from .masked_context import _require_h3_mask_support

                _require_h3_mask_support()
            if continuation_mode == "masked_av":
                prepared_conditioning = conditioning
            else:
                prepared_conditioning = _prepare_native_guide_conditioning(
                    conditioning)
            return (
                prepared_conditioning,
                0,
                False,
                latent,
            )
        previous_frames = state.get("previous_frames")
        if previous_frames is None:
            raise ValueError("H3 chain continuation has no previous frame checkpoint.")
        if continuation_mode == "sliding_history":
            from .sliding_context import apply_sliding_history

            use_latent_audio = cfg["audio_mode"] in (
                "generated_audio", "source_plus_timeline")
            previous_latent = (state.get("previous_latent")
                               if use_latent_audio else None)
            previous_audio = (state.get("previous_audio")
                              if use_latent_audio and external_first else None)
            if (use_latent_audio and previous_latent is None
                    and previous_audio is None and not external_first):
                raise ValueError(
                    "H3 sliding-history continuation has no previous AV latent.")
            out_conditioning, out_latent, trim = apply_sliding_history(
                conditioning=conditioning,
                vae=vae,
                latent=latent,
                previous_frames=previous_frames,
                context_length=cfg["context_length"],
                crop=cfg["crop"],
                previous_latent=previous_latent,
                audio_vae=audio_vae,
                previous_audio=previous_audio,
            )
            return (out_conditioning, trim, True, out_latent)
        if continuation_mode == "masked_av":
            from .masked_context import apply_masked_prefix

            previous_latent = state.get("previous_latent")
            out_conditioning, out_latent, trim = apply_masked_prefix(
                conditioning=conditioning,
                vae=vae,
                latent=latent,
                previous_frames=previous_frames,
                context_length=cfg["context_length"],
                crop=cfg["crop"],
                previous_latent=previous_latent,
                audio_vae=audio_vae,
                previous_audio=(state.get("previous_audio")
                                if external_first else None),
            )
            return (out_conditioning, trim, True, out_latent)
        use_latent_audio = cfg["audio_mode"] in (
            "generated_audio", "source_plus_timeline")
        previous_latent = state.get("previous_latent") if use_latent_audio else None
        previous_audio = (state.get("previous_audio")
                          if use_latent_audio and external_first else None)
        if (use_latent_audio and previous_latent is None
                and previous_audio is None and not external_first):
            raise ValueError("H3 chain continuation has no previous AV latent.")
        out, trim = MiniMaxH3MotionContext().apply(
            conditioning=conditioning,
            vae=vae,
            latent=latent,
            context_frames=previous_frames,
            context_length=cfg["context_length"],
            encode_mode=cfg["encode_mode"],
            anchor_mode=cfg["anchor_mode"],
            crop=cfg["crop"],
            audio_context_length=cfg["audio_context_length"],
            audio_mode="timeline",
            context_latent=previous_latent,
            audio_vae=audio_vae,
            context_audio=previous_audio,
        )
        return (out, trim, True, latent)


class MiniMaxH3ChainSegmentSave:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "state": (STATE_TYPE, {
                    "tooltip": "Current state from H3 Chain Current Shot."}),
                "images": ("IMAGE", {
                    "tooltip": "Delivered images AFTER MiniMax H3 Contex "
                               "Loop Trim. "
                               "The frame count must exactly match this scene's "
                               "planned delivered length."}),
                "sampled_latent": ("LATENT", {
                    "tooltip": "Raw sampler output for the current scene, "
                               "before VAE decoding. Its compact AV streams are "
                               "saved for checkpoint resume and audio context."}),
            },
            "optional": {
                "audio": ("AUDIO", {
                    "tooltip": "Delivered decoded audio AFTER MiniMax H3 "
                               "Contex Loop Trim with match_tail enabled. "
                               "Connect it in every audio mode to preserve "
                               "H3's generated sound as WAV sidecars. Required "
                               "for generated_audio and synchronized review."}),
                "images_with_overlap": ("IMAGE", {
                    "tooltip": "Blend-ready output from Loop Trim. Required "
                               "only when Plan video_blend_frames is above 0. "
                               "It contains the retained repeated head followed "
                               "by the normal delivered frames and is saved as "
                               "a separate disk-backed assembly artifact."}),
            },
            "hidden": {
                "prompt": "PROMPT",
                "extra_pnginfo": "EXTRA_PNGINFO",
            },
        }

    RETURN_TYPES = (SEGMENT_TYPE, "STRING")
    RETURN_NAMES = ("segment", "status")
    OUTPUT_TOOLTIPS = (
        "Persisted scene record for Review Gate and Loop End.",
        "Saved scene number, video/checkpoint paths, frame count, and duration.",
    )
    FUNCTION = "save"
    OUTPUT_NODE = True
    CATEGORY = "conditioning/minimax/contex_loop"
    DESCRIPTION = ("Immediately save one delivered H3 clip as an H.264 segment "
                   "plus a safetensors resume checkpoint, exact prompt metadata, "
                   "generated-audio WAV, and workflow recovery sidecars.")

    @classmethod
    def IS_CHANGED(cls, *args, **kwargs):
        return float("NaN")

    def save(self, state, images, sampled_latent, audio=None,
             images_with_overlap=None, prompt=None, extra_pnginfo=None):
        if _st_save is None:
            raise RuntimeError("safetensors is required for H3 chain checkpoints.")
        plan = state["plan"]
        index = int(state["index"])
        shot = plan["shots"][index - 1]
        actual_frames = int(images.shape[0])
        expected_frames = int(shot["delivered_frames"])
        if actual_frames != expected_frames:
            raise ValueError(
                "H3 chain clip %d produced %d delivered frames; expected %d. "
                "Wire decoded images through MiniMax H3 Contex Loop Trim before "
                "Segment Save." % (index, actual_frames, expected_frames))

        configured_blend = int(
            plan["compatibility"].get("video_blend_frames", 0))
        repeated_frames = max(
            0, int(shot["raw_frames"]) - int(shot["delivered_frames"]))
        blend_frames = min(configured_blend, repeated_frames)
        if blend_frames:
            if images_with_overlap is None:
                raise ValueError(
                    "H3 chain clip %d needs %d retained blend frames. Connect "
                    "Plan video_blend_frames to Loop Trim "
                    "retain_overlap_frames, then connect images_with_overlap "
                    "to Segment Save." % (index, blend_frames))
            blend_count = int(images_with_overlap.shape[0])
            expected_blend_count = expected_frames + blend_frames
            if blend_count != expected_blend_count:
                raise ValueError(
                    "H3 chain clip %d received %d blend-ready frames; expected "
                    "%d (%d retained overlap + %d delivered). Check the Plan "
                    "and Loop Trim blend connections." %
                    (index, blend_count, expected_blend_count, blend_frames,
                     expected_frames))

        mode = plan["compatibility"]["audio_mode"]
        if mode == "generated_audio" and audio is None:
            raise ValueError(
                "H3 chain generated_audio mode requires decoded audio on Segment "
                "Save. Wire it through MiniMax H3 Contex Loop Trim first.")
        compact = _compact_latent(sampled_latent)
        context_length = int(plan["compatibility"]["context_length"])
        context_frames = _tensor_cpu_clone(images[-context_length:])
        parts = compact["samples"]
        tensors = {
            "context_frames": context_frames,
            "video": parts[0],
            "audio": parts[1],
        }
        sample_rate = 0
        if audio is not None:
            waveform, sample_rate = _validate_audio(
                audio, "H3 chain clip %d delivered audio" % index,
                expected_frames=expected_frames)
            tensors["delivered_audio"] = _tensor_cpu_clone(waveform)

        paths = _artifact_paths(plan, index)
        os.makedirs(os.path.dirname(paths["segment"]), exist_ok=True)
        os.makedirs(os.path.dirname(paths["blend_segment"]), exist_ok=True)
        os.makedirs(os.path.dirname(paths["checkpoint"]), exist_ok=True)
        archives = _write_run_archives(plan, prompt, extra_pnginfo)
        previous_metadata = None
        if os.path.isfile(paths["metadata"]):
            try:
                previous_metadata = _read_json(paths["metadata"])
            except (OSError, ValueError, json.JSONDecodeError) as exc:
                _LOG.warning("H3 Chain is replacing unreadable clip %d metadata: %s",
                             index, exc)
        previous_revision = _preserve_previous_revision(
            plan, index, previous_metadata)

        transaction = uuid.uuid4().hex
        published_segment = _versioned_path(paths["segment"], transaction)
        published_blend = (
            _versioned_path(paths["blend_segment"], transaction)
            if blend_frames else None)
        published_checkpoint = _versioned_path(paths["checkpoint"], transaction)
        published_audio = (_versioned_path(paths["generated_audio"], transaction)
                           if audio is not None else None)
        published_prompt = os.path.splitext(published_segment)[0] + ".prompt.txt"
        published_metadata = _versioned_path(paths["metadata"], transaction)
        checkpoint_tmp = "%s.%s.tmp" % (published_checkpoint, uuid.uuid4().hex)
        committed = False
        try:
            video_metadata = _archive_media_metadata(archives)
            video_metadata.update({
                "title": "H3 scene %d - %s" % (index, shot["id"]),
                "comment": shot["prompt"],
                "description": shot.get("scene_prompt", ""),
                "synopsis": shot["prompt_hash"],
                "h3_prompt": shot["prompt"],
                "h3_seed": str(shot["seed"]),
            })
            _write_segment_video(
                images, published_segment, FPS, plan["segment_crf"],
                metadata=video_metadata)
            if published_blend is not None:
                _write_segment_video(
                    images_with_overlap, published_blend, FPS,
                    plan["segment_crf"], metadata={
                        **video_metadata,
                        "title": "H3 blend-ready scene %d - %s" %
                                 (index, shot["id"]),
                        "h3_blend_frames": str(blend_frames),
                    })
            if published_audio is not None:
                _atomic_wav(
                    {"waveform": tensors["delivered_audio"],
                     "sample_rate": sample_rate},
                    published_audio)
            _atomic_text(published_prompt, shot["prompt"])
            _st_save(tensors, checkpoint_tmp, metadata={
                "format": "h3_chain_checkpoint_v3",
                "index": str(index),
                "history_hash": _history_hash(plan, index),
                "prompt_prefix": str(plan.get("prompt_prefix") or ""),
                "scene_prompt": str(shot.get("scene_prompt") or ""),
                "prompt": str(shot["prompt"]),
                "prompt_hash": str(shot["prompt_hash"]),
                "seed": str(shot["seed"]),
                "sample_rate": str(sample_rate),
            })
            os.replace(checkpoint_tmp, published_checkpoint)

            segment = {
                "index": index,
                "id": shot["id"],
                "revision": transaction,
                "segment": _relative_output_path(published_segment),
                "checkpoint": _relative_output_path(published_checkpoint),
                "metadata": _relative_output_path(paths["metadata"]),
                "revision_metadata": _relative_output_path(published_metadata),
                "prompt_file": _relative_output_path(published_prompt),
                "raw_frames": shot["raw_frames"],
                "delivered_frames": shot["delivered_frames"],
                "history_hash": _history_hash(plan, index),
                **_prompt_fields(plan, index),
                "archives": archives,
                "seed": shot["seed"],
                "steps": shot["steps"],
                "sample_rate": sample_rate,
                "segment_sha256": _file_sha256(published_segment),
                "checkpoint_sha256": _file_sha256(published_checkpoint),
                "prompt_file_sha256": _file_sha256(published_prompt),
            }
            if published_blend is not None:
                segment.update({
                    "blend_segment": _relative_output_path(published_blend),
                    "blend_segment_sha256": _file_sha256(published_blend),
                    "blend_frames": blend_frames,
                })
            if published_audio is not None:
                segment.update({
                    "generated_audio": _relative_output_path(published_audio),
                    "generated_audio_sha256": _file_sha256(published_audio),
                })
            predecessors = state.get("segments")
            if index > 1 and isinstance(predecessors, list) and predecessors:
                predecessor = predecessors[-1]
                predecessor_revision = str(
                    predecessor.get("revision") or "")
                predecessor_hash = str(
                    predecessor.get("checkpoint_sha256") or "")
                if predecessor_revision:
                    segment["predecessor_revision"] = predecessor_revision
                if predecessor_hash:
                    segment["predecessor_checkpoint_sha256"] = predecessor_hash
            if previous_revision is not None:
                segment["supersedes"] = previous_revision
            metadata = {
                "format": "h3_chain_segment_v3",
                "run_name": plan["run_name"],
                "plan_hash": plan["plan_hash"],
                "history_hash": segment["history_hash"],
                "compatibility": plan["compatibility"],
                "archives": archives,
                "segment": segment,
            }
            # This metadata replacement is the transaction's commit point. Until
            # it succeeds, resume keeps referencing the previous immutable pair.
            _atomic_json(published_metadata, metadata)
            _atomic_json(paths["metadata"], metadata)
            committed = True
        finally:
            _safe_unlink(checkpoint_tmp)
            if not committed:
                _safe_unlink(published_segment)
                if published_blend is not None:
                    _safe_unlink(published_blend)
                _safe_unlink(published_checkpoint)
                if published_audio is not None:
                    _safe_unlink(published_audio)
                _safe_unlink(published_prompt)
                _safe_unlink(published_metadata)

        if relay_cache.artifact_uri(published_checkpoint).startswith(
                relay_cache.CACHE_SCHEME):
            relay_cache.maybe_prune_run(keep_per_shot=2)

        retained = "; previous revision retained" if previous_revision else ""
        audio_status = (" + generated WAV %s" % published_audio
                        if published_audio is not None else "")
        blend_status = (" + %d-frame blend artifact %s" %
                        (blend_frames, published_blend)
                        if published_blend is not None else "")
        status = ("saved clip %d/%d revision %s: %s + checkpoint %s%s%s%s" %
                  (index, len(plan["shots"]), transaction, published_segment,
                   published_checkpoint, audio_status, blend_status, retained))
        _LOG.info("H3 Chain %s", status)
        return {"ui": {"text": [status]}, "result": (segment, status)}


def _review_video(plan: dict[str, Any], segment: dict[str, Any],
                  audio: dict[str, Any] | None) -> tuple[dict[str, str], bool, str]:
    source = _absolute_output_path(segment["segment"])
    relative_source = _relative_output_path(source)
    if audio is None:
        return ({
            "filename": os.path.basename(relative_source),
            "subfolder": os.path.dirname(relative_source),
            "type": "output",
        }, False, "No audio is connected; this review is silent.")

    expected_frames = int(segment["delivered_frames"])
    waveform, sample_rate = _validate_audio(
        audio, "H3 Chain Review audio", expected_frames=expected_frames)
    audio_value = {"waveform": waveform, "sample_rate": sample_rate}
    audio_hash = _audio_fingerprint(audio_value)
    video_hash = str(segment.get("segment_sha256") or _file_sha256(source))
    index = int(segment["index"])
    review_dir = os.path.join(_run_dir(plan), "reviews")
    os.makedirs(review_dir, exist_ok=True)
    name = "clip_%04d.%s.%s.review.mp4" % (
        index, video_hash[:12], audio_hash[:12])
    review_path = os.path.join(review_dir, name)

    if not os.path.isfile(review_path):
        ffmpeg = shutil.which("ffmpeg")
        transaction = uuid.uuid4().hex
        wav_tmp = os.path.join(review_dir, ".review.%s.wav" % transaction)
        video_tmp = os.path.join(review_dir, ".review.%s.mp4" % transaction)
        try:
            if ffmpeg:
                _write_wav(audio_value, wav_tmp)
                _run_ffmpeg([
                    ffmpeg, "-y", "-i", source, "-i", wav_tmp,
                    "-map", "0:v:0", "-map", "1:a:0", "-c:v", "copy",
                    "-c:a", "aac", "-b:a", "192k",
                    "-t", "%.9f" % (expected_frames / float(FPS)),
                    "-movflags", "+faststart", video_tmp,
                ], timeout_seconds=60.0)
            else:
                _LOG.warning(
                    "H3 Chain ffmpeg executable not found; preparing review "
                    "audio with the built-in PyAV fallback")
                _pyav_mux_audio(
                    source, audio_value, video_tmp, 192, expected_frames)
            os.replace(video_tmp, review_path)
        finally:
            _safe_unlink(wav_tmp)
            _safe_unlink(video_tmp)

        prefix = "clip_%04d." % index
        for filename in os.listdir(review_dir):
            if (filename != name and filename.startswith(prefix) and
                    filename.endswith(".review.mp4")):
                _safe_unlink(os.path.join(review_dir, filename))

    return (_video_output_item(review_path), True, "")


def _review_display_id(unique_id: Any, dynprompt: Any) -> str:
    execution_id = str(unique_id)
    if dynprompt is not None:
        try:
            return str(dynprompt.get_display_node_id(execution_id))
        except Exception:
            pass
    return execution_id


def _review_timeout_seconds(minutes: Any) -> float:
    value = float(minutes)
    if not math.isfinite(value) or value < 0:
        raise ValueError("H3 review timeout must be a finite non-negative value.")
    return min(1440.0, value) * 60.0


def _throw_if_review_interrupted() -> None:
    """Honor ComfyUI Stop/Cancel while an async Review Gate is waiting."""
    try:
        import comfy.model_management as model_management
    except ImportError:
        return
    check = getattr(
        model_management, "throw_exception_if_processing_interrupted", None)
    if callable(check):
        check()


async def _await_review_decision(future: asyncio.Future,
                                 timeout_seconds: float) -> dict[str, Any]:
    """Wait with a heartbeat so cross-thread HTTP decisions always wake up.

    ComfyUI executes prompts on a worker thread and serves HTTP on another
    asyncio loop. Some loop/selector combinations queue call_soon_threadsafe
    callbacks without waking an otherwise idle selector. A short shielded wait
    keeps the execution loop responsive without cancelling its decision future,
    and polls ComfyUI's interrupt flag so Stop/Cancel can end an indefinite gate.
    """
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout_seconds if timeout_seconds > 0 else None
    while True:
        _throw_if_review_interrupted()
        if future.done():
            return future.result()
        wait_seconds = 0.25
        if deadline is not None:
            remaining = deadline - loop.time()
            if remaining <= 0:
                return {"action": "approve", "timed_out": True}
            wait_seconds = min(wait_seconds, remaining)
        try:
            return await asyncio.wait_for(
                asyncio.shield(future), timeout=wait_seconds)
        except asyncio.TimeoutError:
            if deadline is not None and loop.time() >= deadline:
                return {"action": "approve", "timed_out": True}


class MiniMaxH3ChainReview:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "state": (STATE_TYPE, {
                    "tooltip": "Current state from H3 Chain Current Shot. It "
                               "identifies the scene whose saved segment is "
                               "being reviewed."}),
                "segment": (SEGMENT_TYPE, {
                    "tooltip": "Persisted scene output from H3 Chain Segment "
                               "Save. Saving before review makes every accepted "
                               "scene recoverable."}),
                "enabled": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Pause after every saved segment for approval."}),
                "play_notification_sound": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Play a browser chime when a segment becomes "
                               "ready for review."}),
                "auto_continue_timeout_minutes": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 1440.0,
                    "step": 0.5,
                    "tooltip": "Automatically approve and continue after this "
                               "many minutes. 0 waits indefinitely."}),
                "unload_models_while_waiting": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Release model weights from VRAM after the review "
                               "appears. Approval remains responsive; continuing "
                               "must reload the model stack."}),
                "assemble_partial_on_stop": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Approve & stop also joins every accepted segment "
                               "through the current scene into a partial MP4."}),
                "partial_audio_source": (["checkpointed", "source", "none"], {
                    "default": "checkpointed",
                    "tooltip": "Audio for the partial MP4. checkpointed uses each "
                               "saved delivered-audio track; source requires the "
                               "full source_audio input."}),
            },
            "optional": {
                "audio": ("AUDIO", {
                    "tooltip": "Wire frame-exact delivered audio from H3 "
                               "MiniMax H3 Contex Loop Trim for synchronized "
                               "review."}),
                "source_audio": ("AUDIO", {
                    "tooltip": "Optional full source track used only when partial "
                               "audio source is source."}),
            },
            "hidden": {
                "dynprompt": "DYNPROMPT",
                "unique_id": "UNIQUE_ID",
            },
        }

    RETURN_TYPES = (SEGMENT_TYPE, "STRING")
    RETURN_NAMES = ("segment", "status")
    OUTPUT_TOOLTIPS = (
        "The approved segment, or the same segment after review is disabled. "
        "Connect to Loop End.",
        "Review decision, retry seed, timeout, stop, or partial-assembly status.",
    )
    FUNCTION = "review"
    OUTPUT_NODE = True
    CATEGORY = "conditioning/minimax/contex_loop"
    DESCRIPTION = ("Pause after a checkpointed H3 segment for synchronized "
                   "video/audio review. Approve, stop, retry an edited scene "
                   "prompt/seed/duration, or reroll its seed while applying "
                   "the edited duration from the node UI.")

    @classmethod
    def IS_CHANGED(cls, *args, **kwargs):
        return float("NaN")

    async def review(self, state, segment, enabled, play_notification_sound,
                     auto_continue_timeout_minutes, unload_models_while_waiting,
                     assemble_partial_on_stop, partial_audio_source, audio=None,
                     source_audio=None,
                     dynprompt=None, unique_id=None):
        plan = state["plan"]
        index = int(state["index"])
        if int(segment.get("index", -1)) != index:
            raise ValueError(
                "H3 Chain Review received the wrong segment for clip %d." % index)
        if not enabled:
            status = "review bypassed for clip %d" % index
            return {"ui": {"text": [status]}, "result": (segment, status)}
        if PromptServer is None or web is None:
            raise RuntimeError("H3 Chain Review requires ComfyUI's prompt server.")

        # Publish the persisted video and pending token BEFORE preparing the
        # optional audiovisual preview. Firefox/proxy websocket differences
        # made this ordering bug look like a dead button: a slow or stuck audio
        # mux meant the browser never received a token at all. Review audio is
        # a convenience, not part of checkpoint validity, so it must never hold
        # the gate controls hostage.
        video, _has_audio, no_audio_warning = _review_video(
            plan, segment, None)
        shot = plan["shots"][index - 1]
        timeout_seconds = _review_timeout_seconds(
            auto_continue_timeout_minutes)
        server_now = time.time()
        deadline = server_now + timeout_seconds if timeout_seconds > 0 else None
        token = uuid.uuid4().hex
        loop = asyncio.get_running_loop()
        future = loop.create_future()
        payload = {
            "token": token,
            "node_id": _review_display_id(unique_id, dynprompt),
            "execution_id": str(unique_id),
            "run_name": str(plan["run_name"]),
            "clip_index": index,
            "clip_count": len(plan["shots"]),
            "shot_id": shot["id"],
            "scene_prompt": shot.get("scene_prompt", shot["prompt"]),
            "prompt_prefix": str(plan.get("prompt_prefix") or ""),
            "seed": str(shot["seed"]),
            "raw_frames": int(shot["raw_frames"]),
            "duration_seconds": int(shot["raw_frames"]) / float(FPS),
            "video": video,
            "has_audio": False,
            "warning": ("Preparing synchronized audio preview…"
                        if audio is not None else no_audio_warning),
            "preview_pending": audio is not None,
            "preview_revision": 0,
            "play_notification_sound": bool(play_notification_sound),
            "unload_models_while_waiting": bool(unload_models_while_waiting),
            "assemble_partial_on_stop": bool(assemble_partial_on_stop),
            "timeout_seconds": timeout_seconds,
            "deadline": deadline,
            "server_now": server_now,
        }
        _PENDING_REVIEWS[token] = {
            "future": future,
            "loop": loop,
            "public": payload,
            "plan": plan,
            "current_seed": int(shot["seed"]),
            "current_length": int(shot["raw_frames"]),
        }
        PromptServer.instance.send_sync(
            "minimax_h3_context_loop_review", dict(payload),
            PromptServer.instance.client_id)

        if audio is not None:
            # Keep tensor-to-WAV conversion on Comfy's execution thread. Some
            # PyTorch builds can deadlock when their CPU tensor pools are first
            # entered from asyncio.to_thread. The token is already public, and
            # The external ffmpeg path is time-bounded, and either media backend
            # can fail into a silent review instead of an unresolvable workflow
            # hang.
            try:
                video, has_audio, warning = _review_video(plan, segment, audio)
            except Exception as exc:
                _LOG.exception("H3 Chain synchronized review preview failed")
                has_audio = False
                warning = (
                    "Synchronized review audio is unavailable (%s). The saved "
                    "segment/checkpoint is valid; this review is silent." % exc)
            payload.update({
                "video": video,
                "has_audio": has_audio,
                "warning": warning,
                "preview_pending": False,
                "preview_revision": 1,
                "server_now": time.time(),
            })
            PromptServer.instance.send_sync(
                "minimax_h3_context_loop_review", dict(payload),
                PromptServer.instance.client_id)

        if unload_models_while_waiting:
            try:
                import comfy.model_management as model_management
                model_management.unload_all_models()
            except Exception as exc:
                _LOG.warning("H3 Chain Review could not unload models: %s", exc)

        try:
            try:
                decision = await _await_review_decision(
                    future, timeout_seconds)
            except BaseException:
                # ComfyUI's InterruptProcessingException intentionally derives
                # from BaseException. Resolve the browser gate before letting
                # the executor emit its normal execution_interrupted event.
                status = "review interrupted for clip %d" % index
                try:
                    PromptServer.instance.send_sync(
                        "minimax_h3_context_loop_review_resolved",
                        {"token": token, "node_id": payload["node_id"],
                         "action": "interrupted", "status": status},
                        PromptServer.instance.client_id)
                except Exception as exc:
                    _LOG.warning(
                        "H3 Chain Review could not publish interruption: %s",
                        exc)
                raise
        finally:
            _PENDING_REVIEWS.pop(token, None)
            if not future.done():
                future.cancel()

        action = decision["action"]
        if action == "approve":
            timed_out = bool(decision.get("timed_out"))
            status = (("review timed out; auto-approved clip %d/%d; continuing")
                      if timed_out else ("approved clip %d/%d; continuing")) % (
                          index, len(plan["shots"]))
            if index == len(plan["shots"]):
                _PENDING_FINAL_REVIEW_PREVIEWS[
                    _final_review_preview_key(plan)
                ] = {
                    "token": token,
                    "node_id": payload["node_id"],
                    "client_id": PromptServer.instance.client_id,
                }
            if timed_out:
                PromptServer.instance.send_sync(
                    "minimax_h3_context_loop_review_resolved",
                    {"token": token, "node_id": payload["node_id"],
                     "action": "timeout_approve", "status": status},
                    PromptServer.instance.client_id)
            return {"ui": {"text": [status]}, "result": (segment, status)}
        if action == "stop":
            if ExecutionBlocker is None:
                raise RuntimeError("This ComfyUI build does not support review blocking.")
            status = "approved clip %d and stopped at its checkpoint" % index
            partial_item = None
            if assemble_partial_on_stop:
                try:
                    partial_path, partial_warning = _assemble_review_partial(
                        state, segment, partial_audio_source, source_audio)
                    partial_item = _video_output_item(partial_path)
                    status += "; partial video: %s" % partial_path
                    if partial_warning:
                        status += "; %s" % partial_warning
                except Exception as exc:
                    _LOG.exception("H3 Chain partial stop assembly failed")
                    status += "; partial assembly failed: %s" % exc
            resolved = {
                "token": token,
                "node_id": payload["node_id"],
                "action": "stop",
                "status": status,
            }
            if partial_item is not None:
                resolved["partial_video"] = partial_item
            PromptServer.instance.send_sync(
                "minimax_h3_context_loop_review_resolved", resolved,
                PromptServer.instance.client_id)
            return {
                "ui": {"text": [status]},
                "result": (ExecutionBlocker(None), status),
            }
        if action != "retry":
            raise RuntimeError("Unknown H3 review decision %r." % action)

        revised_segment = dict(segment)
        revised_segment["_h3_review_decision"] = {
            "action": "retry",
            "scene_prompt": decision["scene_prompt"],
            "seed": int(decision["seed"]),
            "raw_frames": int(decision["raw_frames"]),
        }
        status = "retrying clip %d with seed %d at %d frames" % (
            index, int(decision["seed"]), int(decision["raw_frames"]))
        return {"ui": {"text": [status]},
                "result": (revised_segment, status)}


def _manifest_from_segments(plan: dict[str, Any], values: list[dict[str, Any]],
                            complete: bool) -> dict[str, Any]:
    segments = []
    archives = _available_run_archives(plan)
    for item in values:
        segment = _public_segment(item)
        index = int(segment.get("index", -1))
        if 1 <= index <= len(plan["shots"]):
            for key, value in _prompt_fields(plan, index).items():
                segment.setdefault(key, value)
        if archives:
            segment.setdefault("archives", archives)
        segments.append(segment)
    expected_count = len(plan["shots"]) if complete else len(segments)
    if expected_count < 1:
        raise ValueError("H3 chain manifest requires at least one saved clip.")
    if len(segments) != expected_count:
        raise ValueError(
            "H3 chain manifest is incomplete: found %d persisted clips, expected %d."
            % (len(segments), expected_count))
    indexes = [int(item.get("index", -1)) for item in segments]
    if indexes != list(range(1, expected_count + 1)):
        raise ValueError("H3 chain manifest segment indexes are not contiguous.")
    total_frames = int(plan["total_delivered_frames"]) if complete else sum(
        int(item.get(
            "delivered_frames",
            plan["shots"][int(item["index"]) - 1]["delivered_frames"]))
        for item in segments)
    manifest = {
        "format": ("h3_chain_manifest_v3" if complete
                   else "h3_chain_partial_manifest_v3"),
        "run_name": plan["run_name"],
        "plan_hash": plan["plan_hash"],
        "prompt_prefix": str(plan.get("prompt_prefix") or ""),
        "compatibility": plan["compatibility"],
        "clip_count": expected_count,
        "total_delivered_frames": total_frames,
        "duration_seconds": total_frames / float(FPS),
        "segments": segments,
    }
    if archives:
        manifest["archives"] = archives
    if isinstance(plan.get("prelude"), dict):
        manifest["prelude"] = _json_document(plan["prelude"])
    if not complete:
        manifest["planned_clip_count"] = len(plan["shots"])
        manifest["last_completed_clip"] = len(segments)
    return manifest


def _manifest_from_state(state: dict[str, Any]) -> dict[str, Any]:
    return _manifest_from_segments(state["plan"], state["segments"], True)


def _partial_manifest(state: dict[str, Any],
                      segment: dict[str, Any]) -> dict[str, Any]:
    plan = state["plan"]
    index = int(state["index"])
    values = list(state.get("segments", [])) + [_public_segment(segment)]
    if index != len(values):
        raise ValueError(
            "H3 partial manifest expected clip %d after %d predecessors." %
            (index, len(values) - 1))
    return _manifest_from_segments(plan, values, False)


def _manifest_path(plan: dict[str, Any]) -> str:
    return os.path.join(_run_dir(plan), "manifest.json")


class MiniMaxH3ChainLoopEnd:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "flow": (FLOW_TYPE, {
                    "rawLink": True,
                    "tooltip": "Connect DIRECTLY from Loop Start's flow "
                               "output. This raw link defines the recursive body "
                               "that Loop End clones for later scenes."}),
                "state": (STATE_TYPE, {
                    "tooltip": "Current state from Current Shot. Loop End adds "
                               "the accepted segment and advances its scene "
                               "index."}),
                "images": ("IMAGE", {
                    "tooltip": "Delivered current-scene images after Motion "
                               "Context Trim. Their tail becomes the next "
                               "scene's visual context."}),
                "sampled_latent": ("LATENT", {
                    "tooltip": "Current sampler output. Its AV streams become "
                               "the next scene's generated-audio context when "
                               "the selected audio mode requires it."}),
                "segment": (SEGMENT_TYPE, {
                    "tooltip": "Approved persisted segment from Review Gate, "
                               "or directly from Segment Save when no review is "
                               "wanted."}),
            },
            "hidden": {
                "dynprompt": "DYNPROMPT",
                "unique_id": "UNIQUE_ID",
            },
        }

    RETURN_TYPES = (MANIFEST_TYPE, "STRING", "IMAGE", "LATENT")
    RETURN_NAMES = ("manifest", "manifest_json", "last_context_frames",
                    "last_context_latent")
    OUTPUT_TOOLTIPS = (
        "Completed chain manifest for H3 Chain Assemble. Produced only when "
        "the final scene is accepted.",
        "Human-readable JSON form of the completed manifest.",
        "Delivered tail frames from the final scene for optional chaining into "
        "another workflow.",
        "Final sampled H3 AV latent for optional continuation outside this loop.",
    )
    FUNCTION = "end"
    CATEGORY = "conditioning/minimax/contex_loop"
    DESCRIPTION = ("Finish one persisted clip, carry only its context tail and "
                   "AV latent, then recursively execute the next shot.")

    def _explore_dependencies(self, node_id: str, dynprompt: Any,
                              upstream: dict[str, list[str]],
                              parent_ids: list[str]) -> None:
        node_info = dynprompt.get_node(node_id)
        for value in node_info.get("inputs", {}).values():
            if not is_link(value):
                continue
            parent_id = value[0]
            display_id = dynprompt.get_display_node_id(parent_id)
            display_node = dynprompt.get_node(display_id)
            if display_node["class_type"] != "MiniMaxH3ChainLoopEnd":
                parent_ids.append(display_id)
            if parent_id not in upstream:
                upstream[parent_id] = []
                self._explore_dependencies(parent_id, dynprompt, upstream,
                                           parent_ids)
            upstream[parent_id].append(node_id)

    def _explore_output_nodes(self, dynprompt: Any,
                              upstream: dict[str, list[str]],
                              parent_ids: list[str]) -> None:
        try:
            import nodes as comfy_nodes
            mappings = comfy_nodes.NODE_CLASS_MAPPINGS
        except Exception:
            return
        output_nodes: dict[str, Any] = {}
        for node_id, node in dynprompt.get_original_prompt().items():
            class_def = mappings.get(node.get("class_type"))
            if not class_def or not getattr(class_def, "OUTPUT_NODE", False):
                continue
            for value in node.get("inputs", {}).values():
                if is_link(value):
                    output_nodes[node_id] = value
        for parent_id in list(upstream):
            display_id = dynprompt.get_display_node_id(parent_id)
            for output_id, link in output_nodes.items():
                linked_id = link[0]
                if (linked_id in parent_ids and display_id == linked_id and
                        output_id not in upstream[parent_id]):
                    if "." in parent_id:
                        parts = parent_id.split(".")
                        parts[-1] = output_id
                        upstream[parent_id].append(".".join(parts))
                    else:
                        upstream[parent_id].append(output_id)

    def _collect_contained(self, node_id: str,
                           upstream: dict[str, list[str]],
                           contained: dict[str, bool]) -> None:
        for child_id in upstream.get(node_id, []):
            if child_id in contained:
                continue
            contained[child_id] = True
            self._collect_contained(child_id, upstream, contained)

    def _recurse(self, flow, next_state, dynprompt, unique_id):
        if GraphBuilder is None:
            raise RuntimeError("H3 Chain Loop requires ComfyUI GraphBuilder.")
        unique_id = str(unique_id)
        upstream: dict[str, list[str]] = {}
        parent_ids: list[str] = []
        self._explore_dependencies(unique_id, dynprompt, upstream, parent_ids)
        parent_ids = list(set(parent_ids))
        self._explore_output_nodes(dynprompt, upstream, parent_ids)

        open_node = str(flow[0])
        start_info = dynprompt.get_node(open_node)
        if start_info["class_type"] != "MiniMaxH3ChainLoopStart":
            raise ValueError("H3 Chain Loop End must receive flow from H3 Chain Loop Start.")
        contained: dict[str, bool] = {unique_id: True, open_node: True}
        self._collect_contained(open_node, upstream, contained)

        graph = GraphBuilder()
        for node_id in contained:
            original = dynprompt.get_node(node_id)
            clone_id = "Recurse" if node_id == unique_id else node_id
            node = graph.node(original["class_type"], clone_id)
            node.set_override_display_id(node_id)
        for node_id in contained:
            original = dynprompt.get_node(node_id)
            clone_id = "Recurse" if node_id == unique_id else node_id
            node = graph.lookup_node(clone_id)
            for key, value in original.get("inputs", {}).items():
                if is_link(value) and value[0] in contained:
                    parent = graph.lookup_node(value[0])
                    node.set_input(key, parent.out(value[1]))
                else:
                    node.set_input(key, value)
        graph.lookup_node(open_node).set_input("initial_state", next_state)
        # The imported source may contain thousands of decoded frames. Once
        # Loop Start has reduced it to typed state, recursive iterations must
        # not keep the adapter dependency alive or prepare the prelude again.
        if "external_context" in start_info.get("inputs", {}):
            graph.lookup_node(open_node).set_input("external_context", None)
        recurse = graph.lookup_node("Recurse")
        return {
            "result": tuple(recurse.out(index)
                            for index in range(len(self.RETURN_TYPES))),
            "expand": graph.finalize(),
        }

    def end(self, flow, state, images, sampled_latent, segment,
            dynprompt=None, unique_id=None):
        plan = state["plan"]
        index = int(state["index"])
        if int(segment.get("index", -1)) != index:
            raise ValueError("H3 Chain End received the wrong segment for clip %d."
                             % index)
        review = segment.get("_h3_review_decision")
        if isinstance(review, dict) and review.get("action") == "retry":
            revised_plan = _plan_with_review_revision(
                plan, index, review.get("scene_prompt", ""),
                int(review.get("seed", plan["shots"][index - 1]["seed"])),
                int(review.get(
                    "raw_frames", plan["shots"][index - 1]["raw_frames"])))
            retry_state = dict(state)
            retry_state["plan"] = revised_plan
            # Keep the predecessor context and accepted segment list unchanged.
            # Segment Save makes the new take active when this index completes
            # again while retaining the rejected take as an immutable revision.
            return self._recurse(flow, retry_state, dynprompt, unique_id)
        context_length = int(plan["compatibility"]["context_length"])
        next_state = {
            "plan": plan,
            "index": index + 1,
            "range_start": int(state.get("range_start", 1)),
            "end_clip": int(state.get("end_clip", len(plan["shots"]))),
            # clone: a tensor view would retain the entire decoded clip
            "previous_frames": _tensor_cpu_clone(images[-context_length:]),
            "previous_latent": _compact_latent(sampled_latent),
            "segments": list(state.get("segments", [])) +
                        [_public_segment(segment)],
            "resumed_from": state.get("resumed_from", 0),
        }
        end_clip = int(next_state["end_clip"])
        if index < end_clip:
            return self._recurse(flow, next_state, dynprompt, unique_id)

        complete = end_clip == len(plan["shots"])
        manifest = _manifest_from_segments(
            plan, next_state["segments"], complete=complete)
        # A normal chain has already created its run directory in Segment Save.
        # Keeping this conditional also permits lightweight/custom segment sinks
        # that deliberately do not use the disk-backed saver.
        if os.path.isdir(_run_dir(plan)):
            if complete:
                manifest_path = _manifest_path(plan)
            else:
                manifest_path = os.path.join(
                    _run_dir(plan), "partial",
                    "through_clip_%04d.manifest.json" % end_clip)
            _atomic_json(manifest_path, manifest)
        manifest_json = json.dumps(manifest, ensure_ascii=False, indent=2,
                                   sort_keys=True)
        return (manifest, manifest_json, next_state["previous_frames"],
                next_state["previous_latent"])


class MiniMaxH3ChainManifestLoad:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "plan": (PLAN_TYPE, {
                    "tooltip": "The same validated H3 Chain Plan used for the "
                               "original render. Plan and generation "
                               "fingerprints are checked against every saved "
                               "scene."}),
            },
            "optional": {
                "source_audio": ("AUDIO", {
                    "tooltip": "The original full source track when the plan "
                               "uses source_track or source_plus_timeline. Its "
                               "fingerprint must match the saved checkpoints."}),
                "external_context": (EXTERNAL_CONTEXT_TYPE, {
                    "tooltip": "Reconnect the same Existing Video Context used "
                               "for scene 1. Its tail fingerprint restores the "
                               "correct resume contract and its persisted "
                               "prelude remains available to Assemble."}),
            },
        }

    RETURN_TYPES = (MANIFEST_TYPE, "STRING", "STRING")
    RETURN_NAMES = ("manifest", "manifest_json", "status")
    OUTPUT_TOOLTIPS = (
        "Verified completed manifest reconstructed from saved scene "
        "checkpoints; connect to H3 Chain Assemble.",
        "Human-readable JSON form of the reconstructed manifest.",
        "Number of verified scenes and checkpoint directory used.",
    )
    FUNCTION = "load"
    CATEGORY = "conditioning/minimax/contex_loop"
    DESCRIPTION = ("Validate every saved clip and rebuild a completed chain "
                   "manifest without rerendering the final clip.")

    @classmethod
    def IS_CHANGED(cls, *args, **kwargs):
        return float("NaN")

    def load(self, plan, source_audio=None, external_context=None):
        prepared_plan = _plan_with_external_context(plan, external_context)
        prepared_plan = _plan_with_source_audio(prepared_plan, source_audio)
        completed = _load_resume_state(
            prepared_plan, len(prepared_plan["shots"]) + 1)
        manifest = _manifest_from_state(completed)
        _atomic_json(_manifest_path(prepared_plan), manifest)
        manifest_json = json.dumps(
            manifest, ensure_ascii=False, indent=2, sort_keys=True)
        status = "loaded and verified %d saved clips from %s" % (
            len(manifest["segments"]), _run_dir(prepared_plan))
        return (manifest, manifest_json, status)


def _generated_audio(manifest: dict[str, Any]) -> dict[str, Any]:
    if _st_load is None or torch is None:
        raise RuntimeError("Generated-audio assembly requires safetensors and torch.")
    waveforms = []
    sample_rate = None
    # Cumulative boundary budgeting was inspired by seitanism's
    # ComfyUI-H3-Motion-Context-MultiRef. Reconcile each saved scene against
    # the full delivered timeline so independent rounding cannot accumulate.
    cumulative_frames = 0
    cumulative_samples = 0
    for segment in manifest["segments"]:
        checkpoint = _absolute_output_path(segment["checkpoint"])
        tensors = _st_load(checkpoint)
        if "delivered_audio" not in tensors:
            raise ValueError(
                "Checkpoint for clip %d has no delivered audio. Wire decoded "
                "audio through Trim and Segment Save." % segment["index"])
        current_rate = int(segment.get("sample_rate", 0))
        if current_rate <= 0:
            raise ValueError("Checkpoint for clip %d has no audio sample rate."
                             % segment["index"])
        if sample_rate is None:
            sample_rate = current_rate
        elif current_rate != sample_rate:
            raise ValueError("Generated segment audio sample rates do not match.")
        waveform = tensors["delivered_audio"]
        expected = int(round(
            int(segment["delivered_frames"]) / float(FPS) * current_rate))
        if int(waveform.shape[-1]) != expected:
            raise ValueError(
                "Checkpoint for clip %d has %d delivered audio samples; expected "
                "%d for %d frames." %
                (segment["index"], int(waveform.shape[-1]), expected,
                 int(segment["delivered_frames"])))
        cumulative_frames += int(segment["delivered_frames"])
        next_boundary = int(round(
            cumulative_frames / float(FPS) * current_rate))
        budget = next_boundary - cumulative_samples
        have = int(waveform.shape[-1])
        if have > budget:
            waveform = waveform[..., :budget]
        elif have < budget:
            waveform = torch.nn.functional.pad(waveform, (0, budget - have))
        waveforms.append(waveform)
        cumulative_samples = next_boundary
    return {"waveform": torch.cat(waveforms, dim=-1),
            "sample_rate": int(sample_rate)}


def _validate_prelude(manifest: dict[str, Any]) -> dict[str, Any] | None:
    value = manifest.get("prelude")
    if value is None:
        return None
    if not isinstance(value, dict) or not bool(value.get("prepend")):
        raise ValueError("H3 chain manifest has an invalid prelude record.")
    frames = int(value.get("frame_count", 0))
    if frames < 1 or int(value.get("fps", 0)) != FPS:
        raise ValueError(
            "H3 chain prelude must contain at least one frame at %d fps." % FPS)
    compatibility = manifest.get("compatibility") or {}
    if (int(value.get("width", 0)) != int(compatibility.get("width", 0)) or
            int(value.get("height", 0)) !=
            int(compatibility.get("height", 0))):
        raise ValueError(
            "H3 chain prelude dimensions do not match generated segments.")
    video_value = value.get("video")
    expected_video_hash = str(value.get("video_sha256") or "")
    if not isinstance(video_value, str) or not expected_video_hash:
        raise ValueError("H3 chain prelude has no verified video artifact.")
    video_path = _absolute_output_path(video_value)
    if not os.path.isfile(video_path):
        raise FileNotFoundError("H3 chain prelude video is missing: %s" % video_path)
    if _file_sha256(video_path) != expected_video_hash:
        raise ValueError("H3 chain prelude video failed its SHA-256 integrity check.")
    audio_value = value.get("audio")
    if audio_value is not None:
        expected_audio_hash = str(value.get("audio_sha256") or "")
        if not isinstance(audio_value, str) or not expected_audio_hash:
            raise ValueError("H3 chain prelude has an unverified audio artifact.")
        audio_path = _absolute_output_path(audio_value)
        if not os.path.isfile(audio_path):
            raise FileNotFoundError(
                "H3 chain prelude audio is missing: %s" % audio_path)
        if _file_sha256(audio_path) != expected_audio_hash:
            raise ValueError(
                "H3 chain prelude audio failed its SHA-256 integrity check.")
    return value


def _prelude_audio(record: dict[str, Any]) -> dict[str, Any] | None:
    value = record.get("audio")
    if value is None:
        return None
    if _st_load is None:
        raise RuntimeError("safetensors is required to load H3 prelude audio.")
    tensors = _st_load(_absolute_output_path(value))
    waveform = tensors.get("waveform")
    if waveform is None:
        raise ValueError("H3 chain prelude audio contains no waveform tensor.")
    sample_rate = int(record.get("audio_sample_rate", 0))
    audio = {"waveform": waveform, "sample_rate": sample_rate}
    _validate_audio(audio, "H3 chain prelude audio",
                    expected_frames=int(record["frame_count"]))
    return audio


def _audio_with_prelude(
    audio: dict[str, Any],
    extension_frames: int,
    prelude: dict[str, Any],
) -> dict[str, Any]:
    waveform, sample_rate = _audio_waveform_3d(
        audio, "H3 extension assembly audio")
    channels = int(waveform.shape[1])
    prelude_frames = int(prelude["frame_count"])
    prelude_samples = int(round(
        prelude_frames / float(FPS) * sample_rate))
    total_samples = int(round(
        (prelude_frames + int(extension_frames)) /
        float(FPS) * sample_rate))
    extension_samples = total_samples - prelude_samples
    normalized_extension = _resample_audio_exact(
        {"waveform": waveform, "sample_rate": sample_rate},
        sample_rate, extension_samples, channels,
        "H3 extension assembly audio")
    saved = _prelude_audio(prelude)
    if saved is None:
        prefix = torch.zeros(
            (1, channels, prelude_samples), dtype=torch.float32)
    else:
        prefix = _resample_audio_exact(
            saved, sample_rate, prelude_samples, channels,
            "H3 chain prelude audio")["waveform"]
    return {
        "waveform": torch.cat(
            (prefix, normalized_extension["waveform"]), dim=-1),
        "sample_rate": sample_rate,
    }


def _validate_manifest(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    segments = manifest.get("segments") or []
    clip_count = int(manifest.get("clip_count", 0))
    if clip_count < 1 or len(segments) != clip_count:
        raise ValueError(
            "H3 chain manifest contains %d segments; expected %d." %
            (len(segments), clip_count))
    total_frames = 0
    for index, segment in enumerate(segments, start=1):
        _verify_segment_artifacts(segment, index)
        total_frames += int(segment.get("delivered_frames", 0))
    expected_frames = int(manifest.get("total_delivered_frames", -1))
    if total_frames != expected_frames:
        raise ValueError(
            "H3 chain manifest segment durations total %d frames; expected %d."
            % (total_frames, expected_frames))
    return segments


def _run_ffmpeg(command: list[str], timeout_seconds: float | None = None) -> None:
    try:
        # ffmpeg writes UTF-8. text=True alone decodes with the locale's
        # preferred encoding, which on a non-UTF-8 Windows console (cp932,
        # cp1251, ...) raises UnicodeDecodeError inside subprocess's reader
        # threads. Those threads die, result.stderr comes back truncated or
        # empty, and a genuine ffmpeg failure below reports no reason at all.
        # Decode as UTF-8 and never let diagnostics be the thing that fails.
        result = subprocess.run(
            command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
            encoding="utf-8", errors="replace", timeout=timeout_seconds)
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(
            "ffmpeg timed out after %.1f seconds" % float(timeout_seconds)) from exc
    if result.returncode:
        tail = "\n".join(result.stderr.splitlines()[-20:])
        raise RuntimeError("ffmpeg failed (%d):\n%s" % (result.returncode, tail))


def _pyav_video_signature(stream: Any) -> tuple[Any, ...]:
    codec = stream.codec_context
    return (
        str(codec.name or ""),
        int(codec.width),
        int(codec.height),
        str(codec.format.name if codec.format is not None else ""),
    )


def _pyav_shift_packet(packet: Any, stream: Any,
                        offset_seconds: Fraction) -> None:
    """Move one remuxed segment packet onto the joined video timeline."""
    time_base = packet.time_base or stream.time_base
    if time_base is None:
        raise RuntimeError(
            "PyAV could not determine an H3 segment video time base.")
    time_base = Fraction(time_base)
    start_time = int(stream.start_time or 0)
    start_seconds = Fraction(start_time) * Fraction(stream.time_base)
    shift = round((offset_seconds - start_seconds) / time_base)
    if packet.pts is not None:
        packet.pts = int(packet.pts) + shift
    if packet.dts is not None:
        packet.dts = int(packet.dts) + shift


def _pyav_concat_video(segment_paths: list[str], delivered_frames: list[int],
                        path: str, metadata: dict[str, Any]) -> None:
    """Stream-copy compatible H.264 segments without an ffmpeg executable."""
    if av is None:
        raise RuntimeError(
            "H3 Chain Assemble found neither an ffmpeg executable nor PyAV.")
    if len(segment_paths) != len(delivered_frames) or not segment_paths:
        raise ValueError("PyAV H3 assembly requires one duration per segment.")

    output = None
    try:
        output = av.open(
            path, mode="w",
            options={"movflags": "use_metadata_tags+faststart"})
        for key, value in metadata.items():
            if value is not None:
                output.metadata[str(key)] = str(value)

        output_stream = None
        expected_signature = None
        frame_offset = 0
        for index, (source, frames) in enumerate(
                zip(segment_paths, delivered_frames), start=1):
            with av.open(source, mode="r") as current:
                streams = list(current.streams.video)
                if len(streams) != 1:
                    raise ValueError(
                        "H3 chain clip %d contains %d video streams; expected 1."
                        % (index, len(streams)))
                input_stream = streams[0]
                signature = _pyav_video_signature(input_stream)
                if output_stream is None:
                    output_stream = output.add_stream_from_template(input_stream)
                    expected_signature = signature
                elif signature != expected_signature:
                    raise ValueError(
                        "H3 chain clip %d has incompatible video parameters %r; "
                        "the first clip uses %r." %
                        (index, signature, expected_signature))

                offset = Fraction(frame_offset, FPS)
                for packet in current.demux(input_stream):
                    if packet.dts is None:
                        continue
                    _pyav_shift_packet(packet, input_stream, offset)
                    packet.stream = output_stream
                    output.mux(packet)
            frame_offset += int(frames)
        output.close()
        output = None
    except Exception:
        if output is not None:
            output.close()
        _safe_unlink(path)
        raise


def _fit_pyav_audio_samples(waveform: Any, required_samples: int) -> Any:
    """Fit final mux audio, tolerating only a one-sample rounding deficit."""
    required_samples = int(required_samples)
    available_samples = int(waveform.shape[-1])
    missing = required_samples - available_samples
    if missing > 1:
        raise ValueError(
            "PyAV H3 assembly audio contains %d samples; %d are required." %
            (available_samples, required_samples))
    if missing == 1:
        waveform = torch.nn.functional.pad(waveform, (0, 1))
        _LOG.warning(
            "H3 Chain PyAV assembly zero-padded a one-sample audio rounding "
            "deficit (%d -> %d samples).",
            available_samples, required_samples)
    return waveform[..., :required_samples]


def _blend_video_records(
    manifest: dict[str, Any],
    segments: list[dict[str, Any]],
    prelude: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    """Resolve the hard base and each disk-backed overlap continuation."""
    configured = int(
        (manifest.get("compatibility") or {}).get("video_blend_frames", 0))
    if configured < 1:
        return []
    records: list[dict[str, Any]] = []
    has_predecessor = prelude is not None
    if prelude is not None:
        records.append({
            "path": _absolute_output_path(prelude["video"]),
            "input_frames": int(prelude["frame_count"]),
            "delivered_frames": int(prelude["frame_count"]),
            "blend_frames": 0,
        })

    for item in segments:
        delivered = int(item["delivered_frames"])
        repeated = max(0, int(item.get("raw_frames", delivered)) - delivered)
        expected_blend = min(configured, repeated) if has_predecessor else 0
        if expected_blend:
            recorded_blend = int(item.get("blend_frames", 0))
            value = item.get("blend_segment")
            if recorded_blend != expected_blend or not isinstance(value, str):
                raise ValueError(
                    "H3 chain clip %d requires a %d-frame blend artifact, but "
                    "the manifest contains %d. Re-save the scene with Plan "
                    "video_blend_frames connected to Loop Trim and its "
                    "images_with_overlap connected to Segment Save." %
                    (int(item.get("index", 0)), expected_blend,
                     recorded_blend))
            path = _absolute_output_path(value)
        else:
            path = _absolute_output_path(item["segment"])
        if not os.path.isfile(path):
            raise FileNotFoundError("H3 chain blend input is missing: %s" % path)
        records.append({
            "path": path,
            "input_frames": delivered + expected_blend,
            "delivered_frames": delivered,
            "blend_frames": expected_blend,
        })
        has_predecessor = True
    if not records:
        raise ValueError("H3 chain blend assembly has no video inputs.")
    return records if any(int(item["blend_frames"]) for item in records) else []


def _ffmpeg_blend_video(
    ffmpeg: str,
    records: list[dict[str, Any]],
    path: str,
    metadata_path: str,
    total_frames: int,
    crf: int,
) -> None:
    """Cumulatively xfade overlap-bearing segments without changing duration."""
    command = [ffmpeg, "-y"]
    for record in records:
        command.extend(["-i", record["path"]])
    command.extend(["-f", "ffmetadata", "-i", metadata_path])

    filters = []
    for index in range(len(records)):
        filters.append(
            "[%d:v]fps=%d,settb=AVTB,setpts=N/(%d*TB)[v%d]" %
            (index, FPS, FPS, index))
    previous = "v0"
    cumulative = int(records[0]["delivered_frames"])
    for index, record in enumerate(records[1:], start=1):
        blend = int(record["blend_frames"])
        if blend < 1:
            raise ValueError(
                "H3 cumulative blend input %d has no retained overlap." % index)
        output = "blend%d" % index
        filters.append(
            "[%s][v%d]xfade=transition=fade:duration=%.9f:offset=%.9f[%s]" %
            (previous, index, blend / float(FPS),
             (cumulative - blend) / float(FPS), output))
        previous = output
        cumulative += int(record["delivered_frames"])
    filters.append(
        "[%s]fps=%d,trim=end_frame=%d,setpts=N/(%d*TB),format=yuv420p[outv]" %
        (previous, FPS, int(total_frames), FPS))
    command.extend([
        "-filter_complex", ";".join(filters),
        "-map", "[outv]", "-map_metadata", str(len(records)), "-an",
        "-c:v", "libx264", "-preset", "medium", "-crf", str(int(crf)),
        "-pix_fmt", "yuv420p", "-r", str(FPS),
        "-frames:v", str(int(total_frames)),
        "-movflags", "use_metadata_tags+faststart", path,
    ])
    _run_ffmpeg(command)


def _decode_rgb_frames(path: str):
    with av.open(path, mode="r") as container:
        streams = list(container.streams.video)
        if len(streams) != 1:
            raise ValueError(
                "H3 blend input %s contains %d video streams; expected 1." %
                (path, len(streams)))
        for frame in container.decode(streams[0]):
            yield frame.to_ndarray(format="rgb24")


def _pyav_blend_video(
    records: list[dict[str, Any]],
    path: str,
    metadata: dict[str, Any],
    total_frames: int,
    crf: int,
) -> None:
    """Streaming PyAV fallback; memory use is bounded by the blend window."""
    if av is None or np is None:
        raise RuntimeError("PyAV cumulative blending requires PyAV and NumPy.")
    if not records:
        raise ValueError("PyAV H3 blending requires at least one input.")

    output = None
    try:
        with av.open(records[0]["path"], mode="r") as first:
            streams = list(first.streams.video)
            if len(streams) != 1:
                raise ValueError("The first H3 blend input must have one video stream.")
            width = int(streams[0].codec_context.width)
            height = int(streams[0].codec_context.height)

        output = av.open(
            path, mode="w",
            options={"movflags": "use_metadata_tags+faststart"})
        for key, value in metadata.items():
            if value is not None:
                output.metadata[str(key)] = str(value)
        stream = output.add_stream("libx264", rate=Fraction(FPS, 1))
        stream.width = width
        stream.height = height
        stream.pix_fmt = "yuv420p"
        stream.options = {"crf": str(int(crf)), "preset": "medium"}
        pending: deque[Any] = deque()
        written = 0

        def encode(array: Any) -> None:
            nonlocal written
            frame = av.VideoFrame.from_ndarray(array, format="rgb24")
            frame.pts = written
            frame.time_base = Fraction(1, FPS)
            for packet in stream.encode(frame):
                output.mux(packet)
            written += 1

        for record_index, record in enumerate(records):
            iterator = iter(_decode_rgb_frames(record["path"]))
            expected_input = int(record["input_frames"])
            blend = int(record["blend_frames"])
            next_blend = (int(records[record_index + 1]["blend_frames"])
                          if record_index + 1 < len(records) else 0)
            seen = 0
            if record_index == 0:
                for array in iterator:
                    pending.append(array)
                    seen += 1
                    while len(pending) > next_blend:
                        encode(pending.popleft())
            else:
                if len(pending) != blend:
                    raise RuntimeError(
                        "H3 PyAV blend retained %d predecessor frames; expected %d."
                        % (len(pending), blend))
                for offset in range(blend):
                    try:
                        incoming = next(iterator)
                    except StopIteration as exc:
                        raise ValueError(
                            "H3 blend input ended inside its overlap.") from exc
                    previous = pending.popleft()
                    alpha = (offset + 1) / float(blend + 1)
                    mixed = np.clip(
                        previous.astype(np.float32) * (1.0 - alpha) +
                        incoming.astype(np.float32) * alpha,
                        0.0, 255.0).round().astype(np.uint8)
                    encode(mixed)
                    seen += 1
                for array in iterator:
                    pending.append(array)
                    seen += 1
                    while len(pending) > next_blend:
                        encode(pending.popleft())
            if seen != expected_input:
                raise ValueError(
                    "H3 blend input %d decoded %d frames; expected %d." %
                    (record_index + 1, seen, expected_input))
            if len(pending) != next_blend:
                raise RuntimeError(
                    "H3 PyAV blend retained %d frames for the next boundary; "
                    "expected %d." % (len(pending), next_blend))

        while pending:
            encode(pending.popleft())
        if written != int(total_frames):
            raise RuntimeError(
                "H3 PyAV blend wrote %d frames; expected %d." %
                (written, total_frames))
        for packet in stream.encode():
            output.mux(packet)
        output.close()
        output = None
    except Exception:
        if output is not None:
            output.close()
        _safe_unlink(path)
        raise


def _pyav_mux_audio(video_path: str, audio: dict[str, Any], path: str,
                     bitrate_kbps: int, total_frames: int) -> None:
    """Stream-copy joined video and encode frame-locked AAC through PyAV."""
    if av is None or torch is None:
        raise RuntimeError("PyAV H3 audio muxing requires PyAV and torch.")
    waveform, sample_rate = _validate_audio(
        audio, "PyAV H3 Chain Assemble audio")
    if waveform.ndim == 3:
        waveform = waveform[0]
    elif waveform.ndim == 1:
        waveform = waveform.unsqueeze(0)
    if waveform.ndim != 2:
        raise ValueError("H3 chain audio must be [batch,channels,samples].")
    channels = int(waveform.shape[0])
    if channels not in (1, 2):
        raise ValueError(
            "PyAV H3 assembly supports mono or stereo audio; got %d channels."
            % channels)
    required_samples = int(round(
        int(total_frames) / float(FPS) * sample_rate))
    if int(waveform.shape[-1]) < required_samples:
        raise ValueError(
            "PyAV H3 assembly audio contains %d samples; %d are required."
            % (int(waveform.shape[-1]), required_samples))
    waveform = (torch.clamp(waveform[..., :required_samples], -1.0, 1.0)
                .to(device="cpu", dtype=torch.float32).contiguous().numpy())
    layout = "mono" if channels == 1 else "stereo"

    source = output = None
    try:
        source = av.open(video_path, mode="r")
        streams = list(source.streams.video)
        if len(streams) != 1:
            raise ValueError(
                "Joined H3 video contains %d video streams; expected 1."
                % len(streams))
        input_video = streams[0]
        output = av.open(
            path, mode="w",
            options={"movflags": "use_metadata_tags+faststart"})
        for key, value in source.metadata.items():
            output.metadata[str(key)] = str(value)
        output_video = output.add_stream_from_template(input_video)
        output_audio = output.add_stream("aac", rate=sample_rate)
        output_audio.bit_rate = int(bitrate_kbps) * 1000
        output_audio.layout = layout

        for packet in source.demux(input_video):
            if packet.dts is None:
                continue
            packet.stream = output_video
            output.mux(packet)

        chunk_size = 1024
        for start in range(0, required_samples, chunk_size):
            stop = min(required_samples, start + chunk_size)
            frame = av.AudioFrame.from_ndarray(
                waveform[:, start:stop], format="fltp", layout=layout)
            frame.sample_rate = sample_rate
            frame.pts = start
            frame.time_base = Fraction(1, sample_rate)
            for packet in output_audio.encode(frame):
                output.mux(packet)
        for packet in output_audio.encode():
            output.mux(packet)

        output.close()
        output = None
        source.close()
        source = None
    except Exception:
        if output is not None:
            output.close()
        if source is not None:
            source.close()
        _safe_unlink(path)
        raise


def _write_ffmetadata(path: str, metadata: dict[str, Any]) -> None:
    def escape(value: Any) -> str:
        text = str(value).replace("\\", "\\\\")
        for character in ("=", ";", "#"):
            text = text.replace(character, "\\" + character)
        return text.replace("\n", "\\\n")

    lines = [";FFMETADATA1"]
    lines.extend("%s=%s" % (escape(key), escape(value))
                 for key, value in metadata.items() if value is not None)
    _atomic_text(path, "\n".join(lines) + "\n")


def _manifest_media_metadata(manifest: dict[str, Any]) -> dict[str, str]:
    metadata = _archive_media_metadata(manifest.get("archives"))
    metadata.update({
        "title": "MiniMax H3 chain - %s" % manifest.get("run_name", "h3_chain"),
        "comment": "%d H3 scenes; prompts and recovery workflow embedded" %
                   int(manifest.get("clip_count", 0)),
        "h3_manifest": json.dumps(
            manifest, ensure_ascii=False, separators=(",", ":")),
    })
    return metadata


def _checkpoint_export_segments(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    segments = manifest.get("segments") or []
    clip_count = int(manifest.get("clip_count", 0))
    if clip_count < 1 or len(segments) != clip_count:
        raise ValueError(
            "H3 PNG export manifest contains %d segments; expected %d." %
            (len(segments), clip_count))
    delivered_total = 0
    for expected_index, segment in enumerate(segments, start=1):
        if int(segment.get("index", -1)) != expected_index:
            raise ValueError(
                "H3 PNG export requires contiguous segment indexes starting "
                "at 1; expected clip %d." % expected_index)
        checkpoint_value = segment.get("checkpoint")
        if not isinstance(checkpoint_value, str):
            raise ValueError(
                "H3 PNG export clip %d has no checkpoint path." % expected_index)
        checkpoint = _absolute_output_path(checkpoint_value)
        if not os.path.isfile(checkpoint):
            raise FileNotFoundError(
                "H3 PNG export checkpoint is missing: %s" % checkpoint)
        expected_hash = str(segment.get("checkpoint_sha256") or "")
        if expected_hash and _file_sha256(checkpoint) != expected_hash:
            raise ValueError(
                "H3 PNG export clip %d checkpoint failed its SHA-256 integrity "
                "check." % expected_index)
        raw_frames = int(segment.get("raw_frames", 0))
        delivered_frames = int(segment.get("delivered_frames", 0))
        if raw_frames < 1 or delivered_frames < 1 or delivered_frames > raw_frames:
            raise ValueError(
                "H3 PNG export clip %d has invalid raw/delivered frame counts "
                "%d/%d." % (expected_index, raw_frames, delivered_frames))
        delivered_total += delivered_frames
    expected_total = int(manifest.get("total_delivered_frames", -1))
    if delivered_total != expected_total:
        raise ValueError(
            "H3 PNG export segment durations total %d frames; expected %d." %
            (delivered_total, expected_total))
    return segments


def _new_export_directory(manifest: dict[str, Any], export_name: str) -> str:
    run_name = _safe_name(manifest.get("run_name"), "h3_chain")
    name = _safe_name(export_name, "png_sequence")
    base = os.path.abspath(os.path.join(
        _output_root(), "h3_chains", run_name, "frames", name))
    root = _output_root()
    if os.path.commonpath([root, base]) != root:
        raise ValueError("H3 PNG export path escapes the ComfyUI output directory.")
    for suffix in range(0, 10000):
        candidate = base if suffix == 0 else "%s_%04d" % (base, suffix + 1)
        try:
            os.makedirs(candidate, exist_ok=False)
            return candidate
        except FileExistsError:
            continue
    raise RuntimeError("H3 PNG export could not allocate a unique output folder.")


def _write_png(path: str, image: Any, compression: int,
               metadata: dict[str, Any]) -> None:
    if Image is None or PngImagePlugin is None or torch is None:
        raise RuntimeError("H3 PNG export requires Pillow and torch.")
    if not torch.is_tensor(image) or image.ndim != 3 or image.shape[-1] < 3:
        raise ValueError(
            "H3 PNG export expected one [height,width,channels] image; got %r." %
            (getattr(image, "shape", None),))
    pixels = (torch.clamp(image[..., :3], 0.0, 1.0) * 255.0)
    pixels = pixels.round().to(device="cpu", dtype=torch.uint8).numpy()
    pnginfo = PngImagePlugin.PngInfo()
    for key, value in metadata.items():
        if value is not None:
            pnginfo.add_text(str(key), str(value))
    temporary = "%s.%s.tmp" % (path, uuid.uuid4().hex)
    try:
        Image.fromarray(pixels).save(
            temporary, format="PNG", compress_level=int(compression),
            pnginfo=pnginfo)
        os.replace(temporary, path)
    finally:
        _safe_unlink(temporary)


class MiniMaxH3ChainExportPNG:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "manifest": (MANIFEST_TYPE, {
                    "tooltip": "Completed or partial manifest from Loop End or "
                               "Manifest Load. Checkpoint latents are decoded "
                               "scene by scene; the H.264 segments are not used."}),
                "video_vae": ("VAE", {
                    "tooltip": "The same MiniMax H3 video VAE used for the "
                               "original render. Decode precision and VAE "
                               "settings determine whether regenerated pixels "
                               "exactly match the first decode."}),
                "export_name": ("STRING", {
                    "default": "png_sequence",
                    "tooltip": "Folder name under output/h3_chains/<run>/frames. "
                               "An existing folder is never overwritten; a "
                               "numbered sibling is created automatically."}),
                "first_frame_number": ("INT", {
                    "default": 1, "min": 0, "max": 999999999,
                    "tooltip": "Number used by the first exported file. Frames "
                               "then continue across scene boundaries without "
                               "resetting."}),
                "png_compression": ("INT", {
                    "default": 4, "min": 0, "max": 9,
                    "tooltip": "Lossless PNG compression effort. 0 is fastest "
                               "and largest; 9 is slowest and smallest. It does "
                               "not change pixels."}),
                "embed_workflow": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Embed the archived ComfyUI workflow, API graph, "
                               "effective plan, and chain manifest in the first "
                               "PNG. Scene prompt metadata is embedded in the "
                               "first frame of every scene."}),
            },
        }

    RETURN_TYPES = ("STRING", "INT", "STRING")
    RETURN_NAMES = ("output_directory", "frame_count", "status")
    OUTPUT_TOOLTIPS = (
        "Absolute folder containing the continuous PNG sequence and export.json.",
        "Total number of delivered frames written across all decoded scenes.",
        "Export folder, scene count, frame count, and frame-number range.",
    )
    FUNCTION = "export"
    OUTPUT_NODE = True
    CATEGORY = "conditioning/minimax/contex_loop"
    DESCRIPTION = ("Re-decode every saved H3 video checkpoint with the selected "
                   "VAE, remove each scene's repeated context overlap, and write "
                   "a continuous lossless PNG sequence without retaining the "
                   "whole production in memory.")

    @classmethod
    def IS_CHANGED(cls, *args, **kwargs):
        return float("NaN")

    def export(self, manifest, video_vae, export_name, first_frame_number,
               png_compression, embed_workflow):
        if _st_load is None or torch is None:
            raise RuntimeError(
                "H3 PNG export requires safetensors and torch.")
        segments = _checkpoint_export_segments(manifest)
        output_dir = _new_export_directory(manifest, export_name)
        partial_path = os.path.join(output_dir, "export.partial.json")
        final_path = os.path.join(output_dir, "export.json")
        frame_number = int(first_frame_number)
        first_number = frame_number
        written = 0
        clip_records = []
        archive_metadata = (_archive_media_metadata(manifest.get("archives"))
                            if bool(embed_workflow) else {})
        manifest_metadata = json.dumps(
            manifest, ensure_ascii=False, separators=(",", ":"))

        for segment in segments:
            index = int(segment["index"])
            checkpoint = _absolute_output_path(segment["checkpoint"])
            tensors = _st_load(checkpoint)
            video = tensors.get("video")
            if video is None:
                raise ValueError(
                    "H3 PNG export checkpoint for clip %d has no video latent." %
                    index)
            images = video_vae.decode(video)
            if not torch.is_tensor(images):
                raise ValueError(
                    "H3 PNG export VAE returned %r instead of an image tensor." %
                    type(images))
            if images.ndim == 5:
                images = images.reshape(
                    -1, images.shape[-3], images.shape[-2], images.shape[-1])
            if images.ndim != 4:
                raise ValueError(
                    "H3 PNG export VAE returned image shape %s; expected "
                    "[frames,height,width,channels]." % (tuple(images.shape),))

            raw_frames = int(segment["raw_frames"])
            delivered_frames = int(segment["delivered_frames"])
            trim_frames = raw_frames - delivered_frames
            if int(images.shape[0]) != raw_frames:
                raise ValueError(
                    "H3 PNG export decoded %d frames for clip %d; its manifest "
                    "requires %d raw frames before trimming %d overlap frames." %
                    (int(images.shape[0]), index, raw_frames, trim_frames))
            images = images[trim_frames:trim_frames + delivered_frames]
            clip_first = frame_number
            prompt = str(segment.get("prompt") or "")
            scene_metadata = json.dumps({
                "index": index,
                "id": str(segment.get("id") or "clip_%04d" % index),
                "prompt_prefix": str(segment.get("prompt_prefix") or ""),
                "scene_prompt": str(segment.get("scene_prompt") or ""),
                "prompt": prompt,
                "prompt_hash": str(segment.get("prompt_hash") or ""),
                "seed": str(segment.get("seed") or ""),
                "raw_frames": raw_frames,
                "delivered_frames": delivered_frames,
                "trim_frames": trim_frames,
            }, ensure_ascii=False, separators=(",", ":"))

            for scene_frame, image in enumerate(images):
                filename = "frame_%08d.png" % frame_number
                png_metadata = {
                    "h3_run_name": str(manifest.get("run_name") or ""),
                    "h3_clip_index": str(index),
                    "h3_clip_frame": str(scene_frame + 1),
                    "h3_frame_number": str(frame_number),
                    "h3_prompt_hash": str(segment.get("prompt_hash") or ""),
                }
                if scene_frame == 0:
                    png_metadata["h3_scene"] = scene_metadata
                    png_metadata["h3_prompt"] = prompt
                if written == 0 and bool(embed_workflow):
                    png_metadata.update(archive_metadata)
                    png_metadata["h3_manifest"] = manifest_metadata
                _write_png(
                    os.path.join(output_dir, filename), image,
                    int(png_compression), png_metadata)
                frame_number += 1
                written += 1

            clip_records.append({
                "index": index,
                "id": str(segment.get("id") or "clip_%04d" % index),
                "checkpoint": segment["checkpoint"],
                "prompt": prompt,
                "prompt_hash": str(segment.get("prompt_hash") or ""),
                "seed": segment.get("seed"),
                "trim_frames": trim_frames,
                "delivered_frames": delivered_frames,
                "first_frame_number": clip_first,
                "last_frame_number": frame_number - 1,
            })
            progress = {
                "format": "h3_chain_png_export_v1",
                "complete": False,
                "run_name": manifest.get("run_name"),
                "source_manifest_format": manifest.get("format"),
                "first_frame_number": first_number,
                "frame_count": written,
                "clips": clip_records,
                "archives": manifest.get("archives", {}),
            }
            _atomic_json(partial_path, progress)
            del images, video, tensors

        export_record = {
            "format": "h3_chain_png_export_v1",
            "complete": True,
            "run_name": manifest.get("run_name"),
            "source_manifest_format": manifest.get("format"),
            "source_plan_hash": manifest.get("plan_hash"),
            "first_frame_number": first_number,
            "last_frame_number": frame_number - 1,
            "frame_count": written,
            "clips": clip_records,
            "archives": manifest.get("archives", {}),
        }
        _atomic_json(final_path, export_record)
        _safe_unlink(partial_path)
        status = ("exported %d clips / %d PNG frames (%d..%d) -> %s" %
                  (len(segments), written, first_number, frame_number - 1,
                   output_dir))
        _LOG.info("H3 Chain %s", status)
        return {"ui": {"text": [status]},
                "result": (output_dir, written, status)}


class MiniMaxH3ChainAssemble:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "manifest": (MANIFEST_TYPE, {
                    "tooltip": "Completed manifest from Loop End or Manifest "
                               "Load. Every segment file is verified before "
                               "joining."}),
                "audio_source": (["plan", "source", "generated", "none"],
                                 {
                                     "default": "plan",
                                     "tooltip": "plan follows the plan's audio "
                                                "mode; source muxes the external "
                                                "track; generated joins saved "
                                                "delivered scene audio; none "
                                                "creates a silent MP4."}),
                "filename": ("STRING", {
                    "default": "final",
                    "tooltip": "Final MP4 basename inside this chain's output "
                               "folder. The .mp4 extension is added "
                               "automatically. Supports date tokens such as "
                               "%date:yyyy-MM-dd%. Existing files are preserved "
                               "by adding _001, _002, and so on."}),
                "audio_bitrate": ("INT", {
                    "default": 256, "min": 64, "max": 512,
                    "tooltip": "AAC bitrate in kilobits per second for the "
                               "final mux. It does not re-encode the saved H.264 "
                               "video segments."}),
            },
            "optional": {
                "source_audio": ("AUDIO", {
                    "tooltip": "Full original source track. Required when "
                               "audio_source resolves to source; it is trimmed "
                               "or safely silent-padded to the final duration."}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("video_path",)
    OUTPUT_TOOLTIPS = (
        "Absolute path of the assembled final MP4.",
    )
    FUNCTION = "assemble"
    OUTPUT_NODE = True
    CATEGORY = "conditioning/minimax/contex_loop"
    DESCRIPTION = ("Stream-copy saved H3 segments into one MP4 and mux either "
                   "the original source track or checkpointed generated audio.")

    @classmethod
    def IS_CHANGED(cls, *args, **kwargs):
        return float("NaN")

    def assemble(self, manifest, audio_source, filename, audio_bitrate,
                 source_audio=None, overwrite_existing=False):
        segments = _validate_manifest(manifest)
        prelude = _validate_prelude(manifest)
        selected = audio_source
        if selected == "plan":
            mode = manifest["compatibility"]["audio_mode"]
            selected = ("source" if mode in
                        ("source_track", "source_plus_timeline")
                        else "generated")
        preserve_generated = manifest.get("format") == "h3_chain_manifest_v3"
        generated_track = None
        generated_warning = ""
        if preserve_generated or selected == "generated":
            try:
                generated_track = _generated_audio(manifest)
            except Exception as exc:
                if selected == "generated":
                    raise
                generated_warning = (
                    "generated audio sidecar unavailable: %s" % exc)
                _LOG.warning("H3 Chain %s", generated_warning)
        audio = None
        if selected == "source":
            _validate_source_audio_hash(
                manifest["compatibility"], source_audio,
                "H3 Chain Assemble")
            waveform, sample_rate = _validate_audio(
                source_audio, "H3 Chain Assemble source audio")
            required_samples = int(round(
                int(manifest["total_delivered_frames"]) /
                float(FPS) * sample_rate))
            if int(waveform.shape[-1]) < required_samples:
                if manifest["compatibility"].get(
                        "source_audio_silent_padding") and _audio_is_silent(waveform):
                    audio = _pad_audio_to_samples(
                        source_audio, required_samples,
                        "H3 Chain Assemble silent placeholder audio")
                else:
                    raise ValueError(
                        "H3 Chain Assemble source audio has %d samples; at least "
                        "%d are required for %d video frames." %
                        (int(waveform.shape[-1]), required_samples,
                         int(manifest["total_delivered_frames"])))
            else:
                audio = source_audio
        elif selected == "generated":
            audio = generated_track
        elif selected != "none":
            raise ValueError("Unknown H3 chain assembly audio source %r."
                             % selected)
        extension_frames = int(manifest["total_delivered_frames"])
        prelude_frames = int(prelude["frame_count"]) if prelude is not None else 0
        total_output_frames = prelude_frames + extension_frames
        if audio is not None and prelude is not None:
            audio = _audio_with_prelude(audio, extension_frames, prelude)
        generated_sidecar_audio = generated_track if preserve_generated else None
        if generated_sidecar_audio is not None and prelude is not None:
            generated_sidecar_audio = _audio_with_prelude(
                generated_sidecar_audio, extension_frames, prelude)

        run_name = _safe_name(manifest.get("run_name"), "h3_chain")
        run_dir = os.path.join(_output_root(), "h3_chains", run_name)
        final_dir = os.path.join(run_dir, "final")
        os.makedirs(final_dir, exist_ok=True)
        final_name = _safe_name(_expand_filename_date(filename), "final")
        final_path = os.path.join(final_dir, final_name + ".mp4")
        if not overwrite_existing:
            final_path = _available_versioned_path(final_path)
        generated_sidecar_path = (
            os.path.splitext(final_path)[0] + ".generated.wav"
            if generated_sidecar_audio is not None else None)
        concat_path = os.path.join(final_dir, ".concat.txt")
        video_tmp = os.path.join(final_dir, ".video.tmp.mp4")
        final_tmp = os.path.join(final_dir, ".final.tmp.mp4")
        wav_tmp = os.path.join(final_dir, ".audio.tmp.wav")
        metadata_tmp = os.path.join(final_dir, ".metadata.tmp.txt")

        segment_paths = []
        delivered_frames = []
        if prelude is not None:
            segment_paths.append(_absolute_output_path(prelude["video"]))
            delivered_frames.append(prelude_frames)
        for item in segments:
            path = _absolute_output_path(item["segment"])
            if not os.path.isfile(path):
                raise FileNotFoundError("H3 chain segment is missing: %s" % path)
            segment_paths.append(path)
            delivered_frames.append(int(item["delivered_frames"]))
        blend_records = _blend_video_records(manifest, segments, prelude)
        blend_enabled = bool(blend_records)
        ffmpeg = shutil.which("ffmpeg")
        if not ffmpeg and av is None:
            raise RuntimeError(
                "H3 Chain Assemble found neither an ffmpeg executable nor "
                "PyAV. Install ffmpeg or restore ComfyUI's av package.")

        for temporary in (video_tmp, final_tmp, wav_tmp, metadata_tmp):
            if os.path.exists(temporary):
                os.unlink(temporary)
        backend = "ffmpeg"
        try:
            media_metadata = _manifest_media_metadata(manifest)
            if blend_enabled:
                _write_ffmetadata(metadata_tmp, media_metadata)
                if ffmpeg:
                    try:
                        _ffmpeg_blend_video(
                            ffmpeg, blend_records, video_tmp, metadata_tmp,
                            total_output_frames,
                            int(manifest["compatibility"].get(
                                "segment_crf", 18)))
                        backend = "ffmpeg cumulative linear blend"
                    except Exception as exc:
                        if av is None:
                            raise
                        backend = "PyAV cumulative linear blend fallback"
                        _LOG.warning(
                            "H3 Chain ffmpeg blending failed; retrying with "
                            "the built-in PyAV fallback: %s", exc)
                        _safe_unlink(video_tmp)
                        _pyav_blend_video(
                            blend_records, video_tmp, media_metadata,
                            total_output_frames,
                            int(manifest["compatibility"].get(
                                "segment_crf", 18)))
                else:
                    backend = "PyAV cumulative linear blend fallback"
                    _LOG.warning(
                        "H3 Chain ffmpeg executable not found; blending with "
                        "the built-in PyAV fallback")
                    _pyav_blend_video(
                        blend_records, video_tmp, media_metadata,
                        total_output_frames,
                        int(manifest["compatibility"].get(
                            "segment_crf", 18)))
            elif ffmpeg:
                with open(concat_path, "w", encoding="utf-8") as handle:
                    for path in segment_paths:
                        escaped = path.replace("\\", "\\\\").replace(
                            "'", "'\\''")
                        handle.write("file '%s'\n" % escaped)
                _write_ffmetadata(metadata_tmp, media_metadata)
                _run_ffmpeg([
                    ffmpeg, "-y", "-f", "concat", "-safe", "0", "-i",
                    concat_path, "-f", "ffmetadata", "-i", metadata_tmp,
                    "-map", "0:v:0", "-map_metadata", "1", "-c", "copy",
                    "-movflags", "use_metadata_tags+faststart", video_tmp,
                ])
            else:
                backend = "PyAV fallback"
                _LOG.warning(
                    "H3 Chain ffmpeg executable not found; assembling with "
                    "the built-in PyAV stream-copy fallback")
                _pyav_concat_video(
                    segment_paths, delivered_frames, video_tmp, media_metadata)

            if audio is None:
                os.replace(video_tmp, final_tmp)
            elif ffmpeg:
                _write_wav(audio, wav_tmp)
                _run_ffmpeg([
                    ffmpeg, "-y", "-i", video_tmp, "-i", wav_tmp,
                    "-map", "0:v:0", "-map", "1:a:0", "-c:v", "copy",
                    "-c:a", "aac", "-b:a", "%dk" % int(audio_bitrate),
                    "-t", "%.9f" % (total_output_frames / float(FPS)),
                    "-map_metadata", "0",
                    "-movflags", "use_metadata_tags+faststart", final_tmp,
                ])
            else:
                _pyav_mux_audio(
                    video_tmp, audio, final_tmp, int(audio_bitrate),
                    total_output_frames)
            if generated_sidecar_path is not None:
                _atomic_wav(generated_sidecar_audio, generated_sidecar_path)
            os.replace(final_tmp, final_path)
        finally:
            for temporary in (concat_path, video_tmp, final_tmp, wav_tmp,
                              metadata_tmp):
                if os.path.exists(temporary):
                    os.unlink(temporary)

        sidecar_status = (
            "; generated audio -> %s" % generated_sidecar_path
            if generated_sidecar_path is not None else
            ("; %s" % generated_warning if generated_warning else ""))
        blend_status = (
            "; %d-frame cumulative visual blend" % int(
                manifest["compatibility"].get("video_blend_frames", 0))
            if blend_enabled else "; hard cuts")
        status = "assembled %d generated clips%s with %s%s -> %s%s" % (
            len(segments), " + existing-video prelude" if prelude else "",
            backend, blend_status, final_path, sidecar_status)
        _LOG.info("H3 Chain %s", status)
        _publish_final_review_preview(manifest, final_path, status)
        return {"ui": {"text": [status]}, "result": (final_path,)}


class MiniMaxH3ChainVideoOutput:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video_path": ("STRING", {
                    "forceInput": True,
                    "tooltip": "Connect the video_path output from H3 Chain "
                               "Assemble. The completed MP4 is shown with "
                               "ComfyUI's standard playback controls.",
                }),
            },
        }

    RETURN_TYPES = ()
    FUNCTION = "preview"
    OUTPUT_NODE = True
    CATEGORY = "conditioning/minimax/contex_loop"
    DESCRIPTION = ("Display an assembled H3 chain MP4 directly in the "
                   "workflow without copying or re-encoding it.")

    @classmethod
    def IS_CHANGED(cls, *args, **kwargs):
        return float("NaN")

    def preview(self, video_path):
        path = _absolute_output_path(str(video_path or "").strip())
        if not os.path.isfile(path):
            raise FileNotFoundError(
                "H3 Chain Final Video Output could not find: %s" % path)
        if os.path.splitext(path)[1].lower() not in (
                ".mp4", ".webm", ".mov", ".mkv"):
            raise ValueError(
                "H3 Chain Final Video Output requires a browser-playable "
                "video file, not %s." % path)
        item = _video_output_item(path)
        return {
            "ui": {
                # PreviewVideo uses the standard image-preview payload with
                # animated=True; the frontend selects its video player for
                # MP4/WebM/MOV/MKV artifacts.
                "images": [item],
                "animated": (True,),
                "text": ["final video -> %s" % path],
            },
            "result": (),
        }


LTX_ROLLING_CONTEXT_TYPE = "H3_RELAY_LTX_ROLLING_CONTEXT"


def _decode_video_images(path: str) -> Any:
    if av is None or np is None or torch is None:
        raise RuntimeError(
            "Incremental LTX input requires PyAV, NumPy, and PyTorch.")
    frames = []
    with av.open(path, mode="r") as container:
        streams = list(container.streams.video)
        if len(streams) != 1:
            raise ValueError(
                "Incremental LTX input %s contains %d video streams; "
                "expected one." % (path, len(streams)))
        for frame in container.decode(streams[0]):
            array = frame.to_ndarray(format="rgb24")
            frames.append(torch.from_numpy(array).to(dtype=torch.float32).div_(255.0))
    if not frames:
        raise ValueError("Incremental LTX input contains no video frames: %s" % path)
    return torch.stack(frames, dim=0)


class MiniMaxH3LTXRollingInput:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "current_segment": ("STRING", {
                    "default": "",
                    "tooltip": "Relative or absolute path to the current "
                               "delivered H3 segment inside ComfyUI output.",
                }),
                "previous_checkpoint": ("STRING", {
                    "default": "",
                    "tooltip": "For shot 2+, connect or enter the previous "
                               "accepted H3 checkpoint. Its exact decoded "
                               "tail becomes LTX context. Leave blank for "
                               "the first shot.",
                }),
                "context_frames": ("INT", {
                    "default": 17, "min": 1, "max": 257, "step": 1,
                    "tooltip": "Pixel frames carried into LTX. Seventeen "
                               "maps exactly to three LTX temporal latents.",
                }),
            },
        }

    RETURN_TYPES = ("IMAGE", LTX_ROLLING_CONTEXT_TYPE, "INT", "INT", "STRING")
    RETURN_NAMES = (
        "padded_images", "rolling_context", "original_frames",
        "delivered_frames", "status")
    FUNCTION = "prepare"
    CATEGORY = "conditioning/minimax/contex_loop/enhance"
    DESCRIPTION = ("Build one incremental LTX enhancement window from the "
                   "current delivered H3 segment and the exact tail stored "
                   "in the previous accepted checkpoint.")

    def prepare(self, current_segment, previous_checkpoint, context_frames):
        segment_path = _absolute_output_path(str(current_segment or "").strip())
        if not os.path.isfile(segment_path):
            raise FileNotFoundError(
                "Incremental LTX current segment is missing: %s" % segment_path)
        current = _decode_video_images(segment_path)
        delivered = int(current.shape[0])
        context_frames = int(context_frames)
        checkpoint_value = str(previous_checkpoint or "").strip()
        carried = 0
        if checkpoint_value:
            if _st_load is None:
                raise RuntimeError(
                    "Incremental LTX checkpoint context requires safetensors.")
            checkpoint_path = _absolute_output_path(checkpoint_value)
            if not os.path.isfile(checkpoint_path):
                raise FileNotFoundError(
                    "Incremental LTX previous checkpoint is missing: %s" %
                    checkpoint_path)
            tensors = _st_load(checkpoint_path, device="cpu")
            previous = tensors.get("context_frames")
            if previous is None or previous.ndim != 4:
                raise ValueError(
                    "Incremental LTX previous checkpoint has no valid "
                    "context_frames tensor.")
            if int(previous.shape[0]) < context_frames:
                raise ValueError(
                    "Incremental LTX requested %d context frames, but the "
                    "checkpoint contains %d." %
                    (context_frames, int(previous.shape[0])))
            previous = previous[-context_frames:].to(dtype=current.dtype)
            if tuple(previous.shape[1:]) != tuple(current.shape[1:]):
                raise ValueError(
                    "Incremental LTX context shape %s does not match current "
                    "segment shape %s." %
                    (tuple(previous.shape[1:]), tuple(current.shape[1:])))
            images = torch.cat((previous, current), dim=0)
            carried = context_frames
        else:
            images = current

        original_frames = int(images.shape[0])
        padded_frames = int(math.ceil(max(0, original_frames - 1) / 8.0) * 8 + 1)
        padding = padded_frames - original_frames
        if padding:
            images = torch.cat(
                (images, images[-1:].repeat(padding, 1, 1, 1)), dim=0)
        rolling = {
            "format": "h3_ltx_rolling_context_v1",
            "current_segment": _relative_output_path(segment_path),
            "previous_checkpoint": (
                _relative_output_path(_absolute_output_path(checkpoint_value))
                if checkpoint_value else ""),
            "context_frames": carried,
            "delivered_frames": delivered,
            "original_frames": original_frames,
            "padded_frames": padded_frames,
            "padding_frames": padding,
        }
        status = (
            "LTX rolling input: %d carried + %d delivered = %d frames; "
            "padded by %d to %d (8n+1)" %
            (carried, delivered, original_frames, padding, padded_frames))
        return images, rolling, original_frames, delivered, status


def _ltx_rolling_checkpoint_path(run_name: str, shot_index: int) -> str:
    normalized = _safe_name(run_name, "h3_ltx_rolling")
    directory = os.path.join(
        _run_dir({"run_name": normalized}), "enhanced_ltx", "checkpoints")
    os.makedirs(directory, exist_ok=True)
    return os.path.join(directory, "clip_%04d.safetensors" % int(shot_index))


class MiniMaxH3LTXRollingInject:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "latent": ("LATENT",),
                "run_name": ("STRING", {"default": "h3_ltx_rolling"}),
                "shot_index": ("INT", {"default": 1, "min": 1, "max": 100000}),
                "context_latent_steps": ("INT", {
                    "default": 3, "min": 1, "max": 64,
                    "tooltip": "Three LTX latent steps represent 17 pixel "
                               "frames and are the recommended rolling context.",
                }),
            },
        }

    RETURN_TYPES = ("LATENT", "STRING")
    RETURN_NAMES = ("latent", "status")
    FUNCTION = "inject"
    CATEGORY = "conditioning/minimax/contex_loop/enhance"
    DESCRIPTION = ("Freeze the previous accepted LTX latent tail at the head "
                   "of the current enhancement target.")

    def inject(self, latent, run_name, shot_index, context_latent_steps):
        shot_index = int(shot_index)
        if shot_index == 1:
            return latent, "LTX rolling context: first shot, no prior latent"
        if _st_load is None or torch is None:
            raise RuntimeError("Incremental LTX latent injection requires safetensors.")
        previous_path = _ltx_rolling_checkpoint_path(run_name, shot_index - 1)
        if not os.path.isfile(previous_path):
            raise FileNotFoundError(
                "Incremental LTX shot %d requires the previous enhancement "
                "checkpoint: %s" % (shot_index, previous_path))
        previous = _st_load(previous_path, device="cpu").get("context_latent")
        samples = latent.get("samples") if isinstance(latent, dict) else None
        if previous is None or samples is None or samples.ndim != 5:
            raise ValueError("Incremental LTX latent checkpoint or target is invalid.")
        steps = int(context_latent_steps)
        if int(previous.shape[2]) != steps:
            raise ValueError(
                "Incremental LTX checkpoint contains %d latent steps; "
                "expected %d." % (int(previous.shape[2]), steps))
        if tuple(previous.shape[:2] + previous.shape[3:]) != tuple(
                samples.shape[:2] + samples.shape[3:]):
            raise ValueError(
                "Incremental LTX checkpoint shape %s is incompatible with "
                "target %s." % (tuple(previous.shape), tuple(samples.shape)))
        if int(samples.shape[2]) < steps:
            raise ValueError("Incremental LTX target is shorter than its context.")
        out = latent.copy()
        out_samples = samples.clone()
        out_samples[:, :, :steps] = previous.to(
            device=out_samples.device, dtype=out_samples.dtype)
        noise_mask = out.get("noise_mask")
        if noise_mask is None:
            noise_mask = torch.ones(
                (int(out_samples.shape[0]), 1, int(out_samples.shape[2]), 1, 1),
                device=out_samples.device, dtype=torch.float32)
        else:
            noise_mask = noise_mask.clone()
        noise_mask[:, :, :steps] = 0.0
        out["samples"] = out_samples
        out["noise_mask"] = noise_mask
        return out, "LTX rolling context: injected and froze %d latent steps from shot %d" % (
            steps, shot_index - 1)


class MiniMaxH3LTXRollingCheckpoint:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "latent": ("LATENT",),
                "run_name": ("STRING", {"default": "h3_ltx_rolling"}),
                "shot_index": ("INT", {"default": 1, "min": 1, "max": 100000}),
                "context_latent_steps": ("INT", {"default": 3, "min": 1, "max": 64}),
            },
        }

    RETURN_TYPES = ("LATENT", "STRING")
    RETURN_NAMES = ("latent", "status")
    FUNCTION = "save"
    CATEGORY = "conditioning/minimax/contex_loop/enhance"
    DESCRIPTION = ("Persist the accepted LTX latent tail for the next "
                   "incremental enhancement job.")

    def save(self, latent, run_name, shot_index, context_latent_steps):
        if _st_save is None or torch is None:
            raise RuntimeError("Incremental LTX checkpoint saving requires safetensors.")
        samples = latent.get("samples") if isinstance(latent, dict) else None
        if samples is None or samples.ndim != 5:
            raise ValueError("Incremental LTX checkpoint requires a 5D latent.")
        steps = int(context_latent_steps)
        if int(samples.shape[2]) < steps:
            raise ValueError("Incremental LTX output is shorter than its context.")
        path = _ltx_rolling_checkpoint_path(run_name, int(shot_index))
        temporary = "%s.%s.tmp" % (path, uuid.uuid4().hex)
        try:
            _st_save({
                "context_latent": samples[:, :, -steps:].detach().cpu().contiguous(),
            }, temporary, metadata={
                "format": "h3_ltx_rolling_checkpoint_v1",
                "shot_index": str(int(shot_index)),
                "context_latent_steps": str(steps),
            })
            os.replace(temporary, path)
        finally:
            _safe_unlink(temporary)
        return latent, "saved %d-step LTX rolling context -> %s" % (steps, path)


class MiniMaxH3LTXRollingCrop:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "rolling_context": (LTX_ROLLING_CONTEXT_TYPE,),
            },
        }

    RETURN_TYPES = ("IMAGE", "INT", "INT", "STRING")
    RETURN_NAMES = ("images", "context_frames", "delivered_frames", "status")
    FUNCTION = "crop"
    CATEGORY = "conditioning/minimax/contex_loop/enhance"
    DESCRIPTION = "Remove terminal LTX alignment padding after decode."

    def crop(self, images, rolling_context):
        if rolling_context.get("format") != "h3_ltx_rolling_context_v1":
            raise ValueError("Incremental LTX crop received an invalid context.")
        original = int(rolling_context["original_frames"])
        if int(images.shape[0]) < original:
            raise ValueError(
                "Incremental LTX decoded %d frames; expected at least %d." %
                (int(images.shape[0]), original))
        context = int(rolling_context["context_frames"])
        delivered = int(rolling_context["delivered_frames"])
        cropped = images[:original]
        return cropped, context, delivered, (
            "cropped LTX padding: %d -> %d frames; %d context + %d delivered" %
            (int(images.shape[0]), original, context, delivered))


STEER_SEQUENCE_TYPE = "H3_RELAY_SEQUENCE"
STEER_SEQUENCE_FORMAT = "h3_steer_sequence_v1"
STEER_ENHANCED_FPS = 48
STEER_H3_CONTEXT_FRAMES = 18
STEER_H3_CONTEXT_CHOICES = (18, 35, 52, 69)
STEER_LTX_CONTEXT_FRAMES = 17
STEER_LTX_CONTEXT_STEPS = 3
STEER_CACHE_FORMAT = "h3_steer_enhanced_checkpoint_v2"


def _steer_sequence(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict) or value.get("format") != STEER_SEQUENCE_FORMAT:
        raise ValueError(
            "Steerable H3 Segment requires a sequence created by MiniMax H3 "
            "Steerable Sequence Start or a preceding segment.")
    shots = value.get("shots")
    segments = value.get("segments")
    if not isinstance(shots, list) or not isinstance(segments, list):
        raise ValueError("Steerable H3 sequence metadata is incomplete.")
    if len(shots) != len(segments):
        raise ValueError(
            "Steerable H3 sequence has %d shots but %d accepted segments." %
            (len(shots), len(segments)))
    context_frames = int(value.get(
        "h3_context_frames", STEER_H3_CONTEXT_FRAMES))
    if context_frames not in STEER_H3_CONTEXT_CHOICES:
        raise ValueError(
            "Steerable H3 overlap must be one of %s frames." %
            (STEER_H3_CONTEXT_CHOICES,))
    return value


def _steer_resume_state(plan: dict[str, Any], start_clip: int,
                        sequence: dict[str, Any]) -> dict[str, Any]:
    """Resume from revisions pinned by the accepted sequence token.

    The generic chain resume path intentionally follows a run's mutable latest
    checkpoint. An interactive steerable sequence is revisioned differently:
    rerolling or canceling an earlier shot must not silently replace the
    predecessor accepted by an existing downstream token.
    """
    if _st_load is None:
        raise RuntimeError("safetensors is required to resume H3 sequences.")
    previous_index = int(start_clip) - 1
    records = sequence.get("segments") or []
    if len(records) != previous_index:
        raise ValueError(
            "Steerable H3 shot %d requires %d pinned predecessors; got %d." %
            (start_clip, previous_index, len(records)))
    restored_segments = []
    last_segment = None
    for index, record in enumerate(records, 1):
        segment = record.get("h3_segment") if isinstance(record, dict) else None
        if not isinstance(segment, dict):
            raise ValueError(
                "Steerable H3 predecessor %d has no pinned H3 checkpoint." %
                index)
        expected = _history_hash(plan, index)
        if str(segment.get("history_hash") or "") != expected:
            raise ValueError(
                "Steerable H3 predecessor %d was generated from different "
                "settings, prompts, seeds, or durations." % index)
        _verify_segment_artifacts(segment, index)
        restored = _public_segment(segment)
        for key, value in _prompt_fields(plan, index).items():
            restored.setdefault(key, value)
        restored_segments.append(restored)
        last_segment = segment
    if last_segment is None:
        raise RuntimeError("Steerable H3 predecessor checkpoint is unavailable.")
    checkpoint = _absolute_output_path(str(last_segment["checkpoint"]))
    tensors = _st_load(checkpoint)
    required = {"context_frames", "video", "audio"}
    missing = sorted(required - set(tensors))
    if missing:
        raise ValueError(
            "Steerable H3 predecessor checkpoint is missing tensors: %s" %
            missing)
    expected_context = min(
        int(plan["compatibility"]["context_length"]),
        int(plan["shots"][previous_index - 1]["delivered_frames"]))
    if int(tensors["context_frames"].shape[0]) != expected_context:
        raise ValueError(
            "Steerable H3 predecessor checkpoint contains %d context frames; "
            "expected %d." %
            (int(tensors["context_frames"].shape[0]), expected_context))
    return {
        "plan": plan,
        "index": int(start_clip),
        "previous_frames": tensors["context_frames"],
        "previous_latent": {"samples": [tensors["video"], tensors["audio"]]},
        "segments": restored_segments,
        "resumed_from": previous_index,
        "range_start": int(start_clip),
        "end_clip": int(start_clip),
    }


def _steer_state(sequence: dict[str, Any], shot_name: str, prompt: str,
                 seed: int, raw_frames: int, steps: int) -> tuple[
                     dict[str, Any], dict[str, Any]]:
    sequence = _steer_sequence(sequence)
    index = len(sequence["shots"]) + 1
    shot_id = _safe_name(shot_name, "shot_%04d" % index)
    if any(str(item.get("id")) == shot_id for item in sequence["shots"]):
        raise ValueError(
            "Steerable H3 shot name %r is already used in this sequence." %
            shot_id)
    raw_frames = _validate_h3_length(
        raw_frames, "Steerable H3 shot %d length" % index)
    steps = int(steps)
    if steps < 1 or steps > 10000:
        raise ValueError("Steerable H3 steps must be between 1 and 10000.")
    seed = int(seed)
    if seed < 0 or seed > MAX_SEED:
        raise ValueError("Steerable H3 seed is outside the uint64 range.")
    scene_prompt = _prompt_text(
        prompt, "Steerable H3 shot %d prompt" % index)
    if not scene_prompt and not str(sequence.get("global_prompt") or "").strip():
        raise ValueError("Steerable H3 Segment requires a shot or global prompt.")

    shot = {
        "id": shot_id,
        "prompt": scene_prompt,
        "length": raw_frames,
        "steps": steps,
        "seed": str(seed),
    }
    shots = [dict(item) for item in sequence["shots"]] + [shot]
    context_frames = int(sequence.get(
        "h3_context_frames", STEER_H3_CONTEXT_FRAMES))
    plan_json = json.dumps({
        "prompt_prefix": str(sequence.get("global_prompt") or ""),
        "shots": shots,
    }, ensure_ascii=False)
    plan = _normalize_plan(
        plan_json=plan_json,
        run_name=str(sequence["run_name"]),
        width=int(sequence["width"]),
        height=int(sequence["height"]),
        context_length=context_frames,
        encode_mode="video",
        anchor_mode="head",
        crop="disabled",
        audio_mode="generated_audio",
        audio_context_length=context_frames,
        default_duration_seconds=raw_frames / float(FPS),
        default_steps=steps,
        base_seed=0,
        segment_crf=int(sequence["h3_crf"]),
        generation_fingerprint=str(sequence["generation_fingerprint"]),
        video_blend_frames=0,
        continuation_mode="sliding_history",
    )
    state = (_steer_resume_state(plan, index, sequence)
             if index > 1 else _initial_state(plan, index, index))
    return state, plan["shots"][index - 1]


def _steer_media_fingerprint(value: Any, label: str,
                             audio: bool = False) -> str | None:
    if value is None:
        return None
    try:
        return _audio_fingerprint(value) if audio else _tensor_fingerprint(value)
    except (KeyError, TypeError, ValueError, RuntimeError) as exc:
        raise ValueError(
            "Steerable H3 could not fingerprint %s for its durable cache: %s" %
            (label, exc)) from exc


def _steer_cache_key(sequence: dict[str, Any], shot: dict[str, Any],
                     ref_image_size: str, first_frame: Any = None,
                     last_frame: Any = None,
                     reference_image_1: Any = None,
                     reference_image_2: Any = None,
                     reference_image_3: Any = None,
                     reference_video: Any = None,
                     reference_video_audio: Any = None,
                     reference_audio: Any = None) -> str:
    """Fingerprint only inputs that can change the rendered shot.

    ComfyUI preview widgets are deliberately absent. The preceding accepted
    revisions are included so changing an earlier shot invalidates every
    continuation that depends on its visual/audio context.
    """
    predecessor = [{
        "index": int(item["index"]),
        "id": str(item["id"]),
        "revision": str(item["revision"]),
        "enhanced_segment_sha256": str(item["enhanced_segment_sha256"]),
        "h3_checkpoint_sha256": str(
            item.get("h3_segment", {}).get("checkpoint_sha256") or ""),
    } for item in sequence["segments"]]
    contract = {
        "version": 1,
        "sequence": {
            key: sequence.get(key) for key in (
                "run_name", "global_prompt", "width", "height", "h3_fps",
                "enhanced_fps", "h3_context_frames", "ltx_context_frames",
                "ltx_context_steps", "enhancement_prompt", "h3_crf",
                "enhanced_crf", "h3_sampling_profile", "output_profile",
                "generation_fingerprint")
        },
        "predecessor": predecessor,
        "shot": {
            "id": str(shot["id"]),
            "prompt": str(shot["prompt"]),
            "seed": str(shot["seed"]),
            "raw_frames": int(shot["raw_frames"]),
            "steps": int(shot["steps"]),
            "ref_image_size": str(ref_image_size),
        },
        "media": {
            "first_frame": _steer_media_fingerprint(
                first_frame, "first frame"),
            "last_frame": _steer_media_fingerprint(last_frame, "last frame"),
            "reference_image_1": _steer_media_fingerprint(
                reference_image_1, "reference image 1"),
            "reference_image_2": _steer_media_fingerprint(
                reference_image_2, "reference image 2"),
            "reference_image_3": _steer_media_fingerprint(
                reference_image_3, "reference image 3"),
            "reference_video": _steer_media_fingerprint(
                reference_video, "reference video"),
            "reference_video_audio": _steer_media_fingerprint(
                reference_video_audio, "reference video audio", audio=True),
            "reference_audio": _steer_media_fingerprint(
                reference_audio, "reference audio", audio=True),
        },
    }
    return _fingerprint(contract)


def _steer_cached_result(sequence: dict[str, Any], state: dict[str, Any],
                         cache_key: str) -> tuple[Any, ...] | None:
    index = int(state["index"])
    lookup = _steer_enhanced_paths(
        str(state["plan"]["run_name"]), index, "lookup")
    try:
        payload = _read_json(lookup["metadata"])
    except (FileNotFoundError, json.JSONDecodeError, OSError, ValueError):
        return None
    if (payload.get("format") != STEER_CACHE_FORMAT or
            str(payload.get("cache_key") or "") != cache_key):
        return None
    record = payload.get("segment")
    if not isinstance(record, dict) or int(record.get("index", -1)) != index:
        return None
    path = _absolute_output_path(str(record.get("enhanced_segment") or ""))
    expected_hash = str(record.get("enhanced_segment_sha256") or "")
    if (not os.path.isfile(path) or not expected_hash or
            _file_sha256(path) != expected_hash):
        return None
    h3_segment = record.get("h3_segment")
    if not isinstance(h3_segment, dict):
        return None
    try:
        _verify_segment_artifacts(h3_segment, index)
    except (FileNotFoundError, OSError, ValueError):
        return None
    if InputImpl is None:
        raise RuntimeError(
            "Steerable H3 cached VIDEO output requires comfy_api.latest.")
    updated = dict(sequence)
    updated["shots"] = _effective_editor_plan(state["plan"])["shots"]
    updated["segments"] = (
        [dict(item) for item in sequence["segments"]] + [dict(record)])
    updated["total_enhanced_frames"] = sum(
        int(item["enhanced_frames"]) for item in updated["segments"])
    status = (
        "reused cached shot %d from disk; generation inputs and predecessor "
        "context are unchanged -> %s" % (index, path))
    _LOG.info("Steerable H3 %s", status)
    return updated, InputImpl.VideoFromFile(path), path, status


class MiniMaxH3SteerSequenceStart:
    DEFAULT_ENHANCEMENT_PROMPT = (
        "Preserve the exact subjects, action, composition, camera motion, "
        "geometry, timing, illumination, color palette, and scene identity "
        "of the supplied reference video. Restore fine spatial detail, clean "
        "edges, natural material texture, and stable frame-to-frame detail. "
        "Maintain exact temporal continuity through incoming context. Do not "
        "introduce new objects, text, people, cuts, reframing, or motion.")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "run_name": ("STRING", {"default": "h3_steerable_movie"}),
                "global_prompt": ("STRING", {
                    "default": "", "multiline": True,
                    "tooltip": "Style, characters, world rules, and other "
                               "direction prepended to every shot prompt.",
                }),
                "width": ("INT", {
                    "default": 832, "min": 32, "max": 2048, "step": 32,
                    "tooltip": "Native H3 generation width before the 2x "
                               "LTX enhancement.",
                }),
                "height": ("INT", {
                    "default": 480, "min": 32, "max": 2048, "step": 32,
                    "tooltip": "Native H3 generation height before the 2x "
                               "LTX enhancement.",
                }),
                "enhancement_prompt": ("STRING", {
                    "default": cls.DEFAULT_ENHANCEMENT_PROMPT,
                    "multiline": True,
                }),
                "h3_crf": ("INT", {"default": 18, "min": 0, "max": 51}),
                "enhanced_crf": ("INT", {
                    "default": 18, "min": 0, "max": 51,
                }),
            },
            "optional": {
                "h3_sampling_profile": ([
                    "turbo_auto",
                    "res_multistep",
                    "euler",
                    "native_spectrum_euler_beta57",
                ], {
                    "default": "turbo_auto",
                    "tooltip": "turbo_auto keeps the existing 4/8-step "
                               "Turbo LoRA path. euler and res_multistep "
                               "remove the Turbo LoRA and use their native "
                               "ComfyUI samplers with the beta scheduler. "
                               "native_spectrum_euler_beta57 "
                               "removes the Turbo LoRA and uses Spectrum with "
                               "Euler plus the exact supplied beta57 manual "
                               "sigma schedule at 16 steps. Other step counts "
                               "use the equivalent beta57 curve (alpha 0.5, "
                               "beta 0.7).",
                }),
                "output_profile": ([
                    "enhanced_ltx_rife",
                    "raw_h3",
                ], {
                    "default": "enhanced_ltx_rife",
                    "tooltip": "enhanced_ltx_rife applies the existing 2x "
                               "LTX enhancement and RIFE 48fps pass. raw_h3 "
                               "accepts native H3 frames directly at 24fps.",
                }),
            },
        }

    RETURN_TYPES = (STEER_SEQUENCE_TYPE, "STRING")
    RETURN_NAMES = ("sequence", "status")
    FUNCTION = "start"
    CATEGORY = "conditioning/minimax/contex_loop/steerable"
    DESCRIPTION = (
        "Initialize a disk-backed incremental H3 movie. Width, height, H3 "
        "sliding-history overlap, output profile, and global direction become "
        "sequence-wide settings.")

    def start(self, run_name, global_prompt, width, height, enhancement_prompt,
              h3_crf, enhanced_crf, h3_sampling_profile="turbo_auto",
              output_profile="enhanced_ltx_rife"):
        width, height = int(width), int(height)
        if width < 32 or height < 32 or width % 32 or height % 32:
            raise ValueError(
                "Steerable H3 width and height must be positive multiples of 32.")
        normalized = _safe_name(run_name, "h3_steerable_movie")
        h3_sampling_profile = str(h3_sampling_profile or "turbo_auto")
        allowed_profiles = {
            "turbo_auto",
            "res_multistep",
            "euler",
            "native_euler_beta",
            "native_spectrum_euler_beta57",
        }
        if h3_sampling_profile not in allowed_profiles:
            raise ValueError(
                "Unknown H3 sampling profile: %s" % h3_sampling_profile)
        output_profile = str(output_profile or "enhanced_ltx_rife")
        if output_profile not in {"enhanced_ltx_rife", "raw_h3"}:
            raise ValueError("Unknown H3 output profile: %s" % output_profile)
        raw_output = output_profile == "raw_h3"
        sequence = {
            "format": STEER_SEQUENCE_FORMAT,
            "run_name": normalized,
            "global_prompt": str(global_prompt or "").strip(),
            "width": width,
            "height": height,
            "h3_fps": FPS,
            "enhanced_fps": FPS if raw_output else STEER_ENHANCED_FPS,
            "h3_context_frames": STEER_H3_CONTEXT_FRAMES,
            "ltx_context_frames": 0 if raw_output else STEER_LTX_CONTEXT_FRAMES,
            "ltx_context_steps": 0 if raw_output else STEER_LTX_CONTEXT_STEPS,
            "enhancement_prompt": str(enhancement_prompt or "").strip(),
            "h3_crf": int(h3_crf),
            "enhanced_crf": int(enhanced_crf),
            "h3_sampling_profile": h3_sampling_profile,
            "output_profile": output_profile,
            "generation_fingerprint": (
                "h3-hybrid-fl2va-ref2va-%s-sliding18-%s-v1"),
            "shots": [],
            "segments": [],
            "total_enhanced_frames": 0,
        }
        sequence["generation_fingerprint"] = (
            sequence["generation_fingerprint"] % (
                h3_sampling_profile, output_profile))
        if raw_output:
            output_summary = "%dx%d/%dfps native H3" % (width, height, FPS)
            context_summary = "LTX and RIFE bypassed"
        else:
            output_summary = "%dx%d/%dfps enhanced" % (
                width * 2, height * 2, STEER_ENHANCED_FPS)
            context_summary = "17-frame LTX rolling context"
        return sequence, (
            "initialized %s with %s at %dx%d H3 -> %s; "
            "18-frame H3 sliding history; %s" %
            (normalized, h3_sampling_profile, width, height,
             output_summary, context_summary))


class MiniMaxH3SteerSegmentPaths:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "state": (STATE_TYPE,),
                "segment": (SEGMENT_TYPE,),
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING", "INT")
    RETURN_NAMES = (
        "current_segment", "current_checkpoint", "previous_checkpoint",
        "run_name", "shot_index")
    FUNCTION = "paths"
    CATEGORY = "conditioning/minimax/contex_loop/steerable"

    def paths(self, state, segment):
        index = int(state["index"])
        if int(segment.get("index", -1)) != index:
            raise ValueError("Steerable H3 received the wrong saved segment.")
        previous = state.get("segments") or []
        previous_checkpoint = (
            str(previous[-1]["checkpoint"]) if previous else "")
        return (
            str(segment["segment"]), str(segment["checkpoint"]),
            previous_checkpoint, str(state["plan"]["run_name"]), index)


def _steer_fit_audio(audio: Any, frame_count: int,
                     fps: int) -> dict[str, Any] | None:
    if audio is None:
        return None
    waveform, sample_rate = _validate_audio(audio, "Steerable H3 segment audio")
    required = int(round(int(frame_count) / float(fps) * sample_rate))
    if int(waveform.shape[-1]) < required:
        audio = _pad_audio_to_samples(audio, required,
                                      "Steerable H3 segment audio")
        waveform = audio["waveform"]
    return {
        "waveform": waveform[..., :required].detach().cpu().contiguous(),
        "sample_rate": sample_rate,
    }


def _steer_enhanced_paths(run_name: str, index: int,
                          revision: str) -> dict[str, str]:
    root = os.path.join(
        _run_dir({"run_name": _safe_name(run_name)}), "enhanced")
    return {
        "segment": os.path.join(
            root, "segments", "clip_%04d.%s.mp4" % (index, revision)),
        "metadata": os.path.join(
            root, "checkpoints", "clip_%04d.json" % index),
        "sequence": os.path.join(root, "sequence.json"),
    }


def _write_steer_enhanced_segment(images: Any, audio: Any, path: str,
                                   fps: int, crf: int,
                                   metadata: dict[str, Any]) -> None:
    ffmpeg = shutil.which("ffmpeg")
    if audio is None:
        _write_segment_video(images, path, fps, crf, metadata=metadata)
        return
    if ffmpeg is None:
        raise RuntimeError(
            "Steerable enhanced segment audio muxing requires ffmpeg.")
    stem = os.path.splitext(path)[0]
    silent = stem + ".video.tmp.mp4"
    wav = stem + ".audio.tmp.wav"
    final = stem + ".final.tmp.mp4"
    for temporary in (silent, wav, final):
        _safe_unlink(temporary)
    try:
        _write_segment_video(images, silent, fps, crf, metadata=metadata)
        _write_wav(audio, wav)
        _run_ffmpeg([
            ffmpeg, "-y", "-i", silent, "-i", wav,
            "-map", "0:v:0", "-map", "1:a:0", "-c:v", "copy",
            "-c:a", "aac", "-b:a", "256k",
            "-t", "%.9f" % (int(images.shape[0]) / float(fps)),
            "-map_metadata", "0", "-movflags", "use_metadata_tags+faststart",
            final,
        ])
        os.replace(final, path)
    finally:
        for temporary in (silent, wav, final):
            _safe_unlink(temporary)


class MiniMaxH3SteerAcceptRaw:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "sequence": (STEER_SEQUENCE_TYPE,),
                "state": (STATE_TYPE,),
                "segment": (SEGMENT_TYPE,),
            },
        }

    RETURN_TYPES = (STEER_SEQUENCE_TYPE, "VIDEO", "STRING", "STRING")
    RETURN_NAMES = ("sequence", "video", "video_path", "status")
    FUNCTION = "accept"
    CATEGORY = "conditioning/minimax/contex_loop/steerable"
    DESCRIPTION = (
        "Accept native H3 frames directly, bypassing LTX enhancement and "
        "RIFE interpolation while retaining the continuation checkpoint.")

    def accept(self, sequence, state, segment):
        sequence = _steer_sequence(sequence)
        if str(sequence.get("output_profile")) != "raw_h3":
            raise ValueError(
                "Raw H3 acceptance requires the raw_h3 output profile.")
        index = int(state["index"])
        if len(sequence["segments"]) != index - 1:
            raise ValueError(
                "Steerable H3 shot %d expected %d preceding accepted "
                "segments; received %d." %
                (index, index - 1, len(sequence["segments"])))
        if int(segment.get("index", -1)) != index:
            raise ValueError("Steerable H3 raw accept received the wrong segment.")
        _verify_segment_artifacts(segment, index)

        path = _absolute_output_path(str(segment["segment"]))
        delivered_count = int(segment["delivered_frames"])
        record = {
            "index": index,
            "id": str(segment["id"]),
            "revision": str(segment["revision"]),
            "h3_segment": _public_segment(segment),
            # Preserve the established steerable record contract so caching
            # and assembly can consume either enhanced or native segments.
            "enhanced_segment": _relative_output_path(path),
            "enhanced_segment_sha256": str(segment["segment_sha256"]),
            "enhanced_frames": delivered_count,
            "enhanced_fps": FPS,
            "context_prefix_frames": 0,
            "cache_key": str(state.get("steer_cache_key") or ""),
        }
        updated = dict(sequence)
        updated["shots"] = _effective_editor_plan(state["plan"])["shots"]
        updated["segments"] = (
            [dict(item) for item in sequence["segments"]] + [record])
        updated["total_enhanced_frames"] = sum(
            int(item["enhanced_frames"]) for item in updated["segments"])

        paths = _steer_enhanced_paths(
            state["plan"]["run_name"], index, str(segment["revision"]))
        _atomic_json(paths["metadata"], {
            "format": STEER_CACHE_FORMAT,
            "cache_key": str(state.get("steer_cache_key") or ""),
            "segment": record,
        })
        _atomic_json(paths["sequence"], updated)
        status = (
            "accepted raw H3 shot %d: saved %d frames (%.3fs) at %dfps; "
            "LTX and RIFE bypassed -> %s" %
            (index, delivered_count, delivered_count / float(FPS), FPS, path))
        if InputImpl is None:
            raise RuntimeError(
                "Steerable H3 VIDEO output requires comfy_api.latest.")
        return updated, InputImpl.VideoFromFile(path), path, status


class MiniMaxH3SteerAcceptEnhanced:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "sequence": (STEER_SEQUENCE_TYPE,),
                "state": (STATE_TYPE,),
                "segment": (SEGMENT_TYPE,),
                "images": ("IMAGE",),
                "rolling_context": (LTX_ROLLING_CONTEXT_TYPE,),
            },
            "optional": {
                "audio": ("AUDIO",),
            },
        }

    RETURN_TYPES = (STEER_SEQUENCE_TYPE, "VIDEO", "STRING", "STRING")
    RETURN_NAMES = ("sequence", "video", "video_path", "status")
    FUNCTION = "accept"
    CATEGORY = "conditioning/minimax/contex_loop/steerable"
    DESCRIPTION = (
        "Remove the duplicated RIFE overlap exactly, save the accepted "
        "enhanced segment, and emit the disk-backed token for the next shot.")

    def accept(self, sequence, state, segment, images, rolling_context,
               audio=None):
        sequence = _steer_sequence(sequence)
        index = int(state["index"])
        if len(sequence["segments"]) != index - 1:
            raise ValueError(
                "Steerable H3 shot %d expected %d preceding accepted "
                "segments; received %d." %
                (index, index - 1, len(sequence["segments"])))
        if int(segment.get("index", -1)) != index:
            raise ValueError("Steerable H3 accept received the wrong H3 segment.")
        if rolling_context.get("format") != "h3_ltx_rolling_context_v1":
            raise ValueError("Steerable H3 accept received invalid LTX context.")

        carried = int(rolling_context["context_frames"])
        prefix = 2 * (carried - 1) + 1 if carried else 0
        actual = int(images.shape[0])
        expected = 2 * (int(rolling_context["original_frames"]) - 1) + 1
        if actual != expected:
            raise ValueError(
                "Steerable H3 RIFE output contains %d frames; expected %d." %
                (actual, expected))
        if prefix >= actual:
            raise ValueError("Steerable H3 enhanced context consumes the clip.")
        delivered = images[prefix:].detach().cpu().contiguous()
        delivered_count = int(delivered.shape[0])
        fitted_audio = _steer_fit_audio(
            audio, delivered_count, STEER_ENHANCED_FPS)

        revision = uuid.uuid4().hex
        paths = _steer_enhanced_paths(
            state["plan"]["run_name"], index, revision)
        os.makedirs(os.path.dirname(paths["segment"]), exist_ok=True)
        metadata = {
            "title": "Steerable H3 shot %d - %s" %
                     (index, segment.get("id", "shot")),
            "comment": segment.get("prompt", ""),
            "h3_seed": str(segment.get("seed", "")),
            "h3_source_segment": str(segment["segment"]),
            "h3_context_prefix_frames": str(prefix),
        }
        _write_steer_enhanced_segment(
            delivered, fitted_audio, paths["segment"], STEER_ENHANCED_FPS,
            int(sequence["enhanced_crf"]), metadata)

        record = {
            "index": index,
            "id": str(segment["id"]),
            "revision": revision,
            "h3_segment": _public_segment(segment),
            "enhanced_segment": _relative_output_path(paths["segment"]),
            "enhanced_segment_sha256": _file_sha256(paths["segment"]),
            "enhanced_frames": delivered_count,
            "enhanced_fps": STEER_ENHANCED_FPS,
            "context_prefix_frames": prefix,
            "cache_key": str(state.get("steer_cache_key") or ""),
        }
        updated = dict(sequence)
        updated["shots"] = _effective_editor_plan(state["plan"])["shots"]
        updated["segments"] = [dict(item) for item in sequence["segments"]] + [record]
        updated["total_enhanced_frames"] = sum(
            int(item["enhanced_frames"]) for item in updated["segments"])
        _atomic_json(paths["metadata"], {
            "format": STEER_CACHE_FORMAT,
            "cache_key": str(state.get("steer_cache_key") or ""),
            "segment": record,
        })
        _atomic_json(paths["sequence"], updated)
        status = (
            "accepted shot %d: removed %d repeated 48fps context frames; "
            "saved %d frames (%.3fs) -> %s" %
            (index, prefix, delivered_count,
             delivered_count / float(STEER_ENHANCED_FPS), paths["segment"]))
        if InputImpl is None:
            raise RuntimeError(
                "Steerable H3 VIDEO output requires comfy_api.latest.")
        return (updated, InputImpl.VideoFromFile(paths["segment"]),
                paths["segment"], status)


class MiniMaxH3SteerableSegment:
    NEGATIVE_PROMPT = (
        "new objects, altered composition, camera cut, reframing, identity "
        "drift, geometry drift, color shift, flicker, temporal inconsistency, "
        "blur, oversharpening, halos, ringing, compression artifacts, text, "
        "watermark")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "sequence": (STEER_SEQUENCE_TYPE,),
                "prompt": ("STRING", {
                    "default": "Continue the current action into the next beat.",
                    "multiline": True,
                    "dynamicPrompts": True,
                }),
                "shot_name": ("STRING", {"default": ""}),
                "seed": ("INT", {
                    "default": 424242, "min": 0, "max": MAX_SEED,
                }),
                "raw_frames": ("INT", {
                    "default": 124, "min": 124, "max": 362, "step": 17,
                    "tooltip": "124 is about 5 seconds. Continuations spend "
                               "18 frames on H3 sliding history.",
                }),
                "h3_steps": ("INT", {
                    "default": 4, "min": 1, "max": 100, "step": 1,
                }),
                "ref_image_size": (["match", "max"], {"default": "match"}),
            },
            "optional": {
                "first_frame": ("IMAGE", {
                    "tooltip": "Opening frame for the first segment only. "
                               "Later segments use the accepted rolling context."}),
                "last_frame": ("IMAGE",),
                "reference_image_1": ("IMAGE",),
                "reference_image_2": ("IMAGE",),
                "reference_image_3": ("IMAGE",),
                "reference_video": ("IMAGE",),
                "reference_video_audio": ("AUDIO",),
                "reference_audio": ("AUDIO",),
            },
        }

    RETURN_TYPES = (STEER_SEQUENCE_TYPE, "VIDEO", "STRING", "STRING")
    RETURN_NAMES = ("sequence", "video", "video_path", "status")
    FUNCTION = "generate"
    OUTPUT_NODE = True
    CATEGORY = "conditioning/minimax/contex_loop/steerable"
    DESCRIPTION = (
        "Generate one independently rerunnable H3 shot, continue from the "
        "preceding accepted AV checkpoint, immediately enhance it with LTX "
        "2.5 at 2x, interpolate to 48fps, and emit the next sequence token.")

    def generate(self, sequence, prompt, shot_name, seed, raw_frames,
                 h3_steps, ref_image_size="match", first_frame=None,
                 last_frame=None, reference_image_1=None,
                 reference_image_2=None, reference_image_3=None,
                 reference_video=None, reference_video_audio=None,
                 reference_audio=None, relay_model=None):
        if GraphBuilder is None:
            raise RuntimeError("Steerable H3 Segment requires ComfyUI GraphBuilder.")
        state, shot = _steer_state(
            sequence, shot_name, prompt, seed, raw_frames, h3_steps)
        index = int(state["index"])
        cache_key = _steer_cache_key(
            sequence, shot, ref_image_size,
            first_frame=first_frame, last_frame=last_frame,
            reference_image_1=reference_image_1,
            reference_image_2=reference_image_2,
            reference_image_3=reference_image_3,
            reference_video=reference_video,
            reference_video_audio=reference_video_audio,
            reference_audio=reference_audio)
        cached = _steer_cached_result(sequence, state, cache_key)
        if cached is not None:
            return cached
        state["steer_cache_key"] = cache_key
        graph = GraphBuilder()

        if relay_model is None:
            h3_model = graph.node("H3RelayInternalHybridLoader", "H3Hybrid")
            for name, value in (
                    ("base_model", "minimax_h3_fl2va_int8_convrot.safetensors"),
                    ("overlay_model", "minimax_h3_ref2va_int8_convrot.safetensors"),
                    ("overlay_preset", "block_range_adaln"),
                    ("block_range_start", 25), ("block_range_end", 49),
                    ("final_adaln_from_overlay", False),
                    ("custom_overlays", ""), ("custom_base", ""),
                    ("weight_dtype", "default")):
                h3_model.set_input(name, value)
            attention = graph.node("ModelAttentionBackend", "H3Attention")
            attention.set_input("model", h3_model.out(0))
            attention.set_input("attention", "comfy kitchen attention")
            model_before_shift = attention.out(0)
        else:
            model_before_shift = relay_model
        sampling_profile = str(
            sequence.get("h3_sampling_profile") or "turbo_auto")
        explicit_sampler = sequence.get("h3_sampler")
        if explicit_sampler is not None:
            sampler_name = str(explicit_sampler)
            scheduler_name = str(sequence.get("h3_scheduler") or "simple")
            use_spectrum = bool(sequence.get("h3_spectrum_enabled", False))
            use_turbo = False
        else:
            sampler_name = (
                "res_multistep"
                if sampling_profile == "res_multistep" else "euler"
            )
            if sampling_profile == "native_spectrum_euler_beta57":
                scheduler_name = "beta57"
            elif sampling_profile in {"res_multistep", "euler"}:
                scheduler_name = "simple"
            else:
                scheduler_name = "beta"
            use_spectrum = sampling_profile == "native_spectrum_euler_beta57"
            use_turbo = sampling_profile == "turbo_auto"
        use_eight_step_turbo = use_turbo and int(shot["steps"]) >= 8
        if use_turbo:
            h3_lora = graph.node("LoraLoaderModelOnly", "H3Turbo")
            h3_lora.set_input("model", model_before_shift)
            h3_lora.set_input(
                "lora_name",
                ("h3/minimax_h3_fl2v_turbo_8step_v1.0_comfyui_"
                 "resized_avg_rank_21_bf16.safetensors")
                if use_eight_step_turbo else
                ("h3/minimax_h3_fl2v_turbo_4step_v1.0_768p_comfyui_"
                 "resized_avg_rank_21_bf16.safetensors"))
            h3_lora.set_input(
                "strength_model", 0.75 if use_eight_step_turbo else 1.0)
            model_before_shift = h3_lora.out(0)
        h3_shift = graph.node("MiniMaxH3SigmaShift", "H3Shift")
        h3_shift.set_input("model", model_before_shift)
        h3_shift.set_input(
            "shift_video", 12.0 if sampling_profile != "turbo_auto" or use_eight_step_turbo else 6.0)
        h3_shift.set_input("shift_audio", 3.0)
        sampling_model = h3_shift.out(0)
        if use_spectrum:
            spectrum = graph.node("H3RelayInternalSpectrum", "H3Spectrum")
            spectrum.set_input("model", sampling_model)
            for name, value in (
                    ("enabled", True), ("blend_weight", 0.50),
                    ("degree", 1), ("ridge_lambda", 0.10),
                    ("window_size", 2.0), ("flex_window", 0.75),
                    ("warmup_steps", 1), ("tail_actual_steps", 1),
                    ("max_history", 8), ("debug", False),
                    ("history_storage", "system_ram"),
                    ("bootstrap_first_forecast", True),
                    ("anchor_residual_feedback", False),
                    ("selective_rollback_correction", False),
                    ("offline_smoothing_replay", True),
                    ("audio_blend_weight", 0.0),
                    ("offline_archive_storage", "system_ram"),
                    ("model_aware_mode", "off"),
                    ("model_aware_risk_threshold", 0.65),
                    ("model_aware_trust_shrinkage", False),
                    ("model_aware_replay_generic_correction", False),
                    ("generic_correction_mode", "coordinate_rls"),
                    ("generic_correction_limiter", "hard_clip"),
                    ("generic_correction_limit", 0.40),
                    ("generic_correction_attenuation", "no_attenuation")):
                spectrum.set_input(name, value)
            sampling_model = spectrum.out(0)

        h3_clip = graph.node("CLIPLoader", "H3Text")
        h3_clip.set_input(
            "clip_name", "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors")
        h3_clip.set_input("type", "minimax")
        h3_clip.set_input("device", "default")
        h3_vae = graph.node("VAELoader", "H3VideoVAE")
        h3_vae.set_input("vae_name", "minimax_h3_video_vae_fp16.safetensors")
        h3_audio_vae = graph.node("VAELoader", "H3AudioVAE")
        h3_audio_vae.set_input(
            "vae_name", "minimax_h3_audio_vae_fp32.safetensors")

        ref2va = graph.node("MiniMaxH3ReferenceToVideo", "H3Conditioning")
        for name, value in (
                ("clip", h3_clip.out(0)), ("vae", h3_vae.out(0)),
                ("audio_vae", h3_audio_vae.out(0)),
                ("prompt", shot["prompt"]),
                ("width", int(sequence["width"])),
                ("height", int(sequence["height"])),
                ("length", int(shot["raw_frames"])),
                ("ref_image_size", ref_image_size)):
            ref2va.set_input(name, value)
        for ref_index, image in enumerate((
                reference_image_1, reference_image_2,
                reference_image_3)):
            if image is not None:
                ref2va.set_input(
                    "ref_images.ref_image_%d" % ref_index, image)
        if reference_video is not None:
            ref2va.set_input("ref_videos.ref_video_0", reference_video)
            if reference_video_audio is not None:
                ref2va.set_input(
                    "ref_video_audios.ref_video_audio_0",
                    reference_video_audio)
        if reference_audio is not None:
            ref2va.set_input("ref_audios.ref_audio_0", reference_audio)

        conditioning = ref2va.out(0)
        if index == 1 and first_frame is not None:
            first_guide = graph.node("MiniMaxH3AddGuide", "OpeningFrame")
            for name, value in (
                    ("positive", conditioning), ("vae", h3_vae.out(0)),
                    ("latent", ref2va.out(1)), ("image", first_frame),
                    ("frame_idx", 0)):
                first_guide.set_input(name, value)
            conditioning = first_guide.out(0)
        if last_frame is not None:
            last_guide = graph.node("MiniMaxH3AddGuide", "EndingFrame")
            for name, value in (
                    ("positive", conditioning), ("vae", h3_vae.out(0)),
                    ("latent", ref2va.out(1)), ("image", last_frame),
                    ("frame_idx", -1)):
                last_guide.set_input(name, value)
            conditioning = last_guide.out(0)

        context = graph.node("H3RelayInternalChainContext", "H3RollingContext")
        for name, value in (
                ("state", state), ("conditioning", conditioning),
                ("vae", h3_vae.out(0)), ("latent", ref2va.out(1)),
                ("audio_vae", h3_audio_vae.out(0))):
            context.set_input(name, value)
        noise = graph.node("RandomNoise", "H3Noise")
        noise.set_input("noise_seed", int(shot["seed"]))
        guider = graph.node("BasicGuider", "H3Guider")
        guider.set_input("model", sampling_model)
        guider.set_input("conditioning", context.out(0))
        sampler_select = graph.node("KSamplerSelect", "H3Sampler")
        sampler_select.set_input("sampler_name", sampler_name)
        if scheduler_name == "beta57":
            if int(shot["steps"]) == 16:
                scheduler = graph.node("ManualSigmas", "H3Beta57ManualSigmas")
                scheduler.set_input(
                    "sigmas",
                    "1.0000, 0.9964, 0.9898, 0.9806, 0.9686, 0.9530, "
                    "0.9327, 0.9065, 0.8719, 0.8257, 0.7635, 0.6775, "
                    "0.5631, 0.4158, 0.2353, 0.0780, 0.0000")
            else:
                scheduler = graph.node(
                    "BetaSamplingScheduler", "H3Beta57Scheduler")
                scheduler.set_input("model", sampling_model)
                scheduler.set_input("steps", int(shot["steps"]))
                scheduler.set_input("alpha", 0.5)
                scheduler.set_input("beta", 0.7)
        else:
            scheduler = graph.node("BasicScheduler", "H3Scheduler")
            scheduler.set_input("model", sampling_model)
            scheduler.set_input("scheduler", scheduler_name)
            scheduler.set_input("steps", int(shot["steps"]))
            scheduler.set_input("denoise", 1.0)
        sampler = graph.node("SamplerCustomAdvanced", "H3Sample")
        for name, value in (
                ("noise", noise.out(0)), ("guider", guider.out(0)),
                ("sampler", sampler_select.out(0)),
                ("sigmas", scheduler.out(0)),
                ("latent_image", context.out(3))):
            sampler.set_input(name, value)
        decode = graph.node("VAEDecode", "H3DecodeVideo")
        decode.set_input("samples", sampler.out(0))
        decode.set_input("vae", h3_vae.out(0))
        decode_audio = graph.node("VAEDecodeAudio", "H3DecodeAudio")
        decode_audio.set_input("samples", sampler.out(0))
        decode_audio.set_input("vae", h3_audio_vae.out(0))
        trim = graph.node("H3RelayInternalLoopTrim", "H3ExactTrim")
        trim.set_input("images", decode.out(0))
        trim.set_input("audio", decode_audio.out(0))
        trim.set_input("trim_frames", context.out(1))
        trim.set_input("retain_overlap_frames", 0)
        segment = graph.node("H3RelayInternalSegmentSave", "H3Checkpoint")
        segment.set_input("state", state)
        segment.set_input("images", trim.out(0))
        segment.set_input("sampled_latent", sampler.out(0))
        segment.set_input("audio", trim.out(1))
        if str(sequence.get("output_profile")) == "raw_h3":
            accept_raw = graph.node(
                "H3RelayInternalAcceptRaw", "AcceptRawSegment")
            accept_raw.set_input("sequence", sequence)
            accept_raw.set_input("state", state)
            accept_raw.set_input("segment", segment.out(0))
            preview = graph.node(
                "H3RelayInternalVideoOutput", "SegmentPreview")
            preview.set_input("video_path", accept_raw.out(2))
            return {
                "result": (
                    accept_raw.out(0), accept_raw.out(1),
                    accept_raw.out(2), accept_raw.out(3)),
                "expand": graph.finalize(),
            }
        paths = graph.node("H3RelayInternalSegmentPaths", "H3Paths")
        paths.set_input("state", state)
        paths.set_input("segment", segment.out(0))

        rolling = graph.node("H3RelayInternalLTXRollingInput", "LTXRollingInput")
        rolling.set_input("current_segment", paths.out(0))
        rolling.set_input("previous_checkpoint", paths.out(2))
        rolling.set_input("context_frames", STEER_LTX_CONTEXT_FRAMES)
        ltx_vae = graph.node("VAELoader", "LTXVAE")
        ltx_vae.set_input("vae_name", "ltx-2.5-video-vae-bf16.safetensors")
        ltx_encode = graph.node("VAEEncode", "LTXEncode")
        ltx_encode.set_input("pixels", rolling.out(0))
        ltx_encode.set_input("vae", ltx_vae.out(0))
        upscale_loader = graph.node("LatentUpscaleModelLoader", "LTXUpscaler")
        upscale_loader.set_input(
            "model_name",
            "ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors")
        upscale = graph.node("LTXVLatentUpsampler", "LTXUpscale")
        upscale.set_input("samples", ltx_encode.out(0))
        upscale.set_input("upscale_model", upscale_loader.out(0))
        upscale.set_input("vae", ltx_vae.out(0))
        inject = graph.node("H3RelayInternalLTXRollingInject", "LTXContext")
        inject.set_input("latent", upscale.out(0))
        inject.set_input("run_name", paths.out(3))
        inject.set_input("shot_index", paths.out(4))
        inject.set_input("context_latent_steps", STEER_LTX_CONTEXT_STEPS)

        ltx_model = graph.node("UNETLoader", "LTXModel")
        ltx_model.set_input(
            "unet_name", "ltx-2.5-22b-dev-transformer-comfy-int8-convrot.safetensors")
        ltx_model.set_input("weight_dtype", "default")
        ltx_distilled = graph.node("LoraLoaderModelOnly", "LTXDistilled")
        ltx_distilled.set_input("model", ltx_model.out(0))
        ltx_distilled.set_input(
            "lora_name", "ltx-2.5-22b-distilled-lora-450-bf16.safetensors")
        ltx_distilled.set_input("strength_model", 1.0)
        ltx_ic = graph.node("LoraLoaderModelOnly", "LTXPixelUpscale")
        ltx_ic.set_input("model", ltx_distilled.out(0))
        ltx_ic.set_input(
            "lora_name",
            "ltx-2.5-22b-ic-lora-pixel-spatial-upscaler-x2-1.0.safetensors")
        ltx_ic.set_input("strength_model", 1.0)
        windows = graph.node("LTXVContextWindows", "LTXWindows")
        for name, value in (
                # H3's 124-frame shot pads to 129 real frames / 17 LTX
                # latent frames. A 121-frame window covers only 16 latents and
                # forces two overlapping diffusion windows for every shot.
                # 129 fits the entire five-second segment in one pass.
                ("model", ltx_ic.out(0)), ("context_length", 129),
                ("context_overlap", 32),
                ("context_schedule", "standard_uniform"),
                ("context_stride", 1), ("closed_loop", False),
                ("fuse_method", "pyramid"), ("freenoise", True),
                ("retain_first_frame", False),
                ("split_conds_to_windows", False)):
            windows.set_input(name, value)
        ltx_clip = graph.node("CLIPLoader", "LTXText")
        ltx_clip.set_input(
            "clip_name", "gemma4-12b-with-proj-ltx-2.5-comfy-int8-convrot.safetensors")
        ltx_clip.set_input("type", "ltxv")
        ltx_clip.set_input("device", "default")
        ltx_positive = graph.node("CLIPTextEncode", "LTXPositive")
        ltx_positive.set_input("clip", ltx_clip.out(0))
        ltx_positive.set_input("text", sequence["enhancement_prompt"])
        ltx_negative = graph.node("CLIPTextEncode", "LTXNegative")
        ltx_negative.set_input("clip", ltx_clip.out(0))
        ltx_negative.set_input("text", self.NEGATIVE_PROMPT)
        ltx_conditioning = graph.node("LTXVConditioning", "LTXConditioning")
        ltx_conditioning.set_input("positive", ltx_positive.out(0))
        ltx_conditioning.set_input("negative", ltx_negative.out(0))
        ltx_conditioning.set_input("frame_rate", 24.0)
        ic_params = graph.node("GetICLoRAParameters", "LTXICParameters")
        ic_params.set_input("iclora_model", ltx_ic.out(0))
        ltx_guide = graph.node("LTXVAddGuide", "LTXGuide")
        for name, value in (
                ("positive", ltx_conditioning.out(0)),
                ("negative", ltx_conditioning.out(1)),
                ("vae", ltx_vae.out(0)), ("latent", inject.out(0)),
                ("image", rolling.out(0)), ("frame_idx", 0),
                ("strength", 1.0),
                ("iclora_parameters", ic_params.out(0))):
            ltx_guide.set_input(name, value)
        ltx_noise = graph.node("RandomNoise", "LTXNoise")
        ltx_noise.set_input("noise_seed", int(shot["seed"]))
        ltx_guider = graph.node("LTXVDualCFGGuider", "LTXGuider")
        for name, value in (
                ("model", windows.out(0)), ("positive", ltx_guide.out(0)),
                ("negative", ltx_guide.out(1)), ("video_cfg", 1.0),
                ("audio_cfg", 1.0)):
            ltx_guider.set_input(name, value)
        ltx_sampler_select = graph.node("KSamplerSelect", "LTXSampler")
        ltx_sampler_select.set_input("sampler_name", "euler_ancestral")
        ltx_sigmas = graph.node("ManualSigmas", "LTXSigmas")
        ltx_sigmas.set_input("sigmas", "0.85, 0.7250, 0.4219, 0.0")
        ltx_sample = graph.node("SamplerCustomAdvanced", "LTXSample")
        for name, value in (
                ("noise", ltx_noise.out(0)),
                ("guider", ltx_guider.out(0)),
                ("sampler", ltx_sampler_select.out(0)),
                ("sigmas", ltx_sigmas.out(0)),
                ("latent_image", ltx_guide.out(2))):
            ltx_sample.set_input(name, value)
        ltx_crop_guides = graph.node("LTXVCropGuides", "LTXCropGuides")
        ltx_crop_guides.set_input("positive", ltx_guide.out(0))
        ltx_crop_guides.set_input("negative", ltx_guide.out(1))
        ltx_crop_guides.set_input("latent", ltx_sample.out(0))
        ltx_checkpoint = graph.node(
            "H3RelayInternalLTXRollingCheckpoint", "LTXCheckpoint")
        ltx_checkpoint.set_input("latent", ltx_crop_guides.out(2))
        ltx_checkpoint.set_input("run_name", paths.out(3))
        ltx_checkpoint.set_input("shot_index", paths.out(4))
        ltx_checkpoint.set_input(
            "context_latent_steps", STEER_LTX_CONTEXT_STEPS)
        ltx_decode = graph.node("VAEDecodeTiled", "LTXDecode")
        ltx_decode.set_input("samples", ltx_checkpoint.out(0))
        ltx_decode.set_input("vae", ltx_vae.out(0))
        ltx_decode.set_input("tile_size", 768)
        ltx_decode.set_input("overlap", 64)
        # Ten-second steerable shots contain roughly twice the latent history
        # of the original five-second prototype. Decoding the whole temporal
        # axis in one tile can exceed 24 GB after the LTX model pass. Keep the
        # same spatial tiles but decode 64-frame temporal tiles with the VAE's
        # standard overlap so long shots remain within a 4090's VRAM budget.
        ltx_decode.set_input("temporal_size", 64)
        ltx_decode.set_input("temporal_overlap", 8)
        ltx_crop = graph.node("H3RelayInternalLTXRollingCrop", "LTXCropPadding")
        ltx_crop.set_input("images", ltx_decode.out(0))
        ltx_crop.set_input("rolling_context", rolling.out(1))
        rife_model = graph.node("FrameInterpolationModelLoader", "RIFEModel")
        rife_model.set_input("model_name", "rife_v4.26_heavy.safetensors")
        rife = graph.node("FrameInterpolate", "RIFE48")
        rife.set_input("interp_model", rife_model.out(0))
        rife.set_input("images", ltx_crop.out(0))
        rife.set_input("multiplier", 2)
        accept = graph.node("H3RelayInternalAcceptEnhanced", "AcceptSegment")
        accept.set_input("sequence", sequence)
        accept.set_input("state", state)
        accept.set_input("segment", segment.out(0))
        accept.set_input("images", rife.out(0))
        accept.set_input("rolling_context", rolling.out(1))
        accept.set_input("audio", trim.out(1))
        preview = graph.node("H3RelayInternalVideoOutput", "SegmentPreview")
        preview.set_input("video_path", accept.out(2))
        return {
            "result": (
                accept.out(0), accept.out(1), accept.out(2), accept.out(3)),
            "expand": graph.finalize(),
        }


class MiniMaxH3SteerAssemble:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "sequence": (STEER_SEQUENCE_TYPE,),
                "filename": ("STRING", {
                    "default": "steerable_movie_%date:yyyy-MM-dd_HH-mm-ss%",
                }),
                "audio_bitrate": ("INT", {
                    "default": 256, "min": 64, "max": 512,
                }),
            },
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("video_path", "status")
    FUNCTION = "assemble"
    OUTPUT_NODE = True
    CATEGORY = "conditioning/minimax/contex_loop/steerable"
    DESCRIPTION = (
        "Assemble every accepted enhanced segment in the connected sequence "
        "token. Connect any intermediate token to export the movie thus far.")

    @classmethod
    def IS_CHANGED(cls, *args, **kwargs):
        return float("NaN")

    def assemble(self, sequence, filename, audio_bitrate):
        sequence = _steer_sequence(sequence)
        records = sequence["segments"]
        if not records:
            raise ValueError("Steerable H3 Assemble needs at least one segment.")
        segment_paths = []
        h3_segments = []
        total_frames = 0
        for expected, record in enumerate(records, 1):
            if int(record.get("index", -1)) != expected:
                raise ValueError("Steerable H3 sequence order is invalid.")
            path = _absolute_output_path(record["enhanced_segment"])
            if not os.path.isfile(path):
                raise FileNotFoundError(
                    "Steerable enhanced segment is missing: %s" % path)
            if _file_sha256(path) != record.get("enhanced_segment_sha256"):
                raise ValueError(
                    "Steerable enhanced segment %d failed its integrity check."
                    % expected)
            segment_paths.append(path)
            total_frames += int(record["enhanced_frames"])
            h3_segments.append(record["h3_segment"])

        ffmpeg = shutil.which("ffmpeg")
        if ffmpeg is None:
            raise RuntimeError("Steerable H3 Assemble requires ffmpeg.")
        output_fps = int(sequence.get("enhanced_fps") or STEER_ENHANCED_FPS)
        if output_fps <= 0:
            raise ValueError("Steerable H3 output frame rate must be positive.")
        output_kind = (
            "raw" if str(sequence.get("output_profile")) == "raw_h3"
            else "enhanced")
        final_dir = os.path.join(
            _output_root(), "h3_chains", sequence["run_name"],
            output_kind, "final")
        os.makedirs(final_dir, exist_ok=True)
        final_name = _safe_name(
            _expand_filename_date(filename), "steerable_movie")
        final_path = _available_versioned_path(
            os.path.join(final_dir, final_name + ".mp4"))
        token = uuid.uuid4().hex
        concat = os.path.join(final_dir, ".%s.concat.txt" % token)
        silent = os.path.join(final_dir, ".%s.video.mp4" % token)
        wav = os.path.join(final_dir, ".%s.audio.wav" % token)
        temporary = os.path.join(final_dir, ".%s.final.mp4" % token)
        try:
            with open(concat, "w", encoding="utf-8") as handle:
                for path in segment_paths:
                    escaped = path.replace("\\", "\\\\").replace(
                        "'", "'\\''")
                    handle.write("file '%s'\n" % escaped)
            _run_ffmpeg([
                ffmpeg, "-y", "-f", "concat", "-safe", "0", "-i", concat,
                "-map", "0:v:0", "-c", "copy", "-an",
                "-movflags", "use_metadata_tags+faststart", silent,
            ])
            generated = _generated_audio({"segments": h3_segments})
            waveform, sample_rate = _validate_audio(
                generated, "Steerable H3 assembled audio")
            required = int(round(
                total_frames / float(output_fps) * sample_rate))
            if int(waveform.shape[-1]) < required:
                generated = _pad_audio_to_samples(
                    generated, required, "Steerable H3 assembled audio")
                waveform = generated["waveform"]
            audio = {
                "waveform": waveform[..., :required],
                "sample_rate": sample_rate,
            }
            _write_wav(audio, wav)
            _run_ffmpeg([
                ffmpeg, "-y", "-i", silent, "-i", wav,
                "-map", "0:v:0", "-map", "1:a:0", "-c:v", "copy",
                "-c:a", "aac", "-b:a", "%dk" % int(audio_bitrate),
                "-t", "%.9f" % (total_frames / float(output_fps)),
                "-map_metadata", "0", "-movflags", "use_metadata_tags+faststart",
                temporary,
            ])
            os.replace(temporary, final_path)
        finally:
            for path in (concat, silent, wav, temporary):
                _safe_unlink(path)
        status = "assembled %d steerable segments, %d frames (%.3fs) -> %s" % (
            len(records), total_frames,
            total_frames / float(output_fps), final_path)
        return {
            "ui": {
                "images": [_video_output_item(final_path)],
                "animated": (True,),
                "text": [status],
            },
            "result": (final_path, status),
        }


def _assemble_review_partial(
    state: dict[str, Any],
    segment: dict[str, Any],
    audio_source: str,
    source_audio: dict[str, Any] | None,
) -> tuple[str, str]:
    manifest = _partial_manifest(state, segment)
    index = int(segment["index"])
    partial_dir = os.path.join(_run_dir(state["plan"]), "partial")
    manifest_path = os.path.join(
        partial_dir, "through_clip_%04d.manifest.json" % index)
    _atomic_json(manifest_path, manifest)

    selected = {
        "checkpointed": "generated",
        "source": "source",
        "none": "none",
    }.get(str(audio_source))
    if selected is None:
        raise ValueError("Unknown H3 partial audio source %r." % audio_source)

    assembler = MiniMaxH3ChainAssemble()
    filename = "partial_through_clip_%04d" % index
    warning = ""
    try:
        result = assembler.assemble(
            manifest, selected, filename, 192, source_audio,
            overwrite_existing=True)
    except Exception as audio_error:
        if selected == "none":
            raise
        _LOG.warning(
            "H3 Chain partial audio assembly failed; saving silent video: %s",
            audio_error)
        result = assembler.assemble(
            manifest, "none", filename, 192, source_audio,
            overwrite_existing=True)
        warning = "audio unavailable, so the partial video is silent (%s)" % audio_error
    return str(result["result"][0]), warning


async def _submit_review_decision(request):
    try:
        body = await request.json()
    except Exception:
        return web.json_response({"error": "Expected a JSON request body."},
                                 status=400)
    token = str(body.get("token") or "")
    pending = _PENDING_REVIEWS.get(token)
    if pending is None:
        return web.json_response(
            {"error": "This H3 review is no longer pending."}, status=404)
    future = pending["future"]
    if future.done():
        return web.json_response(
            {"error": "This H3 review already has a decision."}, status=409)

    action = str(body.get("action") or "")
    if action not in ("approve", "retry", "reroll", "stop"):
        return web.json_response({"error": "Unknown review action."}, status=400)

    decision: dict[str, Any] = {"action": action}
    if action in ("retry", "reroll"):
        scene_prompt = str(body.get("scene_prompt") or "").strip()
        prompt_prefix = str(
            pending.get("public", {}).get("prompt_prefix") or "").strip()
        if not scene_prompt and not prompt_prefix:
            return web.json_response(
                {"error": "Retry requires a scene prompt or shared prompt."},
                status=400)
        if len(scene_prompt) > 200000:
            return web.json_response(
                {"error": "The retry prompt is too large."}, status=400)
        try:
            raw_frames = _validate_h3_length(
                body.get("length", pending.get("current_length")),
                "H3 review retry length")
        except (TypeError, ValueError) as exc:
            return web.json_response({"error": str(exc)}, status=400)
        if action == "reroll":
            seed = secrets.randbits(64)
            while seed == int(pending["current_seed"]):
                seed = secrets.randbits(64)
        else:
            try:
                seed = int(str(body.get("seed")))
            except (TypeError, ValueError):
                return web.json_response(
                    {"error": "The retry seed must be an integer."}, status=400)
            if seed < 0 or seed > MAX_SEED:
                return web.json_response(
                    {"error": "The retry seed is outside the uint64 range."},
                    status=400)
        decision = {
            "action": "retry",
            "scene_prompt": scene_prompt,
            "seed": seed,
            "raw_frames": raw_frames,
        }
        try:
            _plan_with_review_revision(
                pending["plan"],
                int(pending["public"]["clip_index"]),
                scene_prompt, seed, raw_frames)
        except (TypeError, ValueError) as exc:
            return web.json_response({"error": str(exc)}, status=400)

    def resolve_on_execution_loop():
        if not future.done():
            future.set_result(decision)

    try:
        pending["loop"].call_soon_threadsafe(resolve_on_execution_loop)
    except RuntimeError:
        return web.json_response(
            {"error": "This H3 review execution loop is no longer running."},
            status=409)
    return web.json_response({
        "ok": True,
        "action": decision["action"],
        "seed": str(decision.get("seed", pending["current_seed"])),
        "length": int(decision.get(
            "raw_frames", pending["current_length"])),
    })


async def _list_pending_reviews(_request):
    reviews = []
    # HTTP and execution can run on different threads/loops. Snapshot first so
    # a review resolving during recovery cannot invalidate this iteration and
    # turn a browser's reconnect GET into an intermittent 500 response.
    for item in list(_PENDING_REVIEWS.values()):
        if item["future"].done():
            continue
        payload = dict(item["public"])
        payload["server_now"] = time.time()
        reviews.append(payload)
    return web.json_response({"reviews": reviews})


def _checkpoint_review_preview(
        index: int, segment: dict[str, Any], review_dir: str,
        review_filenames: list[str]) -> str | None:
    video_hash = str(segment.get("segment_sha256") or "")[:12]
    if not video_hash or not review_filenames:
        return None
    review_prefix = "clip_%04d.%s." % (int(index), video_hash)
    previews = []
    for candidate in review_filenames:
        if (not candidate.startswith(review_prefix) or
                not candidate.endswith(".review.mp4")):
            continue
        preview_path = os.path.join(review_dir, candidate)
        try:
            previews.append((os.path.getmtime(preview_path), preview_path))
        except OSError:
            # Review replacement is atomic, but an old revision may disappear
            # while the browser is refreshing.
            continue
    return max(previews, key=lambda item: item[0])[1] if previews else None


def _checkpoint_revision_owned_paths(
        metadata_path: str, segment: dict[str, Any],
        review_dir: str) -> list[str]:
    """Return only immutable files owned by one checkpoint revision."""
    paths = {os.path.abspath(metadata_path)}
    for key in ("segment", "checkpoint", "prompt_file", "generated_audio",
                "blend_segment", "revision_metadata"):
        value = segment.get(key)
        if isinstance(value, str) and value:
            paths.add(_absolute_output_path(value))
    index = int(segment.get("index", 0))
    video_hash = str(segment.get("segment_sha256") or "")[:12]
    if index > 0 and video_hash and os.path.isdir(review_dir):
        prefix = "clip_%04d.%s." % (index, video_hash)
        for candidate in os.listdir(review_dir):
            if candidate.startswith(prefix) and candidate.endswith(".review.mp4"):
                paths.add(os.path.abspath(os.path.join(review_dir, candidate)))
    return sorted(paths)


def _checkpoint_revision_size(
        metadata_path: str, segment: dict[str, Any], review_dir: str) -> int:
    total = 0
    for path in _checkpoint_revision_owned_paths(
            metadata_path, segment, review_dir):
        try:
            total += os.path.getsize(path)
        except OSError:
            continue
    return total


def _checkpoint_review_is_shared(
        checkpoint_dir: str, metadata_path: str,
        segment: dict[str, Any]) -> bool:
    selected_hash = str(segment.get("segment_sha256") or "")
    if not selected_hash or not os.path.isdir(checkpoint_dir):
        return False
    selected_metadata = os.path.abspath(metadata_path)
    for candidate in os.listdir(checkpoint_dir):
        if re.fullmatch(r"clip_\d{4}\.[0-9a-f]{32}\.json", candidate) is None:
            continue
        path = os.path.abspath(os.path.join(checkpoint_dir, candidate))
        if path == selected_metadata:
            continue
        try:
            other = _read_json(path).get("segment")
            if (isinstance(other, dict) and
                    str(other.get("segment_sha256") or "") == selected_hash):
                return True
        except (OSError, TypeError, ValueError, json.JSONDecodeError,
                AttributeError):
            continue
    return False


def _load_checkpoint_revision(
        run_name: str, scene: Any, revision: Any
) -> tuple[dict[str, Any], str]:
    index = int(scene)
    token = str(revision or "").strip().lower()
    if index < 1 or index > MAX_SHOTS:
        raise ValueError("Checkpoint scene must be between 1 and %d." % MAX_SHOTS)
    if re.fullmatch(r"[0-9a-f]{32}", token) is None:
        raise ValueError("Checkpoint revision must be a 32-character revision id.")
    checkpoint_dir = os.path.join(
        _output_root(), "h3_chains", run_name, "checkpoints")
    metadata_path = os.path.join(
        checkpoint_dir, "clip_%04d.%s.json" % (index, token))
    if not os.path.isfile(metadata_path):
        raise FileNotFoundError(
            "Scene %d revision %s is no longer available." % (index, token[:8]))
    metadata = _read_json(metadata_path)
    if not isinstance(metadata, dict):
        raise ValueError("Checkpoint revision metadata is not a JSON object.")
    stored_run = str(metadata.get("run_name") or run_name)
    if _safe_name(stored_run, "") != run_name:
        raise ValueError("Checkpoint revision belongs to a different run.")
    segment = metadata.get("segment")
    if not isinstance(segment, dict):
        raise ValueError("Checkpoint revision has no segment record.")
    if int(segment.get("index", -1)) != index:
        raise ValueError("Checkpoint revision belongs to a different scene.")
    if str(segment.get("revision") or "").lower() != token:
        raise ValueError("Checkpoint revision id does not match its metadata.")
    _verify_segment_artifacts(segment, index)
    return metadata, metadata_path


def _active_checkpoint_revision(run_name: str, scene: int) -> str:
    canonical = os.path.join(
        _output_root(), "h3_chains", run_name, "checkpoints",
        "clip_%04d.json" % int(scene))
    if not os.path.isfile(canonical):
        return ""
    try:
        metadata = _read_json(canonical)
        segment = metadata.get("segment")
        return str(segment.get("revision") or "").lower()
    except (OSError, TypeError, ValueError, json.JSONDecodeError, AttributeError):
        return ""


def _checkpoint_plan_revision(segment: dict[str, Any]) -> dict[str, Any]:
    return {
        "scene": int(segment["index"]),
        "scene_id": str(segment.get("id") or ""),
        "revision": str(segment.get("revision") or ""),
        "prompt_prefix": str(segment.get("prompt_prefix") or ""),
        "scene_prompt": str(segment.get("scene_prompt") or ""),
        "seed": str(segment.get("seed") or "0"),
        "steps": int(segment.get("steps", 0)),
        "raw_frames": int(segment.get("raw_frames", 0)),
        "video": _video_output_item(_absolute_output_path(segment["segment"])),
    }


async def _restore_checkpoint_revisions(request):
    try:
        body = await request.json()
    except (json.JSONDecodeError, TypeError):
        return web.json_response(
            {"error": "Checkpoint recovery requires a JSON request."},
            status=400)
    try:
        run_name = _safe_name(body.get("run_name", ""), "")
        if not run_name:
            raise ValueError("A non-empty H3 chain run_name is required.")
        resume_scene = int(body.get("resume_scene", 0))
        if resume_scene < 2 or resume_scene > MAX_SHOTS:
            raise ValueError("Resume scene must be between 2 and %d." % MAX_SHOTS)
        selections = body.get("revisions")
        if not isinstance(selections, list):
            raise ValueError("Checkpoint recovery requires a revision list.")
        by_scene = {}
        for selection in selections:
            if not isinstance(selection, dict):
                raise ValueError("Each checkpoint selection must be an object.")
            scene = int(selection.get("scene", 0))
            if scene in by_scene:
                raise ValueError("Checkpoint scene %d was selected twice." % scene)
            by_scene[scene] = str(selection.get("revision") or "")
        expected = set(range(1, resume_scene))
        if set(by_scene) != expected:
            raise ValueError(
                "Select exactly scenes 1 through %d before restoring this "
                "resume chain." % (resume_scene - 1))

        loaded = []
        compatibility = None
        prompt_prefix = None
        for scene in range(1, resume_scene):
            metadata, metadata_path = _load_checkpoint_revision(
                run_name, scene, by_scene[scene])
            current_compatibility = metadata.get("compatibility")
            if compatibility is None:
                compatibility = current_compatibility
            elif _canonical_json(current_compatibility) != _canonical_json(
                    compatibility):
                raise ValueError(
                    "Selected checkpoint revisions use different Plan "
                    "compatibility settings.")
            segment = metadata["segment"]
            current_prefix = str(segment.get("prompt_prefix") or "")
            if prompt_prefix is None:
                prompt_prefix = current_prefix
            elif current_prefix != prompt_prefix:
                raise ValueError(
                    "Selected checkpoint revisions use different shared prompts.")
            if loaded:
                predecessor = loaded[-1][1]["segment"]
                expected_revision = str(
                    segment.get("predecessor_revision") or "")
                expected_hash = str(
                    segment.get("predecessor_checkpoint_sha256") or "")
                if (expected_revision and expected_revision != str(
                        predecessor.get("revision") or "")):
                    raise ValueError(
                        "Scene %d revision was generated from a different "
                        "scene %d revision." % (scene, scene - 1))
                if (expected_hash and expected_hash != str(
                        predecessor.get("checkpoint_sha256") or "")):
                    raise ValueError(
                        "Scene %d revision was generated from a different "
                        "scene %d checkpoint." % (scene, scene - 1))
            loaded.append((scene, metadata, metadata_path))

        checkpoint_dir = os.path.join(
            _output_root(), "h3_chains", run_name, "checkpoints")
        originals = {}
        committed = []
        try:
            for scene, metadata, _metadata_path in loaded:
                canonical = os.path.join(
                    checkpoint_dir, "clip_%04d.json" % scene)
                originals[canonical] = (_read_json(canonical)
                                        if os.path.isfile(canonical) else None)
                _atomic_json(canonical, metadata)
                committed.append(canonical)
        except Exception:
            for canonical in reversed(committed):
                original = originals.get(canonical)
                try:
                    if original is None:
                        _safe_unlink(canonical)
                    else:
                        _atomic_json(canonical, original)
                except Exception:
                    _LOG.exception(
                        "Could not roll back checkpoint pointer %s", canonical)
            raise

        restored = [
            _checkpoint_plan_revision(metadata["segment"])
            for _scene, metadata, _metadata_path in loaded
        ]
        return web.json_response({
            "ok": True,
            "run_name": run_name,
            "resume_scene": resume_scene,
            "restored": restored,
            "message": "Restored scenes 1 through %d; resume scene %d is ready."
                       % (resume_scene - 1, resume_scene),
        })
    except FileNotFoundError as exc:
        return web.json_response({"error": str(exc)}, status=404)
    except (OSError, TypeError, ValueError, json.JSONDecodeError, KeyError) as exc:
        return web.json_response({"error": str(exc)}, status=400)


async def _delete_checkpoint_revision(request):
    try:
        body = await request.json()
    except (json.JSONDecodeError, TypeError):
        return web.json_response(
            {"error": "Checkpoint revision deletion requires JSON."},
            status=400)
    try:
        run_name = _safe_name(body.get("run_name", ""), "")
        if not run_name:
            raise ValueError("A non-empty H3 chain run_name is required.")
        scene = int(body.get("scene", 0))
        revision = str(body.get("revision") or "").strip().lower()
        if _active_checkpoint_revision(run_name, scene) == revision:
            return web.json_response(
                {"error": "The active checkpoint revision cannot be deleted. "
                          "Restore a different revision first."}, status=409)
        metadata, metadata_path = _load_checkpoint_revision(
            run_name, scene, revision)
        run_dir = os.path.join(_output_root(), "h3_chains", run_name)
        review_dir = os.path.join(run_dir, "reviews")
        owned = _checkpoint_revision_owned_paths(
            metadata_path, metadata["segment"], review_dir)
        checkpoint_dir = os.path.join(run_dir, "checkpoints")
        if _checkpoint_review_is_shared(
                checkpoint_dir, metadata_path, metadata["segment"]):
            absolute_review_dir = os.path.abspath(review_dir)
            owned = [path for path in owned if os.path.commonpath(
                [absolute_review_dir, path]) != absolute_review_dir]
        canonical = os.path.abspath(os.path.join(
            run_dir, "checkpoints", "clip_%04d.json" % scene))
        allowed_roots = [os.path.abspath(os.path.join(run_dir, name)) for name in (
            "segments", "checkpoints", "generated_audio", "blend_segments",
            "reviews")]
        expected_prefix = "clip_%04d.%s" % (scene, revision)
        absolute_review_dir = os.path.abspath(review_dir)
        for path in owned:
            if path == canonical:
                raise ValueError("Refusing to delete an active checkpoint pointer.")
            if not any(os.path.commonpath([root, path]) == root
                       for root in allowed_roots):
                raise ValueError("Checkpoint revision owns an unexpected path.")
            if (os.path.commonpath([absolute_review_dir, path]) !=
                    absolute_review_dir and
                    not os.path.basename(path).startswith(expected_prefix)):
                raise ValueError(
                    "Checkpoint revision references a file owned by another "
                    "revision.")

        existing = [path for path in owned if os.path.isfile(path)]
        owned_sizes = {path: os.path.getsize(path) for path in existing}
        transaction = uuid.uuid4().hex
        staged = []
        try:
            for path in existing:
                temporary = "%s.delete.%s.tmp" % (path, transaction)
                os.replace(path, temporary)
                staged.append((path, temporary))
        except Exception:
            for original, temporary in reversed(staged):
                try:
                    os.replace(temporary, original)
                except Exception:
                    _LOG.exception(
                        "Could not roll back staged checkpoint deletion %s",
                        original)
            raise
        failed = []
        for _original, temporary in staged:
            try:
                os.unlink(temporary)
            except OSError:
                failed.append(temporary)
        if failed:
            _LOG.warning(
                "Checkpoint revision deletion left %d staged files: %s",
                len(failed), ", ".join(failed))
        failed_set = set(failed)
        reclaimed = sum(
            owned_sizes[original] for original, temporary in staged
            if temporary not in failed_set)
        return web.json_response({
            "ok": True,
            "run_name": run_name,
            "scene": scene,
            "revision": revision,
            "deleted_files": len(existing) - len(failed),
            "reclaimed_bytes": reclaimed,
            "message": "Deleted scene %d revision %s." % (
                scene, revision[:8]),
        })
    except FileNotFoundError as exc:
        return web.json_response({"error": str(exc)}, status=404)
    except (OSError, TypeError, ValueError, json.JSONDecodeError, KeyError) as exc:
        return web.json_response({"error": str(exc)}, status=400)


async def _list_saved_checkpoints(request):
    run_name = _safe_name(request.query.get("run_name", ""), "")
    if not run_name:
        return web.json_response(
            {"error": "A non-empty H3 chain run_name is required."}, status=400)
    checkpoint_dir = os.path.join(
        _output_root(), "h3_chains", run_name, "checkpoints")
    review_dir = os.path.join(
        _output_root(), "h3_chains", run_name, "reviews")
    try:
        review_filenames = os.listdir(review_dir) if os.path.isdir(review_dir) else []
    except OSError:
        review_filenames = []
    checkpoints = []
    revisions_by_key: dict[tuple[int, str], dict[str, Any]] = {}
    if os.path.isdir(checkpoint_dir):
        for filename in sorted(os.listdir(checkpoint_dir)):
            match = re.fullmatch(r"clip_(\d{4})\.json", filename)
            if match is None:
                continue
            try:
                metadata = _read_json(os.path.join(checkpoint_dir, filename))
                segment = metadata.get("segment")
                if not isinstance(segment, dict):
                    continue
                index = int(segment.get("index", int(match.group(1))))
                if index != int(match.group(1)):
                    continue
                segment_path = _absolute_output_path(segment["segment"])
                checkpoint_path = _absolute_output_path(segment["checkpoint"])
                ready = (os.path.isfile(segment_path) and
                         os.path.isfile(checkpoint_path))
                item = {
                    "scene": index,
                    "scene_id": str(segment.get("id") or "clip_%04d" % index),
                    "revision": str(segment.get("revision") or ""),
                    "resume_scene": index + 1,
                    "ready": ready,
                    "raw_frames": int(segment.get("raw_frames", 0)),
                    "delivered_frames": int(segment.get("delivered_frames", 0)),
                }
                if os.path.isfile(segment_path):
                    item["video"] = _video_output_item(segment_path)
                    preview = _checkpoint_review_preview(
                        index, segment, review_dir, review_filenames)
                    if preview is not None:
                        item["preview_video"] = _video_output_item(preview)
                partial_path = os.path.join(
                    _output_root(), "h3_chains", run_name, "final",
                    "partial_through_clip_%04d.mp4" % index)
                if os.path.isfile(partial_path):
                    item["partial_video"] = _video_output_item(partial_path)
                checkpoints.append(item)
            except (OSError, TypeError, ValueError, json.JSONDecodeError,
                    KeyError):
                continue
        active_revisions = {
            int(item["scene"]): str(item.get("revision") or "")
            for item in checkpoints
        }
        for filename in sorted(os.listdir(checkpoint_dir)):
            match = re.fullmatch(
                r"clip_(\d{4})\.([0-9a-f]{32})\.json", filename)
            if match is None:
                continue
            try:
                metadata_path = os.path.join(checkpoint_dir, filename)
                metadata = _read_json(metadata_path)
                segment = metadata.get("segment")
                if not isinstance(segment, dict):
                    continue
                index = int(segment.get("index", int(match.group(1))))
                revision = str(segment.get("revision") or match.group(2))
                if (index != int(match.group(1)) or
                        revision != match.group(2)):
                    continue
                segment_path = _absolute_output_path(segment["segment"])
                checkpoint_path = _absolute_output_path(segment["checkpoint"])
                ready = (os.path.isfile(segment_path) and
                         os.path.isfile(checkpoint_path))
                size_bytes = _checkpoint_revision_size(
                    metadata_path, segment, review_dir)
                item = {
                    "scene": index,
                    "scene_id": str(
                        segment.get("id") or "clip_%04d" % index),
                    "revision": revision,
                    "active": active_revisions.get(index) == revision,
                    "ready": ready,
                    "raw_frames": int(segment.get("raw_frames", 0)),
                    "delivered_frames": int(
                        segment.get("delivered_frames", 0)),
                    "seed": str(segment.get("seed") or ""),
                    "steps": int(segment.get("steps", 0)),
                    "predecessor_revision": str(
                        segment.get("predecessor_revision") or ""),
                    "predecessor_checkpoint_sha256": str(
                        segment.get("predecessor_checkpoint_sha256") or ""),
                    "created_at": datetime.fromtimestamp(
                        os.path.getmtime(metadata_path)).isoformat(
                            timespec="seconds"),
                    "size_bytes": size_bytes,
                }
                prompt = str(segment.get("scene_prompt") or "").strip()
                if prompt:
                    item["prompt_preview"] = re.sub(
                        r"\s+", " ", prompt)[:180]
                if os.path.isfile(segment_path):
                    item["video"] = _video_output_item(segment_path)
                    preview = _checkpoint_review_preview(
                        index, segment, review_dir, review_filenames)
                    if preview is not None:
                        item["preview_video"] = _video_output_item(preview)
                revisions_by_key[(index, revision)] = item
            except (OSError, TypeError, ValueError, json.JSONDecodeError,
                    KeyError):
                continue
    return web.json_response({
        "run_name": run_name,
        "checkpoints": checkpoints,
        "revisions": sorted(
            revisions_by_key.values(),
            key=lambda item: (int(item["scene"]), item["created_at"],
                              item["revision"])),
    })


async def _open_run_folder(request):
    try:
        body = await request.json()
    except Exception:
        return web.json_response(
            {"error": "The output-folder request must contain JSON."},
            status=400)
    if not isinstance(body, dict):
        return web.json_response(
            {"error": "The output-folder request must contain a JSON object."},
            status=400)
    try:
        payload = await asyncio.to_thread(
            _open_run_output_directory, body.get("run_name"))
    except ValueError as exc:
        return web.json_response({"error": str(exc)}, status=400)
    except OSError as exc:
        return web.json_response(
            {"error": "Could not create the H3 run folder: %s" % exc},
            status=500)
    return web.json_response(payload)


async def _get_prompt_history(request):
    run_name = request.query.get("run_name", "")
    scene_id = request.query.get("scene_id", "")
    revision = request.query.get("revision", "")
    store = PromptHistoryStore(_output_root())
    try:
        if revision:
            payload = await asyncio.to_thread(
                store.get, run_name, scene_id, revision)
        else:
            payload = await asyncio.to_thread(store.list, run_name, scene_id)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return web.json_response({"error": str(exc)}, status=400)
    return web.json_response(payload)


async def _update_prompt_history(request):
    try:
        body = await request.json()
    except Exception:
        return web.json_response(
            {"error": "The prompt-history request must contain JSON."},
            status=400)
    if not isinstance(body, dict):
        return web.json_response(
            {"error": "The prompt-history request must contain a JSON object."},
            status=400)
    action = str(body.get("action") or "")
    store = PromptHistoryStore(_output_root())
    try:
        if action == "save":
            payload = await asyncio.to_thread(
                store.save_draft,
                body.get("run_name"), body.get("scene_id"),
                body.get("prompt", ""), body.get("parent_revision"))
        elif action == "activate":
            payload = await asyncio.to_thread(
                store.activate,
                body.get("run_name"), body.get("scene_id"),
                body.get("revision"))
        else:
            return web.json_response(
                {"error": "Unknown prompt-history action."}, status=400)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return web.json_response({"error": str(exc)}, status=400)
    return web.json_response(payload)


async def _list_saved_runs(_request):
    try:
        runs = await asyncio.to_thread(
            RunArchiveManager(_output_root(), _input_root()).list_runs)
    except OSError as exc:
        return web.json_response(
            {"error": "Could not scan H3 runs: %s" % exc}, status=500)
    return web.json_response({"runs": runs})


async def _load_saved_run(request):
    run_name = request.query.get("run_name", "")
    try:
        payload = await asyncio.to_thread(
            RunArchiveManager(_output_root(), _input_root()).load_run, run_name)
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        return web.json_response({"error": str(exc)}, status=400)
    return web.json_response(payload)


async def _save_run_assets(request):
    try:
        body = await request.json()
    except (json.JSONDecodeError, TypeError):
        return web.json_response(
            {"error": "The asset request must contain JSON."}, status=400)
    if not isinstance(body, dict):
        return web.json_response(
            {"error": "The asset request must contain a JSON object."}, status=400)
    try:
        payload = await asyncio.to_thread(
            RunAssetStore(_output_root(), _input_root()).save,
            body.get("run_name"), body.get("bindings"), {
                "images": bool(body.get("archive_images", True)),
                "audio": bool(body.get("archive_audio", True)),
                "video": bool(body.get("archive_video", False)),
            })
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        return web.json_response({"error": str(exc)}, status=400)
    return web.json_response(payload)


async def _optimize_scene_prompt(request):
    try:
        body = await request.json()
    except (json.JSONDecodeError, TypeError):
        return web.json_response(
            {"error": "The prompt optimizer request must contain JSON."},
            status=400)
    try:
        payload = await optimize_prompt_payload(body)
    except ValueError as exc:
        return web.json_response({"error": str(exc)}, status=400)
    except RuntimeError as exc:
        return web.json_response({"error": str(exc)}, status=502)
    except Exception:
        _LOG.exception("Direct prompt optimization failed")
        return web.json_response(
            {"error": "Direct prompt optimization failed unexpectedly. "
             "Check the ComfyUI server log."}, status=500)
    return web.json_response(payload)


if (False and PromptServer is not None and web is not None and
        getattr(PromptServer, "instance", None) is not None):
    PromptServer.instance.routes.post(
        "/minimax_h3_context_loop/review")(_submit_review_decision)
    PromptServer.instance.routes.get(
        "/minimax_h3_context_loop/reviews")(_list_pending_reviews)
    PromptServer.instance.routes.get(
        "/minimax_h3_context_loop/checkpoints")(_list_saved_checkpoints)
    PromptServer.instance.routes.post(
        "/minimax_h3_context_loop/checkpoint-revisions/restore")(
            _restore_checkpoint_revisions)
    PromptServer.instance.routes.post(
        "/minimax_h3_context_loop/checkpoint-revisions/delete")(
            _delete_checkpoint_revision)
    PromptServer.instance.routes.post(
        "/minimax_h3_context_loop/open-run-folder")(_open_run_folder)
    PromptServer.instance.routes.get(
        "/minimax_h3_context_loop/prompt-history")(_get_prompt_history)
    PromptServer.instance.routes.post(
        "/minimax_h3_context_loop/prompt-history")(_update_prompt_history)
    PromptServer.instance.routes.get(
        "/minimax_h3_context_loop/runs")(_list_saved_runs)
    PromptServer.instance.routes.get(
        "/minimax_h3_context_loop/run")(_load_saved_run)
    PromptServer.instance.routes.post(
        "/minimax_h3_context_loop/run-assets")(_save_run_assets)
    PromptServer.instance.routes.post(
        "/minimax_h3_context_loop/prompt-optimize")(_optimize_scene_prompt)


CHAIN_NODE_CLASS_MAPPINGS = {
    "MiniMaxH3ChainPlan": MiniMaxH3ChainPlan,
    "MiniMaxH3ChainScenePromptEditor": MiniMaxH3ChainScenePromptEditor,
    "MiniMaxH3ChainRichScenePromptEditor": MiniMaxH3ChainRichScenePromptEditor,
    "MiniMaxH3ChainPlanStudio": MiniMaxH3ChainPlanStudio,
    "MiniMaxH3ChainRunManager": MiniMaxH3ChainRunManager,
    "MiniMaxH3ChainFirstSceneImage": MiniMaxH3ChainFirstSceneImage,
    "MiniMaxH3ChainFrameIndexSwitch": MiniMaxH3ChainFrameIndexSwitch,
    "MiniMaxH3ReferenceVideoPrepare": MiniMaxH3ReferenceVideoPrepare,
    "MiniMaxH3ScheduledPictureReference": MiniMaxH3ScheduledPictureReference,
    "MiniMaxH3ScheduledVideoReference": MiniMaxH3ScheduledVideoReference,
    "MiniMaxH3ScheduledAudioReference": MiniMaxH3ScheduledAudioReference,
    "MiniMaxH3ScheduledReferenceToVideo": MiniMaxH3ScheduledReferenceToVideo,
    "MiniMaxH3TaggedPictureReference": MiniMaxH3TaggedPictureReference,
    "MiniMaxH3TaggedVideoReference": MiniMaxH3TaggedVideoReference,
    "MiniMaxH3TaggedAudioReference": MiniMaxH3TaggedAudioReference,
    "MiniMaxH3TaggedReferenceToVideo": MiniMaxH3TaggedReferenceToVideo,
    "MiniMaxH3ChainExternalVideo": MiniMaxH3ChainExternalVideo,
    "MiniMaxH3ChainLoopStart": MiniMaxH3ChainLoopStart,
    "MiniMaxH3ChainCurrent": MiniMaxH3ChainCurrent,
    "MiniMaxH3PatchPriority": MiniMaxH3PatchPriority,
    "MiniMaxH3ChainContext": MiniMaxH3ChainContext,
    "MiniMaxH3ChainSegmentSave": MiniMaxH3ChainSegmentSave,
    "MiniMaxH3ChainReview": MiniMaxH3ChainReview,
    "MiniMaxH3ChainLoopEnd": MiniMaxH3ChainLoopEnd,
    "MiniMaxH3ChainManifestLoad": MiniMaxH3ChainManifestLoad,
    "MiniMaxH3ChainExportPNG": MiniMaxH3ChainExportPNG,
    "MiniMaxH3ChainAssemble": MiniMaxH3ChainAssemble,
    "MiniMaxH3ChainVideoOutput": MiniMaxH3ChainVideoOutput,
    "MiniMaxH3LTXRollingInput": MiniMaxH3LTXRollingInput,
    "MiniMaxH3LTXRollingInject": MiniMaxH3LTXRollingInject,
    "MiniMaxH3LTXRollingCheckpoint": MiniMaxH3LTXRollingCheckpoint,
    "MiniMaxH3LTXRollingCrop": MiniMaxH3LTXRollingCrop,
    "MiniMaxH3SteerSequenceStart": MiniMaxH3SteerSequenceStart,
    "MiniMaxH3SteerSegmentPaths": MiniMaxH3SteerSegmentPaths,
    "MiniMaxH3SteerAcceptRaw": MiniMaxH3SteerAcceptRaw,
    "MiniMaxH3SteerAcceptEnhanced": MiniMaxH3SteerAcceptEnhanced,
    "MiniMaxH3SteerableSegment": MiniMaxH3SteerableSegment,
    "MiniMaxH3SteerAssemble": MiniMaxH3SteerAssemble,
}

CHAIN_NODE_DISPLAY_NAME_MAPPINGS = {
    "MiniMaxH3ChainPlan": "MiniMax H3 Contex Loop Plan",
    "MiniMaxH3ChainScenePromptEditor": "MiniMax H3 Scene Prompt Editor",
    "MiniMaxH3ChainRichScenePromptEditor": (
        "MiniMax H3 Rich Scene Prompt Editor (Experimental)"),
    "MiniMaxH3ChainPlanStudio": "MiniMax H3 Plan Studio (Experimental)",
    "MiniMaxH3ChainRunManager": "MiniMax H3 Run Manager",
    "MiniMaxH3ChainFirstSceneImage": "MiniMax H3 Frame Gate",
    "MiniMaxH3ChainFrameIndexSwitch": "MiniMax H3 Frame Index Switch",
    "MiniMaxH3ReferenceVideoPrepare": "MiniMax H3 Reference Video Prep",
    "MiniMaxH3ScheduledPictureReference": "MiniMax H3 Scheduled Picture Ref",
    "MiniMaxH3ScheduledVideoReference": "MiniMax H3 Scheduled Video Ref",
    "MiniMaxH3ScheduledAudioReference": "MiniMax H3 Scheduled Audio Ref",
    "MiniMaxH3ScheduledReferenceToVideo": "MiniMax H3 Scheduled Ref2VA",
    "MiniMaxH3TaggedPictureReference": "MiniMax H3 Tagged Picture Ref",
    "MiniMaxH3TaggedVideoReference": "MiniMax H3 Tagged Video Ref",
    "MiniMaxH3TaggedAudioReference": "MiniMax H3 Tagged Audio Ref",
    "MiniMaxH3TaggedReferenceToVideo": "MiniMax H3 Tagged Ref2VA",
    "MiniMaxH3ChainExternalVideo": "MiniMax H3 Existing Video Context",
    "MiniMaxH3ChainLoopStart": "MiniMax H3 Contex Loop Start",
    "MiniMaxH3ChainCurrent": "MiniMax H3 Contex Loop Current Shot",
    "MiniMaxH3PatchPriority": "MiniMax H3 Patch Priority",
    "MiniMaxH3ChainContext": "MiniMax H3 Contex Loop Context",
    "MiniMaxH3ChainSegmentSave": "MiniMax H3 Contex Loop Segment + Checkpoint",
    "MiniMaxH3ChainReview": "MiniMax H3 Contex Loop Review Gate",
    "MiniMaxH3ChainLoopEnd": "MiniMax H3 Contex Loop End",
    "MiniMaxH3ChainManifestLoad": "MiniMax H3 Contex Loop Load Manifest",
    "MiniMaxH3ChainExportPNG": "MiniMax H3 Contex Loop Export PNG Sequence",
    "MiniMaxH3ChainAssemble": "MiniMax H3 Contex Loop Assemble",
    "MiniMaxH3ChainVideoOutput": "MiniMax H3 Final Video Output",
    "MiniMaxH3LTXRollingInput": "MiniMax H3 LTX Rolling Input",
    "MiniMaxH3LTXRollingInject": "MiniMax H3 LTX Rolling Context",
    "MiniMaxH3LTXRollingCheckpoint": "MiniMax H3 LTX Rolling Checkpoint",
    "MiniMaxH3LTXRollingCrop": "MiniMax H3 LTX Rolling Crop",
    "MiniMaxH3SteerSequenceStart": "MiniMax H3 Steerable Sequence Start",
    "MiniMaxH3SteerSegmentPaths": "MiniMax H3 Steerable Segment Paths",
    "MiniMaxH3SteerAcceptRaw": "MiniMax H3 Accept Raw Segment",
    "MiniMaxH3SteerAcceptEnhanced": "MiniMax H3 Accept Enhanced Segment",
    "MiniMaxH3SteerableSegment": "MiniMax H3 Steerable Segment",
    "MiniMaxH3SteerAssemble": "MiniMax H3 Assemble Steerable Sequence",
}
