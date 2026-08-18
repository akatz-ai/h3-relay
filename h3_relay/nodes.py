# SPDX-License-Identifier: GPL-3.0-only
"""Public H3 Relay nodes and namespaced internal graph stages."""

from __future__ import annotations

import math
import os
import shutil
import uuid
from fractions import Fraction
from typing import Any

import folder_paths
import comfy.ldm.modules.attention as attention_module
import comfy.samplers as sampler_module

from comfy_execution.graph_utils import GraphBuilder
from comfy_extras.nodes_frame_interpolation import FrameInterpolate

try:
    from comfy_api.latest import InputImpl
except ImportError:
    InputImpl = None

from .vendor.context_loop import chain_nodes as context
from .vendor.context_loop import nodes as context_nodes
from .vendor.hybrid import MiniMaxH3HybridLoader
from .vendor.spectrum.nodes import SpectrumApplyMiniMaxH3
from . import cache as relay_cache


CATEGORY = "H3 Relay"
FINISH_TYPE = "H3_RELAY_ENHANCED"
FINISH_FORMAT = "h3_relay_enhanced_v1"
LEGACY_FINISH_FORMAT = "h3_relay_finish_v1"
LEGACY_ENHANCED_FORMAT = "h3_relay_enhanced_sequence_v1"
MODEL_BUNDLE_TYPE = "H3_RELAY_MODEL"
MODEL_BUNDLE_FORMAT = "h3_relay_model_v1"
INTERPOLATION_BUNDLE_TYPE = "H3_RELAY_INTERPOLATION"
INTERPOLATION_BUNDLE_FORMAT = "h3_relay_interpolation_v1"
FPS = 24
LTX_CONTEXT_FRAMES = 17
LTX_CONTEXT_STEPS = 3

H3_BASE_MODEL = "minimax_h3_fl2va_int8_convrot.safetensors"
H3_OVERLAY_MODEL = "minimax_h3_ref2va_int8_convrot.safetensors"
H3_TEXT_ENCODER = "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors"
H3_VIDEO_VAE = "minimax_h3_video_vae_fp16.safetensors"
H3_AUDIO_VAE = "minimax_h3_audio_vae_fp32.safetensors"
LTX_VAE = "ltx-2.5-video-vae-bf16.safetensors"
LTX_UPSCALER = "ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors"
LTX_MODEL = "ltx-2.5-22b-dev-transformer-comfy-int8-convrot.safetensors"
LTX_DISTILLED_LORA = "ltx-2.5-22b-distilled-lora-450-bf16.safetensors"
LTX_UPSCALE_LORA = (
    "ltx-2.5-22b-ic-lora-pixel-spatial-upscaler-x2-1.0.safetensors"
)
LTX_TEXT_ENCODER = "gemma4-12b-with-proj-ltx-2.5-comfy-int8-convrot.safetensors"
H3_SAMPLERS = list(sampler_module.SAMPLER_NAMES)
H3_SCHEDULERS = ["beta57"] + [
    name for name in sampler_module.SCHEDULER_NAMES if name != "beta57"
]


def _manual_cache_revision_input():
    return ("STRING", {
        "default": "v1",
        "advanced": True,
        "tooltip": (
            "Manual cache reset only. Change this when replacing model "
            "contents without changing the filename; normal model, LoRA, "
            "dtype, strength, and attention changes are tracked automatically."
        ),
    })


def _require_video_api() -> None:
    if InputImpl is None:
        raise RuntimeError("H3 Relay requires ComfyUI 0.32.0 or newer.")


def _duration_frames(seconds: float) -> int:
    seconds = float(seconds)
    if seconds < 1.0 or seconds > 15.0:
        raise ValueError("H3 Relay shot duration must be between 1 and 15 seconds.")
    requested = int(math.ceil(seconds * FPS))
    frames = 5 + 17 * max(0, int(math.ceil((requested - 5) / 17.0)))
    if frames > 362:
        frames = 362
    if frames <= context.STEER_H3_CONTEXT_FRAMES:
        raise ValueError("H3 Relay continuation window is shorter than its context.")
    return frames


def _validate_ltx_tiling(
    context_window_frames: int,
    context_overlap_frames: int,
    vae_temporal_tile_frames: int,
    vae_temporal_overlap_frames: int,
) -> tuple[int, int, int, int]:
    window = int(context_window_frames)
    overlap = int(context_overlap_frames)
    vae_tile = int(vae_temporal_tile_frames)
    vae_overlap = int(vae_temporal_overlap_frames)
    if window < 65 or (window - 1) % 8:
        raise ValueError("LTX context window must be at least 65 and satisfy 8n+1.")
    if overlap < 0 or overlap % 8 or overlap >= window:
        raise ValueError(
            "LTX context overlap must be a non-negative multiple of 8 smaller than the window."
        )
    if vae_tile < 32 or vae_tile % 8:
        raise ValueError("LTX VAE temporal tile must be a multiple of 8 at least 32.")
    if vae_overlap < 8 or vae_overlap % 8 or vae_overlap * 2 >= vae_tile:
        raise ValueError(
            "LTX VAE temporal overlap must be a multiple of 8 and less than half the tile."
        )
    return window, overlap, vae_tile, vae_overlap


def _raw_sequence(value: Any) -> dict[str, Any]:
    sequence = context._steer_sequence(value)
    if str(sequence.get("output_profile")) != "raw_h3":
        raise ValueError("H3 Relay finishing nodes require a raw H3 sequence.")
    return sequence


def _finish(value: Any) -> dict[str, Any]:
    if (
        not isinstance(value, dict)
        or value.get("format") not in {
            FINISH_FORMAT,
            LEGACY_FINISH_FORMAT,
            LEGACY_ENHANCED_FORMAT,
        }
    ):
        raise ValueError("H3 Relay received an invalid finishing token.")
    if not isinstance(value.get("ltx_segments"), list):
        raise ValueError("H3 Relay finishing token has no LTX segment list.")
    if not isinstance(value.get("delivery_segments"), list):
        raise ValueError("H3 Relay finishing token has no delivery segment list.")
    return value


def _model_bundle(value: Any, kind: str | None = None) -> dict[str, Any]:
    if not isinstance(value, dict) or value.get("format") != MODEL_BUNDLE_FORMAT:
        raise ValueError("H3 Relay received an invalid model bundle.")
    if kind is not None and value.get("kind") != kind:
        raise ValueError(
            "H3 Relay expected a %s model bundle, received %s."
            % (kind, value.get("kind"))
        )
    if value.get("model") is None or not str(value.get("cache_tag") or ""):
        raise ValueError("H3 Relay model bundle is incomplete.")
    return value


def _interpolation_bundle(value: Any) -> dict[str, Any]:
    if (
        not isinstance(value, dict)
        or value.get("format") != INTERPOLATION_BUNDLE_FORMAT
        or value.get("model") is None
        or not str(value.get("cache_tag") or "")
    ):
        raise ValueError("H3 Relay received an invalid interpolation bundle.")
    return value


def _sequence_with_h3_model(sequence: Any, h3_model: Any) -> tuple[dict[str, Any], Any]:
    bundle = _model_bundle(h3_model, "h3")
    sequence = context._steer_sequence(sequence)
    tag = str(bundle["cache_tag"])
    current = sequence.get("model_cache_tag")
    if current is None:
        if sequence.get("segments"):
            raise ValueError(
                "This legacy sequence has accepted shots but no H3 model identity. "
                "Start again from Sequence Start so cache ownership is explicit."
            )
        sequence = dict(sequence)
        sequence["model_cache_tag"] = tag
        sequence["generation_fingerprint"] = "%s-model-%s" % (
            sequence["generation_fingerprint"],
            context._fingerprint(tag)[:16],
        )
    elif str(current) != tag:
        raise ValueError(
            "The connected H3 model/LoRA/attention chain changed. Re-run from "
            "Sequence Start so every continuation uses one model identity."
        )
    return sequence, bundle["model"]


def _raw_revision(record: dict[str, Any]) -> str:
    return str(record.get("revision") or record.get("h3_segment", {}).get("revision") or "")


def _finish_base(raw_sequence: dict[str, Any], previous: Any) -> dict[str, Any]:
    index = len(raw_sequence["segments"])
    if index < 1:
        raise ValueError("H3 Relay finishing requires at least one raw shot.")
    if previous is None:
        if index != 1:
            raise ValueError(
                "H3 Relay shot %d enhancement requires the previous finishing token."
                % index
            )
        return {
            "format": FINISH_FORMAT,
            "run_name": str(raw_sequence["run_name"]),
            "raw_sequence": raw_sequence,
            "raw_revisions": [],
            "ltx_segments": [],
            "delivery_segments": [],
        }

    previous = _finish(previous)
    expected = index - 1
    if len(previous["ltx_segments"]) != expected:
        raise ValueError(
            "H3 Relay shot %d enhancement requires %d preceding LTX results."
            % (index, expected)
        )
    previous_revisions = list(previous.get("raw_revisions") or [])
    if len(previous_revisions) != expected:
        raise ValueError("H3 Relay finishing history is incomplete.")
    actual = [_raw_revision(item) for item in raw_sequence["segments"][:expected]]
    if previous_revisions != actual:
        raise ValueError(
            "An upstream raw H3 shot changed. Re-run finishing from the first stale shot."
        )
    updated = dict(previous)
    updated["raw_sequence"] = raw_sequence
    updated["ltx_segments"] = [dict(item) for item in previous["ltx_segments"]]
    updated["delivery_segments"] = [
        dict(item) for item in previous["delivery_segments"]
    ]
    return updated


def _rebuild_state(raw_sequence: dict[str, Any]):
    index = len(raw_sequence["segments"])
    current = raw_sequence["shots"][index - 1]
    predecessor = dict(raw_sequence)
    predecessor["shots"] = [dict(item) for item in raw_sequence["shots"][:-1]]
    predecessor["segments"] = [
        dict(item) for item in raw_sequence["segments"][:-1]
    ]
    predecessor["total_enhanced_frames"] = sum(
        int(item["enhanced_frames"]) for item in predecessor["segments"]
    )
    prompt = str(current.get("scene_prompt") or current.get("prompt") or "")
    raw_frames = int(current.get("length") or current.get("raw_frames"))
    state, shot = context._steer_state(
        predecessor,
        str(current["id"]),
        prompt,
        int(current["seed"]),
        raw_frames,
        int(current["steps"]),
    )
    segment = raw_sequence["segments"][index - 1].get("h3_segment")
    if not isinstance(segment, dict):
        raise ValueError("H3 Relay raw shot is missing its H3 checkpoint record.")
    if str(segment.get("revision")) != str(
        raw_sequence["segments"][index - 1].get("revision")
    ):
        raise ValueError("H3 Relay raw shot revision metadata is inconsistent.")
    return state, shot, segment


def _relay_root(run_name: str, stage: str) -> str:
    normalized = context._safe_name(run_name, "h3_relay")
    cached = relay_cache.cache_path("h3_relay", normalized, stage)
    legacy = os.path.join(
        folder_paths.get_output_directory(), "h3_relay", normalized, stage
    )
    path = cached if os.path.isdir(cached) or not os.path.isdir(legacy) else legacy
    os.makedirs(path, exist_ok=True)
    return path


def _stage_paths(run_name: str, stage: str, index: int, revision: str) -> dict[str, str]:
    root = _relay_root(run_name, stage)
    return {
        "full": os.path.join(root, "full", "clip_%04d.%s.mp4" % (index, revision)),
        "segment": os.path.join(
            root, "segments", "clip_%04d.%s.mp4" % (index, revision)
        ),
        "metadata": os.path.join(root, "checkpoints", "clip_%04d.json" % index),
        "sequence": os.path.join(root, "sequence.json"),
    }


def _valid_file(path_value: str, expected_hash: str) -> bool:
    try:
        path = context._absolute_output_path(path_value)
        return bool(expected_hash) and os.path.isfile(path) and context._file_sha256(path) == expected_hash
    except (OSError, ValueError):
        return False


def _append_ltx(base: dict[str, Any], raw_sequence: dict[str, Any], record: dict[str, Any]):
    updated = dict(base)
    updated["format"] = FINISH_FORMAT
    updated["raw_sequence"] = raw_sequence
    updated["raw_revisions"] = [
        _raw_revision(item) for item in raw_sequence["segments"]
    ]
    updated["ltx_segments"] = [dict(item) for item in base["ltx_segments"]] + [
        dict(record)
    ]
    updated["delivery_segments"] = [
        dict(item) for item in base["delivery_segments"]
    ]
    updated["stage"] = "ltx"
    return updated


def _append_delivery(finish: dict[str, Any], record: dict[str, Any]):
    updated = dict(finish)
    updated["format"] = FINISH_FORMAT
    updated["ltx_segments"] = [dict(item) for item in finish["ltx_segments"]]
    updated["delivery_segments"] = [
        dict(item) for item in finish["delivery_segments"]
    ] + [dict(record)]
    updated["stage"] = "interpolated"
    return updated


def _truncate_raw_sequence(sequence: dict[str, Any], shot_index: int) -> dict[str, Any]:
    sequence = _raw_sequence(sequence)
    shot_index = int(shot_index)
    if shot_index < 1 or len(sequence["segments"]) < shot_index:
        raise ValueError(
            "H3 Relay disk restore requires raw shot %d, but only %d are saved."
            % (shot_index, len(sequence["segments"]))
        )
    updated = dict(sequence)
    updated["shots"] = [dict(item) for item in sequence["shots"][:shot_index]]
    updated["segments"] = [
        dict(item) for item in sequence["segments"][:shot_index]
    ]
    updated["total_enhanced_frames"] = sum(
        int(item["enhanced_frames"]) for item in updated["segments"]
    )
    for index, record in enumerate(updated["segments"], 1):
        segment = record.get("h3_segment")
        if not isinstance(segment, dict):
            raise ValueError("Restored raw shot %d has no H3 checkpoint." % index)
        context._verify_segment_artifacts(segment, index)
        if not _valid_file(
            str(record.get("enhanced_segment") or ""),
            str(record.get("enhanced_segment_sha256") or ""),
        ):
            raise ValueError("Restored raw shot %d preview failed integrity validation." % index)
    return updated


def _restore_raw_from_disk(run_name: str, shot_index: int) -> dict[str, Any]:
    path = _stage_paths(str(run_name), "raw", int(shot_index), "restore")["sequence"]
    try:
        sequence = context._read_json(path)
    except (FileNotFoundError, OSError, ValueError) as exc:
        raise ValueError(
            "H3 Relay raw sequence is not available on disk for %s: %s"
            % (run_name, exc)
        ) from exc
    return _truncate_raw_sequence(sequence, int(shot_index))


def _restore_finish_from_disk(
    run_name: str, shot_index: int, stage: str,
    delivery_count: int | None = None,
) -> dict[str, Any]:
    stage = str(stage)
    if stage not in {"ltx", "interpolated"}:
        raise ValueError("H3 Relay restore stage must be ltx or interpolated.")
    path = _stage_paths(str(run_name), stage, int(shot_index), "restore")["sequence"]
    try:
        finish = _finish(context._read_json(path))
    except (FileNotFoundError, OSError, ValueError) as exc:
        raise ValueError(
            "H3 Relay %s sequence is not available on disk for %s: %s"
            % (stage, run_name, exc)
        ) from exc
    shot_index = int(shot_index)
    if len(finish["ltx_segments"]) < shot_index:
        raise ValueError(
            "H3 Relay %s restore needs shot %d, but only %d LTX shots are saved."
            % (stage, shot_index, len(finish["ltx_segments"]))
        )
    if delivery_count is None:
        delivery_count = shot_index if stage == "interpolated" else shot_index - 1
    delivery_count = int(delivery_count)
    if delivery_count < 0 or delivery_count > shot_index:
        raise ValueError(
            "H3 Relay restore delivery_count must be between 0 and shot_index."
        )
    if len(finish["delivery_segments"]) < delivery_count:
        raise ValueError(
            "H3 Relay %s restore is missing completed delivery shots."
            % stage
        )
    raw_sequence = _truncate_raw_sequence(finish["raw_sequence"], shot_index)
    updated = dict(finish)
    updated["raw_sequence"] = raw_sequence
    updated["raw_revisions"] = list(finish["raw_revisions"][:shot_index])
    updated["ltx_segments"] = [
        dict(item) for item in finish["ltx_segments"][:shot_index]
    ]
    updated["delivery_segments"] = [
        dict(item) for item in finish["delivery_segments"][:delivery_count]
    ]
    updated["stage"] = stage
    for index, record in enumerate(updated["ltx_segments"], 1):
        if not _valid_file(
            str(record.get("full_segment") or ""),
            str(record.get("full_segment_sha256") or ""),
        ):
            raise ValueError("Restored LTX shot %d failed integrity validation." % index)
    for index, record in enumerate(updated["delivery_segments"], 1):
        if not _valid_file(
            str(record.get("delivery_segment") or ""),
            str(record.get("delivery_segment_sha256") or ""),
        ):
            raise ValueError(
                "Restored interpolated shot %d failed integrity validation." % index
            )
    return updated


class H3RelayRestoreRawSequence:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "run_name": ("STRING", {"forceInput": True}),
                "shot_index": ("INT", {"default": 1, "min": 1, "max": 128}),
            }
        }

    RETURN_TYPES = (context.STEER_SEQUENCE_TYPE,)
    RETURN_NAMES = ("sequence",)
    FUNCTION = "restore"
    CATEGORY = CATEGORY + "/internal"

    def restore(self, run_name, shot_index):
        return (_restore_raw_from_disk(str(run_name), int(shot_index)),)


class H3RelayRestoreEnhanced:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "run_name": ("STRING", {"forceInput": True}),
                "shot_index": ("INT", {"default": 1, "min": 1, "max": 128}),
                "stage": (["ltx", "interpolated"], {"default": "interpolated"}),
                "delivery_count": ("INT", {"default": 1, "min": 0, "max": 128}),
            }
        }

    RETURN_TYPES = (FINISH_TYPE,)
    RETURN_NAMES = ("enhanced",)
    FUNCTION = "restore"
    CATEGORY = CATEGORY + "/internal"

    def restore(self, run_name, shot_index, stage, delivery_count):
        return (_restore_finish_from_disk(
            str(run_name), int(shot_index), str(stage), int(delivery_count)
        ),)


def _has_audio_stream(path: str) -> bool:
    if context.av is None:
        return False
    try:
        with context.av.open(path, mode="r") as container:
            return bool(container.streams.audio)
    except (OSError, ValueError):
        return False


def _transcode_video(source: str, destination: str, output_crf: int) -> None:
    output_crf = int(output_crf)
    if output_crf < 0 or output_crf > 51:
        raise ValueError("H3 Relay output_crf must be between 0 and 51.")
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("H3 Relay cached video encoding requires ffmpeg.")
    os.makedirs(os.path.dirname(destination), exist_ok=True)
    temporary = "%s.%s.tmp.mp4" % (destination, uuid.uuid4().hex)
    context._safe_unlink(temporary)
    try:
        context._run_ffmpeg([
            ffmpeg, "-y", "-i", source,
            "-map", "0:v:0", "-map", "0:a?",
            "-c:v", "libx264", "-crf", str(output_crf),
            "-preset", "medium", "-c:a", "copy",
            "-map_metadata", "0",
            "-movflags", "use_metadata_tags+faststart",
            temporary,
        ])
        os.replace(temporary, destination)
    finally:
        context._safe_unlink(temporary)


def _video_ui(path: str, status: str, result: tuple[Any, ...]):
    return {
        "ui": {
            "images": [context._video_output_item(path)],
            "animated": (True,),
            "text": [status],
        },
        "result": result,
    }


def _raw_preview_with_audio(sequence: dict[str, Any], output_crf: int = 18):
    """Mux the canonical generated WAV into the raw checkpoint preview.

    H3 checkpoint MP4s deliberately contain picture only; continuation audio
    remains in the lossless WAV and AV latent checkpoint. The user-facing
    VIDEO must point at a separate muxed artifact so previews and downstream
    core Save Video nodes receive both streams.
    """
    _require_video_api()
    sequence = _raw_sequence(sequence)
    index = len(sequence["segments"])
    if index < 1:
        raise ValueError("H3 Relay raw preview requires an accepted shot.")
    record = dict(sequence["segments"][index - 1])
    segment = record.get("h3_segment")
    if not isinstance(segment, dict):
        raise ValueError("H3 Relay raw preview is missing its H3 segment record.")
    source = context._absolute_output_path(str(segment["segment"]))
    audio = context._absolute_output_path(str(segment["generated_audio"]))
    if not os.path.isfile(source):
        raise FileNotFoundError("H3 Relay raw checkpoint video is missing: %s" % source)
    if not os.path.isfile(audio):
        raise FileNotFoundError("H3 Relay generated audio is missing: %s" % audio)

    output_crf = int(output_crf)
    if output_crf < 0 or output_crf > 51:
        raise ValueError("H3 Relay output_crf must be between 0 and 51.")
    revision = str(segment["revision"])
    preview_revision = (
        revision if output_crf == 18 else "%s.crf%02d" % (revision, output_crf)
    )
    paths = _stage_paths(sequence["run_name"], "raw", index, preview_revision)
    preview = paths["segment"]
    os.makedirs(os.path.dirname(preview), exist_ok=True)
    if not _has_audio_stream(preview):
        ffmpeg = shutil.which("ffmpeg")
        if ffmpeg is None:
            raise RuntimeError("H3 Relay raw audio preview requires ffmpeg.")
        temporary = "%s.%s.tmp.mp4" % (preview, uuid.uuid4().hex)
        context._safe_unlink(temporary)
        try:
            video_codec = ["-c:v", "copy"]
            if output_crf != 18:
                video_codec = [
                    "-c:v", "libx264", "-crf", str(output_crf),
                    "-preset", "medium",
                ]
            context._run_ffmpeg([
                ffmpeg,
                "-y",
                "-i",
                source,
                "-i",
                audio,
                "-map",
                "0:v:0",
                "-map",
                "1:a:0",
                *video_codec,
                "-c:a",
                "aac",
                "-b:a",
                "256k",
                "-t",
                "%.9f" % (int(record["enhanced_frames"]) / float(FPS)),
                "-map_metadata",
                "0",
                "-movflags",
                "use_metadata_tags+faststart",
                temporary,
            ])
            os.replace(temporary, preview)
        finally:
            context._safe_unlink(temporary)

    record["enhanced_segment"] = context._relative_output_path(preview)
    record["enhanced_segment_sha256"] = context._file_sha256(preview)
    record["output_crf"] = output_crf
    updated = dict(sequence)
    updated["segments"] = [dict(item) for item in sequence["segments"][:-1]] + [
        record
    ]
    metadata_paths = context._steer_enhanced_paths(
        sequence["run_name"], index, revision
    )
    context._atomic_json(metadata_paths["metadata"], {
        "format": context.STEER_CACHE_FORMAT,
        "cache_key": str(record.get("cache_key") or ""),
        "segment": record,
    })
    context._atomic_json(metadata_paths["sequence"], updated)
    context._atomic_json(paths["sequence"], updated)
    status = (
        "raw H3 preview encoded at CRF %d without repeating inference -> %s"
        % (output_crf, preview)
    )
    return updated, preview, status


class H3RelayModelBundlePack:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "kind": ("STRING", {"forceInput": True}),
                "model": ("MODEL",),
                "cache_tag": ("STRING", {"forceInput": True}),
            },
            "optional": {
                "vae": ("VAE",),
                "upscale_model": ("LATENT_UPSCALE_MODEL",),
                "clip": ("CLIP",),
            },
        }

    RETURN_TYPES = (MODEL_BUNDLE_TYPE,)
    RETURN_NAMES = ("model",)
    FUNCTION = "pack"
    CATEGORY = CATEGORY + "/internal"

    def pack(self, kind, model, cache_tag, vae=None, upscale_model=None, clip=None):
        return ({
            "format": MODEL_BUNDLE_FORMAT,
            "kind": str(kind),
            "model": model,
            "vae": vae,
            "upscale_model": upscale_model,
            "clip": clip,
            "cache_tag": str(cache_tag),
        },)


class H3RelayInterpolationBundlePack:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("INTERP_MODEL",),
                "cache_tag": ("STRING", {"forceInput": True}),
            }
        }

    RETURN_TYPES = (INTERPOLATION_BUNDLE_TYPE,)
    RETURN_NAMES = ("interpolation",)
    FUNCTION = "pack"
    CATEGORY = CATEGORY + "/internal"

    def pack(self, model, cache_tag):
        return ({
            "format": INTERPOLATION_BUNDLE_FORMAT,
            "model": model,
            "cache_tag": str(cache_tag),
        },)


class H3RelayH3HybridModelLoader(MiniMaxH3HybridLoader):
    @classmethod
    def INPUT_TYPES(cls):
        schema = super().INPUT_TYPES()
        required = dict(schema["required"])
        base_options, base_config = required["base_model"]
        overlay_options, overlay_config = required["overlay_model"]
        required["base_model"] = (
            base_options,
            {**base_config, "default": H3_BASE_MODEL},
        )
        required["overlay_model"] = (
            overlay_options,
            {**overlay_config, "default": H3_OVERLAY_MODEL},
        )
        required["manual_cache_revision"] = _manual_cache_revision_input()
        return {"required": required, "optional": dict(schema.get("optional", {}))}

    RETURN_TYPES = (MODEL_BUNDLE_TYPE,)
    RETURN_NAMES = ("h3_model",)
    FUNCTION = "load_relay"
    CATEGORY = CATEGORY + "/loaders"

    def load_relay(
        self,
        base_model,
        overlay_model,
        overlay_preset,
        manual_cache_revision,
        block_range_start=0,
        block_range_end=49,
        final_adaln_from_overlay=False,
        custom_overlays="",
        custom_base="",
        weight_dtype="default",
    ):
        (model,) = super().load_hybrid(
            base_model,
            overlay_model,
            overlay_preset,
            block_range_start,
            block_range_end,
            final_adaln_from_overlay,
            custom_overlays,
            custom_base,
            weight_dtype,
        )
        fingerprint = context._fingerprint({
            "kind": "h3_hybrid",
            "base": base_model,
            "overlay": overlay_model,
            "preset": overlay_preset,
            "block_range": [int(block_range_start), int(block_range_end)],
            "final_adaln": bool(final_adaln_from_overlay),
            "custom_overlays": str(custom_overlays),
            "custom_base": str(custom_base),
            "weight_dtype": str(weight_dtype),
            "manual": str(manual_cache_revision),
        })
        return ({
            "format": MODEL_BUNDLE_FORMAT,
            "kind": "h3",
            "model": model,
            "vae": None,
            "upscale_model": None,
            "clip": None,
            "cache_tag": "h3-hybrid:%s" % fingerprint,
        },)


class H3RelayH3ModelLoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_name": (folder_paths.get_filename_list("diffusion_models"),),
                "weight_dtype": (["default", "fp8_e4m3fn", "fp8_e4m3fn_fast", "fp8_e5m2"], {"default": "default"}),
                "manual_cache_revision": _manual_cache_revision_input(),
            }
        }

    RETURN_TYPES = (MODEL_BUNDLE_TYPE,)
    RETURN_NAMES = ("h3_model",)
    FUNCTION = "load"
    CATEGORY = CATEGORY + "/loaders"

    def load(self, model_name, weight_dtype, manual_cache_revision):
        graph = GraphBuilder()
        loader = graph.node("UNETLoader", "H3Model")
        loader.set_input("unet_name", str(model_name))
        loader.set_input("weight_dtype", str(weight_dtype))
        tag = "h3-model:%s" % context._fingerprint({
            "model": str(model_name),
            "weight_dtype": str(weight_dtype),
            "manual": str(manual_cache_revision),
        })
        pack = graph.node("H3RelayInternalModelBundlePack", "H3Bundle")
        pack.set_input("kind", "h3")
        pack.set_input("model", loader.out(0))
        pack.set_input("cache_tag", tag)
        return {"result": (pack.out(0),), "expand": graph.finalize()}


class H3RelayLTXModelLoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_name": (folder_paths.get_filename_list("diffusion_models"), {"default": LTX_MODEL}),
                "vae_name": (folder_paths.get_filename_list("vae"), {"default": LTX_VAE}),
                "latent_2x_model_name": (folder_paths.get_filename_list("latent_upscale_models"), {
                    "default": LTX_UPSCALER,
                    "tooltip": "Learned 2x latent spatial expansion. This establishes the high-resolution target latent before diffusion refinement.",
                }),
                "text_encoder_name": (folder_paths.get_filename_list("text_encoders"), {"default": LTX_TEXT_ENCODER}),
                "distilled_lora": (folder_paths.get_filename_list("loras"), {
                    "default": LTX_DISTILLED_LORA,
                    "tooltip": "Adapts the LTX dev transformer for the fast low-step schedule. It does not perform the spatial upscale.",
                }),
                "distilled_strength": ("FLOAT", {"default": 1.0, "min": -4.0, "max": 4.0, "step": 0.05}),
                "pixel_upscale_ic_lora": (folder_paths.get_filename_list("loras"), {
                    "default": LTX_UPSCALE_LORA,
                    "tooltip": "Reference-conditioned IC-LoRA. It guides diffusion from the original low-resolution pixel video after the latent has been expanded 2x.",
                }),
                "pixel_upscale_ic_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05}),
                "weight_dtype": (["default", "fp8_e4m3fn", "fp8_e4m3fn_fast", "fp8_e5m2"], {"default": "default"}),
                "manual_cache_revision": _manual_cache_revision_input(),
            }
        }

    RETURN_TYPES = (MODEL_BUNDLE_TYPE,)
    RETURN_NAMES = ("ltx_model",)
    FUNCTION = "load"
    CATEGORY = CATEGORY + "/loaders"

    def load(
        self,
        model_name,
        vae_name,
        latent_2x_model_name,
        text_encoder_name,
        distilled_lora,
        distilled_strength,
        pixel_upscale_ic_lora,
        pixel_upscale_ic_strength,
        weight_dtype,
        manual_cache_revision,
    ):
        graph = GraphBuilder()
        model = graph.node("UNETLoader", "LTXModel")
        model.set_input("unet_name", str(model_name))
        model.set_input("weight_dtype", str(weight_dtype))
        distilled = graph.node("LoraLoaderModelOnly", "LTXDistilled")
        distilled.set_input("model", model.out(0))
        distilled.set_input("lora_name", str(distilled_lora))
        distilled.set_input("strength_model", float(distilled_strength))
        upscaler_lora = graph.node("LoraLoaderModelOnly", "LTXPixelUpscale")
        upscaler_lora.set_input("model", distilled.out(0))
        upscaler_lora.set_input("lora_name", str(pixel_upscale_ic_lora))
        upscaler_lora.set_input("strength_model", float(pixel_upscale_ic_strength))
        vae = graph.node("VAELoader", "LTXVAE")
        vae.set_input("vae_name", str(vae_name))
        upscale = graph.node("LatentUpscaleModelLoader", "LTXUpscaler")
        upscale.set_input("model_name", str(latent_2x_model_name))
        clip = graph.node("CLIPLoader", "LTXText")
        clip.set_input("clip_name", str(text_encoder_name))
        clip.set_input("type", "ltxv")
        clip.set_input("device", "default")
        tag = "ltx-stack:%s" % context._fingerprint({
            "model": str(model_name),
            "vae": str(vae_name),
            # Retain the v0.3 fingerprint field names so this UI rename does
            # not invalidate otherwise identical durable LTX artifacts.
            "upscale_model": str(latent_2x_model_name),
            "text_encoder": str(text_encoder_name),
            "distilled_lora": str(distilled_lora),
            "distilled_strength": float(distilled_strength),
            "upscale_lora": str(pixel_upscale_ic_lora),
            "upscale_strength": float(pixel_upscale_ic_strength),
            "weight_dtype": str(weight_dtype),
            "manual": str(manual_cache_revision),
        })
        pack = graph.node("H3RelayInternalModelBundlePack", "LTXBundle")
        pack.set_input("kind", "ltx")
        pack.set_input("model", upscaler_lora.out(0))
        pack.set_input("vae", vae.out(0))
        pack.set_input("upscale_model", upscale.out(0))
        pack.set_input("clip", clip.out(0))
        pack.set_input("cache_tag", tag)
        return {
            "result": (pack.out(0),),
            "expand": graph.finalize(),
        }


class H3RelayLTXModelAdapter:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL", {
                    "tooltip": "LTX diffusion model after any native LoRA, attention, or model-patch nodes.",
                }),
                "vae": ("VAE",),
                "latent_2x_model": ("LATENT_UPSCALE_MODEL", {
                    "tooltip": "Learned latent spatial upscaler used to create the 2x target latent.",
                }),
                "clip": ("CLIP", {
                    "tooltip": "LTX-compatible text encoder, normally the projected Gemma4 12B encoder.",
                }),
                "cache_identity": ("STRING", {
                    "default": "custom-ltx-v1",
                    "advanced": True,
                    "tooltip": (
                        "Stable identity for this custom native model chain. "
                        "Change it whenever an upstream checkpoint, LoRA, "
                        "strength, patch, VAE, latent upscaler, or CLIP changes."
                    ),
                }),
            }
        }

    RETURN_TYPES = (MODEL_BUNDLE_TYPE,)
    RETURN_NAMES = ("ltx_model",)
    FUNCTION = "pack"
    CATEGORY = CATEGORY + "/loaders"
    DESCRIPTION = "Pack native LTX MODEL, VAE, latent 2x upscaler, and CLIP outputs into one cache-aware H3 Relay LTX model bundle."

    def pack(self, model, vae, latent_2x_model, clip, cache_identity):
        identity = str(cache_identity).strip()
        if not identity:
            raise ValueError("H3 Relay custom LTX cache identity cannot be empty.")
        tag = "ltx-custom:%s" % context._fingerprint({
            "identity": identity,
            "model_type": "%s.%s" % (type(model).__module__, type(model).__qualname__),
            "vae_type": "%s.%s" % (type(vae).__module__, type(vae).__qualname__),
            "latent_2x_type": "%s.%s" % (
                type(latent_2x_model).__module__, type(latent_2x_model).__qualname__
            ),
            "clip_type": "%s.%s" % (type(clip).__module__, type(clip).__qualname__),
        })
        return ({
            "format": MODEL_BUNDLE_FORMAT,
            "kind": "ltx",
            "model": model,
            "vae": vae,
            "upscale_model": latent_2x_model,
            "clip": clip,
            "cache_tag": tag,
        },)


class H3RelayModelLoRA:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": (MODEL_BUNDLE_TYPE,),
                "lora_name": (folder_paths.get_filename_list("loras"),),
                "strength": ("FLOAT", {"default": 1.0, "min": -100.0, "max": 100.0, "step": 0.01}),
            }
        }

    RETURN_TYPES = (MODEL_BUNDLE_TYPE,)
    RETURN_NAMES = ("model",)
    FUNCTION = "apply"
    CATEGORY = CATEGORY + "/model"

    def apply(self, model, lora_name, strength):
        bundle = _model_bundle(model)
        graph = GraphBuilder()
        lora = graph.node("LoraLoaderModelOnly", "RelayLoRA")
        lora.set_input("model", bundle["model"])
        lora.set_input("lora_name", str(lora_name))
        lora.set_input("strength_model", float(strength))
        tag = "lora:%s" % context._fingerprint({
            "parent": str(bundle["cache_tag"]),
            "lora": str(lora_name),
            "strength": float(strength),
        })
        pack = graph.node("H3RelayInternalModelBundlePack", "PatchedBundle")
        pack.set_input("kind", str(bundle["kind"]))
        pack.set_input("model", lora.out(0))
        pack.set_input("cache_tag", tag)
        if bundle.get("vae") is not None:
            pack.set_input("vae", bundle["vae"])
        if bundle.get("upscale_model") is not None:
            pack.set_input("upscale_model", bundle["upscale_model"])
        if bundle.get("clip") is not None:
            pack.set_input("clip", bundle["clip"])
        return {"result": (pack.out(0),), "expand": graph.finalize()}


class H3RelayAttention:
    @classmethod
    def INPUT_TYPES(cls):
        choices = []
        for label, key in (
            ("comfy kitchen attention", "comfy_kitchen_int8"),
            ("pytorch attention", "pytorch"),
            ("sage attention", "sage"),
            ("sage attention 3", "sage3"),
        ):
            if attention_module.get_attention_function(key, None) is not None:
                choices.append(label)
        return {
            "required": {
                "model": (MODEL_BUNDLE_TYPE,),
                "attention": (choices,),
            }
        }

    RETURN_TYPES = (MODEL_BUNDLE_TYPE,)
    RETURN_NAMES = ("model",)
    FUNCTION = "apply"
    CATEGORY = CATEGORY + "/model"

    def apply(self, model, attention):
        bundle = _model_bundle(model)
        key = {
            "comfy kitchen attention": "comfy_kitchen_int8",
            "pytorch attention": "pytorch",
            "sage attention": "sage",
            "sage attention 3": "sage3",
        }.get(str(attention))
        function = attention_module.get_attention_function(key, None)
        if function is None:
            raise ValueError("H3 Relay attention backend is unavailable: %s" % attention)
        patched = bundle["model"].clone()
        patched.set_model_optimized_attention(function)
        tag = "attention:%s" % context._fingerprint({
            "parent": str(bundle["cache_tag"]),
            "attention": str(attention),
        })
        updated = dict(bundle)
        updated["model"] = patched
        updated["cache_tag"] = tag
        return (updated,)


class H3RelayInterpolationModelLoader:
    @classmethod
    def INPUT_TYPES(cls):
        models = folder_paths.get_filename_list("frame_interpolation")
        return {
            "required": {
                "model_name": (models, {"default": models[0] if models else ""}),
                "manual_cache_revision": _manual_cache_revision_input(),
            }
        }

    RETURN_TYPES = (INTERPOLATION_BUNDLE_TYPE,)
    RETURN_NAMES = ("interpolation",)
    FUNCTION = "load"
    CATEGORY = CATEGORY + "/loaders"

    def load(self, model_name, manual_cache_revision):
        graph = GraphBuilder()
        loader = graph.node("FrameInterpolationModelLoader", "InterpolationModel")
        loader.set_input("model_name", str(model_name))
        tag = "interpolation:%s" % context._fingerprint({
            "model": str(model_name),
            "manual": str(manual_cache_revision),
        })
        pack = graph.node("H3RelayInternalInterpolationBundlePack", "InterpolationBundle")
        pack.set_input("model", loader.out(0))
        pack.set_input("cache_tag", tag)
        return {"result": (pack.out(0),), "expand": graph.finalize()}


class H3RelayCacheManager:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "action": (["inspect", "prune_superseded"], {
                    "default": "inspect",
                    "tooltip": "Inspect is read-only. Prune removes only superseded immutable revisions while retaining current references.",
                }),
                "keep_revisions_per_shot": ("INT", {
                    "default": 2,
                    "min": 1,
                    "max": 20,
                    "tooltip": "Current protected revisions are always retained in addition to this recent-history count.",
                }),
                "budget_gb": ("FLOAT", {
                    "default": 100.0,
                    "min": 1.0,
                    "max": 10000.0,
                    "step": 1.0,
                    "tooltip": "Reporting threshold. Active/protected cache entries are never deleted merely to meet the budget.",
                }),
            }
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("cache_path", "status")
    FUNCTION = "manage"
    OUTPUT_NODE = True
    CATEGORY = CATEGORY + "/cache"
    DESCRIPTION = "Inspect H3 Relay's managed cache or prune superseded shot revisions without touching published output videos."

    def manage(self, action, keep_revisions_per_shot, budget_gb):
        removed = {
            "removed_bytes": 0,
            "removed_files": 0,
            "removed_revisions": 0,
        }
        if str(action) == "prune_superseded":
            removed = relay_cache.prune_superseded(keep_revisions_per_shot)
        after = relay_cache.cache_stats()
        budget = int(float(budget_gb) * 1024 ** 3)
        status = (
            "H3 Relay cache: %s across %d files and %d revision groups; "
            "removed %s / %d files / %d revisions; budget %s (%s)"
            % (
                relay_cache.format_bytes(after["bytes"]),
                after["files"],
                after["revision_groups"],
                relay_cache.format_bytes(removed["removed_bytes"]),
                removed["removed_files"],
                removed["removed_revisions"],
                relay_cache.format_bytes(budget),
                "over" if after["bytes"] > budget else "within",
            )
        )
        if str(action) == "inspect":
            status += "; inspect is read-only"
        return {
            "ui": {"text": [status]},
            "result": (relay_cache.cache_root(), status),
        }


class H3RelayMemoryRelease:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {}}

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("status",)
    FUNCTION = "release"
    OUTPUT_NODE = True
    CATEGORY = CATEGORY + "/internal"

    def release(self):
        from server import PromptServer

        PromptServer.instance.prompt_queue.set_flag("free_memory", True)
        return ("H3 Relay requested executor-cache and model release.",)


class H3RelaySequenceStart:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "run_name": ("STRING", {"default": "h3_relay_movie"}),
                "global_prompt": ("STRING", {"default": "", "multiline": True}),
                "width": ("INT", {"default": 832, "min": 32, "max": 2048, "step": 32}),
                "height": ("INT", {"default": 480, "min": 32, "max": 2048, "step": 32}),
                "h3_overlap_frames": (list(context.STEER_H3_CONTEXT_CHOICES), {
                    "default": context.STEER_H3_CONTEXT_FRAMES,
                    "tooltip": (
                        "Sequence-wide visual/audio continuation history. "
                        "H3 sliding history requires 17k+1 frame counts. "
                        "Larger values preserve more context but reduce each "
                        "continuation shot's delivered duration."
                    ),
                }),
                "sampler": (H3_SAMPLERS, {
                    "default": "euler",
                    "tooltip": "Native ComfyUI KSamplerSelect options.",
                }),
                "scheduler": (H3_SCHEDULERS, {
                    "default": "beta57",
                    "tooltip": (
                        "beta57 uses the exact H3 manual curve at 16 steps "
                        "and alpha 0.5 / beta 0.7 otherwise. Remaining values "
                        "come from ComfyUI BasicScheduler."
                    ),
                }),
                "spectrum_enabled": ("BOOLEAN", {
                    "default": True,
                    "tooltip": (
                        "Enable Spectrum independently. Supported single-call "
                        "samplers include Euler, res_multistep variants, and "
                        "ER-SDE; other samplers still run with Spectrum bypassed."
                    ),
                }),
            }
        }

    RETURN_TYPES = (context.STEER_SEQUENCE_TYPE, "STRING")
    RETURN_NAMES = ("sequence", "status")
    FUNCTION = "start"
    CATEGORY = CATEGORY

    def start(
        self,
        run_name,
        global_prompt,
        width,
        height,
        h3_overlap_frames,
        sampler,
        scheduler,
        spectrum_enabled,
    ):
        sampler = str(sampler)
        scheduler = str(scheduler)
        spectrum_enabled = bool(spectrum_enabled)
        h3_overlap_frames = int(h3_overlap_frames)
        if h3_overlap_frames not in context.STEER_H3_CONTEXT_CHOICES:
            raise ValueError(
                "H3 overlap must be one of %s frames." %
                (context.STEER_H3_CONTEXT_CHOICES,)
            )
        if sampler not in H3_SAMPLERS:
            raise ValueError("Unknown ComfyUI sampler: %s" % sampler)
        if scheduler not in H3_SCHEDULERS:
            raise ValueError("Unknown ComfyUI scheduler: %s" % scheduler)
        default_pairing = (
            sampler == "euler"
            and scheduler == "beta57"
            and spectrum_enabled
        )
        sequence, status = context.MiniMaxH3SteerSequenceStart().start(
            run_name,
            global_prompt,
            width,
            height,
            "",
            18,
            18,
            ("native_spectrum_euler_beta57"
             if default_pairing else "native_euler_beta"),
            "raw_h3",
        )
        sequence = dict(sequence)
        sequence["h3_context_frames"] = h3_overlap_frames
        sequence["h3_sampler"] = sampler
        sequence["h3_scheduler"] = scheduler
        sequence["h3_spectrum_enabled"] = spectrum_enabled
        if not default_pairing or h3_overlap_frames != context.STEER_H3_CONTEXT_FRAMES:
            sequence["generation_fingerprint"] = (
                "h3-hybrid-fl2va-ref2va-sampler-%s-scheduler-%s-"
                "spectrum-%d-sliding%d-raw_h3-v2"
                % (sampler, scheduler, int(spectrum_enabled), h3_overlap_frames)
            )
        status = status.replace(
            "18-frame H3 sliding history",
            "%d-frame H3 sliding history" % h3_overlap_frames,
        )
        return sequence, (
            "%s; sampler=%s; scheduler=%s; Spectrum %s"
            % (status, sampler, scheduler, "enabled" if spectrum_enabled else "disabled")
        )


class H3RelayGenerateShot(context.MiniMaxH3SteerableSegment):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "h3_model": (MODEL_BUNDLE_TYPE,),
                "sequence": (context.STEER_SEQUENCE_TYPE,),
                "prompt": ("STRING", {
                    "forceInput": True,
                    "tooltip": "Shot-specific direction. Connect a multiline prompt/text node.",
                }),
                "seed": ("INT", {"default": 424242, "min": 0, "max": context.MAX_SEED}),
                "duration_seconds": ("FLOAT", {
                    "default": 5.0,
                    "min": 1.0,
                    "max": 15.0,
                    "step": 0.25,
                    "tooltip": "Rounded upward to MiniMax H3's valid 5+17k frame grid.",
                }),
                "h3_steps": ("INT", {"default": 16, "min": 1, "max": 100}),
                "output_crf": ("INT", {
                    "default": 18,
                    "min": 0,
                    "max": 51,
                    "tooltip": "H.264 quality for this shot's cached review/assembly segment. This is not an H3 model parameter.",
                }),
                "ref_image_size": (["match", "max"], {"default": "match"}),
            },
            "optional": {
                "shot_id": ("STRING", {
                    "default": "",
                    "tooltip": (
                        "Optional stable sequence identifier. Leave blank to "
                        "assign shot_0001, shot_0002, and so on automatically."
                    ),
                }),
                "first_frame": ("IMAGE",),
                "last_frame": ("IMAGE",),
                "reference_image_1": ("IMAGE",),
                "reference_image_2": ("IMAGE",),
                "reference_image_3": ("IMAGE",),
                "reference_video": ("IMAGE",),
                "reference_video_audio": ("AUDIO",),
                "reference_audio": ("AUDIO",),
            },
        }

    FUNCTION = "generate_shot"
    CATEGORY = CATEGORY
    DESCRIPTION = "Generate and cache one native-resolution H3 shot with audio."

    def generate_shot(
        self,
        h3_model,
        sequence,
        prompt,
        seed,
        duration_seconds,
        h3_steps,
        output_crf,
        ref_image_size="match",
        shot_id="",
        first_frame=None,
        last_frame=None,
        reference_image_1=None,
        reference_image_2=None,
        reference_image_3=None,
        reference_video=None,
        reference_video_audio=None,
        reference_audio=None,
    ):
        sequence, model = _sequence_with_h3_model(sequence, h3_model)
        sequence = dict(sequence)
        sequence["relay_output_crf"] = int(output_crf)
        raw_frames = _duration_frames(duration_seconds)
        result = super().generate(
            sequence,
            prompt,
            shot_id,
            seed,
            raw_frames,
            h3_steps,
            ref_image_size,
            first_frame,
            last_frame,
            reference_image_1,
            reference_image_2,
            reference_image_3,
            reference_video,
            reference_video_audio,
            reference_audio,
            relay_model=model,
        )
        if isinstance(result, dict):
            return result
        updated, _, _, status = result
        updated, preview, mux_status = _raw_preview_with_audio(
            updated, output_crf
        )
        final_status = "%s; %s" % (status, mux_status)
        result_tuple = (
            updated,
            InputImpl.VideoFromFile(preview),
            preview,
            final_status,
        )
        return _video_ui(preview, final_status, result_tuple)


class H3RelayAcceptRaw(context.MiniMaxH3SteerAcceptRaw):
    CATEGORY = CATEGORY + "/internal"

    def accept(self, sequence, state, segment):
        output_crf = int(sequence.get("relay_output_crf", 18))
        updated, _, _, status = super().accept(sequence, state, segment)
        updated, preview, mux_status = _raw_preview_with_audio(
            updated, output_crf
        )
        return (
            updated,
            InputImpl.VideoFromFile(preview),
            preview,
            "%s; %s" % (status, mux_status),
        )


class H3RelayAcceptLTX:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "base_enhanced": (FINISH_TYPE,),
                "sequence": (context.STEER_SEQUENCE_TYPE,),
                "segment": (context.SEGMENT_TYPE,),
                "images": ("IMAGE",),
                "rolling_context": (context.LTX_ROLLING_CONTEXT_TYPE,),
                "cache_key": ("STRING", {"forceInput": True}),
                "output_crf": ("INT", {"default": 18, "min": 0, "max": 51}),
                "context_window_frames": ("INT", {"default": 193, "min": 65, "max": 4097, "step": 8}),
                "context_overlap_frames": ("INT", {"default": 64, "min": 0, "max": 1024, "step": 8}),
                "vae_temporal_tile_frames": ("INT", {"default": 128, "min": 32, "max": 4096, "step": 8}),
                "vae_temporal_overlap_frames": ("INT", {"default": 16, "min": 8, "max": 1024, "step": 8}),
            }
        }

    RETURN_TYPES = (FINISH_TYPE, "VIDEO", "STRING", "STRING")
    RETURN_NAMES = ("enhanced", "video", "video_path", "status")
    FUNCTION = "accept"
    CATEGORY = CATEGORY + "/internal"

    def accept(
        self,
        base_enhanced,
        sequence,
        segment,
        images,
        rolling_context,
        cache_key,
        output_crf,
        context_window_frames,
        context_overlap_frames,
        vae_temporal_tile_frames,
        vae_temporal_overlap_frames,
    ):
        _require_video_api()
        raw_sequence = _raw_sequence(sequence)
        index = len(raw_sequence["segments"])
        context_frames = int(rolling_context["context_frames"])
        original_frames = int(rolling_context["original_frames"])
        if int(images.shape[0]) != original_frames:
            raise ValueError(
                "H3 Relay LTX output contains %d frames; expected %d."
                % (int(images.shape[0]), original_frames)
            )
        if context_frames >= original_frames:
            raise ValueError("H3 Relay LTX context consumes the current shot.")

        revision = uuid.uuid4().hex
        paths = _stage_paths(raw_sequence["run_name"], "ltx", index, revision)
        os.makedirs(os.path.dirname(paths["full"]), exist_ok=True)
        os.makedirs(os.path.dirname(paths["segment"]), exist_ok=True)
        metadata = {
            "title": "H3 Relay LTX shot %d" % index,
            "h3_source_segment": str(segment["segment"]),
            "h3_context_prefix_frames": str(context_frames),
            "ltx_context_window_frames": str(context_window_frames),
            "ltx_context_overlap_frames": str(context_overlap_frames),
            "ltx_vae_temporal_tile_frames": str(vae_temporal_tile_frames),
            "ltx_vae_temporal_overlap_frames": str(vae_temporal_overlap_frames),
        }
        context._write_segment_video(
            images.detach().cpu().contiguous(),
            paths["full"],
            FPS,
            18,
            metadata=metadata,
        )
        delivered = images[context_frames:].detach().cpu().contiguous()
        audio = context._generated_audio({"segments": [segment]})
        context._write_steer_enhanced_segment(
            delivered,
            audio,
            paths["segment"],
            FPS,
            18,
            metadata,
        )
        output_crf = int(output_crf)
        delivery_path = paths["segment"]
        if output_crf != 18:
            delivery_path = _stage_paths(
                raw_sequence["run_name"], "ltx", index,
                "%s.crf%02d" % (revision, output_crf),
            )["segment"]
            _transcode_video(paths["segment"], delivery_path, output_crf)
        record = {
            "index": index,
            "id": str(segment["id"]),
            "revision": revision,
            "cache_key": str(cache_key),
            "raw_record": dict(raw_sequence["segments"][index - 1]),
            "full_segment": context._relative_output_path(paths["full"]),
            "full_segment_sha256": context._file_sha256(paths["full"]),
            "master_delivery_segment": context._relative_output_path(paths["segment"]),
            "master_delivery_segment_sha256": context._file_sha256(paths["segment"]),
            "delivery_segment": context._relative_output_path(delivery_path),
            "delivery_segment_sha256": context._file_sha256(delivery_path),
            "output_crf": output_crf,
            "context_frames": context_frames,
            "original_frames": original_frames,
            "delivered_frames": int(delivered.shape[0]),
            "fps": FPS,
            "generated_audio": str(segment["generated_audio"]),
            "context_window_frames": int(context_window_frames),
            "context_overlap_frames": int(context_overlap_frames),
            "vae_temporal_tile_frames": int(vae_temporal_tile_frames),
            "vae_temporal_overlap_frames": int(vae_temporal_overlap_frames),
        }
        updated = _append_ltx(base_enhanced, raw_sequence, record)
        context._atomic_json(paths["metadata"], {
            "format": "h3_relay_ltx_cache_v1",
            "cache_key": str(cache_key),
            "record": record,
        })
        context._atomic_json(paths["sequence"], updated)
        if context._relative_output_path(paths["segment"]).startswith(
                relay_cache.CACHE_SCHEME):
            relay_cache.maybe_prune_run(keep_per_shot=2)
        status = (
            "LTX enhanced shot %d: %d context + %d delivered frames; "
            "%d/%d diffusion window, %d/%d VAE tile -> %s"
            % (
                index,
                context_frames,
                int(delivered.shape[0]),
                int(context_window_frames),
                int(context_overlap_frames),
                int(vae_temporal_tile_frames),
                int(vae_temporal_overlap_frames),
                delivery_path,
            )
        )
        return updated, InputImpl.VideoFromFile(delivery_path), delivery_path, status


class H3RelayEnhanceShot:
    NEGATIVE_PROMPT = context.MiniMaxH3SteerableSegment.NEGATIVE_PROMPT

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "ltx_model": (MODEL_BUNDLE_TYPE,),
                "sequence": (context.STEER_SEQUENCE_TYPE,),
                "enhancement_prompt": ("STRING", {
                    "forceInput": True,
                    "tooltip": "LTX finishing direction. Connect a multiline prompt/text node.",
                }),
                "output_crf": ("INT", {"default": 18, "min": 0, "max": 51}),
                "context_window_frames": ("INT", {
                    "default": 193,
                    "min": 65,
                    "max": 4097,
                    "step": 8,
                    "tooltip": "Real pixel frames per diffusion window. 193 becomes 25 LTX latent frames.",
                }),
                "context_overlap_frames": ("INT", {
                    "default": 64,
                    "min": 0,
                    "max": 1024,
                    "step": 8,
                    "tooltip": "Real pixel-frame overlap. More overlap costs time but can soften window boundaries.",
                }),
                "vae_temporal_tile_frames": ("INT", {
                    "default": 128,
                    "min": 32,
                    "max": 4096,
                    "step": 8,
                    "tooltip": "Frames decoded per VAE temporal tile. Larger values reduce decode seams but use more VRAM.",
                }),
                "vae_temporal_overlap_frames": ("INT", {
                    "default": 16,
                    "min": 8,
                    "max": 1024,
                    "step": 8,
                }),
            },
            "optional": {
                "previous_enhanced": (FINISH_TYPE,),
            },
        }

    RETURN_TYPES = (FINISH_TYPE, "VIDEO", "STRING", "STRING")
    RETURN_NAMES = ("enhanced", "video", "video_path", "status")
    FUNCTION = "enhance"
    OUTPUT_NODE = True
    CATEGORY = CATEGORY
    DESCRIPTION = "Enhance one accepted raw H3 shot with LTX 2.5 at 2x resolution."

    def enhance(
        self,
        ltx_model,
        sequence,
        enhancement_prompt,
        output_crf,
        context_window_frames,
        context_overlap_frames,
        vae_temporal_tile_frames,
        vae_temporal_overlap_frames,
        previous_enhanced=None,
    ):
        _require_video_api()
        bundle = _model_bundle(ltx_model, "ltx")
        model = bundle["model"]
        vae = bundle.get("vae")
        upscale_model = bundle.get("upscale_model")
        clip = bundle.get("clip")
        if vae is None or upscale_model is None or clip is None:
            raise ValueError("H3 Relay LTX model bundle is missing VAE, upscaler, or text encoder.")
        (
            context_window_frames,
            context_overlap_frames,
            vae_temporal_tile_frames,
            vae_temporal_overlap_frames,
        ) = _validate_ltx_tiling(
            context_window_frames,
            context_overlap_frames,
            vae_temporal_tile_frames,
            vae_temporal_overlap_frames,
        )
        raw_sequence = _raw_sequence(sequence)
        base_finish = _finish_base(raw_sequence, previous_enhanced)
        state, shot, segment = _rebuild_state(raw_sequence)
        index = int(state["index"])
        previous_ltx = base_finish["ltx_segments"][-1] if base_finish["ltx_segments"] else None
        cache_contract = {
            "version": 1,
            "raw_revision": str(segment["revision"]),
            "raw_sha256": str(segment["segment_sha256"]),
            "previous_ltx_revision": str(previous_ltx["revision"]) if previous_ltx else "",
            "previous_ltx_sha256": str(previous_ltx["full_segment_sha256"]) if previous_ltx else "",
            "enhancement_prompt": str(enhancement_prompt),
            "seed": str(shot["seed"]),
            "model_cache_tag": str(bundle["cache_tag"]),
            "context_frames": LTX_CONTEXT_FRAMES,
            "context_steps": LTX_CONTEXT_STEPS,
            "context_window_frames": context_window_frames,
            "context_overlap_frames": context_overlap_frames,
            "vae_temporal_tile_frames": vae_temporal_tile_frames,
            "vae_temporal_overlap_frames": vae_temporal_overlap_frames,
        }
        cache_key = context._fingerprint(cache_contract)
        legacy_cache_key = context._fingerprint({
            **cache_contract,
            "enhanced_crf": int(output_crf),
        })
        lookup = _stage_paths(raw_sequence["run_name"], "ltx", index, "lookup")
        try:
            payload = context._read_json(lookup["metadata"])
            record = payload["record"]
        except (FileNotFoundError, KeyError, OSError, ValueError):
            record = None
        if (
            isinstance(record, dict)
            and payload.get("format") == "h3_relay_ltx_cache_v1"
            and str(payload.get("cache_key")) in {
                cache_key,
                legacy_cache_key if int(output_crf) == 18 else "",
            }
            and _valid_file(record.get("full_segment", ""), record.get("full_segment_sha256", ""))
            and _valid_file(
                record.get("master_delivery_segment", record.get("delivery_segment", "")),
                record.get("master_delivery_segment_sha256", record.get("delivery_segment_sha256", "")),
            )
        ):
            record = dict(record)
            requested_crf = int(output_crf)
            master_value = record.get(
                "master_delivery_segment", record["delivery_segment"]
            )
            master_hash = record.get(
                "master_delivery_segment_sha256",
                record["delivery_segment_sha256"],
            )
            master_path = context._absolute_output_path(master_value)
            if requested_crf == 18:
                delivery_path = master_path
            else:
                delivery_path = _stage_paths(
                    raw_sequence["run_name"], "ltx", index,
                    "%s.crf%02d" % (record["revision"], requested_crf),
                )["segment"]
                if not os.path.isfile(delivery_path):
                    _transcode_video(master_path, delivery_path, requested_crf)
            record.update({
                "master_delivery_segment": master_value,
                "master_delivery_segment_sha256": master_hash,
                "delivery_segment": context._relative_output_path(delivery_path),
                "delivery_segment_sha256": context._file_sha256(delivery_path),
                "output_crf": requested_crf,
            })
            updated = _append_ltx(base_finish, raw_sequence, record)
            context._atomic_json(lookup["metadata"], {
                "format": "h3_relay_ltx_cache_v1",
                "cache_key": cache_key,
                "record": record,
            })
            context._atomic_json(lookup["sequence"], updated)
            path = delivery_path
            status = (
                "reused cached LTX inference for shot %d; encoded output at CRF %d"
                % (index, requested_crf)
            )
            return _video_ui(
                path,
                status,
                (updated, InputImpl.VideoFromFile(path), path, status),
            )

        state["steer_cache_key"] = cache_key
        graph = GraphBuilder()
        previous_checkpoint = (
            str(state["segments"][-1]["checkpoint"]) if state.get("segments") else ""
        )
        rolling = graph.node("H3RelayInternalLTXRollingInput", "LTXRollingInput")
        rolling.set_input("current_segment", str(segment["segment"]))
        rolling.set_input("previous_checkpoint", previous_checkpoint)
        rolling.set_input("context_frames", LTX_CONTEXT_FRAMES)

        ltx_encode = graph.node("VAEEncode", "LTXEncode")
        ltx_encode.set_input("pixels", rolling.out(0))
        ltx_encode.set_input("vae", vae)
        upscale = graph.node("LTXVLatentUpsampler", "LTXUpscale")
        upscale.set_input("samples", ltx_encode.out(0))
        upscale.set_input("upscale_model", upscale_model)
        upscale.set_input("vae", vae)
        inject = graph.node("H3RelayInternalLTXRollingInject", "LTXContext")
        inject.set_input("latent", upscale.out(0))
        inject.set_input("run_name", str(raw_sequence["run_name"]))
        inject.set_input("shot_index", index)
        inject.set_input("context_latent_steps", LTX_CONTEXT_STEPS)

        windows = graph.node("LTXVContextWindows", "LTXWindows")
        for name, value in (
            ("model", model),
            ("context_length", context_window_frames),
            ("context_overlap", context_overlap_frames),
            ("context_schedule", "standard_uniform"),
            ("context_stride", 1),
            ("closed_loop", False),
            ("fuse_method", "pyramid"),
            ("freenoise", True),
            ("retain_first_frame", False),
            ("split_conds_to_windows", False),
        ):
            windows.set_input(name, value)

        positive = graph.node("CLIPTextEncode", "LTXPositive")
        positive.set_input("clip", clip)
        positive.set_input("text", str(enhancement_prompt))
        negative = graph.node("CLIPTextEncode", "LTXNegative")
        negative.set_input("clip", clip)
        negative.set_input("text", self.NEGATIVE_PROMPT)
        conditioning = graph.node("LTXVConditioning", "LTXConditioning")
        conditioning.set_input("positive", positive.out(0))
        conditioning.set_input("negative", negative.out(0))
        conditioning.set_input("frame_rate", 24.0)
        ic_params = graph.node("GetICLoRAParameters", "LTXICParameters")
        ic_params.set_input("iclora_model", model)
        guide = graph.node("LTXVAddGuide", "LTXGuide")
        for name, value in (
            ("positive", conditioning.out(0)),
            ("negative", conditioning.out(1)),
            ("vae", vae),
            ("latent", inject.out(0)),
            ("image", rolling.out(0)),
            ("frame_idx", 0),
            ("strength", 1.0),
            ("iclora_parameters", ic_params.out(0)),
        ):
            guide.set_input(name, value)
        noise = graph.node("RandomNoise", "LTXNoise")
        noise.set_input("noise_seed", int(shot["seed"]))
        guider = graph.node("LTXVDualCFGGuider", "LTXGuider")
        guider.set_input("model", windows.out(0))
        guider.set_input("positive", guide.out(0))
        guider.set_input("negative", guide.out(1))
        guider.set_input("video_cfg", 1.0)
        guider.set_input("audio_cfg", 1.0)
        sampler_select = graph.node("KSamplerSelect", "LTXSampler")
        sampler_select.set_input("sampler_name", "euler_ancestral")
        sigmas = graph.node("ManualSigmas", "LTXSigmas")
        sigmas.set_input("sigmas", "0.85, 0.7250, 0.4219, 0.0")
        sample = graph.node("SamplerCustomAdvanced", "LTXSample")
        sample.set_input("noise", noise.out(0))
        sample.set_input("guider", guider.out(0))
        sample.set_input("sampler", sampler_select.out(0))
        sample.set_input("sigmas", sigmas.out(0))
        sample.set_input("latent_image", guide.out(2))
        crop_guides = graph.node("LTXVCropGuides", "LTXCropGuides")
        crop_guides.set_input("positive", guide.out(0))
        crop_guides.set_input("negative", guide.out(1))
        crop_guides.set_input("latent", sample.out(0))
        checkpoint = graph.node("H3RelayInternalLTXRollingCheckpoint", "LTXCheckpoint")
        checkpoint.set_input("latent", crop_guides.out(2))
        checkpoint.set_input("run_name", str(raw_sequence["run_name"]))
        checkpoint.set_input("shot_index", index)
        checkpoint.set_input("context_latent_steps", LTX_CONTEXT_STEPS)
        decode = graph.node("VAEDecodeTiled", "LTXDecode")
        decode.set_input("samples", checkpoint.out(0))
        decode.set_input("vae", vae)
        decode.set_input("tile_size", 768)
        decode.set_input("overlap", 64)
        decode.set_input("temporal_size", vae_temporal_tile_frames)
        decode.set_input("temporal_overlap", vae_temporal_overlap_frames)
        crop = graph.node("H3RelayInternalLTXRollingCrop", "LTXCropPadding")
        crop.set_input("images", decode.out(0))
        crop.set_input("rolling_context", rolling.out(1))
        accept = graph.node("H3RelayInternalAcceptLTX", "AcceptLTX")
        accept.set_input("base_enhanced", base_finish)
        accept.set_input("sequence", raw_sequence)
        accept.set_input("segment", segment)
        accept.set_input("images", crop.out(0))
        accept.set_input("rolling_context", rolling.out(1))
        accept.set_input("cache_key", cache_key)
        accept.set_input("output_crf", int(output_crf))
        accept.set_input("context_window_frames", context_window_frames)
        accept.set_input("context_overlap_frames", context_overlap_frames)
        accept.set_input("vae_temporal_tile_frames", vae_temporal_tile_frames)
        accept.set_input("vae_temporal_overlap_frames", vae_temporal_overlap_frames)
        preview = graph.node("H3RelayInternalVideoOutput", "LTXPreview")
        preview.set_input("video_path", accept.out(2))
        return {
            "result": (accept.out(0), accept.out(1), accept.out(2), accept.out(3)),
            "expand": graph.finalize(),
        }


class H3RelayLoadFrames:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"video_path": ("STRING", {"forceInput": True})}}

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "load"
    CATEGORY = CATEGORY + "/internal"

    def load(self, video_path):
        path = context._absolute_output_path(str(video_path))
        return (context._decode_video_images(path),)


class H3RelayAcceptInterpolation:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "enhanced": (FINISH_TYPE,),
                "images": ("IMAGE",),
                "cache_key": ("STRING", {"forceInput": True}),
                "model_cache_tag": ("STRING", {"forceInput": True}),
                "multiplier": ("INT", {"default": 2, "min": 2, "max": 16}),
                "output_crf": ("INT", {"default": 18, "min": 0, "max": 51}),
            }
        }

    RETURN_TYPES = (FINISH_TYPE, "VIDEO", "STRING", "STRING")
    RETURN_NAMES = ("enhanced", "video", "video_path", "status")
    FUNCTION = "accept"
    CATEGORY = CATEGORY + "/internal"

    def accept(
        self,
        enhanced,
        images,
        cache_key,
        model_cache_tag,
        multiplier,
        output_crf,
    ):
        _require_video_api()
        finish = _finish(enhanced)
        index = len(finish["ltx_segments"])
        source = finish["ltx_segments"][index - 1]
        multiplier = int(multiplier)
        expected = multiplier * (int(source["original_frames"]) - 1) + 1
        if int(images.shape[0]) != expected:
            raise ValueError(
                "H3 Relay interpolation produced %d frames; expected %d."
                % (int(images.shape[0]), expected)
            )
        context_frames = int(source["context_frames"])
        prefix = multiplier * (context_frames - 1) + 1 if context_frames else 0
        if prefix >= expected:
            raise ValueError("H3 Relay interpolated context consumes the shot.")
        delivered = images[prefix:].detach().cpu().contiguous()
        revision = uuid.uuid4().hex
        paths = _stage_paths(finish["run_name"], "interpolated", index, revision)
        os.makedirs(os.path.dirname(paths["segment"]), exist_ok=True)
        raw_record = source["raw_record"]
        h3_segment = raw_record["h3_segment"]
        audio = context._generated_audio({"segments": [h3_segment]})
        fps = FPS * multiplier
        metadata = {
            "title": "H3 Relay interpolated shot %d" % index,
            "interpolation_model": str(model_cache_tag),
            "interpolation_multiplier": str(multiplier),
            "context_prefix_frames": str(prefix),
        }
        context._write_steer_enhanced_segment(
            delivered,
            audio,
            paths["segment"],
            fps,
            18,
            metadata,
        )
        output_crf = int(output_crf)
        delivery_path = paths["segment"]
        if output_crf != 18:
            delivery_path = _stage_paths(
                finish["run_name"], "interpolated", index,
                "%s.crf%02d" % (revision, output_crf),
            )["segment"]
            _transcode_video(paths["segment"], delivery_path, output_crf)
        record = {
            "index": index,
            "id": str(source["id"]),
            "revision": revision,
            "cache_key": str(cache_key),
            "raw_record": dict(raw_record),
            "source_ltx_revision": str(source["revision"]),
            "master_delivery_segment": context._relative_output_path(paths["segment"]),
            "master_delivery_segment_sha256": context._file_sha256(paths["segment"]),
            "delivery_segment": context._relative_output_path(delivery_path),
            "delivery_segment_sha256": context._file_sha256(delivery_path),
            "output_crf": output_crf,
            "delivered_frames": int(delivered.shape[0]),
            "fps": fps,
            "context_prefix_frames": prefix,
            "model_cache_tag": str(model_cache_tag),
            "multiplier": multiplier,
        }
        updated = _append_delivery(finish, record)
        context._atomic_json(paths["metadata"], {
            "format": "h3_relay_interpolation_cache_v1",
            "cache_key": str(cache_key),
            "record": record,
        })
        context._atomic_json(paths["sequence"], updated)
        if context._relative_output_path(paths["segment"]).startswith(
                relay_cache.CACHE_SCHEME):
            relay_cache.maybe_prune_run(keep_per_shot=2)
        status = (
            "interpolated shot %d to %dfps; removed %d context frames -> %s"
            % (index, fps, prefix, delivery_path)
        )
        return updated, InputImpl.VideoFromFile(delivery_path), delivery_path, status


def _video_frame_chunks(path: str, chunk_frames: int):
    if context.av is None or context.torch is None:
        raise RuntimeError("Chunked interpolation requires PyAV and PyTorch.")
    chunk_frames = int(chunk_frames)
    if chunk_frames < 2:
        raise ValueError("Interpolation chunks must contain at least two frames.")
    frames = []
    with context.av.open(path, mode="r") as container:
        streams = list(container.streams.video)
        if len(streams) != 1:
            raise ValueError(
                "H3 Relay interpolation source contains %d video streams; expected one."
                % len(streams)
            )
        for frame in container.decode(streams[0]):
            array = frame.to_ndarray(format="rgb24")
            tensor = context.torch.from_numpy(array).to(
                dtype=context.torch.float32
            ).div_(255.0)
            frames.append(tensor)
            if len(frames) == chunk_frames:
                yield context.torch.stack(frames, dim=0)
                frames = [frames[-1]]
    if len(frames) >= 2:
        yield context.torch.stack(frames, dim=0)
    elif not frames:
        raise ValueError("H3 Relay interpolation source contains no frames.")


def _open_video_encoder(
    path: str, fps: int, width: int, height: int, crf: int,
    metadata: dict[str, Any],
):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    context._safe_unlink(path)
    container = context.av.open(
        path, mode="w", options={"movflags": "use_metadata_tags+faststart"}
    )
    for key, value in metadata.items():
        if value is not None:
            container.metadata[str(key)] = str(value)
    stream = container.add_stream("libx264", rate=Fraction(int(fps), 1))
    stream.width = int(width)
    stream.height = int(height)
    stream.pix_fmt = "yuv420p"
    stream.options = {"crf": str(int(crf)), "preset": "medium"}
    return container, stream


def _encode_video_frame(container, stream, image) -> None:
    array = (
        context.torch.clamp(image[..., :3] * 255.0, 0, 255)
        .to(device="cpu", dtype=context.torch.uint8)
        .numpy()
    )
    frame = context.av.VideoFrame.from_ndarray(array, format="rgb24")
    for packet in stream.encode(frame):
        container.mux(packet)


def _mux_interpolation_audio(
    silent_path: str, final_path: str, audio: dict[str, Any],
    delivered_frames: int, fps: int,
) -> None:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("Chunked interpolation audio muxing requires ffmpeg.")
    wav = "%s.%s.audio.wav" % (final_path, uuid.uuid4().hex)
    temporary = "%s.%s.final.mp4" % (final_path, uuid.uuid4().hex)
    context._safe_unlink(wav)
    context._safe_unlink(temporary)
    try:
        context._write_wav(audio, wav)
        context._run_ffmpeg([
            ffmpeg, "-y", "-i", silent_path, "-i", wav,
            "-map", "0:v:0", "-map", "1:a:0", "-c:v", "copy",
            "-c:a", "aac", "-b:a", "256k",
            "-t", "%.9f" % (int(delivered_frames) / float(fps)),
            "-map_metadata", "0", "-movflags", "use_metadata_tags+faststart",
            temporary,
        ])
        os.replace(temporary, final_path)
    finally:
        context._safe_unlink(wav)
        context._safe_unlink(temporary)
        context._safe_unlink(silent_path)


def _chunked_interpolate(
    finish: dict[str, Any], interpolation_model: Any, model_cache_tag: str,
    multiplier: int, output_crf: int, chunk_frames: int, cache_key: str,
) -> tuple[dict[str, Any], str, str]:
    finish = _finish(finish)
    index = len(finish["ltx_segments"])
    source = finish["ltx_segments"][index - 1]
    source_path = context._absolute_output_path(str(source["full_segment"]))
    multiplier = int(multiplier)
    chunk_frames = int(chunk_frames)
    expected = multiplier * (int(source["original_frames"]) - 1) + 1
    context_frames = int(source["context_frames"])
    prefix = multiplier * (context_frames - 1) + 1 if context_frames else 0
    delivered_frames = expected - prefix
    if delivered_frames < 1:
        raise ValueError("H3 Relay interpolated context consumes the shot.")

    revision = uuid.uuid4().hex
    paths = _stage_paths(finish["run_name"], "interpolated", index, revision)
    os.makedirs(os.path.dirname(paths["segment"]), exist_ok=True)
    silent = "%s.%s.video.mp4" % (paths["segment"], uuid.uuid4().hex)
    metadata = {
        "title": "H3 Relay interpolated shot %d" % index,
        "interpolation_model": str(model_cache_tag),
        "interpolation_multiplier": str(multiplier),
        "context_prefix_frames": str(prefix),
        "interpolation_chunk_frames": str(chunk_frames),
    }
    container = None
    stream = None
    full_output_index = 0
    source_frames = 0
    written_frames = 0
    try:
        for chunk_index, images in enumerate(
                _video_frame_chunks(source_path, chunk_frames)):
            source_frames += int(images.shape[0]) - (1 if chunk_index else 0)
            interpolated = FrameInterpolate.execute(
                interpolation_model, images, multiplier
            )[0]
            if chunk_index:
                interpolated = interpolated[1:]
            if container is None:
                height = int(interpolated.shape[1])
                width = int(interpolated.shape[2])
                container, stream = _open_video_encoder(
                    silent, FPS * multiplier, width, height, 18, metadata
                )
            for image in interpolated:
                if full_output_index >= prefix:
                    _encode_video_frame(container, stream, image)
                    written_frames += 1
                full_output_index += 1
            del interpolated
            del images
        if container is None or stream is None:
            raise ValueError("H3 Relay interpolation produced no frame chunks.")
        for packet in stream.encode():
            container.mux(packet)
        container.close()
        container = None
    except Exception:
        if container is not None:
            container.close()
        context._safe_unlink(silent)
        raise
    if source_frames != int(source["original_frames"]):
        context._safe_unlink(silent)
        raise ValueError(
            "Chunked interpolation decoded %d source frames; expected %d."
            % (source_frames, int(source["original_frames"]))
        )
    if full_output_index != expected or written_frames != delivered_frames:
        context._safe_unlink(silent)
        raise ValueError(
            "Chunked interpolation produced %d/%d full and %d/%d delivered frames."
            % (full_output_index, expected, written_frames, delivered_frames)
        )

    raw_record = source["raw_record"]
    h3_segment = raw_record["h3_segment"]
    audio = context._generated_audio({"segments": [h3_segment]})
    _mux_interpolation_audio(
        silent, paths["segment"], audio, delivered_frames, FPS * multiplier
    )
    output_crf = int(output_crf)
    delivery_path = paths["segment"]
    if output_crf != 18:
        delivery_path = _stage_paths(
            finish["run_name"], "interpolated", index,
            "%s.crf%02d" % (revision, output_crf),
        )["segment"]
        _transcode_video(paths["segment"], delivery_path, output_crf)
    record = {
        "index": index,
        "id": str(source["id"]),
        "revision": revision,
        "cache_key": str(cache_key),
        "raw_record": dict(raw_record),
        "source_ltx_revision": str(source["revision"]),
        "master_delivery_segment": context._relative_output_path(paths["segment"]),
        "master_delivery_segment_sha256": context._file_sha256(paths["segment"]),
        "delivery_segment": context._relative_output_path(delivery_path),
        "delivery_segment_sha256": context._file_sha256(delivery_path),
        "output_crf": output_crf,
        "delivered_frames": delivered_frames,
        "fps": FPS * multiplier,
        "context_prefix_frames": prefix,
        "model_cache_tag": str(model_cache_tag),
        "multiplier": multiplier,
        "chunk_frames": chunk_frames,
    }
    updated = _append_delivery(finish, record)
    context._atomic_json(paths["metadata"], {
        "format": "h3_relay_interpolation_cache_v1",
        "cache_key": str(cache_key),
        "record": record,
    })
    context._atomic_json(paths["sequence"], updated)
    if context._relative_output_path(paths["segment"]).startswith(
            relay_cache.CACHE_SCHEME):
        relay_cache.maybe_prune_run(keep_per_shot=2)
    status = (
        "interpolated shot %d to %dfps in %d-frame chunks; removed %d "
        "context frames -> %s"
        % (index, FPS * multiplier, chunk_frames, prefix, delivery_path)
    )
    return updated, delivery_path, status


class H3RelayInterpolateShot:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "interpolation": (INTERPOLATION_BUNDLE_TYPE,),
                "enhanced": (FINISH_TYPE,),
                "multiplier": ("INT", {"default": 2, "min": 2, "max": 16}),
                "output_crf": ("INT", {"default": 18, "min": 0, "max": 51}),
                "chunk_frames": ("INT", {
                    "default": 48,
                    "min": 8,
                    "max": 256,
                    "step": 8,
                    "tooltip": (
                        "Temporal source frames processed per interpolation "
                        "chunk. Adjacent chunks share one exact boundary frame."
                    ),
                }),
            }
        }

    RETURN_TYPES = (FINISH_TYPE, "VIDEO", "STRING", "STRING")
    RETURN_NAMES = ("enhanced", "video", "video_path", "status")
    FUNCTION = "interpolate"
    OUTPUT_NODE = True
    CATEGORY = CATEGORY
    DESCRIPTION = "Interpolate the latest LTX result with ComfyUI's core frame-interpolation runtime."

    def interpolate(
        self,
        interpolation,
        enhanced,
        multiplier,
        output_crf,
        chunk_frames,
    ):
        _require_video_api()
        interpolation_bundle = _interpolation_bundle(interpolation)
        interpolation_model = interpolation_bundle["model"]
        model_cache_tag = str(interpolation_bundle["cache_tag"])
        chunk_frames = int(chunk_frames)
        if chunk_frames < 8 or chunk_frames > 256 or chunk_frames % 8:
            raise ValueError(
                "H3 Relay interpolation chunk_frames must be a multiple of 8 "
                "between 8 and 256."
            )
        finish = _finish(enhanced)
        index = len(finish["ltx_segments"])
        if index < 1:
            raise ValueError("H3 Relay interpolation requires an LTX result.")
        if len(finish["delivery_segments"]) != index - 1:
            raise ValueError(
                "Interpolate each accepted LTX shot in order before continuing."
            )
        source = finish["ltx_segments"][index - 1]
        cache_contract = {
            "version": 1,
            "ltx_revision": str(source["revision"]),
            "ltx_sha256": str(source["full_segment_sha256"]),
            "model_cache_tag": str(model_cache_tag),
            "multiplier": int(multiplier),
        }
        cache_key = context._fingerprint(cache_contract)
        legacy_cache_key = context._fingerprint({
            **cache_contract,
            "output_crf": int(output_crf),
        })
        lookup = _stage_paths(finish["run_name"], "interpolated", index, "lookup")
        try:
            payload = context._read_json(lookup["metadata"])
            record = payload["record"]
        except (FileNotFoundError, KeyError, OSError, ValueError):
            record = None
        if (
            isinstance(record, dict)
            and payload.get("format") == "h3_relay_interpolation_cache_v1"
            and str(payload.get("cache_key")) in {
                cache_key,
                legacy_cache_key if int(output_crf) == 18 else "",
            }
            and _valid_file(
                record.get("master_delivery_segment", record.get("delivery_segment", "")),
                record.get("master_delivery_segment_sha256", record.get("delivery_segment_sha256", "")),
            )
        ):
            record = dict(record)
            requested_crf = int(output_crf)
            master_value = record.get(
                "master_delivery_segment", record["delivery_segment"]
            )
            master_hash = record.get(
                "master_delivery_segment_sha256",
                record["delivery_segment_sha256"],
            )
            master_path = context._absolute_output_path(master_value)
            if requested_crf == 18:
                path = master_path
            else:
                path = _stage_paths(
                    finish["run_name"], "interpolated", index,
                    "%s.crf%02d" % (record["revision"], requested_crf),
                )["segment"]
                if not os.path.isfile(path):
                    _transcode_video(master_path, path, requested_crf)
            record.update({
                "master_delivery_segment": master_value,
                "master_delivery_segment_sha256": master_hash,
                "delivery_segment": context._relative_output_path(path),
                "delivery_segment_sha256": context._file_sha256(path),
                "output_crf": requested_crf,
            })
            updated = _append_delivery(finish, record)
            context._atomic_json(lookup["metadata"], {
                "format": "h3_relay_interpolation_cache_v1",
                "cache_key": cache_key,
                "record": record,
            })
            context._atomic_json(lookup["sequence"], updated)
            status = (
                "reused cached interpolation for shot %d; encoded output at CRF %d"
                % (index, requested_crf)
            )
            return _video_ui(
                path,
                status,
                (updated, InputImpl.VideoFromFile(path), path, status),
            )

        updated, path, status = _chunked_interpolate(
            finish,
            interpolation_model,
            model_cache_tag,
            int(multiplier),
            int(output_crf),
            chunk_frames,
            cache_key,
        )
        return _video_ui(
            path,
            status,
            (updated, InputImpl.VideoFromFile(path), path, status),
        )


class H3RelayAssemble:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "enhanced": (FINISH_TYPE,),
                "output_stage": (["auto", "interpolated", "ltx"], {"default": "auto"}),
                "filename": ("STRING", {"default": "h3_relay_movie_%date:yyyy-MM-dd_HH-mm-ss%"}),
                "audio_bitrate": ("INT", {"default": 256, "min": 64, "max": 512}),
            }
        }

    RETURN_TYPES = ("VIDEO", "STRING", "STRING")
    RETURN_NAMES = ("video", "video_path", "status")
    FUNCTION = "assemble"
    OUTPUT_NODE = True
    CATEGORY = CATEGORY

    def assemble(self, enhanced, output_stage, filename, audio_bitrate):
        _require_video_api()
        finish = _finish(enhanced)
        ltx = finish["ltx_segments"]
        delivery = finish["delivery_segments"]
        if output_stage == "auto":
            output_stage = "interpolated" if len(delivery) == len(ltx) else "ltx"
        selected = delivery if output_stage == "interpolated" else ltx
        if not selected:
            raise ValueError("H3 Relay assembly has no finished shots.")
        if output_stage == "interpolated" and len(delivery) != len(ltx):
            raise ValueError("Interpolate every LTX shot before interpolated assembly.")
        fps_values = {int(item["fps"]) for item in selected}
        if len(fps_values) != 1:
            raise ValueError("H3 Relay assembly requires one consistent frame rate.")
        raw_sequence = _raw_sequence(finish["raw_sequence"])
        if len(selected) != len(raw_sequence["shots"]):
            raise ValueError("H3 Relay assembly is missing one or more finished shots.")
        records = []
        for item in selected:
            raw_record = item["raw_record"]
            records.append({
                "index": int(item["index"]),
                "id": str(item["id"]),
                "revision": str(item["revision"]),
                "h3_segment": dict(raw_record["h3_segment"]),
                "enhanced_segment": str(item["delivery_segment"]),
                "enhanced_segment_sha256": str(item["delivery_segment_sha256"]),
                "enhanced_frames": int(item["delivered_frames"]),
                "enhanced_fps": int(item["fps"]),
                "context_prefix_frames": int(item.get("context_prefix_frames", 0)),
                "cache_key": str(item.get("cache_key", "")),
            })
        sequence = dict(raw_sequence)
        sequence["segments"] = records
        sequence["enhanced_fps"] = next(iter(fps_values))
        sequence["output_profile"] = "enhanced_ltx_rife"
        sequence["total_enhanced_frames"] = sum(
            int(item["enhanced_frames"]) for item in records
        )
        result = context.MiniMaxH3SteerAssemble().assemble(
            sequence, filename, int(audio_bitrate)
        )
        path, status = result["result"]
        return {
            "ui": result["ui"],
            "result": (InputImpl.VideoFromFile(path), path, status),
        }


NODE_CLASS_MAPPINGS = {
    "H3RelayH3HybridModelLoader": H3RelayH3HybridModelLoader,
    "H3RelayH3ModelLoader": H3RelayH3ModelLoader,
    "H3RelayLTXModelLoader": H3RelayLTXModelLoader,
    "H3RelayLTXModelAdapter": H3RelayLTXModelAdapter,
    "H3RelayInterpolationModelLoader": H3RelayInterpolationModelLoader,
    "H3RelayCacheManager": H3RelayCacheManager,
    "H3RelayModelLoRA": H3RelayModelLoRA,
    "H3RelayAttention": H3RelayAttention,
    "H3RelaySequenceStart": H3RelaySequenceStart,
    "H3RelayGenerateShot": H3RelayGenerateShot,
    "H3RelayEnhanceShot": H3RelayEnhanceShot,
    "H3RelayInterpolateShot": H3RelayInterpolateShot,
    "H3RelayAssemble": H3RelayAssemble,
    "H3RelayInternalHybridLoader": MiniMaxH3HybridLoader,
    "H3RelayInternalModelBundlePack": H3RelayModelBundlePack,
    "H3RelayInternalInterpolationBundlePack": H3RelayInterpolationBundlePack,
    "H3RelayInternalSpectrum": SpectrumApplyMiniMaxH3,
    "H3RelayInternalChainContext": context.MiniMaxH3ChainContext,
    "H3RelayInternalLoopTrim": context_nodes.MiniMaxH3LoopTrim,
    "H3RelayInternalSegmentSave": context.MiniMaxH3ChainSegmentSave,
    "H3RelayInternalAcceptRaw": H3RelayAcceptRaw,
    "H3RelayInternalVideoOutput": context.MiniMaxH3ChainVideoOutput,
    "H3RelayInternalSegmentPaths": context.MiniMaxH3SteerSegmentPaths,
    "H3RelayInternalLTXRollingInput": context.MiniMaxH3LTXRollingInput,
    "H3RelayInternalLTXRollingInject": context.MiniMaxH3LTXRollingInject,
    "H3RelayInternalLTXRollingCheckpoint": context.MiniMaxH3LTXRollingCheckpoint,
    "H3RelayInternalLTXRollingCrop": context.MiniMaxH3LTXRollingCrop,
    "H3RelayInternalAcceptLTX": H3RelayAcceptLTX,
    "H3RelayInternalLoadFrames": H3RelayLoadFrames,
    "H3RelayInternalAcceptInterpolation": H3RelayAcceptInterpolation,
    "H3RelayInternalMemoryRelease": H3RelayMemoryRelease,
    "H3RelayInternalRestoreRawSequence": H3RelayRestoreRawSequence,
    "H3RelayInternalRestoreEnhanced": H3RelayRestoreEnhanced,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "H3RelayH3HybridModelLoader": "H3 Relay · H3 Hybrid Model Loader",
    "H3RelayH3ModelLoader": "H3 Relay · H3 Model Loader",
    "H3RelayLTXModelLoader": "H3 Relay · LTX Upscale Model Loader",
    "H3RelayLTXModelAdapter": "H3 Relay · Pack LTX Model",
    "H3RelayInterpolationModelLoader": "H3 Relay · Interpolation Model Loader",
    "H3RelayCacheManager": "H3 Relay · Cache Manager",
    "H3RelayModelLoRA": "H3 Relay · Apply Model LoRA",
    "H3RelayAttention": "H3 Relay · Attention Backend",
    "H3RelaySequenceStart": "H3 Relay · Sequence Start",
    "H3RelayGenerateShot": "H3 Relay · Generate Shot",
    "H3RelayEnhanceShot": "H3 Relay · LTX 2× Enhance",
    "H3RelayInterpolateShot": "H3 Relay · Interpolate",
    "H3RelayAssemble": "H3 Relay · Assemble",
}
