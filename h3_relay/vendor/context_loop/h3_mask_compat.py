"""Capability-aware static compatibility for ComfyUI MiniMax H3 masks."""

from __future__ import annotations

import inspect
import logging

from . import h3_mask_static as static


_LOG = logging.getLogger("minimax_h3_context_loop.masked_prefix")
_MARKER = "_h3_motion_context_pr15375_compat_v2"


def _mark(fn):
    try:
        setattr(fn, _MARKER, True)
    except Exception:
        pass
    return fn


def _is_ours(fn):
    return bool(getattr(fn, _MARKER, False))


def _signature_has(fn, *names):
    try:
        params = inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return False
    return all(name in params for name in names)


def capability_status():
    import comfy.model_base as model_base
    import comfy.ldm.minimax.model as h3m

    cls = getattr(model_base, "MiniMaxH3", None)
    process = getattr(cls, "process_denoise_mask", None) if cls else None
    scale = getattr(cls, "scale_latent_inpaint", None) if cls else None
    process_native = bool(
        cls and "process_denoise_mask" in cls.__dict__
        and callable(process) and not _is_ours(process)
    )
    scale_native = bool(
        cls and "scale_latent_inpaint" in cls.__dict__
        and callable(scale) and not _is_ours(scale)
    )
    forward = getattr(getattr(h3m, "MiniMaxH3Model", None), "forward", None)
    inner = getattr(getattr(h3m, "MiniMaxH3Model", None), "_forward", None)
    final = getattr(getattr(h3m, "FinalLayer", None), "forward", None)
    indicators = {
        "mask_row_values": callable(getattr(h3m, "mask_row_values", None)),
        "mod_row": callable(getattr(h3m, "_mod_row", None)),
        "forward_masks": callable(forward) and _signature_has(
            forward, "denoise_mask", "audio_denoise_mask"
        ),
        "inner_masks": callable(inner) and _signature_has(
            inner, "denoise_mask", "audio_denoise_mask"
        ),
        "final_layer": callable(final),
    }
    complete = all(indicators.values())
    ours = bool(
        callable(forward) and callable(inner)
        and _is_ours(forward) and _is_ours(inner)
    )
    return {
        "process_denoise_mask_native": process_native,
        "process_denoise_mask_compat": bool(
            callable(process) and _is_ours(process)
        ),
        "scale_latent_inpaint_native": scale_native,
        "scale_latent_inpaint_compat": bool(
            callable(scale) and _is_ours(scale)
        ),
        "mask_engine_complete": complete,
        "mask_engine_native": bool(complete and not ours),
        "mask_engine_compat": ours,
        "mask_engine_indicators": indicators,
    }


def _install_engine_compat(h3m):
    functions = (
        static.mask_row_values,
        static.mod_row,
        static.mod_scale_shift,
        static.mod_gate,
        static.final_forward,
        static.h3_forward,
        static.h3_inner_forward,
    )
    for function in functions:
        _mark(function)
    h3m.mask_row_values = static.mask_row_values
    h3m._mod_row = static.mod_row
    h3m._mod_scale_shift = static.mod_scale_shift
    h3m._mod_gate = static.mod_gate
    h3m.FinalLayer.forward = static.final_forward
    h3m.MiniMaxH3Model.forward = static.h3_forward
    h3m.MiniMaxH3Model._forward = static.h3_inner_forward


def _install_model_base_hooks(_model_base):
    _mark(static.process_denoise_mask)
    _mark(static.scale_latent_inpaint)
    return static.process_denoise_mask, static.scale_latent_inpaint


def ensure_h3_mask_compat():
    import comfy.model_base as model_base
    import comfy.ldm.minimax.model as h3m

    cls = getattr(model_base, "MiniMaxH3", None)
    if cls is None:
        raise RuntimeError("h3_masked_prefix: MiniMaxH3 model class not found.")
    before = capability_status()
    indicators = before["mask_engine_indicators"]
    characteristic = [
        indicators["mask_row_values"],
        indicators["mod_row"],
        indicators["forward_masks"],
        indicators["inner_masks"],
    ]
    if any(characteristic) and not all(characteristic) and not before["mask_engine_compat"]:
        raise RuntimeError(
            "h3_masked_prefix: partial native H3 AV-mask engine detected. "
            "Update this node pack or ComfyUI."
        )
    if not before["mask_engine_complete"]:
        _install_engine_compat(h3m)
        _LOG.info("H3 PR #15375 diffusion-mask compatibility enabled")
    need_process = not (
        "process_denoise_mask" in cls.__dict__
        and callable(getattr(cls, "process_denoise_mask", None))
    )
    need_scale = not (
        "scale_latent_inpaint" in cls.__dict__
        and callable(getattr(cls, "scale_latent_inpaint", None))
    )
    if need_process or need_scale:
        process_fn, scale_fn = _install_model_base_hooks(model_base)
        if need_process:
            cls.process_denoise_mask = process_fn
        if need_scale:
            cls.scale_latent_inpaint = scale_fn
    after = capability_status()
    ready = (
        after["mask_engine_complete"]
        and (after["process_denoise_mask_native"] or after["process_denoise_mask_compat"])
        and (after["scale_latent_inpaint_native"] or after["scale_latent_inpaint_compat"])
    )
    if not ready:
        raise RuntimeError("H3 AV-mask compatibility is incomplete after patching.")
    return True


def ensure_h3_guide_engine_compat():
    import comfy.ldm.minimax.model as h3m

    before = capability_status()
    if not before["mask_engine_complete"]:
        _install_engine_compat(h3m)
        _LOG.info("H3 packed-guide engine compatibility enabled")
    if not capability_status()["mask_engine_complete"]:
        raise RuntimeError(
            "H3 packed-guide engine compatibility is incomplete after patching."
        )
    return True


def is_ready():
    try:
        status = capability_status()
    except Exception:
        return False
    return bool(
        status["mask_engine_complete"]
        and (status["process_denoise_mask_native"] or status["process_denoise_mask_compat"])
        and (status["scale_latent_inpaint_native"] or status["scale_latent_inpaint_compat"])
    )
