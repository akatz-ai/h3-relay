"""Lazy AV denoise-mask payload compatibility for H3 masked continuation.

Normal guide continuation never installs this wrapper. Native ComfyUI support
always wins; the wrapper only adds the two AV mask conditions when the running
MiniMaxH3 implementation does not already expose PR #15375-equivalent payload
handling.

Adapted from seitanism/ComfyUI-H3-Motion-Context-MultiRef (GPL-3.0).
"""

from __future__ import annotations

import functools
import inspect
import logging
import types

import torch

import comfy.conds
import comfy.model_base as model_base
import comfy.utils as utils


_LOG = logging.getLogger("minimax_h3_context_loop.masked_prefix")
# Shared with seitanism's source pack. Ownership remains path-specific through
# `_is_ours`, while `_is_compatible_wrapper` lets a sibling copy stand down.
_MARKER = "_h3_existing_video_av_mask_payload_compat_v2"


def _walk_wrapped(fn):
    seen = set()
    while fn is not None and id(fn) not in seen:
        seen.add(id(fn))
        yield fn
        fn = getattr(fn, "__wrapped__", None)


def _is_ours(fn):
    code = getattr(fn, "__code__", None)
    return bool(
        getattr(fn, _MARKER, False)
        and code is not None
        and code.co_name == "wrapper"
        and code.co_filename == __file__
    )


def _is_compatible_wrapper(fn):
    code = getattr(fn, "__code__", None)
    return bool(
        getattr(fn, _MARKER, False)
        and code is not None
        and code.co_name == "wrapper"
    )


def _signature_has(fn, *names):
    try:
        params = inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return False
    return all(name in params for name in names)


def _code_strings(code):
    """Yield strings/names from a function and nested code objects."""
    if not isinstance(code, types.CodeType):
        return
    yield from code.co_names

    def walk_const(value):
        if isinstance(value, str):
            yield value
        elif isinstance(value, types.CodeType):
            yield from _code_strings(value)
        elif isinstance(value, (tuple, list, set, frozenset)):
            for item in value:
                yield from walk_const(item)

    for value in code.co_consts:
        yield from walk_const(value)


def _function_mentions_native_payload(fn):
    for item in _walk_wrapped(fn):
        if _is_ours(item):
            continue
        strings = set(_code_strings(getattr(item, "__code__", None)) or ())
        if "denoise_mask" in strings and "audio_denoise_mask" in strings:
            return True
    return False


def _native_h3_mask_hooks(cls):
    if cls is None:
        return False
    process_step = cls.__dict__.get("process_timestep")
    process_mask = cls.__dict__.get("process_denoise_mask")
    scale = cls.__dict__.get("scale_latent_inpaint")
    return bool(
        callable(process_step)
        and _signature_has(process_step, "denoise_mask", "audio_denoise_mask")
        and callable(process_mask)
        and callable(scale)
    )


def _native_av_mask_payload(cls, fn):
    return bool(
        fn and (
            _function_mentions_native_payload(fn)
            or _native_h3_mask_hooks(cls)
        )
    )


def capability_status():
    cls = getattr(model_base, "MiniMaxH3", None)
    fn = getattr(cls, "extra_conds", None) if cls is not None else None
    direct = bool(fn and _function_mentions_native_payload(fn))
    native_hooks = _native_h3_mask_hooks(cls)
    native = bool(fn and (direct or native_hooks))
    return {
        "available": fn is not None,
        "native_av_mask_payload": native,
        "native_payload_direct": direct,
        "native_h3_mask_hooks": native_hooks,
        "wrapper_present": bool(
            fn and any(_is_compatible_wrapper(item)
                       for item in _walk_wrapped(fn))),
    }


def _add_av_mask_conditions(out, kwargs):
    if not isinstance(out, dict):
        return
    have_video = "denoise_mask" in out
    have_audio = "audio_denoise_mask" in out
    if have_video and have_audio:
        return

    denoise_mask = kwargs.get("denoise_mask")
    latent_shapes = kwargs.get("latent_shapes")
    if denoise_mask is None or latent_shapes is None or len(latent_shapes) < 2:
        return

    masks = utils.unpack_latents(denoise_mask, latent_shapes)
    if len(masks) < 2:
        return
    if not have_video and torch.amin(masks[0]).item() < 1.0 - 1e-3:
        out["denoise_mask"] = comfy.conds.CONDRegular(
            masks[0][:, :1].clone())
    if not have_audio and torch.amin(masks[1]).item() < 1.0 - 1e-3:
        out["audio_denoise_mask"] = comfy.conds.CONDRegular(
            masks[1][:, :1].clone())


def _make_wrapper(base):
    @functools.wraps(base, updated=())
    def wrapper(self, **kwargs):
        out = base(self, **kwargs)
        _add_av_mask_conditions(out, kwargs)
        return out

    setattr(wrapper, _MARKER, True)
    return wrapper


def ensure_av_mask_payload_compat():
    cls = getattr(model_base, "MiniMaxH3", None)
    if cls is None or not hasattr(cls, "extra_conds"):
        raise RuntimeError(
            "h3_masked_prefix: MiniMaxH3.extra_conds not found.")
    current = cls.extra_conds
    if _native_av_mask_payload(cls, current):
        return True
    if any(_is_compatible_wrapper(item) for item in _walk_wrapped(current)):
        return True
    cls.extra_conds = _make_wrapper(current)
    _LOG.info(
        "h3_masked_prefix: PR #15375 AV-mask payload compatibility enabled")
    return True
