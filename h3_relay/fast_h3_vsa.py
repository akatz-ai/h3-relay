# SPDX-License-Identifier: GPL-3.0-only
"""Relay-owned FastH3 VSA adapter for the public comfy-kitchen runtime.

FastH3 is distilled against FastVideo's VSA layout.  This module intentionally
implements only that checkpoint contract: 4x4x4 video cubes, exact packed
conditioning, ten-percent routed video blocks, no pooled tail, and the learned
``to_gate_compress`` coarse branch.  It does not expose generic Sol-Attention
controls and never silently substitutes dense attention.

The implementation targets the public Sol-Attention API introduced by
Comfy-Org/comfy-kitchen PR #117.  The H3 layout and producer integration are
based on that published API and Kijai's accompanying MiniMax test adapter.
"""

from __future__ import annotations

import importlib.metadata
import logging
import sys
from collections.abc import Callable, Iterator
from typing import Any

import torch

HEAD_DIM = 128
BLOCK_SIZE = 64
PRODUCER_CHUNK = 4096
VSA_KEEP_RATIO = 0.10
VSA_CUBE = (4, 4, 4)
MIN_COMFY_KITCHEN_VERSION = "0.2.31"

_LOG = logging.getLogger("h3_relay.fast_h3_vsa")
_INSTALLED_MODELS: set[int] = set()
_PATCHED_LAYOUTS: set[int] = set()
_SPANS: dict[int, tuple[Any, tuple[int, int], tuple[int, int] | None]] = {}
_PLANS: dict[tuple[Any, ...], dict[str, Any]] = {}
_ROPE_CACHE: dict[tuple[int, int], tuple[Any, Any, torch.Tensor]] = {}
_PRODUCER_STATS: dict[tuple[int, int], tuple[torch.Tensor, torch.Tensor]] = {}
_COUNTERS = {"producer_calls": 0, "patched_blocks": 0, "errors": 0}


def runtime_stats() -> dict[str, int]:
    """Return process-local VSA dispatch counters for diagnostics/tests."""
    return dict(_COUNTERS)


def reset_runtime_stats() -> None:
    """Reset counters and previous-step producer statistics."""
    _COUNTERS.update(producer_calls=0, patched_blocks=0, errors=0)
    _PRODUCER_STATS.clear()


def _version_tuple(value: str) -> tuple[int, ...]:
    parts = []
    for item in str(value).split("."):
        digits = "".join(char for char in item if char.isdigit())
        if not digits:
            break
        parts.append(int(digits))
    return tuple(parts)


def require_runtime():
    """Return the official comfy-kitchen CUDA backend or fail explicitly."""
    try:
        import comfy_kitchen
        from comfy_kitchen.backends import cuda as cuda_backend
    except Exception as exc:
        raise RuntimeError(
            "FastH3 VSA requires the official comfy-kitchen CUDA package "
            f">= {MIN_COMFY_KITCHEN_VERSION}: {exc}"
        ) from exc

    try:
        installed = importlib.metadata.version("comfy-kitchen")
    except importlib.metadata.PackageNotFoundError:
        installed = "unknown"
    if installed != "unknown" and _version_tuple(installed) < _version_tuple(
        MIN_COMFY_KITCHEN_VERSION
    ):
        raise RuntimeError(
            "FastH3 VSA requires comfy-kitchen >= %s; found %s. Update the "
            "ComfyUI environment dependencies." % (MIN_COMFY_KITCHEN_VERSION, installed)
        )
    if not hasattr(comfy_kitchen, "sol_attn") or not hasattr(
        cuda_backend, "sol_attn_chunked"
    ):
        raise RuntimeError(
            "FastH3 VSA requires a comfy-kitchen CUDA build with sol_attn and "
            "sol_attn_chunked from PR #117. The package version alone is not "
            "sufficient; the current PyPI 0.2.31 wheel may lack those symbols."
        )
    backends = comfy_kitchen.list_backends()
    cuda_info = backends.get("cuda", {})
    if not cuda_info.get("available") or "sol_attn" not in cuda_info.get(
        "capabilities", []
    ):
        reason = cuda_info.get("unavailable_reason") or "CUDA sol_attn unavailable"
        raise RuntimeError("FastH3 VSA cannot use comfy-kitchen CUDA: %s" % reason)
    if not torch.cuda.is_available():
        raise RuntimeError("FastH3 VSA requires an available CUDA device.")
    capability = torch.cuda.get_device_capability()
    if capability < (8, 0):
        raise RuntimeError(
            "FastH3 VSA requires an NVIDIA GPU with compute capability 8.0+; "
            "found %d.%d." % capability
        )
    return cuda_backend


def _patch_packed_layout(module: Any) -> None:
    """Observe H3 PackedLayout segment spans without changing its contents."""
    layout_class = getattr(module, "PackedLayout", None)
    if layout_class is None:
        raise RuntimeError("FastH3 VSA requires MiniMax H3 PackedLayout support.")
    if id(layout_class) in _PATCHED_LAYOUTS:
        return
    original_init = layout_class.__init__

    def __init__(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        segments = getattr(self, "segments", []) or []
        video = next(
            ((start, stop) for start, stop, kind in segments if kind == "video"),
            None,
        )
        audio = next(
            ((start, stop) for start, stop, kind in segments if kind == "audio"),
            None,
        )
        positions = getattr(self, "position_ids", None)
        if torch.is_tensor(positions) and video is not None:
            _SPANS[id(positions)] = (self, video, audio)

    layout_class.__init__ = __init__
    _PATCHED_LAYOUTS.add(id(layout_class))


def _install_layout_observer(diffusion_model: Any) -> None:
    """Publish each active H3 packed layout through transformer_options."""
    if id(diffusion_model) in _INSTALLED_MODELS:
        return
    for attribute in ("rope_freqs", "_forward"):
        if not hasattr(diffusion_model, attribute):
            raise RuntimeError(
                "FastH3 VSA requires MiniMax H3 diffusion_model.%s." % attribute
            )

    module = sys.modules.get(type(diffusion_model).__module__)
    if module is None:
        raise RuntimeError("FastH3 VSA could not resolve the H3 model module.")
    _patch_packed_layout(module)
    original_forward = diffusion_model._forward
    original_rope_freqs = diffusion_model.rope_freqs

    def _forward(x, timestep, context, transformer_options=None, **kwargs):
        options = transformer_options if transformer_options is not None else {}
        diffusion_model._h3_relay_vsa_options = options
        try:
            return original_forward(
                x,
                timestep,
                context,
                transformer_options=options,
                **kwargs,
            )
        finally:
            diffusion_model._h3_relay_vsa_options = None
            options.pop("h3_relay_vsa_layout", None)

    def rope_freqs(position_ids, device):
        entry = _SPANS.get(id(position_ids))
        if entry is None:
            raise RuntimeError(
                "FastH3 VSA did not receive an H3 PackedLayout for this forward."
            )
        layout, _video, _audio = entry
        options = getattr(diffusion_model, "_h3_relay_vsa_options", None)
        if options is None:
            raise RuntimeError(
                "FastH3 VSA could not publish its layout into transformer_options."
            )
        options["h3_relay_vsa_layout"] = layout
        return original_rope_freqs(position_ids, device)

    diffusion_model._forward = _forward
    diffusion_model.rope_freqs = rope_freqs
    _INSTALLED_MODELS.add(id(diffusion_model))


def _vsa_plan(layout: Any, device: torch.device) -> dict[str, Any]:
    """Build FastVideo's padded prefix/video-cube row order for one layout."""
    key = (tuple(layout.signature), tuple(layout.segments), str(device))
    cached = _PLANS.get(key)
    if cached is not None:
        return cached

    _text_len, latent_t, latent_h, latent_w, _audio_t = layout.signature
    grid = (int(latent_t), int(latent_h) // 2, int(latent_w) // 2)
    tiled_segments = []
    prefix_blocks = 0
    for start, stop, kind in layout.segments:
        length = int(stop - start)
        if kind != "video":
            blocks = (length + BLOCK_SIZE - 1) // BLOCK_SIZE
            segment = torch.full((blocks * BLOCK_SIZE,), -1, dtype=torch.int64)
            segment[:length] = torch.arange(start, stop)
            tiled_segments.append(segment.view(blocks, BLOCK_SIZE))
            prefix_blocks += blocks
            continue

        if grid[0] * grid[1] * grid[2] != length:
            raise RuntimeError(
                "FastH3 VSA video segment %d does not match latent grid %s."
                % (length, grid)
            )
        cube_t, cube_h, cube_w = VSA_CUBE
        padded_t, padded_h, padded_w = (
            (axis + cube - 1) // cube * cube for axis, cube in zip(grid, VSA_CUBE)
        )
        padded = torch.full((padded_t, padded_h, padded_w), -1, dtype=torch.int64)
        padded[: grid[0], : grid[1], : grid[2]] = torch.arange(start, stop).view(*grid)
        cubes = (
            padded.view(
                padded_t // cube_t,
                cube_t,
                padded_h // cube_h,
                cube_h,
                padded_w // cube_w,
                cube_w,
            )
            .permute(0, 2, 4, 1, 3, 5)
            .reshape(-1, BLOCK_SIZE)
        )
        order = torch.argsort((cubes < 0).to(torch.int8), dim=1, stable=True)
        tiled_segments.append(torch.gather(cubes, 1, order))

    tiles = torch.cat(tiled_segments)
    source_rows = tiles.reshape(-1)
    live = source_rows >= 0
    inverse = torch.empty(layout.seq_len, dtype=torch.int64)
    inverse[source_rows[live]] = torch.nonzero(live).flatten()
    plan = {
        "padded_rows": int(source_rows.numel()),
        "original_rows": int(layout.seq_len),
        "prefix_blocks": prefix_blocks,
        "source_rows": source_rows.to(device),
        "inverse": inverse.to(device),
        "block_len": (tiles >= 0).sum(1).to(torch.int32).to(device),
    }
    if len(_PLANS) >= 4:
        del _PLANS[next(iter(_PLANS))]
    _PLANS[key] = plan
    return plan


def _padded_rope(rope_freqs: torch.Tensor, plan: dict[str, Any]) -> torch.Tensor:
    key = (id(rope_freqs), id(plan))
    cached = _ROPE_CACHE.get(key)
    if cached is not None and cached[0] is rope_freqs and cached[1] is plan:
        return cached[2]
    padded = rope_freqs.new_zeros(
        (1, plan["padded_rows"]) + tuple(rope_freqs.shape[2:])
    )
    padded[0, plan["inverse"]] = rope_freqs[0]
    _ROPE_CACHE.clear()
    _ROPE_CACHE[key] = (rope_freqs, plan, padded)
    return padded


def _padded_chunk(
    values: torch.Tensor,
    plan: dict[str, Any],
    start: int,
    count: int,
) -> torch.Tensor:
    indices = plan["source_rows"][start : start + count]
    return values[indices.clamp_min(0)] * (indices >= 0).unsqueeze(1).to(values.dtype)


def _producer_forward(
    attention: Any,
    cuda_backend: Any,
    block_index: int,
) -> Callable[..., torch.Tensor]:
    """Create an exact FastH3 VSA self-attention forward replacement."""
    import comfy.model_management

    def forward(x, rope_freqs=None, transformer_options=None):
        options = transformer_options if transformer_options is not None else {}
        try:
            if rope_freqs is None:
                raise RuntimeError("missing H3 rope frequencies")
            if x.dim() != 2 or x.device.type != "cuda" or x.dtype != torch.bfloat16:
                raise RuntimeError(
                    "expected a two-dimensional CUDA BF16 activation; got %s %s on %s"
                    % (tuple(x.shape), x.dtype, x.device)
                )
            if int(attention.head_dim) != HEAD_DIM:
                raise RuntimeError(
                    "expected head_dim %d; got %d" % (HEAD_DIM, attention.head_dim)
                )
            layout = options.get("h3_relay_vsa_layout")
            if layout is None or int(layout.seq_len) != int(x.shape[0]):
                raise RuntimeError("missing or mismatched H3 packed layout")
            gate = getattr(attention, "to_gate_compress", None)
            if gate is None:
                raise RuntimeError(
                    "checkpoint/core has no to_gate_compress layer; pin the "
                    "FastH3-capable ComfyUI commit"
                )

            plan = _vsa_plan(layout, x.device)
            padded_rows = int(plan["padded_rows"])
            heads = int(attention.heads)
            padded_rope = _padded_rope(rope_freqs, plan)
            prefix = int(plan["prefix_blocks"])
            sink = [0, prefix]
            q_weight = comfy.model_management.cast_to(
                attention.q_norm.weight, device=x.device
            )
            k_weight = comfy.model_management.cast_to(
                attention.k_norm.weight, device=x.device
            )
            coarse_gate = x.new_empty(padded_rows, heads * HEAD_DIM).view(
                1, padded_rows, heads, HEAD_DIM
            )
            previous_stats = _PRODUCER_STATS.get((id(attention), padded_rows))

            def chunks() -> Iterator[torch.Tensor]:
                for offset in range(0, padded_rows, PRODUCER_CHUNK):
                    chunk = _padded_chunk(x, plan, offset, PRODUCER_CHUNK)
                    coarse_gate.view(padded_rows, heads * HEAD_DIM)[
                        offset : offset + chunk.shape[0]
                    ] = gate(chunk)
                    yield attention.qkv_proj(chunk)

            output, kmean, vscale = cuda_backend.sol_attn_chunked(
                chunks,
                padded_rows,
                heads,
                padded_rope,
                (q_weight, k_weight),
                kmean=None if previous_stats is None else previous_stats[0],
                vscale=None if previous_stats is None else previous_stats[1],
                topk_ratio=VSA_KEEP_RATIO,
                sink_blocks=sink,
                sink_q=sink,
                rope_eps=attention.q_norm.eps,
                tail=False,
                block_len=plan["block_len"],
                coarse_gate=coarse_gate,
            )
            _PRODUCER_STATS[(id(attention), padded_rows)] = (kmean, vscale)
            _COUNTERS["producer_calls"] += 1
            output = output.view(padded_rows, heads * HEAD_DIM)[plan["inverse"]]
            return attention.out_proj(output)
        except Exception as exc:
            _COUNTERS["errors"] += 1
            raise RuntimeError(
                "FastH3 VSA block %d failed; dense fallback is intentionally "
                "disabled: %s" % (block_index, exc)
            ) from exc

    return forward


def apply_fast_h3_vsa(model: Any):
    """Clone a ComfyUI MODEL and install the locked FastH3 VSA contract."""
    cuda_backend = require_runtime()
    diffusion_model = model.get_model_object("diffusion_model")
    if not hasattr(diffusion_model, "rope_freqs") or not hasattr(
        diffusion_model, "_forward"
    ):
        raise RuntimeError("FastH3 VSA requires a MiniMax H3 diffusion model.")
    blocks = getattr(diffusion_model, "blocks", None)
    if not blocks:
        raise RuntimeError("FastH3 VSA found no MiniMax H3 transformer blocks.")

    invalid = []
    for index, block in enumerate(blocks):
        attention = getattr(block, "attn", None)
        if attention is None or not hasattr(attention, "qkv_proj"):
            invalid.append("block %d has no supported self-attention" % index)
        elif not hasattr(attention, "to_gate_compress"):
            invalid.append("block %d has no to_gate_compress" % index)
        elif int(getattr(attention, "head_dim", 0)) != HEAD_DIM:
            invalid.append(
                "block %d head_dim is %s"
                % (index, getattr(attention, "head_dim", None))
            )
    if invalid:
        raise RuntimeError(
            "FastH3 VSA core/checkpoint contract is incomplete: "
            + "; ".join(invalid[:4])
        )

    _install_layout_observer(diffusion_model)
    patched = model.clone()
    existing_patches = getattr(patched, "object_patches", {})
    installed = 0
    for index, block in enumerate(blocks):
        path = "diffusion_model.blocks.%d.attn.forward" % index
        if path in existing_patches:
            raise RuntimeError(
                "FastH3 VSA refuses to replace an existing model patch at %s." % path
            )
        patched.add_object_patch(
            path,
            _producer_forward(block.attn, cuda_backend, index),
        )
        installed += 1
    if installed != len(blocks):
        raise RuntimeError(
            "FastH3 VSA patched %d of %d blocks." % (installed, len(blocks))
        )

    reset_runtime_stats()
    _COUNTERS["patched_blocks"] = installed
    transformer_options = patched.model_options.setdefault("transformer_options", {})
    transformer_options["h3_relay_fast_h3_vsa"] = {
        "version": 1,
        "keep_ratio": VSA_KEEP_RATIO,
        "blocks": installed,
    }
    _LOG.info(
        "H3 Relay installed FastH3 VSA on %d blocks using comfy-kitchen %s",
        installed,
        importlib.metadata.version("comfy-kitchen"),
    )
    return patched
