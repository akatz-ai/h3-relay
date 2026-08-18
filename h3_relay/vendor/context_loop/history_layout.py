"""Guarded MiniMax H3 pre-target history layout compatibility.

H3 Relay's sliding continuation shortens the sampled target latent, then packs
the preceding shot's video/audio tail as conditioning immediately before that
target.  ComfyUI's stock H3 layout can pack the conditioning rows but, until
history anchors are supported in core, places every guide on the target
timeline itself.  This module relocates only the rows belonging to explicitly
marked ``anchor == "history"`` guides and advances the target origin by the
video-history span.

The patch is process-local.  It never edits ComfyUI files, stands down when
core already advertises the capability, self-tests before installation, and
refuses to wrap an unknown third-party layout implementation.
"""

from __future__ import annotations

import logging

import torch

import comfy.ldm.minimax.model as mm

from . import patch_layout


PATCH_MARKER = "_h3_relay_history_layout_patch"
ORIGINAL_MARKER = "_h3_relay_history_layout_original"
_LOG = logging.getLogger("h3_relay.history_layout")
_original_init = None
_legacy_full_builder = False


def _reference_span(block):
    helper = getattr(mm, "_ref_t_span", None)
    if helper is not None:
        return helper(block)
    kind = block["kind"]
    if kind == "image":
        return 1.0
    if kind == "audio":
        return float(block["ref_audio_t"])
    if kind in ("video", "video_audio"):
        return max(
            float(block["ref_audio_t"]),
            sum(mm._video_t_spans(int(block["latent_t"]))),
        )
    raise ValueError("Unknown MiniMax H3 reference kind %r." % kind)


def _build_history_layout(instance, text_len, latent_t, latent_h, latent_w,
                          audio_t, keyframes=None, refs=None):
    """Build the current H3 guide layout on pre-PR #15439 ComfyUI.

    Older H3 layouts emitted one still-image row group per guide and did not
    represent guide audio at all.  A post-position rewrite cannot create the
    missing rows, so history executions use this current-layout snapshot while
    ordinary executions continue through the untouched legacy constructor.
    """
    frame, w_grid = mm._frame_grid(latent_h, latent_w)
    frame_rows = frame.shape[0]
    segments = [("text", text_len)]
    grid = torch.zeros(text_len, 3, dtype=torch.float64)
    grid[:, 0] = torch.arange(text_len, dtype=torch.float64)
    positions = [grid]
    img_pos, img_update = [], []
    audio_pos, audio_update = [], []
    row = text_len

    target_audio_w = (float(w_grid[0]), float(w_grid[-1]))
    cursor = float(text_len)
    for block in refs or ():
        cursor += _reference_span(block)
    history_origin = cursor
    target_origin = history_origin + _history_span(keyframes)
    video_history_cursor = history_origin
    audio_history_cursor = history_origin

    for keyframe in keyframes or ():
        anchor = keyframe.get("anchor", "frame")
        if anchor not in ("history", "first", "frame"):
            raise ValueError("Unknown MiniMax H3 guide anchor %r." % anchor)
        video_latent = keyframe.get("latent")
        if video_latent is not None:
            video_t = int(video_latent.shape[2])
            if anchor == "history":
                cond_t = video_history_cursor
                video_history_cursor += sum(mm._video_t_spans(video_t))
            elif anchor == "first":
                cond_t = target_origin
            else:
                cond_t = (
                    target_origin
                    + mm.FRAME_RESCALE * keyframe["resolved_frame_index"]
                )
            count = video_t * frame_rows
            segments.append(("cond", count))
            positions.append(mm._video_grid(video_t, frame, cond_t))
            img_pos.append(torch.arange(row, row + count))
            img_update.append(torch.zeros(count, dtype=torch.bool))
            row += count
        audio_latent = keyframe.get("audio_latent")
        if audio_latent is not None:
            audio_t_steps = int(audio_latent.shape[-1])
            if anchor == "history":
                cond_t = audio_history_cursor
                audio_history_cursor += audio_t_steps
            elif anchor == "first":
                cond_t = target_origin
            else:
                cond_t = (
                    target_origin
                    + mm.FRAME_RESCALE * keyframe["resolved_frame_index"]
                )
            count = audio_t_steps * 2
            segments.append(("cond_audio", count))
            positions.append(mm._audio_grid(
                cond_t, audio_t_steps, *target_audio_w
            ))
            audio_pos.append(torch.arange(row, row + count))
            audio_update.append(torch.zeros(count, dtype=torch.bool))
            row += count

    if refs:
        cursor = float(text_len)
        for block in refs:
            kind = block["kind"]
            if kind == "image":
                ref_frame, _ = mm._frame_grid(
                    block["latent_h"], block["latent_w"]
                )
                count = ref_frame.shape[0]
                grid = torch.empty(count, 3, dtype=torch.float64)
                grid[:, 0] = cursor
                grid[:, 1:] = ref_frame
                segments.append(("ref_img", count))
                positions.append(grid)
                img_pos.append(torch.arange(row, row + count))
                img_update.append(torch.zeros(count, dtype=torch.bool))
                row += count
                cursor += 1.0
            elif kind == "audio":
                ref_audio_t = int(block["ref_audio_t"])
                if ref_audio_t > 0:
                    count = ref_audio_t * 2
                    segments.append(("ref_audio", count))
                    positions.append(mm._audio_grid(
                        cursor, ref_audio_t, *target_audio_w
                    ))
                    audio_pos.append(torch.arange(row, row + count))
                    audio_update.append(torch.zeros(count, dtype=torch.bool))
                    row += count
                cursor += float(ref_audio_t)
            elif kind in ("video", "video_audio"):
                ref_audio_t = int(block["ref_audio_t"])
                ref_video_t = int(block["latent_t"])
                ref_frame, ref_w_grid = mm._frame_grid(
                    block["latent_h"], block["latent_w"]
                )
                if ref_audio_t > 0:
                    count = ref_audio_t * 2
                    segments.append(("ref_audio", count))
                    positions.append(mm._audio_grid(
                        cursor,
                        ref_audio_t,
                        float(ref_w_grid[0]),
                        float(ref_w_grid[-1]),
                    ))
                    audio_pos.append(torch.arange(row, row + count))
                    audio_update.append(torch.zeros(count, dtype=torch.bool))
                    row += count
                count = ref_video_t * ref_frame.shape[0]
                segments.append(("ref_img", count))
                positions.append(mm._video_grid(
                    ref_video_t, ref_frame, cursor
                ))
                img_pos.append(torch.arange(row, row + count))
                img_update.append(torch.zeros(count, dtype=torch.bool))
                row += count
                cursor += max(
                    float(ref_audio_t), sum(mm._video_t_spans(ref_video_t))
                )
            else:
                raise ValueError("Unknown MiniMax H3 reference kind %r." % kind)

    count = audio_t * 2
    segments.append(("audio", count))
    positions.append(mm._audio_grid(target_origin, audio_t, *target_audio_w))
    audio_pos.append(torch.arange(row, row + count))
    audio_update.append(torch.ones(count, dtype=torch.bool))
    row += count

    count = latent_t * frame_rows
    segments.append(("video", count))
    positions.append(mm._video_grid(latent_t, frame, target_origin))
    img_pos.append(torch.arange(row, row + count))
    img_update.append(torch.ones(count, dtype=torch.bool))
    row += count

    instance.seq_len = row
    instance.position_ids = torch.cat(positions)
    instance.img_pos = torch.cat(img_pos)
    instance.img_update = torch.cat(img_update)
    instance.audio_pos = torch.cat(audio_pos)
    instance.audio_update = torch.cat(audio_update)
    instance.signature = (text_len, latent_t, latent_h, latent_w, audio_t)
    absolute = []
    offset = 0
    for kind, count in segments:
        absolute.append((offset, offset + count, kind))
        offset += count
    instance.segments = absolute


def _target_segments(layout):
    if len(layout.segments) < 2:
        raise RuntimeError("H3 history layout has no target segments.")
    audio = layout.segments[-2]
    video = layout.segments[-1]
    if audio[2] != "audio" or video[2] != "video":
        raise RuntimeError(
            "H3 history layout expected final audio/video segments, found %r/%r."
            % (audio[2], video[2])
        )
    audio_origin = float(layout.position_ids[audio[0], 0])
    video_origin = float(layout.position_ids[video[0], 0])
    if abs(audio_origin - video_origin) > 1e-9:
        raise RuntimeError(
            "H3 target audio/video origins differ: %.9f vs %.9f."
            % (audio_origin, video_origin)
        )
    return audio, video, video_origin


def _guide_segments(layout, keyframes):
    packed = [segment for segment in layout.segments
              if segment[2] in ("cond", "cond_audio")]
    cursor = 0
    mapped = []
    for index, keyframe in enumerate(keyframes or ()):
        video = None
        audio = None
        if keyframe.get("latent") is not None:
            if cursor >= len(packed) or packed[cursor][2] != "cond":
                raise RuntimeError(
                    "H3 history keyframe %d has no matching video segment." % index
                )
            video = packed[cursor]
            cursor += 1
        if keyframe.get("audio_latent") is not None:
            if cursor >= len(packed) or packed[cursor][2] != "cond_audio":
                raise RuntimeError(
                    "H3 history keyframe %d has no matching audio segment." % index
                )
            audio = packed[cursor]
            cursor += 1
        mapped.append((video, audio))
    if cursor != len(packed):
        raise RuntimeError(
            "H3 history layout mapped %d of %d guide segments."
            % (cursor, len(packed))
        )
    return mapped


def _shift_segment(layout, segment, desired_origin):
    start, stop, _ = segment
    current = float(layout.position_ids[start, 0])
    layout.position_ids[start:stop, 0] += float(desired_origin) - current


def _history_span(keyframes):
    return sum(
        sum(mm._video_t_spans(int(keyframe["latent"].shape[2])))
        for keyframe in keyframes or ()
        if (keyframe.get("anchor") == "history"
            and keyframe.get("latent") is not None)
    )


def _relocate_history(layout, keyframes):
    history = [keyframe for keyframe in keyframes or ()
               if keyframe.get("anchor") == "history"]
    if not history:
        return

    audio_target, video_target, history_origin = _target_segments(layout)
    segments = _guide_segments(layout, keyframes)
    target_shift = _history_span(keyframes)
    if target_shift <= 0.0:
        raise RuntimeError(
            "H3 history conditioning must include at least one video latent."
        )

    video_cursor = history_origin
    audio_cursor = history_origin
    for keyframe, (video_segment, audio_segment) in zip(keyframes, segments):
        anchor = keyframe.get("anchor", "frame")
        if anchor not in ("history", "first", "frame"):
            raise ValueError("Unknown MiniMax H3 guide anchor %r." % anchor)
        if video_segment is not None:
            if anchor == "history":
                _shift_segment(layout, video_segment, video_cursor)
                video_cursor += sum(mm._video_t_spans(
                    int(keyframe["latent"].shape[2])
                ))
            else:
                start = video_segment[0]
                _shift_segment(
                    layout,
                    video_segment,
                    float(layout.position_ids[start, 0]) + target_shift,
                )
        if audio_segment is not None:
            if anchor == "history":
                _shift_segment(layout, audio_segment, audio_cursor)
                audio_cursor += int(keyframe["audio_latent"].shape[-1])
            else:
                start = audio_segment[0]
                _shift_segment(
                    layout,
                    audio_segment,
                    float(layout.position_ids[start, 0]) + target_shift,
                )

    _shift_segment(layout, audio_target, history_origin + target_shift)
    _shift_segment(layout, video_target, history_origin + target_shift)


def _normalized_keyframes(keyframes):
    normalized = []
    for keyframe in keyframes or ():
        item = dict(keyframe)
        anchor = item.get("anchor", "frame")
        if anchor == "history":
            item["resolved_frame_index"] = 0
        elif anchor == "first":
            item["resolved_frame_index"] = 0
        elif anchor == "frame":
            if "resolved_frame_index" not in item:
                raise ValueError("MiniMax H3 frame guide has no resolved index.")
        else:
            raise ValueError("Unknown MiniMax H3 guide anchor %r." % anchor)
        normalized.append(item)
    return normalized


def _call_original(instance, text_len, latent_t, latent_h, latent_w, audio_t,
                   keyframes=None, refs=None, *args, **kwargs):
    normalized = _normalized_keyframes(keyframes)
    if _legacy_full_builder and any(
            keyframe.get("anchor") == "history" for keyframe in normalized):
        _build_history_layout(
            instance,
            text_len,
            latent_t,
            latent_h,
            latent_w,
            audio_t,
            keyframes=normalized,
            refs=refs,
        )
        return
    _original_init(
        instance,
        text_len,
        latent_t,
        latent_h,
        latent_w,
        audio_t,
        keyframes=normalized,
        refs=refs,
        *args,
        **kwargs,
    )
    _relocate_history(instance, normalized)


def _patched_init(instance, text_len, latent_t, latent_h, latent_w, audio_t,
                  keyframes=None, refs=None, *args, **kwargs):
    _call_original(
        instance,
        text_len,
        latent_t,
        latent_h,
        latent_w,
        audio_t,
        keyframes,
        refs,
        *args,
        **kwargs,
    )


setattr(_patched_init, PATCH_MARKER, True)


def _build_for_test(keyframes=None, refs=None):
    instance = mm.PackedLayout.__new__(mm.PackedLayout)
    kwargs = {}
    try:
        import inspect
        if "frame_count" in inspect.signature(_original_init).parameters:
            kwargs["frame_count"] = 22
    except (TypeError, ValueError):
        pass
    _call_original(
        instance,
        7,
        7,
        4,
        4,
        12,
        keyframes=keyframes,
        refs=refs,
        **kwargs,
    )
    return instance


def _self_test():
    video_history = torch.zeros(1, 24, 3, 4, 4)
    video_boundary = torch.zeros(1, 24, 1, 4, 4)
    audio_history = torch.zeros(1, 32, 2, 5)
    audio_boundary = torch.zeros(1, 32, 2, 1)
    keyframes = [
        {"anchor": "history", "latent": video_history},
        {"anchor": "first", "resolved_frame_index": 0,
         "latent": video_boundary},
        {"anchor": "history", "audio_latent": audio_history},
        {"anchor": "first", "resolved_frame_index": 0,
         "audio_latent": audio_boundary},
    ]
    refs = [{"kind": "image", "latent_h": 4, "latent_w": 4}]
    layout = _build_for_test(keyframes, refs)
    mapped = _guide_segments(layout, _normalized_keyframes(keyframes))
    _, _, target_origin = _target_segments(layout)
    span = sum(mm._video_t_spans(3))
    history_origin = target_origin - span
    video_history_origin = float(layout.position_ids[mapped[0][0][0], 0])
    video_boundary_origin = float(layout.position_ids[mapped[1][0][0], 0])
    audio_history_origin = float(layout.position_ids[mapped[2][1][0], 0])
    audio_boundary_origin = float(layout.position_ids[mapped[3][1][0], 0])
    if abs(video_history_origin - history_origin) > 1e-9:
        raise RuntimeError("video history does not start before the target")
    if abs(audio_history_origin - history_origin) > 1e-9:
        raise RuntimeError("audio history does not start before the target")
    if abs(video_boundary_origin - target_origin) > 1e-9:
        raise RuntimeError("video boundary is not anchored to the target")
    if abs(audio_boundary_origin - target_origin) > 1e-9:
        raise RuntimeError("audio boundary is not anchored to the target")

    plain_keyframes = [{
        "anchor": "first",
        "resolved_frame_index": 0,
        "latent": video_boundary,
    }]
    plain = _build_for_test(plain_keyframes, refs)
    baseline = mm.PackedLayout.__new__(mm.PackedLayout)
    normalized = _normalized_keyframes(plain_keyframes)
    kwargs = {}
    try:
        import inspect
        if "frame_count" in inspect.signature(_original_init).parameters:
            kwargs["frame_count"] = 22
    except (TypeError, ValueError):
        pass
    _original_init(
        baseline, 7, 7, 4, 4, 12,
        keyframes=normalized, refs=refs, **kwargs,
    )
    if not torch.equal(plain.position_ids, baseline.position_ids):
        raise RuntimeError("history compatibility changed a non-history layout")


def _known_layout_owner(initializer):
    if getattr(initializer, patch_layout.PATCH_MARKER, False):
        return True
    if patch_layout._solattn_wrapped_init(initializer) is not None:
        return True
    home = str(getattr(mm.PackedLayout, "__module__", "") or "")
    where = str(getattr(initializer, "__module__", "") or "")
    return bool(home and home == where and not hasattr(initializer, "__wrapped__"))


def ensure_history_keyframe_support():
    """Install the guarded fallback and return ``native`` or ``compat``."""
    global _legacy_full_builder, _original_init
    cls = getattr(mm, "PackedLayout", None)
    if cls is None:
        raise RuntimeError("MiniMax H3 PackedLayout is unavailable.")
    if getattr(cls, "supports_history_keyframes", False):
        return "native"
    current = getattr(cls, "__init__", None)
    if current is None:
        raise RuntimeError("MiniMax H3 PackedLayout has no constructor.")
    if getattr(current, PATCH_MARKER, False):
        cls.supports_history_keyframes = True
        return "compat"
    if not _known_layout_owner(current):
        raise RuntimeError(
            "H3 Relay cannot enable sliding history because another custom "
            "node owns MiniMax H3 PackedLayout. Disable the conflicting H3 "
            "layout extension and restart ComfyUI."
        )

    native_guides = patch_layout.native_guides_available()
    _legacy_full_builder = not native_guides
    if _legacy_full_builder:
        from .h3_mask_compat import ensure_h3_guide_engine_compat
        ensure_h3_guide_engine_compat()
    _original_init = current
    try:
        _self_test()
    except Exception:
        _original_init = None
        raise
    setattr(_patched_init, ORIGINAL_MARKER, current)
    if native_guides:
        setattr(_patched_init, patch_layout.NATIVE_GUIDES_MARKER, True)
    cls.__init__ = _patched_init
    cls.supports_history_keyframes = True
    _LOG.info("H3 Relay enabled process-local MiniMax H3 history keyframes")
    return "compat"


__all__ = ["ensure_history_keyframe_support"]
