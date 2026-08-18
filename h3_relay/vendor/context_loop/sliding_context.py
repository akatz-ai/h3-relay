"""WanGP-style MiniMax H3 sliding-window continuation.

An 18-frame overlap is represented as 17 frames of history followed by one
frame-zero boundary anchor. The history advances H3's target timeline without
becoming part of the sampled target latent; only the boundary frame is repeated
in the decoded result and trimmed before assembly.
"""

import logging

import torch

from .nodes import (
    AUDIO_HZ,
    FPS,
    FRAME_RESCALE,
    _audio_tail_from_latent,
    _encode_tail_audio,
    _pixel_frames,
    _resize,
    _streams_from_latent,
)


_LOG = logging.getLogger("minimax_h3_context_loop.sliding_history")


def require_sliding_history_support():
    from .nodes import _activate_inline_patches
    from .history_layout import ensure_history_keyframe_support
    from comfy.ldm.minimax.model import PackedLayout

    _activate_inline_patches()
    ensure_history_keyframe_support()
    if not getattr(PackedLayout, "supports_history_keyframes", False):
        raise RuntimeError(
            "h3_sliding_history: this ComfyUI build cannot position H3 history "
            "before the target timeline. Update ComfyUI or check the startup "
            "log for an H3 Relay history-layout compatibility failure."
        )


def _target_latent(latent, history_frames):
    import comfy.nested_tensor

    streams = _streams_from_latent(latent)
    if len(streams) != 2:
        raise ValueError(
            "h3_sliding_history: expected a MiniMax H3 AV latent pair."
        )
    video, audio = streams
    if video.ndim == 4:
        video = video.unsqueeze(0)
    if audio.ndim == 3:
        audio = audio.unsqueeze(0)
    if video.ndim != 5 or audio.ndim != 4:
        raise ValueError(
            "h3_sliding_history: expected video [B,C,T,H,W] and audio "
            "[B,C,2,T], got %s and %s." %
            (tuple(video.shape), tuple(audio.shape))
        )

    history_steps = history_frames // 17 * 5
    target_video_steps = int(video.shape[2]) - history_steps
    if target_video_steps < 2:
        raise ValueError(
            "h3_sliding_history: the history consumes the whole target video."
        )
    target_frames = _pixel_frames(target_video_steps)
    expected_frames = _pixel_frames(int(video.shape[2])) - history_frames
    if target_frames != expected_frames:
        raise RuntimeError(
            "h3_sliding_history: shortening %d video steps by %d history "
            "frames produced %d target frames; expected %d." %
            (int(video.shape[2]), history_frames, target_frames,
             expected_frames)
        )
    target_audio_steps = int(round(target_frames / float(FPS) * AUDIO_HZ))
    if target_audio_steps >= int(audio.shape[-1]):
        raise ValueError(
            "h3_sliding_history: the history does not leave a shorter audio "
            "target."
        )

    out = latent.copy()
    out["samples"] = comfy.nested_tensor.NestedTensor((
        video[:, :, :target_video_steps].clone(),
        audio[..., :target_audio_steps].clone(),
    ))
    out.pop("noise_mask", None)
    return out, target_frames


def _encode_history(vae, frames, boundary):
    history_frames = int(frames.shape[0])
    history_steps = history_frames // 17 * 5

    # ComfyUI's public H3 VAE encoder drops three tail tokens. Appending a
    # valid five-frame terminal chunk makes those the discarded tokens and
    # leaves every 17-frame history chunk intact, matching WanGP's
    # keep_all_latents=True path without bypassing ComfyUI's VAE manager.
    padded = torch.cat((frames, boundary.repeat(5, 1, 1, 1)), dim=0)
    encoded = vae.encode(padded)
    if getattr(encoded, "ndim", 0) != 5:
        raise ValueError(
            "h3_sliding_history: history VAE encode returned shape %s." %
            (tuple(getattr(encoded, "shape", ())),)
        )
    if int(encoded.shape[2]) != history_steps + 2:
        raise RuntimeError(
            "h3_sliding_history: %d history frames plus padding encoded to %d "
            "steps; expected %d." %
            (history_frames, int(encoded.shape[2]), history_steps + 2)
        )
    history = encoded[:, :, :history_steps].clone()
    boundary_latent = vae.encode(boundary)
    if int(boundary_latent.shape[2]) != 1:
        raise RuntimeError(
            "h3_sliding_history: boundary frame encoded to %d video steps." %
            int(boundary_latent.shape[2])
        )
    return history, boundary_latent


def _shift_target_guides(conditioning, history_frames, overlap_frames,
                         continuation_keyframes):
    out = []
    dropped = []
    for embedding, extra in conditioning:
        metadata = extra.copy()
        kept = []
        for keyframe in metadata.get("minimax_keyframes") or ():
            anchor = keyframe.get("anchor", "frame")
            if anchor != "frame":
                raise ValueError(
                    "h3_sliding_history: incoming conditioning already contains "
                    "a continuation history. Wire fresh stock H3 conditioning "
                    "into Chain Context."
                )
            position = float(keyframe.get("resolved_frame_index", 0))
            if position < overlap_frames:
                dropped.append(position)
                continue
            shifted = dict(keyframe)
            shifted["resolved_frame_index"] = position - history_frames
            kept.append(shifted)
        metadata["minimax_keyframes"] = kept + continuation_keyframes
        out.append([embedding, metadata])
    if dropped:
        _LOG.warning(
            "h3_sliding_history: dropped %d target guide(s) inside the "
            "%d-frame overlap at %s.",
            len(dropped), overlap_frames, sorted(set(dropped)),
        )
    return out


def apply_sliding_history(conditioning, vae, latent, previous_frames,
                          context_length, crop, previous_latent=None,
                          audio_vae=None, previous_audio=None):
    require_sliding_history_support()

    overlap = int(context_length)
    if overlap < 18 or (overlap - 1) % 17:
        raise ValueError(
            "h3_sliding_history: context_length must follow 17k+1 "
            "(18, 35, 52, ...)."
        )
    available = int(previous_frames.shape[0])
    if available < overlap:
        raise ValueError(
            "h3_sliding_history: the previous scene has %d delivered frames, "
            "but this continuation needs %d." % (available, overlap)
        )

    target_video = _streams_from_latent(latent)[0]
    if target_video.ndim == 4:
        target_video = target_video.unsqueeze(0)
    width = int(target_video.shape[4]) * 16
    height = int(target_video.shape[3]) * 16

    tail = _resize(previous_frames[available - overlap:], width, height, crop)
    history_frames = overlap - 1
    history = tail[:history_frames]
    boundary = tail[history_frames:]
    history_latent, boundary_latent = _encode_history(
        vae, history, boundary)

    keyframes = [
        {"anchor": "history", "latent": history_latent},
        {"anchor": "first", "resolved_frame_index": 0,
         "latent": boundary_latent},
    ]

    audio_steps = 0
    audio_source = "off"
    if previous_latent is not None or previous_audio is not None:
        if previous_latent is not None:
            audio_latent, audio_steps, overhang = _audio_tail_from_latent(
                previous_latent, overlap)
            audio_source = "sampled latent"
        else:
            if audio_vae is None:
                raise ValueError(
                    "h3_sliding_history: previous_audio requires the H3 audio "
                    "VAE."
                )
            audio_latent, audio_steps = _encode_tail_audio(
                audio_vae, previous_audio, overlap / float(FPS))
            overhang = 0.0
            audio_source = "decoded audio"
        boundary_audio_steps = min(
            audio_steps, max(1, round(AUDIO_HZ / float(FPS))))
        history_audio_steps = audio_steps - boundary_audio_steps
        if history_audio_steps:
            keyframes.append({
                "anchor": "history",
                "audio_latent": audio_latent[..., :history_audio_steps].clone(),
            })
        keyframes.append({
            "anchor": "first",
            "resolved_frame_index": 0,
            "audio_latent": audio_latent[..., history_audio_steps:].clone(),
        })
        if abs(overhang) > 1e-9:
            _LOG.info(
                "h3_sliding_history: predecessor audio ends %.3f latent steps "
                "from its final video frame; retaining WanGP's history/first "
                "split.", overhang)

    out_conditioning = _shift_target_guides(
        conditioning, history_frames, overlap, keyframes)
    out_latent, target_frames = _target_latent(latent, history_frames)

    _LOG.info(
        "h3_sliding_history: %d-frame overlap = %d history + 1 boundary; "
        "target shortened to %d frames at %dx%d; trim 1; audio %s%s",
        overlap, history_frames, target_frames, width, height, audio_source,
        " (%d latent steps)" % audio_steps if audio_steps else "",
    )
    return out_conditioning, out_latent, 1


__all__ = ["apply_sliding_history", "require_sliding_history_support"]
