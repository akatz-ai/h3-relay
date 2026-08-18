"""Masked target-prefix continuation for recursive MiniMax H3 chains.

The previous scene's decoded video tail is VAE-encoded into the beginning of
the next scene's real target latent.  The matching audio tail is copied from
the previous sampled H3 latent (or encoded from imported audio for scene 1).
A nested AV denoise mask then protects that prefix while the sampler generates
only the future portion.

The mask design follows ComfyUI PR #15375 and seitanism's GPL-3.0
ComfyUI-H3-Motion-Context-MultiRef masked-extension work.  Runtime support is
enabled lazily so ordinary guide-mode chains do not patch H3 mask handling.
"""

from __future__ import annotations

import logging

import torch

from .nodes import (
    AUDIO_HZ,
    FPS,
    VIDEO_RUN_GRID,
    _audio_tail_from_latent,
    _pixel_frames,
    _resize,
    _streams_from_latent,
)


_LOG = logging.getLogger("minimax_h3_context_loop.masked_prefix")


def _require_h3_mask_support():
    """Prefer native H3 AV masks, otherwise install the vendored fallback."""
    from .patch_layout import native_guides_available

    if not native_guides_available():
        raise RuntimeError(
            "h3_masked_prefix: masked AV continuation requires the native "
            "MiniMax H3 Add Guide / MultiRef core from ComfyUI PR #15439. "
            "Update ComfyUI before using this experimental mode."
        )
    from .h3_mask_compat import ensure_h3_mask_compat
    from .h3_mask_payload_compat import ensure_av_mask_payload_compat

    ensure_h3_mask_compat()
    ensure_av_mask_payload_compat()

    import comfy.model_base

    cls = getattr(comfy.model_base, "MiniMaxH3", None)
    if (
        cls is None
        or "process_denoise_mask" not in cls.__dict__
        or "scale_latent_inpaint" not in cls.__dict__
    ):
        raise RuntimeError(
            "h3_masked_prefix: H3 per-stream AV mask support could not be "
            "enabled. Check the ComfyUI console capability report."
        )


def _snap_prefix_length(requested, available, target_frames):
    """Resolve a context window to an exact H3 video-VAE run."""
    cap = min(int(requested), int(available), int(target_frames) - 1)
    run = next((value for value in VIDEO_RUN_GRID if value <= cap), 0)
    if run < 5:
        raise ValueError(
            "h3_masked_prefix: masked continuation needs at least 5 previous "
            "frames and a target longer than the preserved prefix."
        )
    if run != int(requested):
        _LOG.warning(
            "h3_masked_prefix: context_length %d -> exact H3 prefix %d "
            "(valid runs are 5, 22, 39, 56, ...; 39/90/141/... align AV "
            "clocks exactly)",
            int(requested), run,
        )
    return run


def _validate_target_streams(latent):
    video, audio = _streams_from_latent(latent)[:2]
    if video.ndim == 4:
        video = video.unsqueeze(0)
    if audio.ndim == 3:
        audio = audio.unsqueeze(0)
    if video.ndim != 5:
        raise ValueError(
            "h3_masked_prefix: video latent must be [B,C,T,H,W], got %s" %
            (tuple(video.shape),)
        )
    if audio.ndim != 4:
        raise ValueError(
            "h3_masked_prefix: audio latent must be [B,C,2,T], got %s" %
            (tuple(audio.shape),)
        )
    if int(video.shape[0]) != 1 or int(audio.shape[0]) != 1:
        raise ValueError(
            "h3_masked_prefix: masked continuation supports H3 batch size 1."
        )
    target_frames = _pixel_frames(int(video.shape[2]))
    expected_audio = int(round(target_frames / float(FPS) * AUDIO_HZ))
    if int(audio.shape[-1]) != expected_audio:
        raise RuntimeError(
            "h3_masked_prefix: target latent has %d audio steps for %d video "
            "frames; expected %d on H3's 40 Hz grid." %
            (int(audio.shape[-1]), target_frames, expected_audio)
        )
    return video, audio, target_frames


def _encode_imported_audio(audio_vae, audio, frames):
    if audio_vae is None:
        raise ValueError(
            "h3_masked_prefix: imported-video scene 1 needs the H3 audio VAE "
            "connected to Chain Context."
        )
    if audio is None:
        raise ValueError(
            "h3_masked_prefix: imported-video scene 1 has no context audio. "
            "Reconnect source audio to Existing Video Context."
        )
    waveform = audio["waveform"]
    sample_rate = int(audio["sample_rate"])
    vae_rate = int(getattr(audio_vae, "audio_sample_rate", 32000))
    if sample_rate != vae_rate:
        try:
            import torchaudio
        except ImportError as exc:
            raise RuntimeError(
                "h3_masked_prefix: imported audio is %d Hz but the H3 audio "
                "VAE wants %d Hz and torchaudio is unavailable." %
                (sample_rate, vae_rate)
            ) from exc
        waveform = torchaudio.functional.resample(
            waveform, sample_rate, vae_rate)
    wanted = int(round(int(frames) / float(FPS) * vae_rate))
    if int(waveform.shape[-1]) < wanted:
        raise ValueError(
            "h3_masked_prefix: imported context audio is shorter than the "
            "%d-frame masked prefix." % int(frames)
        )
    encoded = audio_vae.encode(waveform[:1, ..., -wanted:].movedim(1, -1))
    if getattr(encoded, "ndim", 0) != 4:
        raise ValueError(
            "h3_masked_prefix: audio VAE returned %s; expected [B,C,2,T]." %
            (tuple(getattr(encoded, "shape", ())),)
        )
    steps = int(round(int(frames) / float(FPS) * AUDIO_HZ))
    if int(encoded.shape[-1]) < steps:
        raise RuntimeError(
            "h3_masked_prefix: %d frames need %d audio steps but the audio "
            "VAE produced %d." % (int(frames), steps, int(encoded.shape[-1]))
        )
    return encoded[:1, ..., -steps:], steps, "imported decoded audio"


def _existing_mask_streams(latent, video, audio):
    mask = latent.get("noise_mask")
    if mask is None:
        return (
            torch.ones(
                (1, 1, int(video.shape[2]), int(video.shape[3]),
                 int(video.shape[4])),
                device=video.device, dtype=torch.float32,
            ),
            torch.ones(
                (1, 1, int(audio.shape[2]), int(audio.shape[3])),
                device=audio.device, dtype=torch.float32,
            ),
        )
    if hasattr(mask, "unbind"):
        parts = list(mask.unbind())
    elif isinstance(mask, (tuple, list)):
        parts = list(mask)
    else:
        raise ValueError(
            "h3_masked_prefix: an existing target noise_mask is not a nested "
            "H3 video/audio mask and cannot be composed safely."
        )
    if len(parts) < 2:
        raise ValueError(
            "h3_masked_prefix: existing target noise_mask has no audio stream."
        )
    video_mask, audio_mask = parts[:2]
    if video_mask.ndim == 4:
        video_mask = video_mask.unsqueeze(0)
    if audio_mask.ndim == 3:
        audio_mask = audio_mask.unsqueeze(0)
    expected_video = (1, 1, int(video.shape[2]), int(video.shape[3]),
                      int(video.shape[4]))
    expected_audio = (1, 1, int(audio.shape[2]), int(audio.shape[3]))
    try:
        video_mask = torch.broadcast_to(video_mask, expected_video).clone()
        audio_mask = torch.broadcast_to(audio_mask, expected_audio).clone()
    except RuntimeError as exc:
        raise ValueError(
            "h3_masked_prefix: existing AV noise-mask shapes %s / %s cannot "
            "broadcast to target %s / %s." %
            (tuple(video_mask.shape), tuple(audio_mask.shape), expected_video,
             expected_audio)
        ) from exc
    return video_mask.float(), audio_mask.float()


def _drop_prefix_guides(conditioning, prefix_frames):
    """Remove target guides that conflict with the preserved latent prefix."""
    out = []
    dropped = []
    for embedding, extra in conditioning:
        metadata = extra.copy()
        kept = []
        for guide in metadata.get("minimax_keyframes") or []:
            position = float(guide.get(
                "resolved_frame_index", guide.get("frame_index", 0)))
            if 0 <= position < int(prefix_frames):
                dropped.append(position)
            else:
                kept.append(guide)
        if "minimax_keyframes" in metadata:
            metadata["minimax_keyframes"] = kept
        out.append([embedding, metadata])
    if dropped:
        _LOG.warning(
            "h3_masked_prefix: dropped %d target guide(s) inside preserved "
            "frames 0..%d; the clean target latent already owns that prefix.",
            len(dropped), int(prefix_frames) - 1,
        )
    return out


def apply_masked_prefix(
    conditioning,
    vae,
    latent,
    previous_frames,
    context_length,
    crop,
    previous_latent=None,
    audio_vae=None,
    previous_audio=None,
):
    """Return conditioning, masked target latent, and repeated trim length."""
    _require_h3_mask_support()
    target_video, target_audio, target_frames = _validate_target_streams(latent)

    if getattr(previous_frames, "ndim", 0) != 4:
        raise ValueError(
            "h3_masked_prefix: previous frames must be IMAGE [N,H,W,C]."
        )
    available = int(previous_frames.shape[0])
    frames = _snap_prefix_length(context_length, available, target_frames)
    width = int(target_video.shape[4]) * 16
    height = int(target_video.shape[3]) * 16
    video_tail = _resize(
        previous_frames[available - frames:], width, height, crop)
    video_prefix = vae.encode(video_tail)
    if getattr(video_prefix, "ndim", 0) != 5:
        raise ValueError(
            "h3_masked_prefix: video VAE returned %s; expected [B,C,T,H,W]." %
            (tuple(getattr(video_prefix, "shape", ())),)
        )
    video_steps = int(video_prefix.shape[2])
    covered = _pixel_frames(video_steps)
    if covered != frames:
        raise RuntimeError(
            "h3_masked_prefix: %d context frames encoded to %d video steps "
            "covering %d frames; refusing a phase-shifted seam." %
            (frames, video_steps, covered)
        )
    if video_steps >= int(target_video.shape[2]):
        raise ValueError(
            "h3_masked_prefix: video prefix consumes the whole target latent."
        )

    if previous_latent is not None:
        audio_prefix, audio_steps, overhang = _audio_tail_from_latent(
            previous_latent, frames)
        audio_source = "previous sampled latent"
        if abs(overhang) > 1e-9:
            _LOG.warning(
                "h3_masked_prefix: predecessor audio grid ends %.3f latent "
                "steps from its last video frame; the copied %d-step prefix "
                "is end-aligned. Use 39/90/141/... context frames for an exact "
                "prefix duration.",
                overhang, audio_steps,
            )
    else:
        audio_prefix, audio_steps, audio_source = _encode_imported_audio(
            audio_vae, previous_audio, frames)
    expected_audio_steps = int(round(frames / float(FPS) * AUDIO_HZ))
    if audio_steps != expected_audio_steps:
        raise RuntimeError(
            "h3_masked_prefix: %d video frames require %d audio steps, got %d."
            % (frames, expected_audio_steps, audio_steps)
        )
    if audio_steps >= int(target_audio.shape[-1]):
        raise ValueError(
            "h3_masked_prefix: audio prefix consumes the whole target latent."
        )

    out_video = target_video.clone()
    out_audio = target_audio.clone()
    vp = video_prefix[:1].to(out_video.device, out_video.dtype)
    ap = audio_prefix[:1].to(out_audio.device, out_audio.dtype)
    if (int(vp.shape[1]) != int(out_video.shape[1])
            or tuple(vp.shape[3:]) != tuple(out_video.shape[3:])):
        raise ValueError(
            "h3_masked_prefix: encoded video prefix shape %s does not match "
            "target %s." % (tuple(vp.shape), tuple(out_video.shape))
        )
    if tuple(ap.shape[1:3]) != tuple(out_audio.shape[1:3]):
        raise ValueError(
            "h3_masked_prefix: audio prefix shape %s does not match target %s."
            % (tuple(ap.shape), tuple(out_audio.shape))
        )
    out_video[:, :, :video_steps] = vp
    out_audio[..., :audio_steps] = ap

    video_mask, audio_mask = _existing_mask_streams(
        latent, out_video, out_audio)
    video_mask[:, :, :video_steps] = 0.0
    audio_mask[..., :audio_steps] = 0.0

    import comfy.nested_tensor

    out_latent = latent.copy()
    out_latent["samples"] = comfy.nested_tensor.NestedTensor(
        (out_video, out_audio))
    out_latent["noise_mask"] = comfy.nested_tensor.NestedTensor(
        (video_mask, audio_mask))
    out_conditioning = _drop_prefix_guides(conditioning, frames)

    _LOG.info(
        "h3_masked_prefix: preserved %d target frames = %d video steps / %d "
        "audio steps (%.3fs, audio from %s); target %d frames at %dx%d; trim %d",
        frames, video_steps, audio_steps, frames / float(FPS), audio_source,
        target_frames, width, height, frames,
    )
    return out_conditioning, out_latent, frames
