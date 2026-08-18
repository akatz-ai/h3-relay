"""Pin previous-clip motion at the head of an H3 clip.

Wire it between a stock H3 conditioning node and the sampler:

    MiniMaxH3ImageToVideo / MiniMaxH3ReferenceToVideo (or the t2v path)
        -> H3 Motion Context
        -> guider / sampler

Two axes to test, both cheap.

encode_mode
  frames  one VAE call per frame, each pinned as its own cond block. The
          model sees N snapshots at N instants.
  video   one VAE call for the whole run. The H3 video VAE has latent_dim
          3, so it reads the batch axis as time and compresses the run
          into fewer latent steps (5 pixel frames -> 2 steps, 22 -> 7).
          Each step becomes one cond block, so the motion between frames
          lives inside the latent instead of being implied across separate
          stills. Far fewer rows and one VAE load.

anchor_mode
  head    pinned frames occupy indices 0..N-1 of the delivered timeline.
          They come back in the output, so trim that many frames off the
          front before concatenating.
  before  pinned frames sit at negative indices, ending at -1, so
          delivered frame 0 continues from them and nothing is wasted.
          Their time coordinates land below text_len, which is the range
          the text rows occupy. Whether that collision matters is exactly
          what this mode is asking.
"""

import logging
import os

import comfy.utils
import folder_paths
import node_helpers
import torch

try:
    from safetensors.torch import load_file as _st_load, save_file as _st_save
except ImportError:  # ComfyUI always ships safetensors; belt and braces
    _st_load = _st_save = None

from .patch_layout import (
    MC_KEY,
    MC_AUDIO_KEY,
    apply_patch as apply_layout_patch,
    claim_patch_ownership as claim_layout_patch_ownership,
    is_applied,
    native_guides_available,
)
from .patch_payload import (
    apply_patch as apply_payload_patch,
    claim_patch_ownership as claim_payload_patch_ownership,
    is_applied as payload_patch_applied,
)

try:
    import torchaudio
except ImportError:
    torchaudio = None

_LOG = logging.getLogger("minimax_h3_context_loop")
_legacy_core_warning_emitted = False

FRAME_PER_TOKEN = (1, 4, 4, 4, 4)
FPS = 24  # H3's native rate; audio latents run at 40 Hz, hence FRAME_RESCALE 5/3
FRAME_RESCALE = 5.0 / 3.0
AUDIO_HZ = 40.0


def _activate_inline_patches():
    """Use core-native guides or opt into the legacy guarded fallback.

    Importing the node pack only registers nodes. Activation happens here,
    immediately before Motion Context produces conditioning. Updated ComfyUI
    remains completely unmodified. Older builds receive a one-time upgrade
    warning before the marker-gated compatibility wrappers are activated.
    """
    global _legacy_core_warning_emitted
    if native_guides_available():
        return "native"
    if not _legacy_core_warning_emitted:
        _LOG.warning(
            "MiniMax H3 Contex Loop 0.4: this ComfyUI build does not include "
            "the native H3 Add Guide API merged in Comfy-Org/ComfyUI PR "
            "#15439. Update ComfyUI for the supported 0.4 path. Falling back "
            "to legacy compatibility patches for this session; they are more "
            "likely to conflict with other H3 extensions.")
        _legacy_core_warning_emitted = True
    layout_ok = apply_layout_patch()
    if not layout_ok or not is_applied():
        raise RuntimeError(
            "h3_motion_context: the inline layout patch could not be enabled, "
            "so continuation guides cannot be placed safely. Check the log "
            "for the self-test failure reason.")
    payload_ok = apply_payload_patch()
    if not payload_ok or not payload_patch_applied():
        raise RuntimeError(
            "h3_motion_context: the inline payload patch could not be enabled. "
            "Ref2VA refs could overwrite the motion-context video latents. "
            "Check the log for the compatibility failure reason.")
    return "legacy"


def _claim_inline_patch_ownership():
    """Explicitly make this pack the active compatible patch-family owner."""
    if native_guides_available():
        return "native guides; core-owned; no compatibility patch required"

    # Patch Priority also acts as an early compatibility check, so surface the
    # same one-time update warning as Motion Context before claiming fallback
    # ownership.
    _activate_inline_patches()
    layout_ok, layout_detail = claim_layout_patch_ownership()
    if not layout_ok or not is_applied():
        raise RuntimeError(
            "h3_motion_context: could not claim the H3 layout patch: %s" %
            layout_detail)
    payload_ok, payload_detail = claim_payload_patch_ownership()
    if not payload_ok or not payload_patch_applied():
        raise RuntimeError(
            "h3_motion_context: could not claim the H3 payload patch: %s" %
            payload_detail)
    return "legacy guides; %s; %s" % (layout_detail, payload_detail)


def _prepare_native_guide_conditioning(conditioning):
    """Select the guide implementation and pass scene conditioning through.

    The merged native API aligns downstream Add Guide nodes relative to the
    Ref2VA target timeline itself, so scene 1 no longer needs the old empty
    marker keyframe. Legacy builds still activate their guarded fallback here.
    """
    _activate_inline_patches()
    return conditioning

# Run lengths the video VAE's downscale formula max(1, (n - 5) // 17 * 5 + 2)
# actually distinguishes. Anything between two grid points encodes to the same
# number of latent steps as the lower one, but the steps then cover the FIRST
# `covered` frames of the input rather than the last: encoding 10 frames yields
# the same 2 steps as encoding 5, representing frames [-10..-6] of the source
# clip instead of [-5..-1]. The pinned run would end five frames early and the
# delivered clip would continue from the wrong instant. So off-grid requests
# are snapped DOWN before slicing, keeping content and coverage in agreement.
VIDEO_RUN_GRID = (
    243, 226, 209, 192, 175, 158, 141, 124,
    107, 90, 73, 56, 39, 22, 5, 1,
)


def _pixel_frames(latent_t):
    """Pixel frames covered by latent_t latent steps."""
    return sum(FRAME_PER_TOKEN[k % 5] for k in range(latent_t))


def _step_offsets(latent_t):
    """Pixel-frame index at which each latent step begins."""
    out, acc = [], 0
    for k in range(latent_t):
        out.append(acc)
        acc += FRAME_PER_TOKEN[k % 5]
    return out


def _resize(image, width, height, crop):
    # image [B, H, W, C] -> [B, height, width, 3]; matches the stock helper
    samples = image[..., :3].movedim(-1, 1)
    samples = comfy.utils.common_upscale(samples, width, height, "lanczos", crop)
    return samples.movedim(1, -1)


def _encode_tail_audio(audio_vae, audio, seconds):
    """Encode the last `seconds` of a clip's audio with the H3 audio VAE.

    Returns ([1, 32, 2, T] latent, T) where T counts 40 Hz latent steps,
    matching what the layout calls ref_audio_t.
    """
    waveform = audio["waveform"]  # [B, C, L]
    sr = int(audio["sample_rate"])
    vae_sr = int(getattr(audio_vae, "audio_sample_rate", 32000))
    if sr != vae_sr:
        if torchaudio is None:
            raise RuntimeError(
                "h3_motion_context: context_audio is %d Hz but the VAE wants %d Hz "
                "and torchaudio is not available to resample." % (sr, vae_sr))
        waveform = torchaudio.functional.resample(waveform, sr, vae_sr)
    want = int(round(seconds * vae_sr))
    have = int(waveform.shape[-1])
    if have < want:
        _LOG.warning("h3_motion_context: context_audio is %.3fs, shorter than the "
                     "%.3fs of pinned video. Pinning what there is.",
                     have / vae_sr, seconds)
    else:
        waveform = waveform[..., have - want:]
    z = audio_vae.encode(waveform[:1].movedim(1, -1))  # [1, 32, 2, T]
    return z, int(z.shape[-1])


def _streams_from_latent(latent):
    """Unpack an H3 AV latent into its contained streams.

    NestedTensor.__getitem__ broadcasts the index into every contained
    tensor rather than selecting one, so samples[0] would strip the batch
    dimension off both streams. unbind() returns the pair.
    """
    samples = latent["samples"]
    if hasattr(samples, "unbind"):
        parts = list(samples.unbind())
    elif isinstance(samples, (tuple, list)):
        parts = list(samples)
    else:
        raise ValueError(
            "h3_motion_context: expected a MiniMax H3 AV latent (a nested "
            "video/audio pair), got %r" % type(samples))
    if not parts:
        raise ValueError("h3_motion_context: AV latent contains no streams")
    return parts


def _video_from_latent(latent):
    """Pull the video stream out of an H3 AV latent."""
    video = _streams_from_latent(latent)[0]
    if video.ndim == 4:  # unbatched [C,T,H,W]
        video = video.unsqueeze(0)
    if video.ndim != 5:
        raise ValueError("h3_motion_context: expected video latent [B,C,T,H,W], "
                         "got shape %s" % (tuple(video.shape),))
    return video


def _audio_tail_from_latent(latent, a_frames):
    """Slice the last `a_frames` worth of audio steps straight out of a
    generated H3 latent, skipping the decode -> re-encode round trip.

    Returns (tail latent [1, C, 2, rt], rt, overhang) where rt counts
    40 Hz latent steps and overhang is the signed fraction of a step between
    the clip's audio grid and its last pixel frame. H3 rounds to the NEAREST
    audio step: 124 frames want 206.67 and become 207 (+0.33), while 260
    frames want 433.33 and become 433 (-0.33). The caller compensates the
    placement with this signed offset so the pinned content lands exactly
    where its samples actually sit.
    """
    parts = _streams_from_latent(latent)
    if len(parts) < 2:
        raise ValueError(
            "h3_motion_context: context_latent has no audio stream. Wire the "
            "sampler output of an H3 AV graph, not a video-only latent.")
    video, audio = parts[0], parts[1]
    if video.ndim == 4:
        video = video.unsqueeze(0)
    if audio.ndim == 3:  # unbatched [C,2,T]
        audio = audio.unsqueeze(0)
    if audio.ndim != 4:
        raise ValueError("h3_motion_context: expected audio latent [B,C,2,T], "
                         "got shape %s" % (tuple(audio.shape),))
    total_t = int(audio.shape[-1])
    frames = _pixel_frames(int(video.shape[2]))
    overhang = total_t - FRAME_RESCALE * frames
    if not (-0.5 < overhang < 0.5):
        _LOG.warning(
            "h3_motion_context: context_latent audio grid is unexpected "
            "(%d steps for %d frames); assuming no overhang.", total_t, frames)
        overhang = 0.0
    rt = int(round(a_frames / float(FPS) * AUDIO_HZ))
    if rt > total_t:
        _LOG.warning("h3_motion_context: asked for %d audio steps, the latent "
                     "has %d. Pinning all of it.", rt, total_t)
        rt = total_t
    if rt < 1:
        raise ValueError("h3_motion_context: audio window is empty")
    tail = audio[:1, ..., total_t - rt:].clone()
    return tail, rt, float(overhang)


class MiniMaxH3MotionContext:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "conditioning": ("CONDITIONING", {
                    "tooltip": "Conditioning from the stock MiniMax H3 Image "
                               "to Video or Reference to Video node. Motion "
                               "Context preserves existing Ref2VA references "
                               "and appends its continuation data."}),
                "vae": ("VAE", {
                    "tooltip": "MiniMax H3 video VAE used to encode the "
                               "previous clip's context frames."}),
                "latent": ("LATENT", {
                    "tooltip": "The NEW clip's empty H3 AV latent from the "
                               "stock conditioning node. Do not connect the "
                               "previous sampled latent here; that belongs on "
                               "the optional context_latent input."}),
                "context_frames": ("IMAGE", {
                    "tooltip": "Decoded, delivered frames from the PREVIOUS "
                               "clip. Supplying the whole clip is safe: only "
                               "the requested tail is used."}),
                "context_length": ("INT", {
                    "default": 5, "min": 1, "max": 243,
                    "tooltip": "Frames of the previous clip to carry over. In "
                               "video mode only native 17-frame grid values "
                               "(1, 5, 22, ... 243) are distinct; "
                               "anything else is snapped DOWN to the nearest so "
                               "the pinned run always ends at the clip's last "
                               "frame."}),
                "encode_mode": (["video", "frames"], {
                    "default": "video",
                    "tooltip": "video: one VAE call, motion lives inside the "
                               "latent, fewer rows. frames: one call per frame, "
                               "each pinned as a separate still."}),
                "anchor_mode": (["head", "before"], {
                    "default": "head",
                    "tooltip": "head: pinned frames occupy the first indices and "
                               "come back in the output, so trim them. before: "
                               "negative indices, nothing wasted, but the "
                               "coordinates overlap the text rows."}),
                "crop": (["disabled", "center"], {
                    "default": "disabled",
                    "tooltip": "How context frames are fitted to the new "
                               "latent canvas. disabled stretches to the "
                               "target size; center preserves aspect ratio "
                               "and center-crops."}),
                "audio_context_length": ("INT", {
                    "default": 22, "min": 0, "max": 240,
                    "tooltip": "Frames of tail audio to pin, independent of the "
                               "video window. 0 follows context_length. In "
                               "timeline mode the window is END-aligned with "
                               "the pinned video, so 22 with a 22-frame video "
                               "window overlays it exactly; longer windows "
                               "extend backwards into vacated coordinate "
                               "space (untested)."}),
                "audio_mode": (["timeline", "ref"], {
                    "default": "timeline",
                    "tooltip": "timeline: pinned audio gets coordinates on "
                               "this clip's own timeline, end-aligned with "
                               "the pinned video, so the model reads it as "
                               "this clip's sound so far and continues it. "
                               "ref: stock placement in a span before the "
                               "clip, which the model imitates (similar "
                               "music, not phase-locked) rather than "
                               "continues."}),
            },
            "optional": {
                "context_latent": ("LATENT", {
                    "tooltip": "Previous clip's SAMPLER OUTPUT latent (the same "
                               "one you wire into the decode nodes). When "
                               "supplied, the pinned audio is sliced straight "
                               "from it, skipping the decode/re-encode round "
                               "trip that dulls sound a little more at every "
                               "link of a chain. Takes priority over "
                               "context_audio; audio_vae is not needed on "
                               "this path."}),
                "audio_vae": ("VAE", {
                    "tooltip": "H3 audio VAE. Supply with context_audio to carry "
                               "the previous clip's tail sound across the join. "
                               "Not needed when context_latent is wired."}),
                "context_audio": ("AUDIO", {
                    "tooltip": "Audio of the previous clip. The tail matching the "
                               "pinned frames is encoded and pinned alongside "
                               "them. Ignored when context_latent is wired."}),
            },
        }

    RETURN_TYPES = ("CONDITIONING", "INT")
    RETURN_NAMES = ("conditioning", "trim_frames")
    OUTPUT_TOOLTIPS = (
        "Conditioning with the previous clip's motion and optional timeline "
        "audio appended. Connect this to the guider/sampler path.",
        "Number of pinned leading frames reproduced in head mode. Connect to "
        "MiniMax H3 Contex Loop Trim; before mode returns 0.",
    )
    FUNCTION = "apply"
    CATEGORY = "conditioning/minimax"
    DESCRIPTION = ("Pin a run of consecutive frames from a previous clip as "
                   "never-denoised conditioning rows, so the model reads real "
                   "motion instead of guessing it from a single still. A stock "
                   "last_frame target is preserved outside the carried head.")

    def apply(self, conditioning, vae, latent, context_frames, context_length,
              encode_mode, anchor_mode, crop, audio_context_length=22,
              audio_mode="timeline", context_latent=None, audio_vae=None,
              context_audio=None):
        guide_api = _activate_inline_patches()
        native_guides = guide_api == "native"

        video = _video_from_latent(latent)
        latent_t = int(video.shape[2])
        width = int(video.shape[4]) * 16
        height = int(video.shape[3]) * 16
        frame_count = _pixel_frames(latent_t)

        available = int(context_frames.shape[0])
        n = min(int(context_length), available)
        if n < 1:
            raise ValueError("h3_motion_context: context_frames is empty")
        if n < context_length:
            _LOG.warning("h3_motion_context: only %d frames supplied, pinning %d",
                         available, n)

        if encode_mode == "video":
            # snap down to the VAE grid BEFORE slicing, so the frames encoded
            # are exactly the frames the latent steps will cover (see
            # VIDEO_RUN_GRID). Slicing the last n and letting the VAE keep the
            # first `covered` of them would pin a run ending before the clip
            # does, and the join would jump by the difference.
            run = next(g for g in VIDEO_RUN_GRID if g <= n)
            if run != n:
                _LOG.warning(
                    "h3_motion_context: %d frames is off the VAE grid; pinning "
                    "the last %d instead (usable runs: %s)",
                    n, run, ", ".join(str(value) for value in
                                     reversed(VIDEO_RUN_GRID)))
            n = run

        if n >= frame_count:
            raise ValueError(
                "h3_motion_context: asked to pin %d frames into a %d frame clip. "
                "The pinned run must be a small fraction of the timeline."
                % (n, frame_count))

        # the LAST n frames of the incoming clip become the pinned run
        tail = _resize(context_frames[available - n:], width, height, crop)

        if encode_mode == "video":
            # one call; the VAE reads the batch axis as time and compresses
            enc = vae.encode(tail)
            if getattr(enc, "ndim", 0) != 5:
                raise ValueError(
                    "h3_motion_context: video-mode encode returned shape %s, "
                    "expected [B,C,T,H,W]. Try encode_mode=frames."
                    % (tuple(getattr(enc, "shape", ())),))
            steps = int(enc.shape[2])
            offsets = _step_offsets(steps)
            covered = _pixel_frames(steps)
            if covered != n:
                # n was snapped to the grid above, so a mismatch here means
                # the VAE's downscale formula changed underneath us and the
                # pinned content no longer lines up with the positions we
                # would write. Refuse rather than render a shifted join.
                raise RuntimeError(
                    "h3_motion_context: %d frames encoded to %d latent steps "
                    "covering %d frames; the VAE grid no longer matches "
                    "VIDEO_RUN_GRID. Upstream VAE change, refusing to run."
                    % (n, steps, covered))
            if native_guides:
                # Core Add Guide accepts one multi-frame latent and lays out
                # all of its video steps. Keeping the encoded run intact is
                # both cheaper and exactly the API PR #15439 intends.
                blocks = [enc]
                offsets = [0]
            else:
                blocks = [enc[:, :, k:k + 1] for k in range(steps)]
            span = covered
        else:
            blocks, offsets = [], []
            for i in range(n):
                blocks.append(vae.encode(tail[i:i + 1]))
                offsets.append(i)
            span = n

        if anchor_mode == "before":
            indices = [o - span for o in offsets]
        else:
            indices = list(offsets)

        keyframes = []
        for p, blk in zip(indices, blocks):
            if native_guides:
                keyframes.append({
                    "resolved_frame_index": p,
                    "latent": blk,
                })
            else:
                keyframes.append({
                    # Legacy core accepts only the first/last frame here; the
                    # real position rides under MC_KEY for the layout patch.
                    "resolved_frame_index": 0,
                    MC_KEY: p,
                    "latent": blk,
                })

        ref_audio_t = 0
        motion_context_audio_ref = None
        timeline_end_frame = None
        a_frames = 0
        audio_src = "off"
        if context_latent is not None or context_audio is not None:
            # the audio window is independent of the video one: audio cond
            # rows cost rows but never cost delivered frames
            a_frames = int(audio_context_length) or span
            if context_latent is not None:
                if context_audio is not None:
                    _LOG.info("h3_motion_context: both context_latent and "
                              "context_audio wired; using the latent (skips "
                              "one VAE round trip).")
                audio_latent, ref_audio_t, overhang = _audio_tail_from_latent(
                    context_latent, a_frames)
                audio_src = "latent"
            else:
                if audio_vae is None:
                    raise ValueError(
                        "h3_motion_context: context_audio supplied without "
                        "audio_vae. Wire the H3 audio VAE, or wire "
                        "context_latent instead.")
                audio_latent, ref_audio_t = _encode_tail_audio(
                    audio_vae, context_audio, a_frames / float(FPS))
                overhang = 0.0  # decoded audio was match_tail-cut at the frame
                audio_src = "vae"
            if audio_mode == "timeline":
                # end-align the audio window with the pinned video: both are
                # the tail of clip A, so both must end at the same instant
                # of the new timeline -- frame `span` in head mode (where
                # A's last frame sits), frame 0 in before mode. On the
                # latent path the sliced content reaches `overhang` of a
                # step past A's last frame (H3 rounds its audio grid up),
                # so the end coordinate moves by exactly that much; the
                # layout patch takes a fractional frame index.
                timeline_end_frame = float(
                    span if anchor_mode == "head" else 0)
                timeline_end_frame += overhang / FRAME_RESCALE
                if native_guides:
                    # Native audio guides are start-anchored in units of video
                    # frames. Convert our end-aligned continuation window to
                    # its exact (occasionally fractional) start position.
                    audio_start = (timeline_end_frame
                                   - ref_audio_t / FRAME_RESCALE)
                    keyframes.append({
                        "resolved_frame_index": audio_start,
                        "audio_latent": audio_latent,
                    })
                else:
                    motion_context_audio_ref = {
                        "kind": "audio",
                        "ref_audio_t": ref_audio_t,
                        "audio_latent": audio_latent,
                        MC_AUDIO_KEY: timeline_end_frame,
                    }
            else:
                motion_context_audio_ref = {
                    "kind": "audio",
                    "ref_audio_t": ref_audio_t,
                    "audio_latent": audio_latent,
                }

        # Preserve a stock last_frame guide. The carried head already owns any
        # first-frame guide at the same coordinates, so guides inside that
        # repeated span are removed. On legacy ComfyUI, tag retained guides with
        # MC_KEY so reference compensation treats the complete target-relative
        # set consistently. Native ComfyUI already anchors the full guide list
        # to the target origin. Ported from NikoDemon80 upstream 0.3.0.
        head_end = span if anchor_mode == "head" else 0
        out = []
        dropped = []
        for embedding, extra in conditioning:
            metadata = extra.copy()
            prior = metadata.get("minimax_keyframes") or []
            prior_frame_count = metadata.get("minimax_frame_count")
            if (prior and prior_frame_count is not None
                    and int(prior_frame_count) != frame_count):
                raise ValueError(
                    "h3_motion_context: the conditioning carries keyframes "
                    "resolved for a %d frame clip, but the latent is %d "
                    "frames. Wire the conditioning and latent from the same "
                    "stock H3 node."
                    % (int(prior_frame_count), frame_count))
            kept = []
            for prior_keyframe in prior:
                position = float(prior_keyframe.get(
                    MC_KEY, prior_keyframe.get("resolved_frame_index", 0)))
                if position < head_end:
                    dropped.append(position)
                    continue
                retained = dict(prior_keyframe)
                if not native_guides:
                    retained[MC_KEY] = position
                kept.append(retained)
            metadata["minimax_keyframes"] = kept + keyframes
            if not native_guides:
                metadata["minimax_frame_count"] = frame_count
            out.append([embedding, metadata])
        if dropped:
            _LOG.warning(
                "h3_motion_context: dropped %d keyframe anchor(s) at "
                "frame(s) %s because the carried head already owns frames "
                "0..%d; a last_frame anchor is preserved.",
                len(dropped), sorted(set(dropped)), head_end - 1)
        if motion_context_audio_ref is not None:
            # Ref2VA multi-reference coexistence design contributed by
            # seitanism in the Banodoco seamless-extension thread. Append so
            # existing image/video/audio refs remain intact.
            out = node_helpers.conditioning_set_values(
                out, {"minimax_refs": [motion_context_audio_ref]}, append=True)

        trim = span if anchor_mode == "head" else 0
        _LOG.info("h3_motion_context: %s/%s, %d frames -> %d cond blocks at "
                  "indices %d..%d, %d frame clip at %dx%d, trim %d, audio %s",
                  encode_mode, anchor_mode, n, len(blocks),
                  indices[0], indices[-1], frame_count, width, height, trim,
                  ("%d frames -> %d latent steps (%.3fs) from %s, %s"
                   % (a_frames, ref_audio_t, ref_audio_t / AUDIO_HZ, audio_src,
                      "on the timeline ending at frame %.3f"
                      % float(timeline_end_frame)
                      if audio_mode == "timeline" else "stock ref placement"))
                  if ref_audio_t else "off")
        return (out, trim)


class MiniMaxH3LoopTrim:
    """Drop the pinned head off a decoded clip, picture and sound together.

    The pinned frames occupy the start of the delivered timeline, so they
    have to come off before concatenating. Trimming only the images would
    leave the audio a full trim_frames longer than the video, and muxing
    those puts the whole soundtrack ahead of the picture by trim_frames/24
    seconds. At 5 frames that is 208ms, silent on ambience but squarely
    offbeat on anything with a pulse.

    So this takes both streams and removes the same span from each: whole
    frames from the images, the matching number of samples from the
    waveform. Wire trim_frames from the motion context node so the count
    follows whatever the encoder actually produced.

    The tail needs the same treatment for a different reason. H3's audio
    latent runs at 40 Hz against 24 fps picture, and FRAME_RESCALE is 5/3.
    ComfyUI rounds the required audio steps to the nearest integer, so some
    valid H3 lengths decode about 8.3 ms long (124 frames) and others about
    8.3 ms short (260 frames). Either error accumulates down a chain. Match
    Tail truncates excess samples or zero-pads a short decode so every
    delivered stream is exactly frames/fps long.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE", {
                    "tooltip": "Decoded images from the CURRENT H3 sample. "
                               "The leading pinned overlap is removed."}),
                "trim_frames": ("INT", {
                    "default": 0, "min": 0, "max": 4096,
                    "tooltip": "Connect trim_frames from MiniMax H3 Contex "
                               "Loop Context. In head mode this removes the "
                               "repeated overlap; before mode supplies 0."}),
            },
            "optional": {
                "audio": ("AUDIO", {
                    "tooltip": "Decoded audio for the same clip. Trimmed by the "
                               "matching duration so sound stays locked to "
                               "picture. Leave unwired for silent clips."}),
                "fps": ("FLOAT", {
                    "default": 24.0, "min": 1.0, "max": 240.0, "step": 0.001,
                    "tooltip": "Frame rate used to convert the trim into an "
                               "audio duration. Must match what you feed "
                               "Create Video."}),
                "match_tail": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Truncate or zero-pad audio so its duration "
                               "equals frames/fps exactly. H3 rounds its 40 Hz "
                               "audio grid to the nearest step, producing about "
                               "8ms of excess or shortage on some lengths."}),
                "retain_overlap_frames": ("INT", {
                    "default": 0, "min": 0, "max": 4096,
                    "tooltip": "Optional visual-only overlap for an external "
                               "stitcher. 0 keeps the normal hard-trim output "
                               "only. A positive value makes the extra image "
                               "output retain up to that many of the final "
                               "repeated context frames. Audio always removes "
                               "the complete overlap."}),
            },
        }

    RETURN_TYPES = ("IMAGE", "AUDIO", "IMAGE", "INT")
    RETURN_NAMES = ("images", "audio", "images_with_overlap",
                    "overlap_frames")
    OUTPUT_TOOLTIPS = (
        "Delivered frames with the repeated leading context removed.",
        "Audio trimmed by the same duration and, when match_tail is enabled, "
        "fitted exactly to the delivered image duration.",
        "Optional blend-ready image stream. When overlap_frames is positive, "
        "this retains only the final requested part of the repeated visual "
        "context before the delivered frames. Audio remains fully trimmed.",
        "Number of repeated leading frames retained in images_with_overlap. "
        "Use as an overlap count in a compatible video stitcher.",
    )
    FUNCTION = "trim"
    CATEGORY = "conditioning/minimax/contex_loop"
    DESCRIPTION = ("Remove the leading pinned frames from a decoded H3 clip, "
                   "trimming picture and sound by the same duration.")

    def trim(self, images, trim_frames, audio=None, fps=24.0, match_tail=True,
             retain_overlap_frames=0):
        n = max(0, int(trim_frames))
        total = int(images.shape[0])
        if n >= total:
            raise ValueError(
                "h3_motion_context: asked to trim %d frames from a %d frame clip"
                % (n, total))
        out_images = images[n:] if n else images
        retained = min(n, max(0, int(retain_overlap_frames)))
        overlap_images = images[n - retained:] if retained else out_images

        out_audio = audio
        if audio is not None:
            waveform = audio["waveform"]
            sr = int(audio["sample_rate"])
            seconds = n / float(fps)
            cut = int(round(seconds * sr))
            length = int(waveform.shape[-1])
            if cut >= length:
                raise ValueError(
                    "h3_motion_context: trimming %.3fs from %.3fs of audio would "
                    "leave nothing. Check that fps matches the clip."
                    % (seconds, length / sr))
            waveform = waveform[..., cut:]

            if match_tail:
                frames_left = total - n
                want = int(round(frames_left / float(fps) * sr))
                have = int(waveform.shape[-1])
                if have > want:
                    over = have - want
                    waveform = waveform[..., :want]
                    _LOG.info("h3_motion_context: tail trimmed %d samples "
                              "(%.2fms) so audio matches %d frames exactly",
                              over, over / sr * 1000.0, frames_left)
                elif have < want:
                    missing = want - have
                    waveform = torch.nn.functional.pad(waveform, (0, missing))
                    _LOG.info("h3_motion_context: tail padded %d zero samples "
                              "(%.2fms) so audio matches %d frames exactly",
                              missing, missing / sr * 1000.0, frames_left)

            out_audio = {"waveform": waveform, "sample_rate": sr}
            _LOG.info("h3_motion_context: %d frames / %.4fs picture, %.4fs sound, "
                      "drift %.2fms",
                      total - n, (total - n) / float(fps),
                      int(waveform.shape[-1]) / sr,
                      abs((total - n) / float(fps) - int(waveform.shape[-1]) / sr) * 1000.0)
        elif n:
            _LOG.info("h3_motion_context: trimmed %d leading frames, %d remain. "
                      "No audio wired; if this clip has sound, mux it through "
                      "this node or it will run %.3fs ahead of the picture.",
                      n, total - n, n / float(fps))

        return (out_images, out_audio, overlap_images, retained)


def _resolve_latent_path(path, clip_index=0):
    """Turn the loader's path input into a concrete file.

    Accepts an absolute path, a path relative to ComfyUI's output folder,
    or a directory (in either form). For a directory:

      clip_index == 0   the NEWEST .safetensors inside is used. Simple,
                        but NOT retry-safe: re-rolling a clip loads the
                        rejected attempt's own save (see the node docs).
                        Its run counter also numbers ATTEMPTS, not clips.
      clip_index  > 0   exactly that clip's slot is loaded: clip 1 is
                        *_00001.safetensors. Auto-mode files carry a
                        trailing underscore (*_00001_.safetensors) and
                        are never matched, because their numbers count
                        runs and could hold a reject.
    """
    p = (path or "").strip().strip('"').strip("'")
    if not p:
        p = "h3_context"
    candidates = [p, os.path.join(folder_paths.get_output_directory(), p)]
    for c in candidates:
        if os.path.isfile(c):
            return c
        if os.path.isdir(c):
            idx = int(clip_index)
            if idx > 0:
                # indexed slots use the natural name: clip 2 lives in
                # *_00002.safetensors. Auto-mode files carry a trailing
                # underscore (*_00002_.safetensors) and are deliberately
                # NOT matched: their numbers count runs, not clips, so a
                # reject could be sitting in any of them.
                endings = ("_%05d.safetensors" % idx,
                           "_clip%03d.safetensors" % idx)  # older versions
                files = [os.path.join(c, f) for f in os.listdir(c)
                         if f.endswith(endings)]
                if not files:
                    near = [f for f in os.listdir(c)
                            if f.endswith("_%05d_.safetensors" % idx)]
                    hint = ""
                    if near:
                        hint = (" Found %s, which is an auto-numbered save "
                                "(trailing underscore = numbered by RUN, so "
                                "it may be a reject). If it really is clip "
                                "%d, rename it to drop the trailing "
                                "underscore: %s" %
                                (near[0], idx,
                                 near[0].replace("_%05d_" % idx,
                                                 "_%05d" % idx)))
                    raise FileNotFoundError(
                        "h3_motion_context: no saved latent for clip %d "
                        "(no *_%05d.safetensors in %s).%s"
                        % (idx, idx, c, hint))
            else:
                files = [os.path.join(c, f) for f in os.listdir(c)
                         if f.endswith(".safetensors")]
                if not files:
                    raise FileNotFoundError(
                        "h3_motion_context: no saved latents in %s. Run a "
                        "clip with the Save Latent node first." % c)
            return max(files, key=os.path.getmtime)
    raise FileNotFoundError(
        "h3_motion_context: %r is neither a file nor a folder (also tried "
        "relative to the ComfyUI output directory)." % p)


class MiniMaxH3MotionContextSaveLatent:
    """Save an H3 AV latent to disk so the NEXT run can load it.

    Wiring the sampler's output straight into context_latent is a cycle:
    the sampler would be consuming its own result. The latent that motion
    context needs is the PREVIOUS clip's, which lives in the previous run
    -- so it has to cross runs through disk, the same way the frames and
    audio already do. Stock Save/Load Latent can't serialise H3's nested
    video/audio pair; this saves the two streams side by side.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "latent": ("LATENT", {
                    "tooltip": "The sampler's output latent (the same one "
                               "you wire into the decode nodes)."}),
                "filename_prefix": ("STRING", {
                    "default": "h3_context/clip",
                    "tooltip": "Saved under the ComfyUI output folder. The "
                               "default keeps all chain latents in one "
                               "folder so the Load node can always pick "
                               "the newest."}),
                "clip_index": ("INT", {
                    "default": 0, "min": 0, "max": 9999,
                    "tooltip": "Which clip of the chain THIS is. Saves to "
                               "that clip's fixed slot, so a re-roll "
                               "overwrites its own reject instead of "
                               "stacking new files. Generating clip 2: "
                               "set 2 here and 1 on the Load node. 0 = "
                               "old behaviour, a new numbered file every "
                               "run (numbers count runs, not clips)."}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("latent_path",)
    OUTPUT_TOOLTIPS = (
        "Absolute path of the saved H3 AV safetensors checkpoint.",
    )
    FUNCTION = "save"
    OUTPUT_NODE = True
    CATEGORY = "conditioning/minimax"
    DESCRIPTION = ("Save the sampler's AV latent so the next run's Motion "
                   "Context node can pin audio from it via the matching "
                   "Load node.")

    def save(self, latent, filename_prefix, clip_index=0):
        if _st_save is None:
            raise RuntimeError("h3_motion_context: safetensors is not "
                               "available; cannot save latents.")
        parts = _streams_from_latent(latent)
        if len(parts) < 2:
            raise ValueError(
                "h3_motion_context: latent has no audio stream; wire the "
                "sampler output of an H3 AV graph.")
        video = parts[0].cpu().contiguous()
        audio = parts[1].cpu().contiguous()
        folder, filename, counter, _, _ = folder_paths.get_save_image_path(
            filename_prefix, folder_paths.get_output_directory())
        if int(clip_index) > 0:
            # fixed slot with the natural name: clip 2 -> *_00002. A
            # re-roll of this clip overwrites its own save, so rejects
            # never accumulate or get loaded later. Auto mode (below)
            # keeps a trailing underscore, which is what excludes its
            # run-numbered files from indexed loading.
            path = os.path.join(folder, "%s_%05d.safetensors"
                                % (filename, int(clip_index)))
        else:
            path = os.path.join(folder, "%s_%05d_.safetensors"
                                % (filename, counter))
        _st_save({"video": video, "audio": audio}, path,
                 metadata={"format": "h3_motion_context_av_v1"})
        _LOG.info("h3_motion_context: saved AV latent to %s (video %s, "
                  "audio %s)", path, tuple(video.shape), tuple(audio.shape))
        return (path,)


class MiniMaxH3MotionContextLoadLatent:
    """Load a saved H3 AV latent for the context_latent input.

    clip_index means exactly what it says: set it to the clip you want to
    CONTINUE FROM, and that clip's slot is loaded. Generating clip 2 from
    clip 1: Load node 1, Save node 2. Re-rolling clip 2 changes nothing --
    it reloads slot 1 and overwrites slot 2's reject. Accept, then bump
    both numbers.

    At 0 it loads the newest file in the folder instead. Simple, but NOT
    retry-safe: a re-roll's newest file is the rejected attempt's own
    save, so the retry gets conditioned on the audio you just rejected.

    The output is ONLY for the Motion Context node's context_latent input.
    It is not a decodable latent -- do not wire it into VAE decode.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "latent_path": ("STRING", {
                    "default": "h3_context",
                    "tooltip": "A saved latent file, or a folder (relative "
                               "paths resolve against the ComfyUI output "
                               "directory). Pointing at a specific FILE "
                               "always loads that file, ignoring "
                               "clip_index."}),
                "clip_index": ("INT", {
                    "default": 0, "min": 0, "max": 9999,
                    "tooltip": "The clip to CONTINUE FROM: that clip's "
                               "slot is loaded. Generating clip 2 from "
                               "clip 1: set 1 here and 2 on the Save "
                               "node. 0 = newest file in the folder "
                               "(NOT retry-safe: a re-roll loads its own "
                               "rejected audio)."}),
            },
        }

    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("context_latent",)
    OUTPUT_TOOLTIPS = (
        "Saved previous-clip AV latent for Motion Context's context_latent "
        "input. Do not send it to VAE Decode.",
    )
    FUNCTION = "load"
    CATEGORY = "conditioning/minimax"
    DESCRIPTION = ("Load a latent saved by H3 Motion Context Save Latent, "
                   "for the context_latent input only.")

    @classmethod
    def IS_CHANGED(cls, latent_path, clip_index=0):
        # the path string stays constant while the file behind it changes
        # (newest save, or an overwritten slot), so cache on the resolved
        # file identity instead -- otherwise ComfyUI would happily serve
        # a stale latent forever
        try:
            p = _resolve_latent_path(latent_path, clip_index)
            return "%s:%d" % (p, os.stat(p).st_mtime_ns)
        except Exception:
            return float("NaN")  # unresolvable: never cache

    def load(self, latent_path, clip_index=0):
        if _st_load is None:
            raise RuntimeError("h3_motion_context: safetensors is not "
                               "available; cannot load latents.")
        path = _resolve_latent_path(latent_path, clip_index)
        data = _st_load(path)
        if "video" not in data or "audio" not in data:
            raise ValueError(
                "h3_motion_context: %s is not an h3_motion_context latent "
                "(missing video/audio streams). Was it saved by the stock "
                "Save Latent node instead?" % path)
        _LOG.info("h3_motion_context: loaded AV latent from %s", path)
        # a plain list, not a NestedTensor: only this repo's context_latent
        # input accepts it, which is the point -- it cannot be mistaken
        # for a decodable latent without failing loudly downstream
        return ({"samples": [data["video"], data["audio"]]},)


# The original public Motion Context / Save / Load ids belong to
# NikoDemon80's upstream pack.  This specialized loop pack keeps its context
# engine internal and exports only the stricter loop trim under a distinct id,
# allowing both custom-node folders to be installed at the same time.
NODE_CLASS_MAPPINGS = {
    "MiniMaxH3LoopTrim": MiniMaxH3LoopTrim,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "MiniMaxH3LoopTrim": "MiniMax H3 Contex Loop Trim",
}
