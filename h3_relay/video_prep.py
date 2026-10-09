"""CPU video preparation for the interactive H3 review workflow.

Models remain in the host-mounted store. No scheduler or generation side effects.
"""
import logging
import json
import math
import subprocess
import tempfile
from fractions import Fraction
from pathlib import Path

import torch
from comfy_api.latest._input_impl.video_types import VideoFromFile, VideoFromComponents
from comfy_api.latest._util import VideoComponents

LOG = logging.getLogger(__name__)
FPS = 24
MAX_FRAMES = 3592  # largest 17n+5 value within the installed H3 node's 3600 limit
PIXEL_BUDGETS = {"480": 500_000, "768": 1_000_000}


def canvas_dimensions(width, height, resolution):
    """Nearest shape to the ideal area-scaled canvas, strictly within the cap.

    Squared log distances give each axis equal relative weight. Searching the
    32px grid avoids independently rounding both axes up past the pixel budget.
    """
    budget = PIXEL_BUDGETS[resolution]
    if not all(math.isfinite(x) and x > 0 for x in (width, height)):
        raise ValueError("Video has invalid display dimensions.")
    ratio = width / height
    ideal_width, ideal_height = math.sqrt(budget * ratio), math.sqrt(budget / ratio)
    candidates = []
    for w in range(32, min(16384, budget // 32) + 1, 32):
        max_h = min(16384, budget // w // 32 * 32)
        for h in {max(32, min(max_h, int(ideal_height // 32) * 32)),
                  max(32, min(max_h, math.ceil(ideal_height / 32) * 32))}:
            score = math.log(w / ideal_width) ** 2 + math.log(h / ideal_height) ** 2
            candidates.append((score, -(w * h), w, h))
    _, _, w, h = min(candidates)
    return w, h


def display_dimensions(stream):
    sar = stream.get("sample_aspect_ratio", "1:1")
    try:
        sar = float(Fraction(sar.replace(":", "/")))
        if sar <= 0:
            sar = 1.0
    except (ValueError, ZeroDivisionError):
        sar = 1.0
    width, height = stream["width"] * sar, stream["height"]
    rotation = next((s["rotation"] for s in stream.get("side_data_list", []) if "rotation" in s), 0)
    if round(rotation) % 180 == 90:
        width, height = height, width
    return width, height


def window_length(count):
    if not 1 <= count <= MAX_FRAMES:
        raise ValueError(f"Video must contain 1–{MAX_FRAMES} frames at 24 fps; got {count}.")
    return 5 + 17 * max(0, (count - 5 + 16) // 17)


def requested_window(duration_seconds):
    """0 keeps the full source; positive durations snap to the nearest H3 grid."""
    if not math.isfinite(duration_seconds) or not 0 <= duration_seconds <= MAX_FRAMES / FPS:
        raise ValueError(f"Duration must be between 0 (full clip) and {MAX_FRAMES/FPS:.3f} seconds.")
    if duration_seconds == 0:
        return None
    n = max(0, math.floor((duration_seconds * FPS - 5) / 17 + 0.5))
    return min(MAX_FRAMES, 5 + 17 * n)


class LabsH3PrepareVideo:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "video": ("VIDEO",),
            "resolution": (["480", "768"], {"default": "480",
                "tooltip": "480 = up to 0.5 megapixels; 768 = up to 1.0 megapixel. Automatically upscale/downscale near the source aspect ratio, with a small centered crop to H3's 32-pixel grid."}),
        }, "optional": {
            "duration_seconds": ("FLOAT", {"default": 0.0, "min": 0.0,
                "max": math.floor(MAX_FRAMES / FPS * 100) / 100, "step": 0.1,
                "tooltip": "0 = full clip. Otherwise keep the beginning and trim the tail to the nearest 17n+5 window at 24 fps: 5s → 124 frames, 6s → 141, 7s → 175. Never extend beyond the source; actual duration is shown in the summary."}),
        }}

    RETURN_TYPES = ("IMAGE", "AUDIO", "INT", "INT", "INT", "INT", "FLOAT", "STRING", "VIDEO")
    RETURN_NAMES = ("reference_frames", "source_audio", "width", "height", "h3_length", "source_frames", "fps", "summary", "prepared_video")
    FUNCTION = "prepare"
    CATEGORY = "Akatz Labs/video"

    def prepare(self, video, resolution, duration_seconds=0.0):
        if resolution not in ("480", "768"):
            raise ValueError("Choose 480 (0.5 MP) or 768 (1.0 MP).")
        requested = requested_window(duration_seconds)
        # Save through the VideoInput API: this honors existing trim/crop wrappers,
        # unlike reading their underlying get_stream_source() directly.
        source = video.get_stream_source()
        start, duration = video.get_active_trim_window()
        with tempfile.TemporaryDirectory(prefix="labs-h3-prepare-") as directory:
            if not isinstance(source, str) or start or duration or getattr(video, "_VideoFromFile__crop", None):
                source = str(Path(directory) / "source.mkv")
                video.save_to(source)
            output = str(Path(directory) / "prepared.mkv")
            probe = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0",
                "-show_entries", "stream=width,height,sample_aspect_ratio:stream_side_data=rotation",
                "-of", "json", source], check=True, capture_output=True, timeout=30)
            streams = json.loads(probe.stdout).get("streams", [])
            if not streams:
                raise ValueError("The upload does not contain a video stream.")
            target_width, target_height = canvas_dimensions(*display_dimensions(streams[0]), resolution)
            # FFmpeg handles display rotation; dar includes non-square pixels.
            # Scale before decoding to a float tensor, then crop (never stretch).
            filters = (
                "setpts=PTS-STARTPTS,fps=fps=24:round=near:eof_action=pass,"
                f"scale=w='ceil(max({target_width},{target_height}*dar))':"
                f"h='ceil(max({target_height},{target_width}/dar))':flags=lanczos,"
                f"setsar=1,crop=w={target_width}:h={target_height}"
            )
            frame_limit = requested if requested is not None else MAX_FRAMES + 1
            audio_filter = "asetpts=PTS-STARTPTS"
            if requested is not None:
                audio_filter += f",atrim=end={requested/FPS:.12f}"
            command = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", "-y",
                "-i", source, "-map", "0:v:0", "-map", "0:a:0?", "-vf", filters,
                "-af", audio_filter, "-frames:v", str(frame_limit),
                "-c:v", "ffv1", "-threads", "4", "-c:a", "pcm_s16le", output]
            try:
                subprocess.run(command, check=True, capture_output=True, timeout=180)
            except subprocess.CalledProcessError as error:
                raise ValueError("Could not prepare the uploaded video: " + error.stderr.decode(errors="replace")[-1200:]) from error
            components = VideoFromFile(output).get_components()
        images, audio = components.images, components.audio
        count, height, width = images.shape[:3]
        length = window_length(count)
        if (width, height) != (target_width, target_height) or width * height > PIXEL_BUDGETS[resolution]:
            raise ValueError(f"Unexpected prepared dimensions: {width} x {height}")
        # Clip audio to the unpadded video timeline; never stretch speech.
        if audio is not None:
            audio = dict(audio, waveform=audio["waveform"][..., :round(count / FPS * audio["sample_rate"])])
        preview = VideoFromComponents(VideoComponents(images=images, audio=audio, frame_rate=Fraction(FPS)))
        if length > count:
            images = torch.cat([images, images[-1:].expand(length - count, -1, -1, -1)], dim=0)
        summary = (f"{width} × {height} · {width*height/1_000_000:.3f} MP / {PIXEL_BUDGETS[resolution]/1_000_000:.1f} MP budget\n"
                   f"24 fps · {count} source frames ({count/FPS:.3f}s)\n"
                   f"H3: {length} frames · {length-count} repeated tail frames · output trimmed to {count}\n"
                   "Aspect-preserving scale; centered crop to 32-pixel grid.")
        if requested is not None:
            summary += f"\nRequested {duration_seconds:g}s → nearest H3 window {requested} frames ({requested/FPS:.3f}s)."
            if count < requested:
                summary += f" Source ends earlier; output stays at {count/FPS:.3f}s."
        else:
            summary += "\nDuration: full clip."
        if count > 362:
            summary += "\nLong clip: beyond H3's documented ~5–15s trained range; higher memory use and quality unverified."
        LOG.info("H3 video prepared width=%s height=%s frames=%s window=%s", width, height, count, length)
        return (images, audio, width, height, length, count, float(FPS), summary, preview)


class LabsH3TrimOutput:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"images": ("IMAGE",), "source_frames": ("INT", {"default": 120, "min": 1, "max": MAX_FRAMES})}}

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("trimmed_frames",)
    FUNCTION = "trim"
    CATEGORY = "Akatz Labs/video"

    def trim(self, images, source_frames):
        if images.shape[0] == source_frames - 1:
            # Installed H3 VAE can decode one fewer frame at an exact window
            # boundary. Hold its final frame for 1/24s to preserve A/V duration.
            LOG.warning("H3 decoded one frame short; repeating the final frame to preserve source duration")
            images = torch.cat([images, images[-1:]], dim=0)
        if images.shape[0] < source_frames:
            raise ValueError(f"H3 returned {images.shape[0]} frames; expected at least {source_frames}. Check the length connections.")
        return (images[:source_frames],)


NODE_CLASS_MAPPINGS = {"LabsH3PrepareVideo": LabsH3PrepareVideo, "LabsH3TrimOutput": LabsH3TrimOutput}
NODE_DISPLAY_NAME_MAPPINGS = {"LabsH3PrepareVideo": "H3 Prepare Video · Auto Size & Length", "LabsH3TrimOutput": "H3 Trim Output to Source Length"}
