"""Source-aligned editing on H3's frame grid, with incremental window previews."""
from __future__ import annotations

WINDOW_SIZES = tuple(range(22, 363, 17))


def validate_window_size(value) -> int:
    frames = int(value)
    if frames not in WINDOW_SIZES:
        raise ValueError("Window size must be 22, 39, 56, ... 362 frames (17n + 5)")
    return frames


def window_starts(count: int, window_frames: int = 22) -> list[int]:
    window_frames = validate_window_size(window_frames)
    if count < 1:
        raise ValueError("Source video has no frames")
    starts = [0]
    stride = window_frames - 1
    while starts[-1] + window_frames < count:
        candidate = min(starts[-1] + stride, count - window_frames)
        # An 18-frame history needs a boundary at frame 17 or later.
        # Retain the original 22-frame scheduling behavior for short tails.
        starts.append(max(stride, candidate))
    return starts


class H3RelayGreenMask:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"images": ("IMAGE",), "mask": ("MASK",),
                "expand_pixels": ("INT", {"default": 5, "min": 0, "max": 64})}}
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "apply"
    CATEGORY = "H3 Relay/person remover"

    def apply(self, images, mask, expand_pixels):
        import torch
        import torch.nn.functional as F
        if tuple(mask.shape) != tuple(images.shape[:3]):
            raise ValueError("Provide one aligned mask per source frame at the same resolution")
        mask = mask.to(device=images.device) > 0.5
        if expand_pixels:
            mask = F.max_pool2d(mask.float().unsqueeze(1), 2 * expand_pixels + 1,
                                stride=1, padding=expand_pixels).squeeze(1) > 0
        result = images.clone()
        result[mask] = torch.tensor([0., 1., 0.], device=images.device, dtype=images.dtype)
        return (result,)


class H3RelaySourceHistory:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"conditioning": ("CONDITIONING",), "vae": ("VAE",),
                "latent": ("LATENT",), "previous_frames": ("IMAGE",),
                "previous_audio": ("AUDIO",), "audio_vae": ("VAE",)}}
    RETURN_TYPES = ("CONDITIONING", "LATENT")
    FUNCTION = "apply"
    CATEGORY = "H3 Relay/internal"

    def apply(self, conditioning, vae, latent, previous_frames, previous_audio, audio_vae):
        import nodes
        state = {"index": 1, "external_context": True, "previous_frames": previous_frames,
                 "previous_latent": None, "previous_audio": previous_audio,
                 "plan": {"compatibility": {"continuation_mode": "sliding_history",
                          "context_length": 18, "crop": "disabled", "audio_mode": "generated_audio"},
                          "shots": [{"continuation_mode": "sliding_history"}]}}
        result = nodes.NODE_CLASS_MAPPINGS["H3RelayInternalChainContext"]().apply(
            state=state, conditioning=conditioning, vae=vae, latent=latent, audio_vae=audio_vae)
        if result[1] != 1 or result[2] is not True:
            raise RuntimeError("H3 Relay history contract changed; refusing to sample")
        return result[0], result[3]


class H3RelayRemovalWindow:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"source": ("IMAGE",), "anchor": ("IMAGE",),
                "start": ("INT", {"default": 0})},
                "optional": {"previous": ("H3_REMOVAL_STATE",),
                             "window_frames": ("INT", {"default": 22})}}
    RETURN_TYPES = ("IMAGE", "IMAGE", "IMAGE", "AUDIO")
    RETURN_NAMES = ("source_window", "anchor", "history_frames", "history_audio")
    FUNCTION = "prepare"
    CATEGORY = "H3 Relay/internal"

    def prepare(self, source, anchor, start, previous=None, window_frames=22):
        import torch
        window_frames = validate_window_size(window_frames)
        window = source[start:start + window_frames]
        if len(window) < window_frames:
            window = torch.cat((window, window[-1:].repeat(window_frames - len(window), 1, 1, 1)))
        if previous is None:
            return window, anchor[:1], anchor[:1], {"waveform": torch.zeros(1, 2, 1), "sample_rate": 48000}
        frames, audio = previous["frames"], previous["audio"]
        if start < 17 or start >= len(frames):
            raise ValueError("Missing generated boundary/history for source window")
        rate = audio["sample_rate"]
        history_audio = {"sample_rate": rate,
            "waveform": audio["waveform"][..., round((start - 17) * rate / 24):round((start + 1) * rate / 24)]}
        return window, frames[start:start + 1], frames[start - 17:start + 1], history_audio


class H3RelayRemovalAppend:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"frames": ("IMAGE",), "audio": ("AUDIO",),
                "start": ("INT", {"default": 0}), "source_count": ("INT", {"default": 22})},
                "optional": {"previous": ("H3_REMOVAL_STATE",),
                             "window_frames": ("INT", {"default": 22}),
                             "preview_run": ("STRING", {"default": ""}),
                             "window_index": ("INT", {"default": 0}),
                             "cache_scope": ("STRING", {"default": ""}),
                             "config_key": ("STRING", {"default": ""}),
                             "window_seed": ("STRING", {"default": "0"}),
                             "reuse_record": ("STRING", {"default": ""})}}
    RETURN_TYPES = ("H3_REMOVAL_STATE", "IMAGE", "AUDIO")
    FUNCTION = "append"
    CATEGORY = "H3 Relay/internal"

    def append(self, frames, audio, start, source_count, previous=None,
               window_frames=22, preview_run="", window_index=0,
               cache_scope="", config_key="", window_seed="0", reuse_record=""):
        import torch
        import torch.nn.functional as F
        window_frames = validate_window_size(window_frames)
        if len(frames) != window_frames:
            raise ValueError(f"Expected {window_frames} decoded frames, received {len(frames)}")
        rate = int(audio["sample_rate"])
        end = min(source_count, start + window_frames)
        retained = 0 if previous is None else len(previous["frames"]) - start
        if not 0 <= retained < window_frames:
            raise ValueError("Invalid source-aligned overlap")
        records = [] if previous is None else previous.get("records", [])
        record = None
        if cache_scope:
            from . import removal_cache
            record = (removal_cache.metadata(cache_scope, reuse_record) if reuse_record else
                      removal_cache.save(cache_scope, config_key, window_index, start, window_frames,
                                         source_count, window_seed, records, frames, audio))
            if record["config_key"] != config_key or record["previous_records"] != records:
                raise ValueError("Saved window history does not match this render")
        if preview_run:
            # This completes before the dependent continuation can start.
            from .removal_previews import publish_window
            publish_window(preview_run, window_index, frames[:end - start], start, retained,
                           record=record, reused=bool(reuse_record))
        origin_sample = round(start * rate / 24)
        samples = round((start + window_frames) * rate / 24) - origin_sample
        waveform = audio["waveform"][..., :samples]
        waveform = F.pad(waveform, (0, max(0, samples - waveform.shape[-1])))
        first_sample = round((start + retained) * rate / 24) - origin_sample
        last_sample = round(end * rate / 24) - origin_sample
        waveform = waveform[..., first_sample:last_sample]
        frames = frames[retained:end - start]
        if previous is not None:
            if previous["audio"]["sample_rate"] != rate:
                raise ValueError("Generated audio sample rate changed between windows")
            frames = torch.cat((previous["frames"], frames))
            waveform = torch.cat((previous["audio"]["waveform"], waveform), dim=-1)
        if waveform.shape[-1] != round(end * rate / 24):
            raise ValueError("Generated audio no longer matches the source clock")
        if len(frames) != end:
            raise ValueError("Assembly no longer matches source frame positions")
        audio = {"waveform": waveform, "sample_rate": rate}
        state = {"frames": frames, "audio": audio,
                 "records": records + ([record["record_id"]] if record else [])}
        return state, frames, audio


class H3RelayRemovalRestore:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"cache_scope": ("STRING",), "record_id": ("STRING",),
                             "config_key": ("STRING",)}}
    RETURN_TYPES = ("IMAGE", "AUDIO")
    FUNCTION = "restore"
    CATEGORY = "H3 Relay/internal"

    def restore(self, cache_scope, record_id, config_key):
        from .removal_cache import load
        return load(cache_scope, record_id, config_key)


class H3RelayPersonRemover:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "model": ("MODEL",), "clip": ("CLIP",), "video_vae": ("VAE",), "audio_vae": ("VAE",),
            "masked_source": ("IMAGE",), "clean_first_frame": ("IMAGE",),
            "fps": ("FLOAT", {"default": 24.}),
            "prompt": ("STRING", {"multiline": True, "default": "Remove the green-masked person and reconstruct the background. Preserve the rest of the video, including its camera motion and frame timing."}),
            "seed": ("INT", {"default": 904234, "min": 0, "max": 0xffffffffffffffff}),
            "steps": ("INT", {"default": 12, "min": 1, "max": 100}),
        }, "optional": {
            "window_frames": ([str(n) for n in WINDOW_SIZES], {
                "default": "22",
                "tooltip": "H3 uses 17n + 5 frames at 24 fps: 22 (0.92s), 39 (1.63s), "
                           "56 (2.33s), 73 (3.04s), 90 (3.75s), 107 (4.46s), 124 (5.17s), "
                           "up to 362 (15.08s). 22 is the proven removal default. "
                           "Larger windows need more memory and may remove people less reliably. "
                           "The relay always carries 18 generated history frames."}),
            "window_controls": ("STRING", {"default": "{}"}),
        }, "hidden": {"unique_id": "UNIQUE_ID", "extra_pnginfo": "EXTRA_PNGINFO",
                       "execution_prompt": "PROMPT"}}
    RETURN_TYPES = ("IMAGE", "AUDIO")
    RETURN_NAMES = ("clean_frames", "generated_audio")
    FUNCTION = "generate"
    CATEGORY = "H3 Relay/person remover"

    def generate(self, model, clip, video_vae, audio_vae, masked_source, clean_first_frame,
                 fps, prompt, seed, steps, window_frames="22", window_controls="{}",
                 unique_id=None, extra_pnginfo=None, execution_prompt=None):
        from comfy_execution.graph_utils import GraphBuilder
        window_frames = validate_window_size(window_frames)
        count, height, width, channels = masked_source.shape
        if abs(fps - 24) > 0.001:
            raise ValueError("Person Remover V1 requires a 24 fps source; resample before masking")
        if width % 32 or height % 32 or channels != 3:
            raise ValueError("Use RGB source dimensions divisible by 32")
        if tuple(clean_first_frame.shape[1:]) != (height, width, channels):
            raise ValueError("Clean anchor must match the source resolution and framing")
        starts = window_starts(count, window_frames)
        from . import removal_cache
        cache_scope = removal_cache.scope_key(unique_id, extra_pnginfo)
        config_key = removal_cache.configuration_key(masked_source, clean_first_frame,
            {"fps": fps, "prompt": prompt, "steps": steps, "window_frames": window_frames},
            execution_prompt, unique_id)
        choices = removal_cache.plan(window_controls, cache_scope, config_key, starts, window_frames, count, seed)
        import logging
        logging.info("H3 Person Remover: Qwen encoder load device=%s; locked windows=%d/%d",
                     getattr(getattr(clip, "patcher", None), "load_device", "unknown"),
                     sum(bool(c["record_id"]) for c in choices), len(starts))
        from .removal_previews import begin_run
        preview_run = begin_run(unique_id, extra_pnginfo, starts, window_frames, count, seed)
        graph = GraphBuilder()
        sampler = graph.node("KSamplerSelect", sampler_name="er_sde")
        sigmas = graph.node("BasicScheduler", model=model, scheduler="simple", steps=steps, denoise=1.)
        previous = None
        for index, start in enumerate(starts):
            args = {} if previous is None else {"previous": previous.out(0)}
            choice = choices[index]
            append_args = dict(start=start, source_count=count, window_frames=window_frames,
                               preview_run=preview_run, window_index=index, cache_scope=cache_scope,
                               config_key=config_key, window_seed=str(choice["seed"]), **args)
            if choice["record_id"]:
                restored = graph.node("H3RelayRemovalRestore", cache_scope=cache_scope,
                                      config_key=config_key, record_id=choice["record_id"])
                previous = graph.node("H3RelayRemovalAppend", frames=restored.out(0), audio=restored.out(1),
                                      reuse_record=choice["record_id"], **append_args)
                continue
            noise = graph.node("RandomNoise", noise_seed=choice["seed"])
            window = graph.node("H3RelayRemovalWindow", source=masked_source,
                                anchor=clean_first_frame, start=start, window_frames=window_frames, **args)
            ref = graph.node("MiniMaxH3ReferenceToVideo", prompt=prompt, width=width, height=height,
                             length=window_frames if previous is None else window_frames + 17,
                             ref_image_size="match", clip=clip,
                             **{"ref_images.ref_image_0": window.out(1), "ref_videos.ref_video_0": window.out(0)})
            conditioning, latent = ref.out(0), ref.out(1)
            if previous is not None:
                history = graph.node("H3RelaySourceHistory", conditioning=conditioning, latent=latent,
                                     vae=video_vae, audio_vae=audio_vae, previous_frames=window.out(2),
                                     previous_audio=window.out(3))
                conditioning, latent = history.out(0), history.out(1)
            guide = graph.node("MiniMaxH3AddGuide", positive=conditioning, latent=latent,
                               vae=video_vae, image=window.out(0), frame_idx=0)
            guider = graph.node("CFGGuider", model=model, positive=guide.out(0), negative=guide.out(0), cfg=1.)
            sample = graph.node("SamplerCustomAdvanced", noise=noise.out(0), guider=guider.out(0),
                                sampler=sampler.out(0), sigmas=sigmas.out(0), latent_image=latent)
            separate = graph.node("LTXVSeparateAVLatent", av_latent=sample.out(0))
            video = graph.node("VAEDecode", samples=separate.out(0), vae=video_vae)
            audio = graph.node("VAEDecodeAudio", samples=separate.out(1), vae=audio_vae)
            previous = graph.node("H3RelayRemovalAppend", frames=video.out(0), audio=audio.out(0),
                                  **append_args)
        return {"result": (previous.out(1), previous.out(2)), "expand": graph.finalize(),
                "ui": {"h3_removal_run": [preview_run]}}


NODE_CLASS_MAPPINGS = {c.__name__: c for c in (H3RelayGreenMask, H3RelaySourceHistory,
    H3RelayRemovalWindow, H3RelayRemovalAppend, H3RelayRemovalRestore, H3RelayPersonRemover)}
NODE_DISPLAY_NAME_MAPPINGS = {
    "H3RelayGreenMask": "H3 Relay Green Mask",
    "H3RelayPersonRemover": "H3 Relay Person Remover",
}
