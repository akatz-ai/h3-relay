"""General source-aligned Ref2VA editing with configurable generated history.

Uses the same native sliding-history engine and append contract as Person Remover,
but keeps character reference conditioning separate from source-video conditioning.
"""
from __future__ import annotations

import json
import logging

WINDOWS = tuple(range(22, 363, 17))
HISTORY = (0, 1, *range(18, 342, 17))
LOG = logging.getLogger(__name__)
WINDOW_POLICY = "adaptive-target-v1"


def validate_settings(window_frames, history_frames):
    window, history = int(window_frames), int(history_frames)
    if window not in WINDOWS:
        raise ValueError("Window must be 22, 39, 56, ... 362 frames (17n+5).")
    if history not in HISTORY:
        raise ValueError("History must be 0, 1, or 18, 35, 52, ... (17n+1).")
    if history >= window:
        raise ValueError("History must be smaller than the generation window.")
    return window, history


def plan_windows(count, window_frames=124, history_frames=18):
    window, history = validate_settings(window_frames, history_frames)
    if count < 1:
        raise ValueError("Source video has no frames.")
    # History precedes the target; only its boundary frame is regenerated.
    # Do not move the tail backwards and resample an already accepted prefix.
    result, covered = [], 0
    while covered < count:
        effective_history = history if result else 0
        start = covered - (1 if effective_history else 0)
        needed = min(count - start, window)
        # The first target has no history, even when the configured history is long.
        size = next(v for v in WINDOWS if v >= needed and v > effective_history)
        end = min(count, start + size)
        result.append({"index": len(result), "source_start": start, "source_end_exclusive": end,
            "window_frames": size, "padded_tail_frames": size - (end - start),
            "history_frames": effective_history,
            "history_start": start - effective_history + 1 if effective_history else None,
            "history_end_exclusive": start + 1 if effective_history else None,
            "discarded_overlap": covered - start, "delivered_frames": end - covered})
        covered = end
    return result


def slice_audio(audio, start, end):
    if audio is None:
        return None
    rate = audio["sample_rate"]
    return {"sample_rate": rate, "waveform": audio["waveform"][..., round(start*rate/24):round(end*rate/24)]}


class H3RelayEditWindow:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"source": ("IMAGE",), "start": ("INT",),
            "window_frames": ("INT",), "history_frames": ("INT",)},
            "optional": {"previous": ("H3_REMOVAL_STATE",), "source_audio": ("AUDIO",)}}
    RETURN_TYPES = ("IMAGE", "IMAGE", "AUDIO", "AUDIO")
    RETURN_NAMES = ("source_window", "generated_history", "generated_history_audio", "source_window_audio")
    FUNCTION = "prepare"
    CATEGORY = "H3 Relay/internal"

    def prepare(self, source, start, window_frames, history_frames, previous=None, source_audio=None):
        import torch
        window, history = validate_settings(window_frames, history_frames)
        if not 0 <= start < len(source):
            raise ValueError("Source window starts outside the video.")
        frames = source[start:start+window]
        if len(frames) < window:
            frames = torch.cat((frames, frames[-1:].expand(window-len(frames), -1, -1, -1)))
        prior_frames, prior_audio = frames[:1], None
        if previous is not None and history:
            if start < history-1 or start >= len(previous["frames"]):
                raise ValueError("Missing generated history at the exact source boundary.")
            prior_frames = previous["frames"][start-history+1:start+1]
            prior_audio = slice_audio(previous["audio"], start-history+1, start+1)
        return frames, prior_frames, prior_audio, slice_audio(source_audio, start, min(len(source), start+window))


class H3RelayEditHistory:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"conditioning": ("CONDITIONING",), "latent": ("LATENT",),
            "vae": ("VAE",), "audio_vae": ("VAE",), "previous_frames": ("IMAGE",),
            "previous_audio": ("AUDIO",), "history_frames": ("INT",)}}
    RETURN_TYPES = ("CONDITIONING", "LATENT")
    FUNCTION = "apply"
    CATEGORY = "H3 Relay/internal"

    def apply(self, conditioning, latent, vae, audio_vae, previous_frames, previous_audio, history_frames):
        from .vendor.context_loop.sliding_context import apply_sliding_history
        positive, target, trim = apply_sliding_history(conditioning, vae, latent,
            previous_frames, int(history_frames), "disabled", audio_vae=audio_vae,
            previous_audio=previous_audio)
        if trim != 1:
            raise RuntimeError("H3 history boundary contract changed.")
        return positive, target


class H3RelayEditAppend:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"frames": ("IMAGE",), "audio": ("AUDIO",), "start": ("INT",),
            "source_count": ("INT",), "window_frames": ("INT",)},
            "optional": {"previous": ("H3_REMOVAL_STATE",),
                "preview_run": ("STRING", {"default": ""}),
                "window_index": ("INT", {"default": 0}),
                "cache_scope": ("STRING", {"default": ""}),
                "config_key": ("STRING", {"default": ""}),
                "window_seed": ("STRING", {"default": "0"}),
                "reuse_record": ("STRING", {"default": ""}),
                "window_metadata": ("STRING", {"default": ""})}}
    RETURN_TYPES = ("H3_REMOVAL_STATE", "IMAGE", "AUDIO")
    FUNCTION = "append"
    CATEGORY = "H3 Relay/internal"

    def append(self, frames, audio, start, source_count, window_frames, previous=None,
               preview_run="", window_index=0, cache_scope="", config_key="",
               window_seed="0", reuse_record="", window_metadata=""):
        import torch
        from .person_remover import H3RelayRemovalAppend
        if len(frames) == window_frames - 1:
            LOG.warning("H3 edit window at %d decoded one frame short; holding its final frame", start)
            frames = torch.cat((frames, frames[-1:]))
        result = H3RelayRemovalAppend().append(frames, audio, start, source_count,
                                              previous, window_frames=window_frames,
                                              preview_run=preview_run, window_index=window_index,
                                              cache_scope=cache_scope, config_key=config_key,
                                              window_seed=window_seed, reuse_record=reuse_record,
                                              window_metadata=json.loads(window_metadata) if window_metadata else None)
        LOG.info("H3 edit assembled source frames [0, %d); window source start=%d size=%d",
                 len(result[1]), start, window_frames)
        return result


class H3RelayWindowedEdit:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "model": ("MODEL",), "clip": ("CLIP",), "video_vae": ("VAE",), "audio_vae": ("VAE",),
            "source": ("IMAGE",), "reference_image": ("IMAGE",),
            "fps": ("FLOAT", {"default": 24.}),
            "prompt": ("STRING", {"multiline": True, "default": "Replace the person in <Video 1> with the character in <Picture 1>. Preserve motion, framing and scene."}),
            "window_frames": ([str(v) for v in WINDOWS], {"default": "124", "tooltip": "Maximum generated frames per window; short clips and tails shrink automatically, excluding past history: 107 = 4.46s, 124 = 5.17s. Valid H3 sizes are 17n+5."}),
            "history_frames": ([str(v) for v in HISTORY], {"default": "18", "tooltip": "0 = independent windows; 1 = generated boundary only; 18/35/52... = H3 sliding history. Includes one boundary frame. Must be smaller than window_frames."}),
            "seed": ("INT", {"default": 904234, "min": 0, "max": 0xffffffffffffffff,
                "tooltip": "Master seed. Window 1 uses this value; later windows use master + window index. Changing it starts fresh windows and clears per-window overrides. Keep fixed to reroll individual windows."}),
            "steps": ("INT", {"default": 8, "min": 1, "max": 100}),
            "cfg": ("FLOAT", {"default": 1., "min": 0., "max": 100.}),
            "sampler_name": ("STRING", {"default": "er_sde"}),
            "scheduler": ("STRING", {"default": "simple"}),
            "denoise": ("FLOAT", {"default": 1., "min": 0., "max": 1.}),
        }, "optional": {"source_audio": ("AUDIO",),
            "source_count": ("INT", {"default": 0, "min": 0, "tooltip": "0 uses all input frames. Connect preparation's source_frames to exclude its model-padding tail."}),
            "window_controls": ("STRING", {"default": "{}"})},
            "hidden": {"unique_id": "UNIQUE_ID", "extra_pnginfo": "EXTRA_PNGINFO",
                       "execution_prompt": "PROMPT"}}
    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("edited_frames", "window_plan")
    FUNCTION = "generate"
    CATEGORY = "H3 Relay"

    def generate(self, model, clip, video_vae, audio_vae, source, reference_image, fps,
                 prompt, window_frames="124", history_frames="18", seed=904234, steps=8,
                 cfg=1., sampler_name="er_sde", scheduler="simple", denoise=1.,
                 source_audio=None, source_count=0, window_controls="{}",
                 unique_id=None, extra_pnginfo=None, execution_prompt=None):
        from comfy_execution.graph_utils import GraphBuilder
        window, history = validate_settings(window_frames, history_frames)
        if abs(fps-24.) > 0.001:
            raise ValueError("Windowed H3 editing requires source video at 24 fps.")
        if source_count < 0 or source_count > len(source):
            raise ValueError("source_count exceeds the prepared source.")
        count = source_count or len(source)
        source = source[:count]
        _, height, width, channels = source.shape
        if channels != 3 or width % 32 or height % 32:
            raise ValueError("Source must be RGB with dimensions divisible by 32.")
        if len(reference_image) < 1:
            raise ValueError("Provide a character/reference image.")
        reference_image = reference_image[:1]
        windows = plan_windows(count, window, history)
        if len(windows) > 1 and history >= 18:
            from .vendor.context_loop.sliding_context import require_sliding_history_support
            require_sliding_history_support()  # fail before the first GPU window
        plan = json.dumps({"fps": 24, "source_frames": count, "window_frames": window,
            "history_frames": history, "window_policy": WINDOW_POLICY, "windows": windows}, indent=2)
        LOG.info("H3 windowed edit plan: %s", plan)
        from . import removal_cache
        from .removal_previews import begin_run
        starts = [entry["source_start"] for entry in windows]
        cache_scope = removal_cache.scope_key(unique_id, extra_pnginfo)
        settings = {"kind": "windowed_ref_edit_v1", "window_policy": WINDOW_POLICY,
            "window_plan": windows, "fps": fps, "prompt": prompt,
            "master_seed": str(seed), "seed_scheme": removal_cache.SEED_SCHEME,
            "steps": steps, "window_frames": window, "history_frames": history,
            "cfg": cfg, "sampler_name": sampler_name, "scheduler": scheduler, "denoise": denoise,
            "source_audio": None if source_audio is None else {
                "sample_rate": source_audio["sample_rate"],
                "sha256": removal_cache.tensor_hash(source_audio["waveform"])}}
        config_key = removal_cache.configuration_key(source, reference_image, settings,
                                                      execution_prompt, unique_id)
        # Migrate pre-binding workflows, including unlocked per-window seeds.
        # Consult only the preview explicitly belonging to this workflow/node.
        previous_config = None
        workflow = (extra_pnginfo or {}).get("workflow", {})
        node = next((n for n in workflow.get("nodes", []) if str(n.get("id")) == str(unique_id)), {})
        previous_run_id = node.get("properties", {}).get("h3_removal_preview_run")
        if previous_run_id:
            from .removal_previews import latest_run
            old = latest_run(str(workflow.get("id", "")), str(unique_id), previous_run_id)
            if old:
                previous_config = old.get("config_key")
                if not previous_config and old.get("segments"):
                    record_id = old["segments"][0].get("record_id")
                    if record_id:
                        try:
                            previous_config = removal_cache.metadata(cache_scope, record_id)["config_key"]
                        except ValueError:
                            # Optional old preview may have been cleaned up.
                            # Actual locked checkpoints are still checked below.
                            pass
        window_controls, controls_reset = removal_cache.controls_for_configuration(
            window_controls, cache_scope, config_key, previous_config)
        if controls_reset:
            LOG.info("H3 edit inputs/settings changed: starting fresh windows; old checkpoints retained")
        choices = removal_cache.plan(window_controls, cache_scope, config_key, starts, window, count, seed,
                                     derive_seeds=True, window_plan=windows)
        preview_run = begin_run(unique_id, extra_pnginfo, starts, window, count, seed,
                                config_key=config_key, controls_reset=controls_reset,
                                window_plan=windows, window_policy=WINDOW_POLICY)
        graph = GraphBuilder()
        sampler = graph.node("KSamplerSelect", sampler_name=sampler_name)
        sigmas = graph.node("BasicScheduler", model=model, scheduler=scheduler, steps=steps, denoise=denoise)
        previous = None
        for entry in windows:
            start, index = entry["source_start"], entry["index"]
            size, effective_history = entry["window_frames"], entry["history_frames"]
            prior = {} if previous is None else {"previous": previous.out(0)}
            choice = choices[index]
            append_args = dict(start=start, source_count=count, window_frames=size,
                window_metadata=json.dumps(entry, sort_keys=True),
                preview_run=preview_run, window_index=index, cache_scope=cache_scope,
                config_key=config_key, window_seed=str(choice["seed"]), **prior)
            if choice["record_id"]:
                restored = graph.node("H3RelayRemovalRestore", cache_scope=cache_scope,
                                      config_key=config_key, record_id=choice["record_id"])
                previous = graph.node("H3RelayEditAppend", frames=restored.out(0), audio=restored.out(1),
                                      reuse_record=choice["record_id"], **append_args)
                continue
            audio_args = {} if source_audio is None else {"source_audio": source_audio}
            part = graph.node("H3RelayEditWindow", source=source, start=start,
                window_frames=size, history_frames=effective_history, **prior, **audio_args)
            length = size + (effective_history-1 if effective_history >= 18 else 0)
            ref_args = {"ref_images.ref_image_0": reference_image, "ref_videos.ref_video_0": part.out(0)}
            if source_audio is not None:
                ref_args["ref_video_audios.ref_video_audio_0"] = part.out(3)
            ref = graph.node("MiniMaxH3ReferenceToVideo", prompt=prompt, width=width, height=height,
                length=length, ref_image_size="match", clip=clip, vae=video_vae,
                audio_vae=audio_vae, **ref_args)
            positive, latent = ref.out(0), ref.out(1)
            if previous is not None and history >= 18:
                carry = graph.node("H3RelayEditHistory", conditioning=positive, latent=latent,
                    vae=video_vae, audio_vae=audio_vae, previous_frames=part.out(1),
                    previous_audio=part.out(2), history_frames=history)
                positive, latent = carry.out(0), carry.out(1)
            elif previous is not None and history == 1:
                guide = graph.node("MiniMaxH3AddGuide", positive=positive, latent=latent,
                    vae=video_vae, image=part.out(1), frame_idx=0)
                positive = guide.out(0)
            noise = graph.node("RandomNoise", noise_seed=choice["seed"])
            guider = graph.node("CFGGuider", model=model, positive=positive, negative=positive, cfg=cfg)
            sample = graph.node("SamplerCustomAdvanced", noise=noise.out(0), guider=guider.out(0),
                sampler=sampler.out(0), sigmas=sigmas.out(0), latent_image=latent)
            streams = graph.node("LTXVSeparateAVLatent", av_latent=sample.out(0))
            video = graph.node("VAEDecode", samples=streams.out(0), vae=video_vae)
            audio = graph.node("VAEDecodeAudio", samples=streams.out(1), vae=audio_vae)
            previous = graph.node("H3RelayEditAppend", frames=video.out(0), audio=audio.out(0),
                **append_args)
        return {"result": (previous.out(1), plan), "expand": graph.finalize(),
                "ui": {"h3_removal_run": [preview_run]}}


NODE_CLASS_MAPPINGS = {c.__name__: c for c in (H3RelayEditWindow, H3RelayEditHistory,
    H3RelayEditAppend, H3RelayWindowedEdit)}
NODE_DISPLAY_NAME_MAPPINGS = {"H3RelayWindowedEdit": "H3 Relay Windowed Edit · Character Swap"}
