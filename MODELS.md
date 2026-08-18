# H3 Relay model manifest

H3 Relay does not redistribute model weights. The reference workflow uses the
following official model repositories and ComfyUI folders.

Installing H3 Relay does not grant permission to use these separately licensed
models. Review each upstream license and obtain any required authorization
before downloading, inference, redistribution, or commercial use.

Authenticate and run the commands from the ComfyUI root:

```bash
hf auth login
cd /path/to/ComfyUI
```

## MiniMax H3

Source: <https://huggingface.co/Comfy-Org/MiniMax-H3>

```bash
hf download Comfy-Org/MiniMax-H3 \
  diffusion_models/minimax_h3_fl2va_int8_convrot.safetensors \
  diffusion_models/minimax_h3_ref2va_int8_convrot.safetensors \
  text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors \
  vae/minimax_h3_video_vae_fp16.safetensors \
  vae/minimax_h3_audio_vae_fp32.safetensors \
  --local-dir models
```

Qwen3-VL 32B is MiniMax H3's matching text encoder. It is not an alternative
to the LTX encoder.

## LTX 2.5

Source: <https://huggingface.co/Lightricks/LTX-2.5>

```bash
hf download Lightricks/LTX-2.5 \
  diffusion_models/ltx-2.5-22b-dev-transformer-comfy-int8-convrot.safetensors \
  text_encoders/gemma4-12b-with-proj-ltx-2.5-comfy-int8-convrot.safetensors \
  vae/ltx-2.5-video-vae-bf16.safetensors \
  latent_upscale_models/ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors \
  loras/ltx-2.5-22b-distilled-lora-450-bf16.safetensors \
  --local-dir models
```

Gemma4 12B is the custom LTX 2.5 encoder and includes its matching projection.
The distilled LoRA adapts the dev transformer for the fast low-step inference
used by the workflow. It is not the upscaler adapter.

## LTX 2x pixel-spatial IC-LoRA

Source:
<https://huggingface.co/Lightricks/LTX-2.5-22b-IC-LoRA-Pixel-Spatial-Upscaler>

```bash
hf download Lightricks/LTX-2.5-22b-IC-LoRA-Pixel-Spatial-Upscaler \
  ltx-2.5-22b-ic-lora-pixel-spatial-upscaler-x2-1.0.safetensors \
  --local-dir models/loras
```

This separate IC-LoRA consumes the low-resolution H3 video as an in-context
reference and performs the creative 2x re-render. The reference workflow uses
both LTX LoRAs at strength 1.0.

The learned latent model performs the spatial 2x expansion first. The IC-LoRA
then guides diffusion from the original pixel video while the model refines
that high-resolution latent. Advanced native loader chains can be combined
with **H3 Relay · Pack LTX Model**; update its `cache_identity` whenever any
upstream component or patch changes.

## RIFE interpolation

Source: <https://huggingface.co/Comfy-Org/frame_interpolation>

```bash
hf download Comfy-Org/frame_interpolation \
  frame_interpolation/rife_v4.26_heavy.safetensors \
  --local-dir models
```

RIFE is optional. Bypass the Interpolate nodes to assemble LTX output at 24
fps. Keep them enabled for the reference 48 fps output.

The complete reference profile occupies approximately 129 GiB. Review and
accept the upstream model licenses before downloading, and restart ComfyUI
after installing files so its loader lists refresh.
