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

## Experimental FastH3 VSA profile

Canonical model card:
<https://huggingface.co/FastVideo/FastVideo-FastH3-4-step-Preview-v1-VSA-DataFree>

Kijai ComfyUI INT8 ConvRot repack:
<https://huggingface.co/Kijai/MiniMax-H3-experimental>

```bash
hf download Kijai/MiniMax-H3-experimental \
  minimax_h3_fastvideo_vsa_datafree_1300step_4step_int8_convrot.safetensors \
  --local-dir models/diffusion_models
```

H3 Relay pins the experimental profile to the exact filename above. The
measured checkpoint SHA-256 is
`7221ae65d78780354d51e5048d29728d9f1f8fb9baf50b1dd3df85f5101413d3`.
Verify the current upstream artifact before assuming that hash applies to a
newer revision.

Native ComfyUI support is under
<https://github.com/Comfy-Org/ComfyUI/pull/15958>. The required Sol-Attention
kernel is merged in <https://github.com/Comfy-Org/comfy-kitchen/pull/117>.
Use the official `comfy-kitchen==0.2.33` CUDA wheel. The ComfyUI model-support
PR remains unmerged; pin Kijai ComfyUI commit
`10febb01d7be73d1491cf5e5347b5ab8b6c2c09e` for this profile. H3
Relay supplies its own locked MiniMax VSA adapter; do not install the temporary
`SolAttnMiniMax` test node. **FastH3 VSA Profile** fails when the core gate
support or CUDA kernel is absent instead of falling back to dense attention,
because the checkpoint was distilled against VSA-H3 at 90 percent sparsity.

FastVideo currently describes this preview as text-to-audio-video only.
FL2VA, Ref2VA, and H3 Relay sliding continuation are experimental inherited
behaviors rather than upstream-supported contracts.

## H3 Ultimate 2x finishing

Learned H3 latent upscaler:
<https://huggingface.co/LBH-123-AI/Minimax_h3_latent_Upscaler>

MMH3 Ultimate Upscale nodes:
<https://github.com/bbaudio-2025/Comfyui-MMH3-UltimateUpscale>

The bundled workflows were validated against MMH3 Ultimate commit
`6db8fa5a4e4ca0718d2ea8d08002ea899fe27721`.

This external node pack is required for every fresh **H3 Ultimate 2× Enhance**
inference. It is not required for raw generation, continuation, raw assembly,
LTX finishing, interpolation, or restoration of an already verified Ultimate
cache result.

```bash
hf download LBH-123-AI/Minimax_h3_latent_Upscaler \
  minimax_h3_latent_upscaler_3d_fp16.safetensors \
  --local-dir models/latent_upscale_models

cd custom_nodes
git clone https://github.com/bbaudio-2025/Comfyui-MMH3-UltimateUpscale.git
```

**H3 Relay · H3 Ultimate 2× Enhance** reuses the connected FastH3 VSA model,
the ordinary MiniMax H3 text encoder and VAEs, the accepted raw AV latent, and
the shot's H3 references. It does not require the LTX model stack. Its default
finishing profile uses the learned latent 2x expansion, one Euler/simple step
at denoise 0.2, 136/17 temporal windows, and 1024-pixel spatial tiles with
128-pixel overlap.

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

RIFE is optional. Bypass the Interpolate nodes to assemble H3 Ultimate or LTX
output at 24 fps. Keep them enabled for the reference 48 fps output.

The complete reference profile occupies approximately 129 GiB. Review and
accept the upstream model licenses before downloading, and restart ComfyUI
after installing files so its loader lists refresh.
