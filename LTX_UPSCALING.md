# LTX 2.5 upscaling notes

## What the logs mean

ComfyUI's `LTXV Context Windows` control is expressed in real pixel frames,
then converted to LTX latent time by `(frames - 1) / 8 + 1`.

The original H3 Relay settings were:

| Control | Pixel frames | LTX latent frames |
| --- | ---: | ---: |
| Diffusion window | 129 | 17 |
| Diffusion overlap | 32 | 4 |
| VAE decode tile | 64 | 8 |
| VAE decode overlap | 8 | 1 |

Therefore, a log line saying `Context length 17 with overlap 4 for 47 frames`
does not mean the model was seeing only 17 video frames. It was seeing 129
video frames, or roughly 5.4 seconds at 24 fps. A 362-frame H3 shot is padded
to 369 pixel frames and represented by 47 LTX latent frames.

## Validated balanced preset

The current H3 Relay defaults are:

| Control | Pixel frames | LTX latent frames |
| --- | ---: | ---: |
| Diffusion window | 193 | 25 |
| Diffusion overlap | 64 | 8 |
| VAE decode tile | 128 | 16 |
| VAE decode overlap | 16 | 2 |

This preset completed on the RTX 4090 in 260 seconds for the cached 15-second
reference H3 shot. It reduced the three-step diffusion pass from fourteen
window evaluations to nine while keeping the same three-sigma LTX refinement
schedule. It also doubled the temporal VAE tile and overlap.

H3 Relay includes these values in its LTX cache key. Changing any one of them
reuses the raw H3 result and reruns only LTX plus later finishing derived from
that LTX revision.

## Limits of the pasted full-attention claim

The official LTX Looping Sampler does not retain full temporal attention over
the complete movie. Its own documentation describes overlapping temporal
tiles, with an 80-frame tile and 24-frame overlap by default. Subsequent tiles
are conditioned on the previous tile's ending frames and blended. It can be a
useful alternative tiler, but it changes the sampler/guider contract and does
not remove temporal segmentation.

Spatial tiling and tiled VAE decode reduce different memory costs:

- spatial diffusion tiling reduces transformer spatial activation memory;
- temporal diffusion windows reduce transformer temporal activation memory;
- VAE spatial/temporal tiling reduces only encode/decode memory.

Using spatial VAE tiles does not make full 15-second transformer attention fit
on a 24 GB GPU by itself.

## Model behavior

The Lightricks model card describes the Pixel Spatial Upscaler as a creative,
generative upsampler rather than a pixel-accurate restoration model. It
synthesizes new high-frequency detail and recommends a very-low-resolution
draft around 280p. It also identifies factual live-action fidelity as outside
the model's intended use. Faces and backgrounds can therefore change even
without a temporal seam.

Lower guidance, fewer steps, and lower LoRA strength generally preserve more
of the reference. H3 Relay already uses CFG 1.0 and a short three-step schedule;
future controls should test LoRA strength and sigma start separately from
window sizing.

## Recommended experiments

1. **Balanced / current default:** `193 / 64`, VAE `128 / 16`.
2. **Larger temporal window:** `257 / 96`, VAE `192 / 32`; expect higher peak
   VRAM and test for OOM before adopting.
3. **No diffusion windows:** use a window at least as long as the padded input
   (`369` for the first 362-frame shot). This is experimental on 24 GB and may
   OOM or offload enough weights to become much slower.
4. **Spatial-only VAE decode:** set a VAE temporal tile larger than the movie
   while retaining spatial tiles. This tests decode seams but does not remove
   diffusion-window boundaries.
5. **Fidelity-first finishing:** for live action, compare a non-generative
   super-resolution backend against LTX rather than assuming every artifact is
   caused by windowing.

## Primary references

- <https://huggingface.co/Lightricks/LTX-2.5-22b-IC-LoRA-Pixel-Spatial-Upscaler>
- <https://github.com/Lightricks/LTX-2/blob/main/packages/ltx-pipelines/src/ltx_pipelines/ic_lora.py>
- <https://github.com/Lightricks/ComfyUI-LTXVideo/blob/master/looping_sampler.md>
- <https://github.com/Lightricks/ComfyUI-LTXVideo/issues/470>
