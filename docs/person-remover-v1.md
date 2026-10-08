# Person Remover V1 workflow

Use `example_workflows/H3-Relay-Person-Remover-V1.json` with the separate
Person Remover V1 LoRA. This is a source-aligned edit: it reconstructs a
background while following the input video's timing and camera movement.

## Inputs

- A short, single-shot **24 fps** RGB video. Width and height must both be
  divisible by 32. Resize and resample before making the mask or clean anchor.
- A clean version of **source frame zero**, with the person removed. Keep its
  exact resolution and framing. Create this externally with an image editor,
  ChatGPT ImageGen, Qwen Image Edit, or another method.
- A short text description in **CLIP Text Encode**, for example `man in gray shirt`
  or `woman wearing a red coat`. Native SAM3 detects and tracks that target.
  The SAM checkpoint includes its text encoder; no point selector or KJ Nodes is needed.
  If multiple people match, use a more specific description and inspect the mask.

SAM3 Video Track propagates the selection. H3 Relay Green Mask expands each
binary mask by 5 pixels by default and fills selected pixels with RGB 0/255/0.
It preserves every unselected source pixel and rejects a mask batch with a
mismatched frame count or resolution. Preview the mask before rendering H3 by
setting the clean-output Save Video node to **Never**, then restore **Always**.

## Relay behavior

`H3RelayPersonRemover` expands into native ComfyUI conditioning, sampling and
VAE nodes. Select **window_frames** directly on the node. Its tooltip
shows the valid `17n + 5` choices and duration at 24 fps: 22, 39, 56, 73, 90, 107,
124, and so on up to 362. The default remains **22 frames (0.92 seconds)**.
39 frames is 1.63 seconds; 124 is 5.17 seconds. The relay starts at 22 because
its continuation needs 18 history frames. Five-frame H3 clips cannot supply that history.
Larger windows use more memory and may remove people less reliably. In the
77-frame test with 39-frame windows, the opening output still contained the
person even though timing, continuation and assembly passed. Keep 22 as the
default; treat larger sizes as experiments and inspect their preview cards.

The first window has the selected length. Each later window selects the clean
boundary frame from the generated sequence, imports 18 decoded video/audio
history frames, and edits the next aligned source window. Continuations ask
native H3 for `window_frames + 17` frames; its history logic shortens the
sampled target to exactly `window_frames`. The history remains 18 frames for all choices.

Usually windows start `window_frames - 1` frames apart (21 with the default). The final full window shifts backward
when necessary; all repeated overlap frames are removed. Very short tails are
padded by repeating their final source frame, then trimmed. Output contains
exactly the source frame count at 24 fps. For 124 input frames the starts are
`0, 21, 42, 63, 84, 102`.

## Per-window previews

A small preview video appears on the Person Remover node after each decoded
window, before the next window proceeds. Cards form a two-column grid and show
the source-frame span and discarded overlap count. They include the overlapping
boundary for inspection and omit padded tail frames.

Hover over a card to play its muted loop; move away to pause. On a touch screen,
tap to toggle playback. Use ComfyUI's normal execution controls to stop a run;
completed cards remain available. Reduce the window size or change the seed,
then run again. The next run gets a fresh grid; previous preview files remain in
the window cache. The grid shows the rendered seed/window size so
previous previews are distinguishable from newly edited settings.

Reopening the same workflow on the same server restores its latest cards.
Previews are H.264 videos capped at 512 pixels; they are for inspection,
not full-quality exports. Save Video still writes the final assembled output.
Existing workflows without `window_frames` continue to use 22 frames.

### Reroll and automatic reuse

Each card has a seed field and **Reroll**. Completed windows are retained
implicitly; normal Run reuses them when inputs/settings match. Reroll keeps
all earlier results and rebuilds the selected window and every later window.
**Regenerate all** generates fresh seeds for all windows and rebuilds everything.
There are no manual locks to manage.

A seed edit is a draft until Reroll is clicked. Enter a different seed to use
that value; otherwise Reroll chooses a fresh random seed. Use ComfyUI's normal
stop controls if a run is active, then reroll when the queue is idle.

The Green Mask node saves the exact prepared green-masked source on disk.
Unchanged runs and rerolls reuse it without requesting SAM detection or tracking,
including after a ComfyUI restart. Changing the source video, target text,
tracking/selection settings, mask expansion or SAM model version prepares a new
mask. H3 seeds, steps and reroll selections do not change this prepared source.
The first run after installing this update prepares and saves the mask once.
Unsupported upstream custom nodes fall back to ordinary evaluation.

The cache stores exact decoded frames and generated audio in
`ComfyUI/output/__h3_removal_cache/`, alongside small previews and manifests.
Checkpoints survive ComfyUI restarts and workflow reloads on that server.
Cache files are local artifacts and are not embedded in the workflow JSON.
If checkpoints were deleted or moved, use Regenerate all to rebuild them.
Old candidates are retained, so disk use grows with rerolls.
Prepared masks live in the `prepared/` subdirectory as lossless tensors; these
can be large. Removing them causes the next run to prepare its mask again.

Reuse checks the actual masked source pixels, clean anchor, prompt, steps,
window size, model/encoder/VAE graph and model file versions. Changed inputs or
settings start a fresh window set automatically. The hidden checkpoint format
still accepts legacy lock fields for saved-workflow compatibility, but the UI
manages them automatically. Use Reroll or Regenerate all for fresh candidates.

The first clean image is a visual reference; generated frame zero is not
pixel-locked to that image. Only generated raw frames/audio feed continuation.
This workflow does not apply the optional seam color grade used in some
comparison exports. Color shifts and geometry changes still need review.

Defaults: B2000 LoRA strength 1, 12 steps, 22-frame windows, er_sde, simple, CFG 1, fixed seed 904234,
video/audio sigma shifts 12/3. No Turbo or VFX LoRA. Keep the default removal
prompt for an initial test.

The workflow routes Load CLIP through native **Select CLIP Device**, set to
**gpu:0**, to run Qwen reference encoding on the GPU. Load CLIP's `default` can
choose CPU under low-VRAM settings; leaving it alone does not guarantee GPU
encoding. This uses built-in ComfyUI nodes. The current encoder can emit a memory
warning even when the tested short clips complete; large-batch stability remains
unverified.

The output video is silent. Generated audio still carries internal history.
Connect the source AUDIO socket to the final Create Video node if you want the
original soundtrack. Visual person removal does not remove that person's voice.

## Requirements and limits

Use current ComfyUI with native MiniMax H3, SAM3, Load Video and Save Video
nodes, plus this H3 Relay build. All model loaders remain visible in the graph;
its Markdown download note lists the model URLs and target folders.

The validated removal runtime used ComfyUI **0.37.0**, frontend **1.53.6**,
PyTorch **2.14.0+cu130** and an RTX 4090. The older FastH3 VSA recipe serves a
different workflow; its pinned runtime does not establish native SAM3 support.

| Model | ComfyUI folder |
| --- | --- |
| [H3 Ref2VA pruned INT8 ConvRot](https://huggingface.co/Comfy-Org/MiniMax-H3/blob/main/diffusion_models/minimax_h3_ref2va_pruned_int8_convrot.safetensors) | `models/diffusion_models` |
| [Qwen3VL 32B NVFP4 AWQ](https://huggingface.co/Comfy-Org/MiniMax-H3/blob/main/text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors) | `models/text_encoders` |
| [H3 video VAE FP16](https://huggingface.co/Comfy-Org/MiniMax-H3/blob/main/vae/minimax_h3_video_vae_fp16.safetensors) | `models/vae` |
| [H3 audio VAE FP32](https://huggingface.co/Comfy-Org/MiniMax-H3/blob/main/vae/minimax_h3_audio_vae_fp32.safetensors) | `models/vae` |
| [SAM 3.1 model and text encoder](https://huggingface.co/Comfy-Org/sam3.1/blob/main/checkpoints/sam3.1_multiplex_fp16.safetensors) | `models/checkpoints` |
| `H3-Person-Remover-V1.safetensors` — separate B2000 adapter | `models/loras` |

The Person Remover adapter is not bundled in this repository. Its public model
release is pending. The validated B2000 candidate is 155,110,584 bytes with
SHA-256 `b01dc9fe888c2f4e455a8eb3b4878dfd4038ee0e2320a7b186c7935e65c93ba5`.
LTX, RIFE, MMH3 Ultimate and the FastH3 VSA profile are not required for this graph.

Start with approximately five-second clips. This workflow uses ComfyUI's normal
expanded-graph cache; it is not the disk-staged long-form Relay runner. CPU RAM
use grows with clip length. Stop at scene cuts and create a new clean anchor.
It regenerates the complete image, so background text, screens, reflections,
hands, shadows, occlusions and camera movement can change. Passing the runtime
checks is not a guarantee of clean removal or acceptance for training.

The LoRA is based on MiniMax H3 and retains the upstream model license. The node
pack's GPL-3.0 license does not change model rights. The upstream H3 standard grant
has territorial restrictions; review the actual license before distribution.
