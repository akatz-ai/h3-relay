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
VAE nodes. Select **window_frames** directly on the node. The built-in guide
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
tap to toggle playback. **Stop run** interrupts only the run shown by that node
and leaves completed cards available. Reduce the window size or change the seed,
then run again. The next run gets a fresh grid; previous preview files remain in
ComfyUI's temporary directory. The grid shows the rendered seed/window size so
previous previews are distinguishable from newly edited settings.

Reopening the same workflow on the same server restores its latest cards.
Previews are temporary H.264 videos capped at 512 pixels; they are for inspection,
not full-quality exports. Save Video still writes the final assembled output.
Existing workflows without `window_frames` continue to use 22 frames.

The first clean image is a visual reference; generated frame zero is not
pixel-locked to that image. Only generated raw frames/audio feed continuation.
This workflow does not apply the optional seam color grade used in some
comparison exports. Color shifts and geometry changes still need review.

Defaults: LoRA strength 1, 20 steps, er_sde, simple, CFG 1, fixed seed 904234,
video/audio sigma shifts 12/3. No Turbo or VFX LoRA. Keep the default removal
prompt for an initial test.

The output video is silent. Generated audio still carries internal history.
Connect the source AUDIO socket to the final Create Video node if you want the
original soundtrack. Visual person removal does not remove that person's voice.

## Requirements and limits

Use current ComfyUI with native MiniMax H3, SAM3, Load Video and Save Video
nodes, plus this H3 Relay build. All model loaders remain visible in the graph;
its Markdown download note lists the model URLs and target folders.

Start with approximately five-second clips. This workflow uses ComfyUI's normal
expanded-graph cache; it is not the disk-staged long-form Relay runner. CPU RAM
use grows with clip length. Stop at scene cuts and create a new clean anchor.
It regenerates the complete image, so background text, screens, reflections,
hands, shadows, occlusions and camera movement can change. Passing the runtime
checks is not a guarantee of clean removal or acceptance for training.

The LoRA is based on MiniMax H3 and retains the upstream model license. The node
pack's MIT license does not change model rights. The upstream H3 standard grant
has territorial restrictions; review the actual license before distribution.
