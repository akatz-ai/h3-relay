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
- Positive and negative pixel coordinates selecting the person in frame zero.
  The two Text (Multiline) nodes feed native SAM3 Detect. Each list uses
  `[{"x": 400, "y": 200}]`; replace the example points for your source.

SAM3 Video Track propagates the selection. H3 Relay Green Mask expands each
binary mask by 5 pixels by default and fills selected pixels with RGB 0/255/0.
It preserves every unselected source pixel and rejects a mask batch with a
mismatched frame count or resolution. Preview the mask before rendering H3 by
setting the clean-output Save Video node to **Never**, then restore **Always**.

## Relay behavior

`H3RelayPersonRemover` expands into native ComfyUI conditioning, sampling and
VAE nodes. The first window has 22 frames. Each later window selects the clean
boundary frame from the generated sequence, imports 18 decoded video/audio
history frames, and edits the next aligned source window. Continuations ask
native H3 for 39 frames before its history logic shortens the target to 22.

Usually windows start 21 frames apart. The final full window shifts backward
when necessary; all repeated overlap frames are removed. Very short tails are
padded by repeating their final source frame, then trimmed. Output contains
exactly the source frame count at 24 fps. For 124 input frames the starts are
`0, 21, 42, 63, 84, 102`.

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
