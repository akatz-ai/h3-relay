# H3 Relay example workflows

`H3-Relay-Person-Remover-V1.json` is a separate source-video editing workflow.
It uses standard H3 Ref2VA, a clean first-frame anchor, native SAM3 text targeting,
and the separate Person Remover B2000 adapter. Start at 12 steps and 22-frame
windows; its cards provide prefix locks and seeded rerolls, and unchanged
rerolls reuse a persistent prepared mask. Use a current ComfyUI with native SAM3;
the older FastH3 runtime pins below apply to the FastH3 examples.
See [the removal guide](../docs/person-remover-v1.md) for setup, model locations,
adapter availability and limitations.

The FastH3 examples are arranged from the smallest useful graph to the full
reference-driven sequence:

1. `H3-Relay-FastH3-VSA-One-Shot.json`
   - one five-second 832x480 FastH3 VSA generation;
   - no reference media, upscaling, interpolation, or assembly;
   - use this first to verify the experimental FastH3 runtime.
2. `H3-Relay-FastH3-VSA-One-Shot-Ultimate-2x.json`
   - the same prompt, seed, and raw generation;
   - one H3 Ultimate 2x pass producing 1664x960 at 24 fps;
   - no references, RIFE, or final sequence assembly.
3. `H3-Relay-FastH3-Akatz-3x10-Ultimate.json`
   - three independently reviewable 10-second FastH3 generation windows;
   - 18-frame visual/audio continuation between shots;
   - five bundled character/environment references;
   - raw and H3 Ultimate sequence assembly, with optional bypassed RIFE nodes.

Every workflow contains black Markdown instruction cards with its exact model
filenames, ComfyUI folders, tested experimental runtime commits, required
custom nodes, and run order.

## Reference assets

The five original generated images under `assets/` belong to the Akatz
three-shot example. Copy them into `ComfyUI/input/` before loading that workflow:

```text
akatz-general-reference.png
akatz-character-sheet.png
akatz-face-closeup.png
cyberpunk-neon-street.jpg
cyberpunk-stairwell.jpg
```

The saved Load Image nodes use those exact filenames. The two one-shot examples
do not require the assets.

## Experimental runtime warning

FastH3 VSA is not yet supported by an ordinary stock ComfyUI release. The
instruction cards pin the ComfyUI VSA commit and official comfy-kitchen 0.2.33
CUDA wheel. Relay supplies its own VSA adapter for generation and Ultimate;
the temporary `SolAttnMiniMax` node is no longer required. Treat the runtime
pins as part of the workflow, not optional performance suggestions.

Regenerate all three FastH3 workflow files after editing their shared builder:

```bash
node scripts/build_fast_h3_akatz_workflow.mjs
```
