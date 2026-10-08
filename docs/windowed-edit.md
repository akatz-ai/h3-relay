# Windowed reference editing

`H3RelayWindowedEdit` expands ordinary ComfyUI sampler/decode nodes. It supports
character swaps and other prompt-driven Ref2VA edits without owning model loading.
Connect the same MODEL, CLIP, video/audio VAEs, source frames and character image
as a working single-window graph. Both VAEs remain connected to Ref2VA encoding.
The reference input uses its first image, matching the single-character contract.

## Controls and source timing

- Source: RGB, 24 fps, both dimensions divisible by 32. `source_count=0` uses all
  frames; connect the preparation node's actual count to exclude padded frames.
- `window_frames`: 22, 39, 56, …, 362 (`17n+5`). Default 124 = 5.167 seconds.
  Use 107 = 4.458 seconds for a window strictly below five seconds.
- `history_frames`: 0 (independent), 1 (generated boundary only), or 18, 35, 52,
  … (`17k+1`). Default 18; history must be smaller than the window.
- Native defaults: seed 904234, eight steps, er_sde, simple, CFG 1, denoise 1.
  Model/LoRA settings remain outside this node and are part of cache compatibility.

H3 sliding history consists of past frames **before** the sampled target plus
one repeated boundary. Eighteen means 17 past frames and one boundary, not an
18-frame reduction of each window's delivered duration. Ordinary continuation
advances by `window_frames - 1`. The final window can move backward to use a full
source segment, provided its preceding generated history exists. Short clips or
unavoidable tails repeat the last source frame only inside the model window.

For a 240-frame (ten-second) source and 124/18 settings:

| Window | Original source frames | Generated history | Delivered frames |
| --- | --- | --- | --- |
| 1 | 0–123 | none | 0–123 |
| 2 | 116–239 | 99–116 | 124–239 |

The second window discards its first eight decoded frames, retaining the already
accepted prefix exactly. The output has 240 frames. `window_plan` describes every
source range, history range, padded tail and discarded overlap. No-history mode
instead uses adjacent, independent windows. The currently installed H3 VAE can
decode one final frame short; the adapter explicitly logs and holds that one
frame. Larger shortages fail rather than silently shortening output.

Generated audio is kept internally for AV continuation. To retain the original
sound, connect prepared source audio directly to the final Create Video node.

## Previews, locks and rerolls

This node uses the **same implementation** as Person Remover: `person_remover.js`,
`removal_previews.py`, `removal_cache.py` and the shared append/restore nodes.
Each completed window publishes an MP4/poster before its continuation starts.
Hover to play (tap on touch), Lock to retain a contiguous prefix, and Reroll to
regenerate the selected window and everything after it. Enter a seed before
rerolling to request that seed; leaving it unchanged chooses a fresh seed.

Checkpoints contain exact floating-point frames and generated audio, not
re-encoded preview pixels. Input pixels, character image, source audio, prompt,
window/history lengths, sampler settings, model graph and model-file versions
must match. Changing them requires unlocking incompatible checkpoints. Workflow
ID and node ID isolate checkpoint ownership. Preview identity is saved in the
workflow, allowing reload and reconnect. Cache files remain on the serving
machine under `output/__h3_removal_cache/`; saved JSON does not contain them.
Large windows accumulate RAM and disk use; this is an experimentation workflow,
not a streaming implementation. Checkpoints are deliberately not auto-deleted.

## Local Labs workflow

The separate [labs-character-swap-turbo8-relay workflow](../development_workflows/labs-character-swap-turbo8-relay.json) preserves the original
prompt, character-swap LoRA, Turbo8 and explicit model/memory controls.
Its Labs H3 Prepare Video node handles the existing 0.5/1 MP presets and optional
total-duration trim. This preparation pack belongs to `akatz-labs-infra`; it is
an additional dependency of that local workflow, not bundled with H3 Relay.
The saved `labs-character-swap-single-window-saved-20261007` baseline is unchanged.

Longer windows and different history lengths are experiments. Correct frame
assembly and successful rendering do not establish character consistency or
seam quality; review the previews and final video before choosing production
defaults.
