# Windowed reference editing

`H3RelayWindowedEdit` expands ordinary ComfyUI sampler/decode nodes. It supports
character swaps and other prompt-driven Ref2VA edits without owning model loading.
Connect the same MODEL, CLIP, video/audio VAEs, source frames and character image
as a working single-window graph. Both VAEs remain connected to Ref2VA encoding.
The reference input uses its first image, matching the single-character contract.

## Controls and source timing

- Source: RGB, 24 fps, both dimensions divisible by 32. `source_count=0` uses all
  frames; connect the preparation node's actual count to exclude padded frames.
- `window_frames`: maximum target size, 22, 39, 56, …, 362 (`17n+5`).
  Default maximum 124 = 5.167 seconds. Short clips and final windows shrink
  automatically to the smallest supported size that fits.
  Use 107 = 4.458 seconds for a window strictly below five seconds.
- `history_frames`: 0 (independent), 1 (generated boundary only), or 18, 35, 52,
  … (`17k+1`). Default 18; history must be smaller than the window.
- Native defaults: seed 904234, eight steps, er_sde, simple, CFG 1, denoise 1.
  Model/LoRA settings remain outside this node and are part of cache compatibility.

H3 sliding history consists of past frames **before** the sampled target plus
one repeated boundary. Eighteen means 17 past frames and one boundary, not an
18-frame reduction of each window's delivered duration. Ordinary continuation
advances by the actual target size minus one. The final window does not shift
backward to fill the maximum size: it starts at the preceding output boundary,
rounds the remaining source frames up to the H3 grid, and pads only the tail.
A continuation's target stays larger than its configured history. The first
window has effective history zero, so long configured history does not inflate
a short single-window clip. No-history mode uses adjacent independent windows.

For a 144-frame (six-second) source and 124/18 settings:

| Window | Target size | Original source frames | Generated history | Delivered frames |
| --- | --- | --- | --- | --- |
| 1 | 124 | 0–123 | none | 0–123 |
| 2 | 22 | 123–143 + one padded frame | 106–123 | 124–143 |

The continuation's Ref2VA allocation is 39 frames: a 22-frame target plus 17
past-history frames. Sliding-history conditioning removes those past frames
from the sampled target. Assembly discards the repeated boundary and final
padding, retaining the accepted 124-frame prefix exactly. Output is 144 frames.
A standalone 24-frame clip instead uses a 39-frame target and trims 15 pad frames.

`window_plan` describes every actual source range, target size, effective history,
padded tail and discarded boundary. Its `window_policy` is `adaptive-target-v1`.
Policy and the complete plan participate in checkpoint compatibility. Changing
policy invalidates prior selections while preserving their artifacts on disk.
Checkpoints and preview segments record actual window sizes; the preview header
labels the selected size as an adaptive maximum. Person Remover retains its
existing fixed-window planner and cache behavior.

The currently installed H3 VAE can decode one final frame short; the adapter
explicitly logs and holds that frame, as before. If that frame is tail padding,
it is discarded and all real source positions remain intact. Without padding,
the final delivered frame is the explicitly logged hold, not a new model frame.
Larger shortages fail rather than silently shortening output.

CPU tests validate source/audio alignment, history bounds, variable graph sizes,
exact checkpoint prefix reuse, stale-plan rejection and preview metadata. They
do not establish seam quality, identity consistency or GPU speedup; those require
a matched render against the previous fixed-size workflow.

Generated audio is kept internally for AV continuation. To retain the original
sound, connect prepared source audio directly to the final Create Video node.

## Master seed and window seeds

The node's `seed` input is the **master seed** (the API/socket name stays `seed`
for saved-workflow compatibility). Window 1 uses the master unchanged; window 2
uses master + 1, window 3 master + 2, and so on. Addition wraps at the unsigned
64-bit limit. This `h3-window-seed-increment-v2` scheme replaces hashed sub-seeds
so the first window can use the same noise seed as a single-window workflow.
Matching seed alone does not match differing inputs, conditioning or settings.

- Same master + inputs/settings: reuse the current window results.
- Changed master: invalidate the entire window set and all per-window overrides;
  derive new defaults and regenerate every window on Run.
- Fixed master + a card's Reroll: keep earlier checkpoints, override that window's
  seed, and regenerate it and all following windows. Later windows retain their
  own seeds but receive the updated generated history.
- Regenerate all: explicitly choose fresh per-window overrides under the current
  master. A subsequent master change clears those overrides too.

Keep **control after generate = fixed** when refining individual windows. Normal
ComfyUI increment/randomize controls intentionally change the master for the next
Run. The preview header identifies the master used for the displayed results,
and each card reports its actual sampled window seed.

The seed-scheme version is part of checkpoint compatibility. Checkpoints from
before this master/sub-seed implementation are retained on disk, but the first
new Run generates a set using the new semantics.

## Previews and rerolls

This node uses the **same implementation** as Person Remover: `person_remover.js`,
`removal_previews.py`, `removal_cache.py` and the shared append/restore nodes.
Each completed window publishes an MP4/poster before its continuation starts.
Hover to play (tap on touch). **Reroll** regenerates that window and everything
after it, automatically keeping earlier results. **Regenerate all** uses fresh
seeds for every window. Ordinary **Run** with unchanged inputs reuses completed
windows, including after reload. There are no manual lock controls.

A card seed edit is a draft until Reroll is clicked. Enter a different seed to
request that exact value; leaving it unchanged chooses a fresh seed.

Checkpoints contain exact floating-point frames and generated audio, not
re-encoded preview pixels. Input pixels, character image, source audio, prompt,
window/history lengths, sampler settings, model graph and model-file versions
must match for reuse. Changing any of them automatically starts a fresh window
set, clears old checkpoint selections and per-window seed overrides, and replaces the preview
grid when Run is clicked. The panel explains that inputs/settings changed.
Existing output files and checkpoints remain on disk; they are never reused
for the changed configuration. Workflow
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

## Native Sol comparison

The separate [native Sol workflow](../development_workflows/labs-character-swap-turbo8-native-sol.json)
adds ComfyUI's built-in `BlockSparseAttention` (display name **Model Sparse
Attention**) after the existing model/memory patches. It selects `sol-attn`,
not SLA or FastH3 VSA; the original Ref2VA weights and both LoRAs are unchanged.
No additional custom-node pack is required in the tested review environment.

Select the sparse node and toggle **Bypass** (Ctrl+B) for the existing dense
baseline. Restore **Always** for Sol. The existing **Model Attention Backend**
continues to supply Comfy Kitchen INT8 attention for dense steps/fallbacks.
Keep the master seed fixed and the inputs, resolution, duration, window/history,
sampler and LoRAs identical. Toggling the node or changing its settings invalidates
window checkpoints; the derived seeds stay the same unless the master changes.
Use untouched derived seeds for matched tests: changing model settings clears
manual per-window overrides as part of the normal configuration reset.

The saved defaults are tau 1.3, start 0.2, end 1.0, min_tokens 12288,
extra_tokens 256, sink_conditioning `exact_kv_and_rows`, and verbose enabled.
Higher tau makes the approximation sparser. Conditioning keys/values remain
exact, and target-audio query rows remain dense; this does not guarantee equal
identity, coherence or audio quality. Short sequences below min_tokens run
dense. Inspect logs for `BlockSparseAttention: sparse producer path` to verify
actual use. Compare warm sampler times independently of loading and encoding.

Validated runtime: ComfyUI `2255709aa0be2deade91c7c80cda49d31b73906f`,
comfy-kitchen 0.2.35 with CUDA Sol support, and the existing pinned KJ MiniMax
memory patch snapshot. This workflow requires a ComfyUI build exposing the
native sparse node; older builds may not load it. Successful local execution
does not establish a cloud speedup or quality parity.
