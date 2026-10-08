# Windowed edit validation — 2026-10-07 Pacific

Environment: `labs-direct-swap` on akatz-arch, owned review container
`akatz-labs-workflow-review-20261007`, ComfyGit 0.7.1, ComfyUI
`2255709aa0be2deade91c7c80cda49d31b73906f`. Models are host-mounted read-only
from `/data/models`; no model bytes were added to the container or this repository.
H3 Relay base: `09722780e3f2ac12ede8059789b7a926b1d81be8` plus this branch.

## Baseline and registration

- Saved `labs-character-swap-single-window-saved-20261007` before editing.
- New `labs-character-swap-turbo8-relay` is a separate workflow with a new UUID.
- Compared 13 retained nodes: model, both LoRAs, CLIP/VAEs, prompt, character image
  and memory-control widgets match the baseline.
- Registered through `cg node dev-link h3-relay-character-swap`.
- Live `/object_info/H3RelayWindowedEdit` and browser preview widget verified.
- Strict ComfyGit resolution completed: six models, four custom-node types,
  no missing dependencies and no downloads.

## CPU validation

- 63 Python tests passed in the ComfyUI environment, including all new timeline,
  expansion and append/preview tests plus existing Person Remover cache/preview
  regressions.
- The remaining Node-based workflow-clone test could not launch because the GPU
  container has no `node` executable; it passed separately on the hub. Total:
  64 tests passed across the two appropriate environments.
- Expanded H3 Relay runtime contract passed against the installed ComfyUI.
- Timeline tests cover zero/boundary/sliding history, 22/39/56/124-frame windows,
  awkward tails, exact source coverage, source audio and generated-history audio.
- Reroll expansion test confirms restored prefix windows omit their samplers;
  later windows use the chosen per-window seed and correct source start.

## Real GPU render and browser previews

Test source `labs-relay-proof-10s.mp4` is a ten-second repeat of the existing
`labs-turbo8-source.mp4` fixture. This deliberately has a source-motion restart;
use a continuous user clip for aesthetic seam evaluation. Reference is the
existing `labs-turbo8-character.png`. No user source was overwritten.

- Prompt ID: `c312cdf5-8a70-44af-8784-4482f74f1d31` — success.
- Defaults: 124-frame window, 18-frame history, seed 904234, 8 steps,
  er_sde/simple, CFG 1, denoise 1, both existing LoRAs strength 1.
- Actual prepared size: 800×608 (0.4864 MP), 24 fps, 240 source frames.
- Source windows: 0–123 and 116–239; history for window 2: 99–116;
  discard its eight overlapping decoded frames.
- Output: `output/labs-swap-relay/result_00001_.mp4`, H264, 800×608,
  exactly 240 frames / 10.000 seconds at 24 fps; original source audio retained.
- Run `c810b5e0270f44e6bd3919673ebd7189` published window 1 before window 2
  completed. Both preview MP4s played in Chromium via the Tailnet HTTPS route.
- Browser reported no uncaught JavaScript errors.
- First preview visually contains the reference character in the test scene.
  This is technical validation, not human acceptance of character or seam quality.

Detailed graphs, prompt histories, manifests, browser screenshots and test
artifacts are retained on the hub at
`/home/akatz/dev/artifacts/labs-character-swap-relay-20261007/`.

## Reroll and restart persistence

- Clicked the shared browser **Reroll window 2** button, submitting prompt
  `169f8c2d-7be4-47eb-a32e-c6ed9413a2e2` — success.
- Reroll run: `557ad69854444f4f9715f19e9c4b9e85`.
- Window 1 reused checkpoint `8cc3a0f7a52f4f2ab4f4785ff296bb23` and retained
  exact floating-point frame hash
  `9e36cd3c2aa145dbc27bec0c3b5a884da29b56ddadcd3f449618cf966945bd3b`.
- Window 2 changed from seed 904234 to 11539398726631347047 and produced a new
  checkpoint and different frame hash. Only its sampler ran.
- Output `result_00002_.mp4` played at 800×608 / 10 seconds through Tailnet HTTPS.
  A ranged download returned HTTP 206.
- Saved the test graph as `labs-character-swap-relay-proof-10s`. After confirming
  the queue was empty, restarted only the owned review container, opened a fresh
  browser and restored both cards, the locked prefix and the per-window seeds.
  Preview playback, WebSocket events and final video playback worked; no uncaught
  browser errors. The queue was empty at handoff.
- Final focused tests passed again after the single-reference normalization.

The 124/18 configuration has real GPU evidence. Other supported window/history
sizes have frame-alignment coverage and use the shared H3 history engine, but
have not all been rendered or aesthetically evaluated in this character-swap graph.


## Automatic fresh sets after input changes

The windowed-edit node now reconciles actual content/configuration identity before
planning locks. Changes reset the whole window-control set, including seed
customizations, and bind browser controls to the new configuration. Unchanged
inputs retain strict checkpoint validation and exact reroll prefix reuse. Legacy
saved workflows migrate using their own preview/checkpoint metadata. Existing
render files are retained; stale pixels are not reused. Reset happens on Run.

Validation on the same Arch review environment:

- 13 cache/input/preview tests, 4 windowed-edit tests and 6 Person Remover tests passed.
- The active user generation completed successfully before the runtime restart.
- Browser submission `e15fcbed-7503-4bf3-b397-af2ebb853fd4` used a changed 22-frame
  video and 22-frame window with stale locked controls. It completed a real GPU
  render with `controls_reset=true`, fresh records, no reused segments and empty
  per-window overrides in the UI.
- A repeated preview event preserved a newly clicked lock rather than clearing it.
- Submission `ee803135-98f3-4799-88e2-2e8bcc00b355` kept the same input/configuration
  and restored that exact locked record with `controls_reset=false` and no sampler.
- No uncaught browser errors. Evidence is in
  `/home/akatz/dev/artifacts/labs-relay-auto-reset-20261007/`.

## Automatic reuse without manual lock controls

The shared preview panel now provides a seed draft and Reroll per window, plus
Regenerate all. Completed prefixes are retained internally; no Lock/Unlock UI
is shown. Explicit reroll intent is protected from stale preview refreshes, and
pending rerolls follow the latest run on reconnect. Seed drafts only take effect
on Reroll. Person Remover uses the same panel and now reconciles changed config
before checkpoint planning, so it no longer needs an Unlock all escape hatch.

Validation used a separate workflow UUID and a 43-frame sample at 800×608 with
22-frame windows / 18-frame history on Arch:

- Initial two-window GPU run: `39f50931-9f56-4612-85e3-cbe61ed6a150`, success.
- Unchanged Run: `c843bee6-79b3-40f5-ac91-6e2c1342dbde`, success; both exact
  checkpoints reused automatically.
- Last-window reroll: `11ed8db5-617f-4b2a-a88f-30b8e1ec5c40`, success; window 1's
  checkpoint unchanged, window 2 resampled with the entered seed 123456789.
- First-window reroll intent tested in the actual browser with queue submission
  intercepted: no prefix retained, and a stale progress event did not erase intent.
- Regenerate all: `6cea4fb9-10f9-41f1-a033-96743630ec00`, success; both windows
  resampled with fresh seeds and no checkpoint reuse.
- Browser reload restored both cards and automatic reuse. No Lock/Unlock labels
  or uncaught browser errors. Six focused Person Remover tests passed.
- Queue empty after testing. Graphs, histories, manifests and screenshot are in
  `/home/akatz/dev/artifacts/labs-relay-controls-20261007/`.

## Master seed and deterministic window seeds

The master seed and derivation scheme now participate in the configuration
identity. Changing the master clears checkpoint selections and per-window seed
overrides. Each default window seed is derived from the master and window index;
fixed-master rerolls retain the preceding checkpoint prefix.

Validation used an isolated workflow UUID, 43 source frames at 800×608,
22-frame windows and 18-frame history in the same Arch review container:

- Initial real GPU run `e64e374f-6f68-4804-ae16-a1eabfa19f06`: both window
  seeds matched independent SHA256 derivation from master 904234.
- Unchanged Run `29d6a85c-41f1-4129-aaf5-f694afa58869`: both exact records reused.
- Last-window reroll `97718dbe-40c2-45a4-83c2-6e0ba7638f58`: first record
  retained; second regenerated with the requested seed 123456789.
- Master changed to 904235, Run `f693b869-4861-49c3-8547-b0c2b25f041b`:
  controls reset, neither record reused, both seeds derived from the new master.
  The previous per-window override did not survive the master change.
- Unchanged new-master Run `3bcda330-91fa-4a81-b1a3-a7a054291583`: both new
  records reused. Browser reload restored both preview cards and reuse controls.
- All five ComfyUI histories report success. No uncaught browser errors.
  The screenshot confirms the master-seed widget label and preview status.
- 25 focused cache, preview, windowed-edit and Person Remover tests passed in
  the runtime Python environment. Queue empty after testing.
- Graph, manifests, browser script, histories and screenshot are retained at
  `/home/akatz/dev/artifacts/labs-relay-master-seed-20261007/`.

This proves the master-change and prefix-reuse behavior with real sampling;
it is not a new visual-quality evaluation of the character-swap model.

## Native Sol toggle (2026-10-08 UTC)

Created `labs-character-swap-turbo8-native-sol` as a separate saved workflow,
preserving the original. The built-in `BlockSparseAttention` node selects
`sol-attn`; bypass returns to the original Comfy Kitchen dense model path.
No runtime restart, package installation or model download was necessary.

The browser serialized both enabled and bypassed graphs correctly. Real GPU
validation used the same 43-frame source, reference, prompt, master seed 904234,
eight steps, 800×608 resolution, 22-frame windows and 18-frame history:

- Dense: `08766ebd-2ff3-4616-9398-ad92d1917e12`, success.
- Sol: `5d869ce1-88cf-417d-b161-57c2f7d108d5`, success.
- Both runs used the same two derived seeds; switching attention generated new
  checkpoints for both windows instead of reusing dense pixels.
- For this short smoke fixture only, min_tokens was 0 to force eligibility.
  The user-facing saved workflow retains 12288 and 124-frame windows.
- Verbose runtime logs show dense early steps followed by `sparse producer path`
  at 8747 tokens in window 1 and 11655 in window 2. The KJ memory patches,
  source/reference conditioning and generated AV history remained connected.
- Browser-observed whole-job times were 47.745s dense and 46.953s Sol. This is
  one short, sequential, unbalanced-loading sample, not a speedup measurement.
  Neither visual quality parity nor normal 124-frame-window speed is established.
- Both MP4 outputs and preview manifests were produced. Browser reported no
  uncaught errors; queue empty after testing.
- Reopened the saved workflow through the Tailnet userdata route and verified
  its contents match the source artifact. Sol output played in the browser;
  both final MP4s support HTTP 206 ranged downloads over Tailnet HTTPS.
- Evidence: `/home/akatz/dev/artifacts/labs-relay-native-sol-20261008/`.

## Incrementing window seeds and single-window parity (2026-10-08 UTC)

Supersedes the hashed seed scheme in the earlier master-seed validation. The
default seed is now `(master + zero_based_window_index) mod 2^64`. The scheme
identifier is part of the cache configuration, so old hashed checkpoints and
overrides are cleared on the next Run while their output files are retained.
Per-window rerolls and unchanged-input reuse keep their existing behavior.

The reported mismatch compared a single-window seed of 904234 with a first
relay-window hashed seed of 1384856719383077662. The user's relay job also had
duration 0 (full clip), whereas the single-window screenshot used duration 5.
The relay's first window has no generated history; later windows do.

Matched real GPU validation used the uploaded portrait video and character
reference from the saved workflows, the same full prompt and model/LoRA
ancestry, seed 904234, 480 preset, duration 5, eight er_sde/simple steps, CFG 1,
and Sol bypassed. All shared API inputs were checked for equality.

- Single-window `labs-character-swap-turbo8-auto` prompt:
  `dadd2f70-ab34-4807-9048-e480e2af16ae`, success.
- Relay prompt: `e9d87011-65d4-415b-9f77-0e247205ddcc`, success; one freshly
  rendered window with seed 904234 and 124 frames.
- Both final videos are 512×928, 124 frames / 5.166667 seconds. FFmpeg PSNR
  comparison over all decoded frames reported infinity for Y/U/V, average,
  minimum and maximum: decoded video pixels were identical in this test.
- 26 focused cache/preview, windowed-edit and Person Remover tests passed,
  including sequential seeds, uint64 wrap, migration from hashed controls,
  master changes and preservation of same-master per-window overrides.
- Both outputs support HTTP 206 ranged playback over Tailnet HTTPS. No uncaught
  browser errors; queue empty after verification.
- Evidence: `/home/akatz/dev/artifacts/labs-relay-sequential-seed-20261008/`.

This verifies the first-window comparison under matched conditions, not general
bitwise determinism across devices, software versions or continuation windows.


## Main integration without latent experiment (2026-10-08 UTC)

The non-experimental integration uses commit `394f9e7` and retains decoded
window previews, automatic checkpoint reuse, dependent rerolls, incrementing
master/window seeds, and the optional Native Sol comparison workflow.
The latent-history implementation from `cfd2be4` is preserved separately on
`experiment/character-swap-latent-20261008`; its toggle, graph helpers and UI
changes are excluded from this integration.

The 71 unit tests passed across the existing worker Python environment (70)
and hub Node-enabled environment (the workflow-cloning test). The complete
worker invocation initially reported that test as an error because the GPU
container has no Node executable; the unchanged test passed on the hub.
The expanded ComfyUI runtime contract, JavaScript syntax and Git whitespace
checks passed. Prior matched GPU evidence
for these exact non-experimental code changes is recorded above.
