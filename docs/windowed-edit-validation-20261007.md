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
