# Validation record

Validation was performed on 2026-08-16 with ComfyUI
`v0.33.0-6-g1c6d8d45`, an RTX 4090, and the model filenames documented in the
example workflow.

## Standalone loading

The following source packs were temporarily disabled before restarting
ComfyUI:

- `ComfyUI-MiniMaxH3-Contex-Loop`
- `ComfyUI-Spectrum-MiniMax-H3`
- `ComfyUI_MinimaxH3HybridLoader`

H3 Relay loaded successfully and completed raw H3, LTX, interpolation, and
assembly without those packs. The source packs were then restored, and their
original node ids coexist with H3 Relay's namespaced ids.

## Staged cache

A one-second/one-step reference shot completed every stage. After restarting
ComfyUI, the identical H3 -> LTX -> interpolation -> assembly request completed
in 545 ms. The raw H3, LTX, and interpolation artifacts were all restored from
their disk-backed content caches.

The raw H3 cache was also migrated from its picture-only checkpoint MP4 to a
user-facing preview artifact containing stream-copied H.264 plus stereo 32-kHz
AAC from the canonical generated WAV. Both the direct `VIDEO` output and a
connected ComfyUI core Save Video node preserved the audio stream. Repairing an
already completed 15-second shot from cache took about one second and repeated
no H3 inference.

## Two-shot continuation

A two-shot test verified both continuation chains:

- raw H3 Shot 2 resumed from Shot 1's AV checkpoint;
- LTX Shot 2 carried 17 decoded H3 frames and three LTX latent steps;
- interpolation removed 33 repeated 48-fps prefix frames;
- the final 1664x960/48-fps movie contained H.264 video and stereo 32-kHz AAC;
- assembly produced 119 delivered frames over 2.479 seconds.

## Published-shot equivalence

The first 15-second shot from the original internal Spectrum-16 workflow
was rerendered through H3 Relay using the same global prompt, shot prompt,
seed 424242, 832x480 dimensions, 362 raw frames, 16-step Spectrum Euler manual
beta57 profile, LTX models/settings, and RIFE 4.26 heavy 2x interpolation.

- Raw H3 decoded-video framemd5 stream: identical SHA-256
  `2c5d3400f98f53106c1cb896e998f6b9e61f3bbb77c05b48f30d231db35e30c0`.
- Generated H3 WAV: identical SHA-256
  `69b644ff7aa60ad5ad5a01d3265f74b316fa479b375d49efc4d25895eb0dcc54`.
- Both finished videos: 1664x960, 48 fps, 723 frames, 15.063 seconds.
- Finished-video comparison: SSIM `0.991877`; average PSNR `45.147389 dB`.

The raw H3 picture and sound are bit-for-bit equivalent after decode. The
finished output is perceptually near-identical; minor LTX/RIFE differences are
expected from the independently queued finishing pass and encode path.

## Larger LTX temporal windows and preview

The original 129/32-pixel diffusion window and 64/8-pixel VAE temporal tile
were compared with a 193/64 diffusion window and 128/16 VAE tile using the
same cached 362-frame H3 shot.

- ComfyUI reported 25 latent frames with 8-frame latent overlap over the full
  47-frame latent video.
- The three-sigma pass used nine window evaluations instead of fourteen.
- The pass completed without OOM in 260 seconds on the RTX 4090.
- H3 inference was not repeated.
- The LTX node published a standard animated-video preview directly in its
  execution output; cached previews use the same UI payload.

The larger-window output is a new cached LTX revision. Visual inspection is
still required before claiming that it eliminates every face/background
artifact because the IC-LoRA itself is a generative re-renderer.

## Public model graph and sequence naming

A one-second end-to-end graph was run through the public H3 Hybrid Model
Loader, cache-aware Comfy Kitchen attention node, H3 generation, public LTX
model-stack loader, PyTorch LTX attention node, LTX enhancement, shared core
frame-interpolation loader, interpolation, and assembly. It completed in 45
seconds and produced raw, LTX, interpolation, and final previews with audio.

The same cached graph was assembled with interpolation omitted. Assembly
selected the LTX records and produced 1664x960 video at 24 fps, confirming that
the type-identical `enhanced` input/output makes interpolation bypass safe.

## Consolidated model bundles

The public contracts were shortened to `enhanced` and `previous_enhanced` and
the finishing type was renamed `H3_RELAY_ENHANCED`. H3 and LTX loaders now
produce one `H3_RELAY_MODEL` bundle each. The LTX bundle carries its diffusion
model, VAE, latent upscaler, text encoder, required LoRAs, and cache identity
over one graph edge. Interpolation uses one equivalent bundle edge.

A one-second full pipeline using the bundled H3, LTX, attention, and
interpolation nodes completed in 45 seconds. A cached run with interpolation
omitted assembled 1664x960 LTX output at 24 fps. The migrated reference graph
validated against every live node input/output schema and contains no public
cache-tag wires.

## Version 0.4 release boundary

The local ComfyUI source modification to `PackedLayout` was removed completely;
`git diff -- comfy/ldm/minimax/model.py` is empty. H3 Relay now provides the
history-keyframe behavior as a guarded process-local compatibility layer.

The runtime contract passed against detached, unmodified checkouts of:

- ComfyUI `v0.32.0`;
- ComfyUI `v0.33.0`;
- current commit `1c6d8d45`.

The older tagged builds exercised the vendored multi-step video/audio guide
engine plus full history layout. The current build exercised the smaller
position-only wrapper around its native guide layout. These checks validate
node loading, patch ownership, history self-tests, and graph expansion; the
live render below validates the current production path.

An isolated ComfyUI `v0.32.0` server was also launched from a detached stock
worktree with only H3 Relay installed. A real two-shot GPU render named
`h3_relay_v4_comfy032_smoke` passed through the full legacy fallback: the
packed-guide engine, AV payload merge, full history layout, and target trim.
Shot 1 was restored from H3 Relay's disk cache and Shot 2 generated in 11.16
seconds. Its 21-frame result is 832x480/24 fps H.264 with stereo 32 kHz AAC;
picture and sound are both exactly 0.875 seconds. The temporary server and
worktree were removed afterward.

The official Comfy-Org MiniMax H3 checkpoints were downloaded under their
published filenames and verified against their Hub LFS SHA-256 values:

- FL2VA int8 ConvRot: `7ad4c73e6e378b822ffd1629f27f632d3787d95f5e468e3af958f98c58df96a5`;
- Ref2VA int8 ConvRot: `9eef934046a0671bc8a5daf87100705e1478419c574cfde70c50fbe6885f76a9`.

After restarting ComfyUI on the clean core, a new two-shot run named
`h3_relay_v4_official_clean_core_smoke` completed using the official hybrid
pair. Shot 2 activated H3 Relay's process-local history layer and logged:

- 18-frame overlap = 17 history frames + one boundary frame;
- target shortened to 22 frames;
- one repeated boundary frame trimmed;
- sampled-latent audio history with 30 latent steps.

The two one-step shots completed in 20.88 seconds. Their raw previews contain
H.264 video at 832x480/24 fps and stereo 32 kHz AAC audio. Shot 2 records Shot
1's revision and checkpoint SHA-256 as its explicit predecessor, proving that
the continuation was neither an independent render nor a stale cache hit.

The example and saved workflows now use official model filenames and contain a
Markdown installation note with exact Hugging Face sources, commands, ComfyUI
folders, and the distinct roles of the LTX distilled and pixel-upscaler LoRAs.

## Advanced LTX adapter and prompt sockets

H3 Relay now exposes **Pack LTX Model**, which accepts native ComfyUI MODEL,
VAE, LATENT_UPSCALE_MODEL, and CLIP values and returns the same `ltx_model`
bundle used by the full loader. Its stable `cache_identity` is included in the
durable cache tag and must be changed when a custom upstream chain changes.
The runtime contract verifies stable tags for identical identities and a new
tag after the identity changes.

The full loader now names the learned component `latent_2x_model_name` and the
diffusion adapter `pixel_upscale_ic_lora`. Tooltips explain that latent
expansion happens first and the IC-LoRA then guides one generative 2x
refinement from the original pixel video. `cache_revision` is now the clearer
advanced `manual_cache_revision`; the internal v0.3 fingerprint field names
remain unchanged so the UI rename alone does not invalidate durable artifacts.

Shot `prompt` and LTX `enhancement_prompt` are required STRING sockets instead
of linked widgets. The live schema advertises `forceInput: true`, and the saved
workflow contains named, connected sockets without widget metadata, preventing
the frontend's unlabeled-dot rendering.

The saved workflow migration preserved the user-added H3 LoRA model path,
current functional-node positions, prompts, and the 24 fps `ltx` assembly
selection.

## Reference note formatting and native samplers

The generated and saved workflows now use the same annotation convention as
ComfyUI's `video_minimax_h3_t2v` template: black `MarkdownNote` cards with
`#222` title bars, `#000` backgrounds, `Note:` titles, folder-grouped direct
model links, a visual `ComfyUI/models` storage tree, issue links, and the same
size-reference table.

The migration consolidated the frontend-renumbered installation note into the
canonical Model Links card, retained the existing Size Settings card position,
and replaced the plain workflow Note with `Note: H3 Relay`. It also reconciled
four frontend-renumbered shot nodes back to their canonical IDs at their
current positions. The user-added H3 LoRA node remains connected between the
hybrid loader and H3 attention; duplicate shot nodes and the direct
loader-to-attention edge were removed. The 24 fps `ltx` assembly choice remains
selected.

Sequence Start now exposes independent `sampler`, `scheduler`, and
`spectrum_enabled` controls. Sampler values come directly from
`comfy.samplers.SAMPLER_NAMES`; scheduler values are `beta57` plus
`comfy.samplers.SCHEDULER_NAMES`. The default is Euler + beta57 + Spectrum.
At 16 steps beta57 uses the exact manual sigma list; other step counts use the
equivalent alpha 0.5 / beta 0.7 scheduler.

Runtime contracts verify independent graph expansion, including Euler/simple
and res_multistep/simple without Spectrum plus the default manual-beta57
Spectrum path. Fresh two-step GPU smoke renders completed successfully for:

- `h3_relay_sampling_controls_euler_simple_no_spectrum`;
- `h3_relay_sampling_controls_res_multistep_simple_no_spectrum`;
- `h3_relay_sampling_controls_euler_beta57_spectrum`.

All three outputs contain 832x480/24 fps H.264 video and stereo 32 kHz AAC
audio over 1.625 seconds.

## Managed cache, publishing, and per-stage CRF

Sequence Start no longer exposes `h3_crf`. Generate Shot, LTX Enhance, and
Interpolate expose the common `output_crf` name. H3 keeps its internal
continuation/source segment at CRF 18; LTX and interpolation keep CRF-18 master
videos. Requested output variants are encoded from those cached masters, so
CRF never participates in H3, LTX diffusion, or RIFE inference identities.

New run `h3_relay_managed_cache_smoke` validated the storage boundary:

- H3 segment, WAV, AV checkpoint, metadata, raw muxed preview, LTX full video,
  LTX delivered video, and rolling latent were written under
  `user/__h3_relay_cache`;
- no intermediate `output/h3_chains/h3_relay_managed_cache_smoke` directory
  existed before explicit publication;
- the browser preview was reported as `type=temp` through a hard link sharing
  the cache file's inode, so it consumed no second data block;
- a connected native Save Video node published an 832x480/24 fps H.264 video
  with stereo 32 kHz AAC under `output/h3_relay_published`;
- LTX remained cached and previewed through temp while Assemble alone
  published a 1664x960/24 fps H.264/AAC final movie under
  `output/h3_chains/.../enhanced/final`.

Run `h3_relay_crf_separation_smoke` validated encoding-only invalidation:

- H3 CRF 18 created one AV checkpoint; CRF 24 completed in about 0.5 seconds,
  created only a `.crf24.mp4` preview variant, and retained that checkpoint;
- LTX CRF 18 performed one 25-second diffusion pass; CRF 24 completed in about
  one second with status `reused cached LTX inference`;
- RIFE CRF 18 completed in about 2.6 seconds; CRF 24 completed in about 0.5
  seconds with status `reused cached interpolation`.

Managed artifact references use `cache://` URIs. Legacy bare output-relative
paths remain readable, and existing legacy H3/LTX runs resolve to their
original output directories instead of being moved or duplicated.

Cache Manager live inspection reported 87.39 MiB across 17 files and three
revision groups for the validation run. Inspect was read-only; protected prune
removed nothing. Runtime tests create three synthetic immutable revisions,
protect the current pointer, prune to one recent revision, and verify that the
two superseded revisions and any preview links are removed while the current
revision remains.

## Version 0.5 bounded-memory staging

The original cold four-shot 58-second graph ran as one prompt in 1,648.97
seconds. It peaked at 23,037 MiB VRAM, 98.54 GiB resident system RAM, 110.32
GiB ComfyUI cgroup memory, and 62.31 GiB swap. ComfyUI retained same-prompt H3,
LTX, decoded-frame, and interpolation tensors as active cache entries until
the queue item ended.

Version 0.5 divides Assemble execution into individual partial prompts and
restores only small, integrity-checked manifests between them. An internal
release prompt resets ComfyUI's execution cache and unloaded models before the
next stage. Cached two-shot staging completes seven jobs in 11.35 seconds;
cached H3 and RIFE checks take about 0.25 seconds each.

Chunked interpolation was compared against the original monolithic RIFE
artifacts at both a real 48-frame boundary and across the complete four-shot
reference. Both comparisons produced SSIM 1.000000 and infinite PSNR. The two
complete 58.064-second assembled MP4 files are bit-for-bit identical:

`78c6dccfbd27b887839f5fad0823ec218bbe60888a770e56bfb0b35323c9f700`

Fresh chunked RIFE over the four cached 15-second LTX shots plus final assembly
completed in 119.84 seconds with a 22.25 GiB system-RAM peak. The old
monolithic finishing path reached 98.54 GiB.

A genuinely new ten-shot H3 -> LTX -> RIFE -> Assemble graph completed 31
separate jobs in 400.23 seconds. H3 and LTX used real inference at one H3 step
for iteration speed; every stage wrote and restored its own artifacts. Peak
VRAM was 23,959 MiB and peak system RAM was 22.40 GiB. Per-shot timings stayed
flat as the sequence grew, and the 1664x960/48 fps output contains ten video
and stereo-audio segments over 9.48 seconds.

Additional live configurations passed:

- three-shot interpolation-bypassed LTX assembly at 24 fps;
- two-shot 35-frame H3 history (34 history + one boundary), LTX, chunked RIFE,
  and assembly with a 22.79 GiB RAM peak;
- stale Shot 2 prompt invalidation, where Shot 1 reused in 0.32 seconds, Shot 2
  and its finishing dependents regenerated, and untouched finishing stages
  remained cached;
- current ComfyUI v0.33 and detached stock ComfyUI v0.32 runtime contracts.

## Version 1.0 public release audit

The public repository and Registry archive contain no internal themed example
names, prompts, filenames, or documentation. The replacement Orbital Storm
workflow uses original characters and dialogue. A shortened fresh four-shot
GPU smoke generated all H3, LTX, chunked-RIFE, and assembly stages in 173.05
seconds; its 1664x960/48 fps H.264 output contains stereo 32 kHz AAC and four
ordered records in every manifest.

The legacy ComfyUI 0.32 packed-guide compatibility installer was rewritten as
ordinary static Python functions. Registry security scans find no dynamic
execution calls. A real two-shot ComfyUI 0.32 GPU continuation completed in
20.17 seconds using the static fallback, with 18-frame visual/audio history
and frame-exact generated sound.

Official `comfy-cli 1.16.0` validation passed. Its `.comfyignore`-filtered
archive contains 66 files, is 410,153 bytes compressed, retains the root GPL
and all three upstream licenses, and excludes benchmarks, tests, migration
scripts, internal source specs, and unused Spectrum evaluation modules. The
exact ZIP loaded in a clean stock ComfyUI 0.33 checkout with only H3 Relay
installed; all 13 public nodes, 20 namespaced internal runtime nodes, the
staged HTTP route, and frontend extension registered successfully.
