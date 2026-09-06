# Validation record

## Official comfy-kitchen 0.2.33 release candidate (2026-09-06 UTC)

Linux RTX 4090 functional validation used a fresh Python 3.11 environment,
the unmodified official PyPI `comfy-kitchen==0.2.33` wheel, PyTorch
`2.13.0+cu130`, Kijai ComfyUI VSA commit
`10febb01d7be73d1491cf5e5347b5ab8b6c2c09e`, and MMH3 Ultimate commit
`6db8fa5a4e4ca0718d2ea8d08002ea899fe27721`. No temporary
`SolAttnMiniMax` node or private kernel wheel was installed.

- FastH3 raw smoke: 416x256, 39 frames, native stereo audio, successful history
  and full video/audio decode.
- Three-shot five-reference continuation: the packaged 3x10-second graph was
  deliberately shortened to three 3-second windows at 832x480. All three raw
  shots and all three one-step Ultimate 2x passes completed. Delivery contains
  73 + 55 + 55 = 183 frames after continuation trimming. Final assembly is
  1664x960 at 24 fps with 7.625 seconds of stereo 32-kHz audio; full decode passes.
  This shortened run is a functional test, not full-duration quality proof.
- Standard H3 hybrid/Spectrum-16 first-shot regression on the same runtime:
  416x256, 39 frames, stereo audio, successful history and full decode.
- 41 Relay unit tests pass, including exact native-audio rounding acceptance
  and rejection of unrelated waveform truncation. Expanded runtime contracts pass.
- Recovery deliberately restored accepted raw checkpoints and completed the
  remaining Ultimate stages after fixing the 266-sample native-audio tail.
  Initial failure histories are retained rather than discarded.

The original 56-frame continuation checkpoint can decode to 74,400 audio
samples because H3 rounds duration to its 40-Hz audio-latent grid; the exact
video clock needs 74,666. Relay now pads only this exact expected native
rounding case before cropping the boundary frame. Other short waveforms fail.

Upstream caveat: the installed wheel does **not** pass its entire upstream
Sol-Attention suite. A clean-process valid-input selection produced 93 passes,
one failure, and 11 deselections. The failure is the synthetic top-k-ties
accuracy case (cosine 0.982592 versus 0.99). The full 105-case run additionally
exposed direct native-binding validation failures; a negative-range case caused
an illegal CUDA access and subsequent process-local cascades. All tested
chunked VSA/coarse-gate valid-input cases passed. These results support this
tested Relay path, not a blanket claim of upstream kernel correctness.

Windows GPU execution remains unvalidated in this pass. The official Windows
wheel's published hash and exported wrapper/native symbols were checked, but
that is not equivalent to running inference. Stock ComfyUI remains insufficient
for FastH3 until its VSA model integration is available; the VSA commit is part
of the required configuration.

Local evidence is retained under `benchmark/results/official-wheel-20260906/`
(ignored): frozen prompts, all histories, kernel JUnit XML, stream metadata,
media hashes, and full-decode results. The desktop remained active and 9 GiB
VRAM was reserved, so timings are not controlled performance benchmarks.

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

## Experimental FastH3 VSA Relay profile

The FastH3 profile was initially validated against ComfyUI PR #15958,
comfy-kitchen PR #117, Kijai's step-1300 INT8 ConvRot checkpoint, and the
temporary `SolAttnMiniMax` VSA node. H3 Relay now owns the equivalent narrow
adapter against the merged comfy-kitchen PR #117 API. Capability testing found
that the PyPI 0.2.31 wheel does not yet include that merged API, despite sharing
the source version number; the adapter rejects that artifact before sampling.
The live expanded graph contains Euler, the simple scheduler, four steps,
shifts 12/3, VSA 10 percent keep over the complete sampling range, no Spectrum
node, and no Turbo LoRA even when the visible Generate Shot `h3_steps` input is
16.

The Relay-owned adapter was validated on `akatzfeyserver` against Kijai
ComfyUI commit `10febb01d7be73d1491cf5e5347b5ab8b6c2c09e`, PyTorch
2.13.0+cu130, and an upstream-PR-#117-capable comfy-kitchen 0.2.31 CUDA wheel.
The historical `sol_attn_minimax_v5` directory was mounted empty. Live
`object_info` contained `H3RelayInternalFastH3VSA` and did not contain
`SolAttnMiniMax`. A fresh uncached one-second request patched all 50 H3 blocks,
executed successfully in 20.37 seconds, and assembled 39 frames / 1.625 seconds
of 416x256 H.264 video with 32 kHz stereo AAC. The repository suite passed 30
tests, including capability, gate, layout, graph-expansion, and fail-closed
contracts.

A second live request restored that accepted first shot, generated only a new
continuation through the Relay-owned adapter, and assembled two segments into
60 frames / 2.500 seconds. The continued segment delivered 21 new frames /
0.875 seconds after its 18-frame history prefix was removed; decoded picture
and sound were both 0.875 seconds with 0.00 ms drift. The request completed in
21.13 seconds and exercised the VSA layout observer together with Relay's
history-anchor and audio-trim paths.

The same live request against the official PyPI 0.2.31 wheel stopped at
`H3RelayInternalFastH3VSA` with a clear missing `sol_attn` /
`sol_attn_chunked` error before diffusion sampling. That negative control
confirms both the upstream artifact mismatch and the absence of a silent dense
fallback.

The VSA adapter and H3 Relay both wrap `PackedLayout`: VSA observes segment
spans, while Relay positions sliding-history anchors. The first continuation
failed closed because the temporary node's Comfy loader module name is a
filesystem path ending in `/sol_attn_minimax_v5`, not its import-style dotted
name. Relay now recognises both exact forms only when the wrapper closure
contains the expected callable `original_init`. A clean restart then composed
both wrappers; the accepted first shot was recovered from disk without
regeneration. The Relay-owned wrapper uses the same narrow composition
contract and retains compatibility with the historical temporary node during
migration.

Three real raw-sequence validations completed at 832x480 with 18-frame visual
and audio overlap:

- two 10-second windows assembled to 468 frames / 19.500 seconds;
- extending the same durable run reused shots 1 and 2, generated only shot 3,
  and assembled 693 frames / 28.875 seconds;
- the original four-prompt Breaking Bad Part 1 sequence assembled to 1,394
  frames / 58.085 seconds in 250.259 seconds end to end.

Each continuation logged 17 history frames plus one boundary, a shortened
target, one-frame trim, 30 audio latent steps, and zero decoded audio-duration
drift. At the two-window seam, mean absolute pixel change was 10.257 versus
8.894-14.149 for adjacent frame pairs. At the added third-window seam it was
23.650 versus 22.608-44.270 nearby. The three Breaking Bad seam changes were
2.946, 3.751, and 4.230, all inside their local neighboring motion ranges.
No visual discontinuity spike was measured at a Relay boundary.

### Direct-graph parity and accepted-latent finishing bridge

The FastH3 public profile was replayed against the frozen five-reference,
1344x768, 362-frame direct graph from `INV-H3-007-016` using the same model,
reference hashes/order, seed 424246, Euler/simple four-forward schedule, shifts
12/3, and VSA 10 percent contract. A fresh direct control reproduced the frozen
video and audio latents exactly. The initial public-node replay differed because
shot prompt normalization removed one trailing newline (4616 model-facing
characters became 4615). Preserving nonblank model-facing shot prompt bytes
restored exact equality across all 10,354,176 video-latent and 38,592
audio-latent values; both maximum absolute differences were 0.0. The corrected
public-node run completed in 202.182 seconds versus 203.191 seconds for the
fresh direct control.

**Accepted Raw Latent** was then live-loaded from the hash-recorded sequence
checkpoint and emitted the expected `[1,24,107,48,84]` video and
`[1,32,2,603]` audio tensors in 1.008 seconds. The browser-loaded Ultimate
example converted to a valid 29-node API graph with the bridge feeding MMH3
Ultimate Upscale. Neither workflow had missing node types, and no finishing
generation was submitted during workflow validation.

### First-class Ultimate finishing and Akatz three-shot example

The public **H3 Ultimate 2× Enhance** node was registered in the same isolated
RTX 4090 ComfyUI 0.34.0 runtime used for the FastH3 reference checks. The live
runtime contract expanded one accepted 243-frame shot into a graph containing:

- the shared FastH3 bundle followed by 12/3 shifts and VSA 10 percent;
- the accepted raw latent bridge with SHA-256 verification;
- matching H3 prompt/reference conditioning at 1664x960;
- the learned H3 latent 2x model;
- 136/17 temporal splitting and 1024x1024 spatial tiles with 128-pixel overlap;
- one Euler/simple refinement step at denoise 0.2;
- tiled video/audio decode and the namespaced Ultimate acceptance stage.

The generated `H3-Relay-FastH3-Akatz-3x10-Ultimate.json` then loaded through
the live ComfyUI frontend with 25 nodes and 58 links. All three Generate Shot
nodes restored seeds, fixed control mode, 10-second windows, four steps, CRF,
reference sizing, shot IDs, and five connected references without `NaN` or
positional shifts. All three Ultimate nodes restored their distinct seeds,
fixed control mode, tiling values, five references, and ordered enhanced-state
links. The three RIFE nodes remained bypassed as authored. Conversion produced
20 executable API nodes with no missing class types or required inputs. The
queue remained empty; this validation did not submit the 30-second generation
or its three finishing passes.

The first real 832x480-to-1664x960 attempt exposed an upstream spatial-parameter
constraint: the fixed 1024-pixel tile height exceeded the 960-pixel output
height. Ultimate tiling now resolves from the sequence-derived 2x canvas. The
live runtime contract confirms that the same saved 1024x1024/128 request becomes
1024x960/128 for this canvas, remains 1024x1024/128 for a 2688x1536 target, and
also clamps overlap below very small target dimensions. The failed job left the
queue empty and did not invalidate its accepted raw H3 cache.

### Release example tiers and reference assets

The FastH3 builder now emits three separate release examples from one source:

- raw one-shot: 7 canvas / 4 executable API nodes;
- one-shot plus H3 Ultimate 2x: 9 canvas / 6 executable API nodes;
- Akatz three-shot sequence: 28 canvas / 58 links / 20 executable API nodes
  after the three intentionally bypassed RIFE nodes are removed.

All three converted against the live ComfyUI 0.34.0 object schema with no
missing class types or required inputs. A browser load restored the focused
workflow's `424246` seed, fixed control mode, 5-second duration, four visible
steps, CRF 18, reference sizing, and shot id. The Ultimate example also
restored `424264`, fixed control mode, 136/17 temporal settings, 0.999 anchor,
1024 tile maxima, and 128 overlap. The complex workflow restored all three
10-second generators, three Ultimate nodes, five references, five instruction
cards, and three bypassed RIFE modes without unknown nodes or `NaN` values.

The five packaged reference images were checked byte-for-byte against the
validated inputs and are guarded by SHA-256 tests. No generation was submitted
during this workflow-documentation pass.

Ultimate dependency handling was also tested explicitly. A complete synthetic
MMH3 class registry passes; removing `MMH3UltimateUpscale` raises the documented
error with the external repository URL, missing class name, restart direction,
and confirmation that other Relay nodes are unaffected. The preflight occurs
after durable cache lookup. The live RTX 4090 service reported all four required
MMH3 classes present, passed the runtime contract, and retained an empty queue.

### Continuation-checkpoint Ultimate frame contract

A real three-shot 1344x768 FastH3 sequence exposed the difference between the
planned generation window and sampled continuation checkpoint. Shot 1 retained
72 video tokens / 243 pixel frames and delivered all 243. Shots 2 and 3 each
retained 67 video tokens / 226 pixel frames: one boundary frame plus 225
delivered frames. The previous Ultimate accept path incorrectly expected the
243-frame plan length and rejected the valid 226-frame decode.

Ultimate now reads the accepted safetensors header, conditions at the actual
checkpoint frame count, validates decoded output against that checkpoint, and
derives the removable prefix as checkpoint minus delivered frames. A targeted
disk-restored replay reused all raw shots and Ultimate shot 1. Ultimate shot 2
completed in 346.91 seconds and shot 3 in 339.69 seconds; both persisted
`original_frames=226`, `context_frames=1`, and `delivered_frames=225`.

The final cached-only assembly completed in 1.38 seconds and published a
2688x1536, 24fps, 693-frame / 28.875-second video with stereo 32kHz AAC of the
same duration. No H3 generation or first-shot Ultimate inference was repeated,
and the queue finished empty.
