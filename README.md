# H3 Relay

H3 Relay provides steerable, resumable MiniMax H3 shot generation for ComfyUI.
It separates native H3 generation from optional H3 Ultimate or LTX 2.5
enhancement and built-in frame interpolation so creators can approve a shot
before paying for finishing work.

The initial node set is:

- **H3 Relay · H3 Hybrid Model Loader**
- **H3 Relay · H3 Model Loader**
- **H3 Relay · FastH3 VSA Profile**
- **H3 Relay · LTX Upscale Model Loader**
- **H3 Relay · Pack LTX Model**
- **H3 Relay · Cache Manager**
- **H3 Relay · Apply Model LoRA**
- **H3 Relay · Attention Backend**
- **H3 Relay · Sequence Start**
- **H3 Relay · Generate Shot**
- **H3 Relay · Accepted Raw Latent**
- **H3 Relay · H3 Ultimate 2× Enhance**
- **H3 Relay · LTX 2× Enhance**
- **H3 Relay · Interpolate**
- **H3 Relay · Assemble Raw Sequence**
- **H3 Relay · Assemble**

The pack is intentionally staged:

```text
Sequence Start -> Generate Shot 1 -> Generate Shot 2 -> ...
                       |                    |
                       v                    v
             Ultimate/LTX Enhance 1 -> Ultimate/LTX Enhance 2
                       |                    |
                       v                    v
                  Interpolate 1 ------> Interpolate 2 -> Assemble
```

The Assemble node includes **Run staged · bounded RAM**. This action derives
the dependency order from the visible graph, then queues each H3 shot,
enhancement, interpolation, and final assembly as separate jobs. Accepted
state is restored from integrity-checked disk manifests, and an explicit
release barrier clears ComfyUI's executor cache and loaded models between
jobs. A failed stage stops the sequence instead of queueing invalid dependents.

The ordinary ComfyUI queue action remains available for users who explicitly
want one monolithic prompt. The staged action is the recommended path for long
sequences because its memory requirement is bounded by one active shot rather
than growing with shot count.

Raw H3 continuation depends only on accepted raw H3 checkpoints. LTX carries
its own 17-frame/three-latent temporal context, and interpolation removes the
corresponding duplicated prefix after running. This lets users review and
reroll native 480p H3 video/audio before loading LTX or the interpolation
model.

The public token names intentionally distinguish the two graph streams:

- `sequence` is native H3 continuation state;
- `enhanced` is the Ultimate/LTX/interpolation/assembly state.

**Accepted Raw Latent** is an optional bridge from `sequence` to ComfyUI's
standard nested H3 `LATENT` type. It loads an accepted shot's integrity-checked
video/audio checkpoint by sequence position (`-1` means latest), so latent-aware
finishers such as MMH3 Ultimate Upscale can reuse reviewed H3 work without
hard-coded cache paths or regeneration. The sequence remains the authority;
the node rejects missing shots, missing hashes, and mismatched checkpoint
content before returning the latent.

**H3 Ultimate 2× Enhance** packages that bridge into the same staged finishing
contract as LTX. It reuses the connected FastH3 VSA `h3_model`, automatically
reuses the accepted shot's complete global and scene prompt, reconnects up to
nine H3 references, expands the accepted latent with the learned H3 2× model,
and performs tiled one-step FastH3 refinement. Later shots retain their raw H3
history prefix during refinement, then remove exactly the same prefix before
publishing the enhanced segment. Its `enhanced` output can connect directly to
the next Ultimate node, to Interpolate, or to Assemble. Tile width, tile height,
and overlap are maxima: the node derives the 2x canvas from `sequence` and
clamps those values automatically (for example, 832x480 becomes 1664x960 and a
1024x1024 request resolves to 1024x960 tiles).

For continuation shots, the sequence's generation-window length can be larger
than the sampled checkpoint latent. With the default 18-frame history, a
243-frame window produces a 226-frame checkpoint containing one retained
boundary frame plus 225 delivered frames. Ultimate reads that checkpoint frame
count directly, refines the complete latent, and removes only the retained
boundary before publishing the shot.

### Required node pack for Ultimate finishing

**H3 Relay · H3 Ultimate 2× Enhance** requires the external
[Comfyui-MMH3-UltimateUpscale](https://github.com/bbaudio-2025/Comfyui-MMH3-UltimateUpscale)
node pack and its H3 latent-upscaler model. H3 Relay deliberately does not copy
or silently download that implementation. Install it under
`ComfyUI/custom_nodes/` and restart ComfyUI before running an Ultimate graph.

H3 Relay itself and every non-Ultimate Relay node continue to load when the
external pack is absent. The Ultimate node remains visible so saved workflows
can open, but a fresh Ultimate execution stops with an actionable error listing
the missing MMH3 classes and installation URL. An already cached, verified
Ultimate result may still be restored without loading the external engine.

Interpolation consumes and returns the same `enhanced` type. Bypass
every interpolation node to assemble enhanced video at 24 fps; enable them to
assemble the interpolated version.

Interpolation is temporally streamed in overlapping source chunks (48 frames
by default). Each chunk shares one exact boundary frame with the next, invokes
ComfyUI's native interpolation implementation, and is encoded immediately.
This preserves adjacent-pair RIFE math without retaining an entire 15-second
1664x960 float-frame tensor in system RAM.

Chunk boundaries are mathematically exact for RIFE: each chunk repeats its
first source boundary frame, the duplicate output frame is discarded, and the
remaining adjacent pairs are identical to a monolithic pass. The chunk size is
an execution-memory setting and deliberately does not invalidate an otherwise
identical cached interpolation result.

H3 and LTX models are loaded once and fanned out to every shot through one
typed model-bundle wire. The LTX bundle contains its model, VAE, latent
upscaler, text encoder, required LoRAs, and internal cache fingerprint. H3
Relay's LoRA and Attention nodes update that hidden fingerprint whenever they
patch a bundle, so a model-chain change invalidates only derived artifacts.
Users never wire cache tags between shot nodes. The H3 Ultimate finisher shares
the same connected H3 bundle used by Generate Shot instead of loading another
transformer implicitly. Loaders retain one advanced
`manual_cache_revision` field only for replacing model contents without
changing filenames.

That fan-out remains the visible graph contract. In bounded-memory staged
mode, each partial prompt constructs only the model family required by its
target; cached prerequisites are restored without traversing their loaders.
The default minimum-RAM release barrier unloads models between jobs, trading a
small reload cost for predictable memory. Cached model patchers remain lazy,
so a cache hit does not materialize checkpoint weights on the GPU.

Advanced users can build an LTX stack with native ComfyUI MODEL, VAE,
LATENT_UPSCALE_MODEL, CLIP, LoRA, attention, and patch nodes, then use
**Pack LTX Model** to convert those four components into H3 Relay's one-wire
bundle. Its advanced `cache_identity` must change whenever that custom native
stack changes because generic loaded objects do not retain stable cross-restart
checkpoint provenance.

The LTX finishing path uses both spatial components once: the learned latent
model expands the target latent 2x, then the pixel-spatial IC-LoRA guides a
generative diffusion refinement from the original low-resolution pixel video.
This is one 2x pipeline, not two successive 2x passes.

The attention node exposes backends actually registered by the installed
ComfyUI environment. Comfy Kitchen is used by the reference H3 graph. Sage is
shown only when its optional package/backend is installed; H3 Relay does not
silently pretend Sage is available.

The checkpoint MP4 and lossless generated WAV remain separate internally.
H3 Relay stream-copies the checkpoint picture and muxes that WAV into a
dedicated raw preview MP4, so the node's `VIDEO` output and connected core
Save Video nodes include audio without re-encoding the H3 picture.

Every stage uses disk-backed, content-addressed records. Re-running unchanged
inputs returns the accepted artifact; changing an H3 shot invalidates only the
finishing work derived from that revision.

## Managed cache and publishing

H3, H3 Ultimate, LTX, and interpolation intermediates for new run names are stored under
ComfyUI's protected `user/__h3_relay_cache` system-user directory. Browser
previews use hard-linked files in ComfyUI `temp`, so they remain playable
without duplicating data blocks or appearing as published outputs. Existing
legacy run names continue reading their established `output/h3_chains` and
`output/h3_relay` paths.

Connect a native **Save Video** node to publish any stage `VIDEO` under
`output/`. **H3 Relay · Assemble** always publishes the final movie there.
Cache Manager reports cache usage and can prune superseded immutable rerolls;
current metadata references and the configured recent revisions are retained.

Encoding controls are named `output_crf` on Generate Shot, H3 Ultimate/LTX
Enhance, and Interpolate. CRF is no longer a sequence-wide H3 setting because it controls
the cached H.264 segment, not model generation or continuation. H3 uses a
fixed internal source segment and LTX/interpolation retain CRF-18 masters;
changing `output_crf` encodes a derived variant without repeating inference.

Sequence Start exposes ComfyUI's complete native sampler list and complete
BasicScheduler list as independent controls. H3 Relay adds `beta57`, which uses
the exact manual H3 curve at 16 steps and alpha 0.5 / beta 0.7 otherwise.
Spectrum is a separate boolean. The default is Euler + beta57 + Spectrum;
sampler and scheduler can otherwise be mixed freely.

Sequence Start also exposes the sequence-wide `h3_overlap_frames` contract.
MiniMax H3 sliding history follows its 17k+1 temporal grid, so the supported
choices are 18, 35, 52, and 69 frames; 18 remains the default. Larger overlaps
carry more matched visual/audio history into each continuation but reduce the
new duration delivered by that generation window.

## Experimental FastH3 VSA profile

**FastH3 VSA Profile** loads Kijai's INT8 ConvRot repack of FastVideo's
step-1300 four-forward VSA checkpoint into the same `h3_model` wire used by
Generate Shot. The profile owns its inference contract: Generate Shot forces
Euler, the simple scheduler, four transformer forwards, CFG 1 through
`BasicGuider`, video/audio shifts 12/3, Spectrum off, and VSA with 10 percent
video-cube keep over the complete denoising range. `h3_steps` and the Sequence
Start sampler controls remain effective for standard H3 bundles but are
overridden by this locked profile.

The profile reuses H3 Relay's accepted raw checkpoint, visual/audio overlap,
cache, reroll, and sequence assembly implementation. **Assemble Raw Sequence**
publishes an accepted native H3 chain without requiring the LTX or RIFE stages.
This allows a graph to chain any number of bounded FastH3 shots while carrying
only the configured recent overlap rather than growing one unbounded model
context.

The same profile can also drive **H3 Ultimate 2× Enhance**. Its one-step
refinement reapplies shifts 12/3 and VSA at 10 percent keep to the shared model,
uses 136-frame temporal windows with 17-frame overlap by default, and uses
1024×1024 spatial tiles with 128-pixel overlap. These are finishing defaults,
not additional raw-generation steps.

FastVideo documents the preview checkpoint as T2VA-only; reference and sliding
continuation use are experimental. H3 Relay includes their media in the durable
cache identity but cannot guarantee that a future FastH3 checkpoint preserves
the inherited H3 reference behavior. Keep the standard FL2VA/Ref2VA profile as
a fallback until a reference-distilled FastH3 release is available.

The integration requires the pinned ComfyUI FastVideo-VSA model support and
the official `comfy-kitchen==0.2.33` CUDA wheel. ComfyUI PR #15958 is still
unmerged; the FastH3 environment pins Kijai's VSA commit instead of assuming
stock ComfyUI supports this checkpoint. H3 Relay checks the CUDA
`sol_attn` capability as well as the package version. H3 Relay owns the
narrow MiniMax VSA adapter and inserts it automatically; users do not install
the temporary `SolAttnMiniMax` test node. Applying another H3 Relay LoRA or
Attention Backend after the FastH3 profile is rejected because those controls
would make the locked profile ambiguous. Missing VSA gates or CUDA kernels fail
explicitly instead of silently generating with dense attention.

## Installation

In ComfyUI Manager, search for **H3 Relay**, install the node pack, and restart
ComfyUI. For a manual installation:

```bash
cd /path/to/ComfyUI/custom_nodes
git clone https://github.com/akatz-ai/h3-relay.git
```

Install the separately licensed model files described in `MODELS.md`, ensure
FFmpeg is available on `PATH`, then restart ComfyUI again.

## Requirements

- ComfyUI 0.32.0 or newer
- MiniMax H3 FL2VA and Ref2VA model files
- Optional experimental FastH3 VSA checkpoint and VSA runtime described in
  `MODELS.md`
- MiniMax H3 text encoder and video/audio VAEs
- H3 latent 2× upscaler and MMH3 Ultimate Upscale custom node pack when using
  **H3 Ultimate 2× Enhance**; neither is required by other Relay nodes
- LTX 2.5 model, VAE, distilled LoRA, pixel-spatial upscaler and text encoder
- A ComfyUI-compatible frame-interpolation checkpoint for interpolation
- FFmpeg

H3 Relay never edits ComfyUI source files. For ComfyUI builds that predate
native H3 history anchors, it installs a guarded process-local packed-layout
compatibility layer immediately before the first sliding continuation. The
fallback self-tests before activation, leaves ordinary layouts unchanged, and
fails closed when another unknown H3 layout extension owns the same hook.

H3 Relay does not redistribute model weights. The example workflow contains
an embedded installation note with the exact filenames, official download
commands, and ComfyUI folders used by the reference RTX 4090 configuration.
The same manifest is available in `MODELS.md`.

The GPL-3.0 license in this repository covers H3 Relay's source code only. It
does not grant a license to MiniMax H3, LTX, RIFE, text-encoder, or other model
weights. Review and accept every upstream model license and obtain any required
authorization before downloading or using those files, especially for
commercial use.

Workflow guidance uses the same black `MarkdownNote` card convention as the
official MiniMax H3 template: an overview card, a folder-grouped direct model
link and storage-tree card, and a size-reference table.

## Example workflow

`example_workflows/H3-Relay-Orbital-Storm-Spectrum16-58s.json` provides an
original four-shot Spectrum-16 reference configuration without subgraphs. It uses
the same prompts, global continuity direction, seeds, H3 dimensions, manual
beta57 schedule, LTX finishing prompt, RIFE model, overlap rules, and assembly
settings as the published 58-second comparison.

Run each raw H3 shot by itself and review it first. Run its LTX and interpolation
nodes only after accepting the raw result. Queueing the final assembler runs or
reuses every missing dependency.

The experimental FastH3 examples are tiered so users can validate one feature
at a time:

- `H3-Relay-FastH3-VSA-One-Shot.json` is the smallest possible five-second
  FastH3 VSA generation graph.
- `H3-Relay-FastH3-VSA-One-Shot-Ultimate-2x.json` keeps the same prompt and
  seed, then adds one H3 Ultimate 2× finishing node.
- `H3-Relay-FastH3-Akatz-3x10-Ultimate.json` is the complete three-window
  action example using the Akatz character and five explicit
  character/environment reference images. It generates three reviewable
  10-second FastH3 VSA windows at 832×480, carries 18-frame AV history between
  them, and provides raw assembly plus an optional per-shot H3 Ultimate 2×
  chain at 1664×960. RIFE is wired after every enhanced shot and bypassed by
  default. Enable all three interpolation nodes for 48 fps, or leave all three
  bypassed for 24 fps.

All three workflows contain exact experimental runtime and model-install cards.
The original generated Akatz references are included under
`example_workflows/assets/`; copy them into `ComfyUI/input/` before loading the
three-shot workflow. See `example_workflows/README.md` for the tier comparison.

Regenerate the checked-in workflows after changing their builders or source
specs:

```bash
node scripts/build_example_workflow.mjs
node scripts/build_fast_h3_akatz_workflow.mjs
```

### Clone and retheme an existing workflow

Use `scripts/clone_retheme_workflow.mjs` when a new story should retain the
runtime settings and layout of an existing saved workflow. The script rebuilds
nodes from the current canonical template, restores settings by widget name,
and writes both ComfyUI's positional and named widget state. This is important
for frontend-only values such as a seed's `control_after_generate` mode and for
saved workflows whose optional-input order predates the current node schema.

Reference images in the retheme spec are connected by input name after the
canonical socket layout is built, rather than by a hard-coded slot number.
The source workflow is always read-only and the output is refused when it
already exists unless `--force` is passed explicitly.

```bash
node scripts/clone_retheme_workflow.mjs \
  --source /path/to/saved-workflow.json \
  --template example_workflows/H3-Relay-Orbital-Storm-Spectrum16-58s.json \
  --spec scripts/fixtures/combustible-lesson-spectrum16-58s.json \
  --output /path/to/new-workflow.json
```

The `h3_relay_retheme_v1` spec contains `run_name`, `output_filename`,
`global_prompt`, `enhancement_prompt`, exactly four `shot_prompts`, and any
reference-loader definitions. Each reference definition names its target input
(`reference_image_1`, `reference_image_2`, or `reference_image_3`) and may set
its loader position and size.

## Tests

Pure workflow tests:

```bash
python -m unittest discover -s tests -p 'test_*.py' -v
```

Expanded runtime contract from a ComfyUI environment:

```bash
COMFYUI_ROOT=/path/to/ComfyUI /path/to/ComfyUI/.venv/bin/python \
  tests/runtime_contract.py
```

See `VALIDATION.md` for the live RTX 4090 staged-cache and reference-output
checks completed for version 0.1.0.

See `LTX_UPSCALING.md` for the distinction between pixel-frame and latent-frame
window sizes, the validated 4090 temporal-window preset, official Looping
Sampler behavior, and the generative upscaler's fidelity limitations.

## License and attribution

H3 Relay is GPL-3.0. See `NOTICE.md`, `UPSTREAMS.md`, and the retained licenses
under `h3_relay/vendor/`.
