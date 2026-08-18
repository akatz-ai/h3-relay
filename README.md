# H3 Relay

H3 Relay provides steerable, resumable MiniMax H3 shot generation for ComfyUI.
It separates low-resolution H3 generation from optional LTX 2.5 enhancement
and built-in frame interpolation so creators can approve a shot before paying
for finishing work.

The initial node set is:

- **H3 Relay · H3 Hybrid Model Loader**
- **H3 Relay · H3 Model Loader**
- **H3 Relay · LTX Upscale Model Loader**
- **H3 Relay · Pack LTX Model**
- **H3 Relay · Cache Manager**
- **H3 Relay · Apply Model LoRA**
- **H3 Relay · Attention Backend**
- **H3 Relay · Sequence Start**
- **H3 Relay · Generate Shot**
- **H3 Relay · LTX 2× Enhance**
- **H3 Relay · Interpolate**
- **H3 Relay · Assemble**

The pack is intentionally staged:

```text
Sequence Start -> Generate Shot 1 -> Generate Shot 2 -> ...
                       |                    |
                       v                    v
                 LTX Enhance 1 ------> LTX Enhance 2
                       |                    |
                       v                    v
                  Interpolate 1 ------> Interpolate 2 -> Assemble
```

The Assemble node includes **Run staged · bounded RAM**. This action derives
the dependency order from the visible graph, then queues each H3 shot, LTX
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
- `enhanced` is the LTX/interpolation/assembly state.

Interpolation consumes and returns the same `enhanced` type. Bypass
every interpolation node to assemble LTX video at 24 fps; enable them to
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
Users never wire cache tags between shot nodes. Loaders retain one advanced
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

H3, LTX, and interpolation intermediates for new run names are stored under
ComfyUI's protected `user/__h3_relay_cache` system-user directory. Browser
previews use hard-linked files in ComfyUI `temp`, so they remain playable
without duplicating data blocks or appearing as published outputs. Existing
legacy run names continue reading their established `output/h3_chains` and
`output/h3_relay` paths.

Connect a native **Save Video** node to publish any stage `VIDEO` under
`output/`. **H3 Relay · Assemble** always publishes the final movie there.
Cache Manager reports cache usage and can prune superseded immutable rerolls;
current metadata references and the configured recent revisions are retained.

Encoding controls are named `output_crf` on Generate Shot, LTX Enhance, and
Interpolate. CRF is no longer a sequence-wide H3 setting because it controls
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
- MiniMax H3 text encoder and video/audio VAEs
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

Regenerate the checked-in workflow after changing its builder or source spec:

```bash
node scripts/build_example_workflow.mjs
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
