# Changelog

## Unreleased

- Provision pinned ComfyUI and CPU-only PyTorch in CI so VSA/audio unit tests
  and expanded runtime contracts run without a GPU or model downloads.
- Add a separate, materialization-tested ComfyGit FastH3/Ultimate recipe with
  immutable public model sources and official kitchen 0.2.33. Keep the older
  standard-H3/LTX/RIFE recipe as a clearly identified historical baseline.
- Use the official comfy-kitchen 0.2.33 CUDA wheel and Relay-owned VSA adapter
  for both FastH3 generation and Ultimate refinement.
- Fit short Ultimate continuation audio to the exact video clock when the
  native 40 Hz audio-latent rounding leaves a sub-frame tail; reject waveforms
  that do not match that native duration.
- Add **H3 Ultimate 2× Enhance**, a cache-aware, sequence-compatible finisher
  that reuses the connected FastH3 VSA model, the accepted AV latent, the full
  accepted shot prompt, and up to nine H3 references. It applies the learned
  H3 latent 2× model plus tiled one-step refinement and emits the same
  `enhanced` stream consumed by Interpolate and Assemble.
- Document MMH3 Ultimate as a required external node pack for fresh Ultimate
  execution and add an actionable runtime preflight listing missing engine
  classes. Other Relay nodes and verified cached Ultimate results remain usable
  without materializing the external engine.
- Treat Ultimate tile dimensions and spatial overlap as safe upper bounds,
  clamping them to the sequence-derived 2x canvas. A 832x480 sequence now
  resolves the standard 1024x1024 request to 1024x960 automatically.
- Derive Ultimate conditioning and acceptance length from the accepted AV
  checkpoint rather than the larger continuation generation window. Later
  shots retain one boundary frame in their sampled latent (226 checkpoint / 1
  boundary / 225 delivered for a 243-frame window), which is now refined and
  cropped without rejecting the valid output.
- Replace the generic FastH3/LTX sample with a reproducible Akatz three-shot,
  three-by-ten-second cyberpunk sequence. The new canvas includes five
  references, raw assembly, per-shot H3 Ultimate finishing, and bypassed
  optional RIFE stages.
- Package three documented FastH3 example tiers: raw one-shot, one-shot plus
  H3 Ultimate 2x, and the complete three-shot reference sequence. Include the
  five original generated reference assets, exact experimental runtime pins,
  per-tier model links/storage paths, and resolution guidance.
- Add **Accepted Raw Latent**, an integrity-checked public bridge from an
  accepted sequence shot to nested H3 video/audio latent finishers.
- Preserve nonblank model-facing shot prompt bytes, including outer
  whitespace, so the public FastH3 profile can reproduce equivalent direct
  graphs at the same seed.
- Add the experimental **FastH3 VSA Profile** loader with a cache-owned,
  four-forward Euler/simple contract, shifts 12/3, Spectrum disabled, and
  VSA-H3 at 10 percent keep.
- Add **Assemble Raw Sequence** for publishing accepted native H3 chains
  without traversing the LTX or interpolation finishing stages.
- Replace the temporary external `SolAttnMiniMax` node with a Relay-owned,
  fail-closed FastH3 adapter against official comfy-kitchen 0.2.33 while
  retaining migration compatibility with the historical layout observer.
- Extend Generate Shot from three direct reference images to nine bounded,
  cache-tracked references without changing the first three socket names.
- Validate two- and three-window FastH3 reference sequences plus the original
  four-window Breaking Bad Part 1 prompts with 18-frame visual/audio overlap.

## 1.0.2

- Declare Generate Shot's seed control explicitly so ComfyUI restores the
  seed, `control_after_generate`, duration, steps, CRF, and reference sizing
  without positional shifts or `NaN` values.
- Synchronize the public Orbital Storm example with the current browser-saved
  workflow, including seed `424243`, Comfy Kitchen LTX attention, named widget
  state, and current H3 Relay project links.
- Add a schema-aware workflow clone/retheme utility that preserves settings by
  widget name, canonicalizes browser-reordered inputs, connects reference
  assets by socket name, and validates link integrity.
- Extend saved-workflow benchmark parsing and regression coverage for named
  widget values, image loaders, frontend-reordered nodes, and rethemed prompts.
- Publish releases automatically when a `vX.Y.Z` tag matches the package
  version, while retaining manual dispatch for recovery.

## 1.0.1

- Add the shared `akatz` Registry icon used by DepthFlow and DepthCrafter so
  H3 Relay displays consistently in Comfy Registry and Extension Manager.

## 1.0.0

- First public Comfy Registry release under publisher `akatz`.
- Replace the internal test story with an original Orbital Storm workflow.
- Replace dynamic legacy H3 compatibility installation with ordinary static,
  reviewable Python functions that meet Registry security standards.
- Add a curated Registry archive, clean-install CI, model-license guidance, and
  a manual publish action.

## 0.5.0

- Add an Assemble-node staged runner that executes each H3, LTX, RIFE, and
  assembly target as a separate queue job with deterministic release barriers.
- Restore raw and finishing sequence tokens from hash-validated disk manifests
  so later jobs do not traverse prior model families or retain prior tensors.
- Stream native ComfyUI RIFE over exact one-frame-overlap temporal chunks and
  encode each result immediately; expose a 48-source-frame default.
- Preserve legacy interpolation caches because chunk size changes memory use,
  not adjacent-pair interpolation math or cache identity.
- Add a hash-guarded workflow updater for the interpolation widget and a
  browser-visible **Run staged · bounded RAM** Assemble action.

## 0.4.0

- Expose sequence-wide H3 sliding-history overlap as an 18/35/52/69-frame
  combo, retaining 18 as the default and cache-compatible legacy value.
- Replace the required `shot_name` field with an optional `shot_id`; blank
  values receive deterministic `shot_0001`, `shot_0002`, and later IDs.
- Move new H3/LTX/interpolation intermediates into a protected managed cache,
  expose temp hard-link previews, and reserve normal output publication for
  Save Video and Assemble.
- Add Cache Manager inspection and protected superseded-revision pruning.
- Move H3 CRF from Sequence Start to per-shot `output_crf` and rename LTX
  `enhanced_crf` to the same stage-level name.
- Match the official MiniMax H3 workflow's black MarkdownNote formatting for
  the overview, direct model links, storage tree, issue links, and size table.
- Replace combined sampling profiles with independent native ComfyUI sampler,
  scheduler, and Spectrum controls. Default to Euler + beta57 + Spectrum and
  retain exact beta57 manual sigmas at 16 steps.
- Add **Pack LTX Model** so advanced users can bundle native MODEL, VAE,
  LATENT_UPSCALE_MODEL, and CLIP chains into H3 Relay's `ltx_model` type.
- Rename LTX loader controls to distinguish the learned latent 2x model from
  the pixel-spatial IC-LoRA, with purpose-specific tooltips.
- Rename `cache_revision` to advanced `manual_cache_revision` and document the
  custom adapter's stable `cache_identity` requirement.
- Make shot and enhancement prompts labeled, required STRING sockets instead
  of connected multiline widgets that ComfyUI rendered as unlabeled dots.
- Add a guarded, process-local MiniMax H3 history-keyframe compatibility layer
  so sliding continuation no longer requires a modified ComfyUI checkout.
- Use the official Comfy-Org MiniMax H3 checkpoint filenames in loaders and the
  example workflow.
- Add an embedded workflow model-installation guide plus `MODELS.md` with exact
  Hugging Face sources, download commands, ComfyUI folders, and the separate
  purposes of the LTX distilled and pixel-upscaler LoRAs.
- Use the official root-level LTX pixel-upscaler LoRA filename.

## 0.3.0

- Shorten `enhanced_sequence` and `previous_enhanced_sequence` to `enhanced`
  and `previous_enhanced`.
- Consolidate H3 and LTX model components plus cache identity into one typed
  model-bundle wire.
- Hide routine cache tags inside loader, LoRA, and attention bundles.
- Add a bundled interpolation loader so interpolation model identity also
  travels over one wire.

## 0.2.0

- Rename the native continuation input/output to `sequence` everywhere.
- Replace the ambiguous finishing token with `enhanced_sequence`.
- Make interpolation input/output type-identical so bypass preserves the
  enhancement chain and 24-fps LTX assembly works naturally.
- Add shared H3 hybrid/basic model loaders and an LTX upscale stack loader.
- Add cache-aware LoRA and attention patch nodes.
- Accept one shared core interpolation model across every shot.
- Preserve durable cache identity across model, LoRA, and attention changes.
- Retain direct raw, LTX, interpolation, and assembly previews.

## 0.1.0

- Initial staged H3 generation, LTX enhancement, interpolation, and assembly.
