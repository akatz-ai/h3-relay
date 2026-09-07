# Experimental FastH3 ComfyGit environment

This separate recipe pins the VSA-capable ComfyUI model integration, official
`comfy-kitchen==0.2.33` PyPI wheel, Relay-owned attention adapter, MMH3 Ultimate,
and five content-identified model files with immutable public download URLs.
It installs all three FastH3 example workflows. It does not install the
temporary `SolAttnMiniMax` node or a private kernel wheel.

The recipe pins Relay commit `33ca879b8697605bedf6ed6875465239dfc46312`.
Prepublication validation used local Git transport for that commit; normal
installation uses the HTTPS repository declared in the manifest. See
[`../VALIDATION.md`](../VALIDATION.md) for the tested Linux scope, upstream
kernel caveats, and unvalidated Windows status.

## Requirements

- Python 3.11 and current ComfyGit with pinned alternate-ComfyUI repository
  materialization support.
- NVIDIA/CUDA 13.0 runtime; this pass is validated on RTX 4090 Linux only.
- Roughly 43 GiB of model bytes plus Python/runtime/cache/output space.
- The VSA ComfyUI commit in this manifest is required. The wheel alone does
  not make an ordinary stock ComfyUI release FastH3-compatible.

From the Relay repository root:

```bash
cg materialize ./comfygit_fast_h3_environment \
  --name h3-relay-fast-h3 \
  --workspace /path/to/comfygit-workspace \
  --models-dir /path/to/comfyui-models \
  --torch-backend cu130 \
  --models all
```

Model reuse is content-checked by ComfyGit. `comfy-kitchen==0.2.33` is specified
both as a dependency and a UV override because the pinned experimental
ComfyUI requirements still name the older kitchen release.
ComfyGit owns resolution of the platform-specific PyTorch trio from the
selected backend; their resolved versions must be recorded with each run.

Before running the three-shot example, copy the five reference images from
the installed node's `example_workflows/assets/` into that environment's
`ComfyUI/input/`. Directory materialization installs workflows, not arbitrary
input media. The two one-shot examples need no reference files.

Optional RIFE is bypassed in the three-shot example. Its model is intentionally
not one of this recipe's five required models; install the documented RIFE
weight separately before enabling interpolation. The validated output is 24 fps.

The existing `../comfygit_environment/` remains the older standard-H3/LTX/RIFE
reference environment and is not a FastH3 installation recipe.
