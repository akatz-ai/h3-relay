# H3 Relay ComfyGit environment

This directory is a portable ComfyGit source for the full H3 Relay reference
workflow. It pins:

- ComfyUI commit `1c6d8d45b3693bfbb32385b410d813a7fd6be216`
- H3 Relay commit `50cfe4ce38726d1590247bcc49ea8edf3bbd6081`
- Python 3.11
- all 12 required MiniMax H3, LTX 2.5, and RIFE model files by path, size,
  content hash, and official source URL

## Prerequisites

- A current ComfyGit checkout containing the H3 Relay materialization fixes.
- About 129 GiB for model weights, plus environment, cache, and output space.
- Hugging Face access to the gated `Lightricks/LTX-2.5` and
  `Lightricks/LTX-2.5-22b-IC-LoRA-Pixel-Spatial-Upscaler` repositories.
- A configured Hugging Face credential (`cg auth set huggingface`).
- A working NVIDIA compute stack. The reference workflow's `comfy kitchen
  attention` mode currently requires a driver and PyTorch build capable of
  CUDA 13.0. On older CUDA 12.8 systems, selecting `pytorch attention` works
  as a substantially slower compatibility mode.

After rebinding a GPU from VFIO to the NVIDIA driver on Linux, ensure UVM is
loaded before launching ComfyUI:

```bash
sudo modprobe nvidia_uvm
```

## Materialize

From the H3 Relay repository root:

```bash
cg materialize ./comfygit_environment \
  --name h3-relay \
  --workspace /path/to/comfygit-workspace \
  --models-dir /path/to/comfyui-models \
  --models all \
  --use
```

ComfyGit auto-selects the local PyTorch backend. Existing weights at the exact
declared paths are reused only when their hashes match. Missing weights are
downloaded from the declared sources, and materialization fails if a required
download, node install, environment sync, or content-hash check fails.

Run the environment with:

```bash
cg -e h3-relay run
```

The saved workflow is installed as
`H3-Relay-Orbital-Storm-Spectrum16-58s`.
