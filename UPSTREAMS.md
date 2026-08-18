# Upstream snapshots

## ComfyUI MiniMax H3 packed layout

- Source: <https://github.com/Comfy-Org/ComfyUI>
- Reference commit: `1c6d8d45`
- License: GPL-3.0
- H3 Relay adaptation: the stock multi-step video/audio guide layout is paired
  with a guarded pre-target history anchor. On older ComfyUI versions the
  compatible guide engine and layout are installed in memory only; on future
  versions advertising native history support the fallback stands down.

## MiniMax H3 Context Loop

- Source: <https://github.com/ethanfel/ComfyUI-MiniMaxH3-Contex-Loop>
- Base commit: `4a982d68faa8`
- License: GPL-3.0
- Snapshot: the locally tested runtime at import time, including the sliding
  history, disk checkpoint, steerable sequence, LTX rolling context, and
  assembly changes used by the reference workflow.
- H3 Relay adaptation: upstream public node registration is disabled; the
  required runtime classes are registered under H3 Relay namespaced ids.

## Spectrum MiniMax H3

- Source: <https://github.com/xmarre/ComfyUI-Spectrum-MiniMax-H3>
- Commit: `59cb4649c298`
- License: GPL-3.0
- H3 Relay adaptation: the runtime snapshot is retained and only the Spectrum
  apply class required by the sampling profile is registered publicly.

## MiniMax H3 Hybrid Loader

- Source: <https://github.com/scottmudge/ComfyUI_MinimaxH3HybridLoader>
- Commit: `a44c69b02242`
- License: MIT
- H3 Relay adaptation: the loader class is registered under a namespaced
  internal node id and used by the staged H3 generation graph.

Future imports should be made as isolated commits, with the upstream commit
updated here and H3 Relay-specific changes reapplied explicitly.
