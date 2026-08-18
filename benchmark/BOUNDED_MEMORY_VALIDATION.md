# Bounded-memory validation

H3 Relay 0.5 replaces monolithic long-sequence execution with disk-restored
partial prompts and exact chunked interpolation. Detailed raw telemetry for the
baseline remains under the locally ignored `benchmark/results/` directory.

| Test | Work | Runtime | Peak VRAM | Peak RAM | Result |
|---|---|---:|---:|---:|---|
| Four-shot baseline | Cold H3 + LTX + monolithic RIFE | 27m 29s | 23,037 MiB | 98.54 GiB | Success |
| Four-shot chunk quality | Cached H3/LTX + fresh chunked RIFE | 1m 59.8s | 3,764 MiB | 22.25 GiB | Bit-identical |
| Ten-shot scaling | Fresh H3 (1 step) + LTX + chunked RIFE | 6m 40.2s | 23,959 MiB | 22.40 GiB | Success |
| 35-frame overlap | Fresh two-shot full pipeline | 1m 14.5s | — | 22.79 GiB | Success |
| LTX-only bypass | Cached three-shot 24 fps assembly | 14.6s | — | — | Success |

The complete old and new four-shot outputs both have SHA-256:

`78c6dccfbd27b887839f5fad0823ec218bbe60888a770e56bfb0b35323c9f700`

The ten-shot result is 1664x960 at 48 fps with stereo 32 kHz AAC, 455 video
frames, and a 9.48-second duration. FFmpeg decoded it without errors; its raw,
LTX, and interpolated manifests each contain exactly ten ordered records.

The observed 23,959 MiB ten-shot VRAM peak comes from a single active H3 model
stage, not accumulation. The near-constant 22 GiB host-memory peak across two,
four, and ten-shot staged configurations demonstrates that memory now scales
with the active shot rather than total sequence length.
