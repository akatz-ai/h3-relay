#!/usr/bin/env node

import crypto from "node:crypto";
import fs from "node:fs/promises";
import path from "node:path";

const ROOT = path.resolve(import.meta.dirname, "..");
const SPEC = path.join(
  ROOT,
  "examples/orbital-storm-spectrum16-58s-api.json",
);
const OUTPUT = path.join(
  ROOT,
  "example_workflows/H3-Relay-Orbital-Storm-Spectrum16-58s.json",
);
const api = JSON.parse(await fs.readFile(SPEC, "utf8"));
const sequenceSpec = api["1"].inputs;
const shots = ["10", "11", "12", "13"].map((id) => api[id].inputs);

const nodeProperties = (name) => ({ "Node name for S&R": name });

function widgetInput(name, type) {
  return { name, type, widget: { name }, link: null };
}

function socketInput(name, type, optional = false) {
  return {
    name,
    type,
    link: null,
    ...(optional ? { shape: 7 } : {}),
  };
}

function output(name, type) {
  return { name, type, links: [] };
}

function promptNode(id, title, value, pos, size = [600, 320]) {
  return {
    id,
    type: "PrimitiveStringMultiline",
    pos,
    size,
    flags: {},
    order: id,
    mode: 0,
    inputs: [widgetInput("value", "STRING")],
    outputs: [output("STRING", "STRING")],
    properties: nodeProperties("PrimitiveStringMultiline"),
    widgets_values: [value],
    title,
  };
}

function markdownNote(id, title, value, pos, size) {
  return {
    id,
    type: "MarkdownNote",
    pos,
    size,
    color: "#222",
    bgcolor: "#000",
    flags: {},
    order: id,
    mode: 0,
    inputs: [],
    outputs: [],
    properties: {},
    widgets_values: [value],
    title,
  };
}

function generateNode(id, index, shot, pos) {
  return {
    id,
    type: "H3RelayGenerateShot",
    pos,
    size: [580, 690],
    flags: {},
    order: id,
    mode: 0,
    inputs: [
      socketInput("h3_model", "H3_RELAY_MODEL"),
      socketInput("sequence", "H3_RELAY_SEQUENCE"),
      socketInput("prompt", "STRING"),
      widgetInput("seed", "INT"),
      widgetInput("duration_seconds", "FLOAT"),
      widgetInput("h3_steps", "INT"),
      widgetInput("output_crf", "INT"),
      widgetInput("ref_image_size", "COMBO"),
      widgetInput("shot_id", "STRING"),
      socketInput("first_frame", "IMAGE", true),
      socketInput("last_frame", "IMAGE", true),
      socketInput("reference_image_1", "IMAGE", true),
      socketInput("reference_image_2", "IMAGE", true),
      socketInput("reference_image_3", "IMAGE", true),
      socketInput("reference_video", "IMAGE", true),
      socketInput("reference_video_audio", "AUDIO", true),
      socketInput("reference_audio", "AUDIO", true),
    ],
    outputs: [
      output("sequence", "H3_RELAY_SEQUENCE"),
      output("video", "VIDEO"),
      output("video_path", "STRING"),
      output("status", "STRING"),
    ],
    properties: nodeProperties("H3RelayGenerateShot"),
    widgets_values: [
      shot.seed,
      "fixed",
      15.0,
      shot.h3_steps,
      shot.output_crf,
      shot.ref_image_size,
      shot.shot_id ?? "",
    ],
    title: index === 0
      ? "SHOT 1 · GENERATE / REVIEW / REROLL RAW H3"
      : `SHOT ${index + 1} · CONTINUE RAW H3 FROM SHOT ${index}`,
  };
}

function enhanceNode(id, index, pos) {
  return {
    id,
    type: "H3RelayEnhanceShot",
    pos,
    size: [530, 300],
    flags: {},
    order: id,
    mode: 0,
    inputs: [
      socketInput("ltx_model", "H3_RELAY_MODEL"),
      socketInput("sequence", "H3_RELAY_SEQUENCE"),
      socketInput("enhancement_prompt", "STRING"),
      widgetInput("output_crf", "INT"),
      widgetInput("context_window_frames", "INT"),
      widgetInput("context_overlap_frames", "INT"),
      widgetInput("vae_temporal_tile_frames", "INT"),
      widgetInput("vae_temporal_overlap_frames", "INT"),
      socketInput("previous_enhanced", "H3_RELAY_ENHANCED", true),
    ],
    outputs: [
      output("enhanced", "H3_RELAY_ENHANCED"),
      output("video", "VIDEO"),
      output("video_path", "STRING"),
      output("status", "STRING"),
    ],
    properties: nodeProperties("H3RelayEnhanceShot"),
    widgets_values: [18, 193, 64, 128, 16],
    title: `SHOT ${index + 1} · OPTIONAL LTX 2× ENHANCE`,
  };
}

function interpolateNode(id, index, pos) {
  return {
    id,
    type: "H3RelayInterpolateShot",
    pos,
    size: [530, 250],
    flags: {},
    order: id,
    mode: 0,
    inputs: [
      socketInput("interpolation", "H3_RELAY_INTERPOLATION"),
      socketInput("enhanced", "H3_RELAY_ENHANCED"),
      widgetInput("multiplier", "INT"),
      widgetInput("output_crf", "INT"),
      widgetInput("chunk_frames", "INT"),
    ],
    outputs: [
      output("enhanced", "H3_RELAY_ENHANCED"),
      output("video", "VIDEO"),
      output("video_path", "STRING"),
      output("status", "STRING"),
    ],
    properties: nodeProperties("H3RelayInterpolateShot"),
    widgets_values: [2, 18, 48],
    title: `SHOT ${index + 1} · OPTIONAL INTERPOLATE TO 48 FPS`,
  };
}

const columns = [-700, 40, 780, 1520];
const promptNodes = shots.map((shot, index) => promptNode(
  10 + index,
  `SHOT ${index + 1} PROMPT`,
  shot.prompt,
  [columns[index], -760],
));
const generateNodes = shots.map((shot, index) => generateNode(
  20 + index,
  index,
  shot,
  [columns[index], -350],
));
const enhanceNodes = shots.map((_, index) => enhanceNode(
  30 + index,
  index,
  [columns[index], 440],
));
const interpolateNodes = shots.map((_, index) => interpolateNode(
  40 + index,
  index,
  [columns[index], 830],
));

const nodes = [
  {
    id: 3,
    type: "H3RelayH3HybridModelLoader",
    pos: [-1480, -1120],
    size: [620, 420],
    flags: {},
    order: 3,
    mode: 0,
    inputs: [
      widgetInput("base_model", "COMBO"),
      widgetInput("overlay_model", "COMBO"),
      widgetInput("overlay_preset", "COMBO"),
      widgetInput("manual_cache_revision", "STRING"),
      widgetInput("block_range_start", "INT"),
      widgetInput("block_range_end", "INT"),
      widgetInput("final_adaln_from_overlay", "BOOLEAN"),
      widgetInput("custom_overlays", "STRING"),
      widgetInput("custom_base", "STRING"),
      widgetInput("weight_dtype", "COMBO"),
    ],
    outputs: [output("h3_model", "H3_RELAY_MODEL")],
    properties: nodeProperties("H3RelayH3HybridModelLoader"),
    widgets_values: [
      "minimax_h3_fl2va_int8_convrot.safetensors",
      "minimax_h3_ref2va_int8_convrot.safetensors",
      "block_range_adaln",
      "v1",
      25,
      49,
      false,
      "",
      "",
      "default",
    ],
    title: "H3 HYBRID · FL2VA BASE + REF2VA OVERLAY",
  },
  {
    id: 4,
    type: "H3RelayAttention",
    pos: [-780, -1040],
    size: [420, 180],
    flags: {},
    order: 4,
    mode: 0,
    inputs: [
      socketInput("model", "H3_RELAY_MODEL"),
      widgetInput("attention", "COMBO"),
    ],
    outputs: [output("model", "H3_RELAY_MODEL")],
    properties: nodeProperties("H3RelayAttention"),
    widgets_values: ["comfy kitchen attention"],
    title: "H3 ATTENTION · COMFY KITCHEN",
  },
  {
    id: 5,
    type: "H3RelayLTXModelLoader",
    pos: [-1480, 1080],
    size: [620, 520],
    flags: {},
    order: 5,
    mode: 0,
    inputs: [
      widgetInput("model_name", "COMBO"),
      widgetInput("vae_name", "COMBO"),
      widgetInput("latent_2x_model_name", "COMBO"),
      widgetInput("text_encoder_name", "COMBO"),
      widgetInput("distilled_lora", "COMBO"),
      widgetInput("distilled_strength", "FLOAT"),
      widgetInput("pixel_upscale_ic_lora", "COMBO"),
      widgetInput("pixel_upscale_ic_strength", "FLOAT"),
      widgetInput("weight_dtype", "COMBO"),
      widgetInput("manual_cache_revision", "STRING"),
    ],
    outputs: [output("ltx_model", "H3_RELAY_MODEL")],
    properties: nodeProperties("H3RelayLTXModelLoader"),
    widgets_values: [
      "ltx-2.5-22b-dev-transformer-comfy-int8-convrot.safetensors",
      "ltx-2.5-video-vae-bf16.safetensors",
      "ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors",
      "gemma4-12b-with-proj-ltx-2.5-comfy-int8-convrot.safetensors",
      "ltx-2.5-22b-distilled-lora-450-bf16.safetensors",
      1.0,
      "ltx-2.5-22b-ic-lora-pixel-spatial-upscaler-x2-1.0.safetensors",
      1.0,
      "default",
      "v1",
    ],
    title: "LTX 2.5 · MODEL / VAE / LATENT 2× / GEMMA4 / 2 LORAS",
  },
  {
    id: 6,
    type: "H3RelayAttention",
    pos: [-780, 1190],
    size: [420, 180],
    flags: {},
    order: 6,
    mode: 0,
    inputs: [
      socketInput("model", "H3_RELAY_MODEL"),
      widgetInput("attention", "COMBO"),
    ],
    outputs: [output("model", "H3_RELAY_MODEL")],
    properties: nodeProperties("H3RelayAttention"),
    widgets_values: ["pytorch attention"],
    title: "LTX ATTENTION · PYTORCH (SWAP OR BYPASS TO TEST)",
  },
  {
    id: 7,
    type: "H3RelayInterpolationModelLoader",
    pos: [2190, 1420],
    size: [540, 120],
    flags: {},
    order: 7,
    mode: 0,
    inputs: [widgetInput("model_name", "COMBO"), widgetInput("manual_cache_revision", "STRING")],
    outputs: [output("interpolation", "H3_RELAY_INTERPOLATION")],
    properties: nodeProperties("H3RelayInterpolationModelLoader"),
    widgets_values: ["rife_v4.26_heavy.safetensors", "v1"],
    title: "INTERPOLATION MODEL · CORE LOADER WRAPPER",
  },
  {
    id: 1,
    type: "H3RelaySequenceStart",
    pos: [-1480, -350],
    size: [620, 650],
    flags: {},
    order: 1,
    mode: 0,
    inputs: [
      widgetInput("run_name", "STRING"),
      widgetInput("global_prompt", "STRING"),
      widgetInput("width", "INT"),
      widgetInput("height", "INT"),
      widgetInput("h3_overlap_frames", "COMBO"),
      widgetInput("sampler", "COMBO"),
      widgetInput("scheduler", "COMBO"),
      widgetInput("spectrum_enabled", "BOOLEAN"),
    ],
    outputs: [
      output("sequence", "H3_RELAY_SEQUENCE"),
      output("status", "STRING"),
    ],
    properties: nodeProperties("H3RelaySequenceStart"),
    widgets_values: [
      "orbital_storm_h3_relay_spectrum16",
      sequenceSpec.global_prompt,
      sequenceSpec.width,
      sequenceSpec.height,
      sequenceSpec.h3_overlap_frames ?? 18,
      sequenceSpec.sampler,
      sequenceSpec.scheduler,
      sequenceSpec.spectrum_enabled,
    ],
    title: "H3 RELAY · GLOBAL RAW H3 SEQUENCE",
  },
  promptNode(
    2,
    "GLOBAL LTX ENHANCEMENT PROMPT",
    sequenceSpec.enhancement_prompt,
    [-1480, 440],
    [620, 520],
  ),
  ...promptNodes,
  ...generateNodes,
  ...enhanceNodes,
  ...interpolateNodes,
  {
    id: 50,
    type: "H3RelayAssemble",
    pos: [2240, 830],
    size: [540, 300],
    flags: {},
    order: 50,
    mode: 0,
    inputs: [
      socketInput("enhanced", "H3_RELAY_ENHANCED"),
      widgetInput("output_stage", "COMBO"),
      widgetInput("filename", "STRING"),
      widgetInput("audio_bitrate", "INT"),
    ],
    outputs: [
      output("video", "VIDEO"),
      output("video_path", "STRING"),
      output("status", "STRING"),
    ],
    properties: nodeProperties("H3RelayAssemble"),
    widgets_values: [
      "interpolated",
      "orbital_storm_h3_relay_spectrum16",
      256,
    ],
    title: "ASSEMBLE FOUR ACCEPTED 2× / 48 FPS SHOTS",
  },
  markdownNote(
    60,
    "Note: H3 Relay",
    `## H3 Relay

H3 Relay is a staged MiniMax H3 long-form video workflow for generating, reviewing, continuing, enhancing, interpolating, and assembling shots without repeating accepted work. Native H3 video and stereo audio remain the source of continuity; LTX 2.5 and RIFE are optional finishing stages.

## Project links

- [MiniMax H3](https://www.minimax.io/blog/minimax-h3)
- [🤗 Comfy-Org/MiniMax-H3](https://huggingface.co/Comfy-Org/MiniMax-H3)
- [LTX 2.5](https://huggingface.co/Lightricks/LTX-2.5)

## About this workflow

**Generation stages**

1. Generate and review each native 832x480/24fps H3 shot with synchronized audio.
2. Continue the next shot from the accepted H3 video/audio checkpoint using configurable sliding history (18 frames by default).
3. Optionally run the learned LTX latent 2x expansion and pixel-spatial IC-LoRA refinement.
4. Optionally run RIFE to convert 24fps LTX output to 48fps.
5. Assemble the selected finishing stage with one final AAC encode.

**Key inputs**

- **global_prompt**: series-wide style, character, environment, voice, and continuity rules
- **prompt**: shot-specific action, camera, timing, dialogue, SFX, and music direction
- **duration_seconds**: rounded upward to H3's valid 5+17k frame grid
- **h3_overlap_frames**: sequence-wide H3 visual/audio history; choose 18, 35, 52, or 69 frames
- **output_crf**: per-stage H.264 cache/assembly quality; it is not a model parameter
- **sequence**: native H3 continuation state passed between accepted shots
- **enhanced**: LTX/interpolation state; bypass Interpolate nodes for 24fps assembly

**Sampling controls**

- **sampler**: every native option exposed by ComfyUI KSamplerSelect
- **scheduler**: beta57 plus every native option exposed by ComfyUI BasicScheduler
- **spectrum_enabled**: independently enables or disables Spectrum forecasting; unsupported samplers still run with Spectrum bypassed
- **default**: Euler + beta57 + Spectrum enabled

Sampler and scheduler can be mixed freely. For example, Euler or res_multistep with the simple scheduler runs without changing any model inputs.

H3, LTX, and interpolation videos are managed cache artifacts, not published outputs. Connect a native Save Video node to publish an individual stage, or run Assemble to publish the final movie. Cache Manager can inspect usage and prune superseded rerolls.

Unchanged H3, LTX, interpolation, and assembly stages are restored from durable disk caches across queues and restarts.`,
    [-2080, 1840],
    [500, 620],
  ),
  markdownNote(
    61,
    "Note: Model Links",
    `## Model Links

**vae**

- [minimax_h3_video_vae_fp16.safetensors](https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main/vae/minimax_h3_video_vae_fp16.safetensors)
- [minimax_h3_audio_vae_fp32.safetensors](https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main/vae/minimax_h3_audio_vae_fp32.safetensors)
- [ltx-2.5-video-vae-bf16.safetensors](https://huggingface.co/Lightricks/LTX-2.5/resolve/main/vae/ltx-2.5-video-vae-bf16.safetensors)

**diffusion_models**

- [minimax_h3_fl2va_int8_convrot.safetensors](https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main/diffusion_models/minimax_h3_fl2va_int8_convrot.safetensors)
- [minimax_h3_ref2va_int8_convrot.safetensors](https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main/diffusion_models/minimax_h3_ref2va_int8_convrot.safetensors)
- [ltx-2.5-22b-dev-transformer-comfy-int8-convrot.safetensors](https://huggingface.co/Lightricks/LTX-2.5/resolve/main/diffusion_models/ltx-2.5-22b-dev-transformer-comfy-int8-convrot.safetensors)

**text_encoders**

- [qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors](https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main/text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors)
- [gemma4-12b-with-proj-ltx-2.5-comfy-int8-convrot.safetensors](https://huggingface.co/Lightricks/LTX-2.5/resolve/main/text_encoders/gemma4-12b-with-proj-ltx-2.5-comfy-int8-convrot.safetensors)

**latent_upscale_models**

- [ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors](https://huggingface.co/Lightricks/LTX-2.5/resolve/main/latent_upscale_models/ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors)

**loras**

- [ltx-2.5-22b-distilled-lora-450-bf16.safetensors](https://huggingface.co/Lightricks/LTX-2.5/resolve/main/loras/ltx-2.5-22b-distilled-lora-450-bf16.safetensors)
- [ltx-2.5-22b-ic-lora-pixel-spatial-upscaler-x2-1.0.safetensors](https://huggingface.co/Lightricks/LTX-2.5-22b-IC-LoRA-Pixel-Spatial-Upscaler/resolve/main/ltx-2.5-22b-ic-lora-pixel-spatial-upscaler-x2-1.0.safetensors)

**frame_interpolation**

- [rife_v4.26_heavy.safetensors](https://huggingface.co/Comfy-Org/frame_interpolation/resolve/main/frame_interpolation/rife_v4.26_heavy.safetensors)

## Model Storage Location

\`\`\`
📂 ComfyUI/
├── 📂 models/
│   ├── 📂 vae/
│   │   ├── minimax_h3_video_vae_fp16.safetensors
│   │   ├── minimax_h3_audio_vae_fp32.safetensors
│   │   └── ltx-2.5-video-vae-bf16.safetensors
│   ├── 📂 diffusion_models/
│   │   ├── minimax_h3_fl2va_int8_convrot.safetensors
│   │   ├── minimax_h3_ref2va_int8_convrot.safetensors
│   │   └── ltx-2.5-22b-dev-transformer-comfy-int8-convrot.safetensors
│   ├── 📂 text_encoders/
│   │   ├── qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors
│   │   └── gemma4-12b-with-proj-ltx-2.5-comfy-int8-convrot.safetensors
│   ├── 📂 latent_upscale_models/
│   │   └── ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors
│   ├── 📂 loras/
│   │   ├── ltx-2.5-22b-distilled-lora-450-bf16.safetensors
│   │   └── ltx-2.5-22b-ic-lora-pixel-spatial-upscaler-x2-1.0.safetensors
│   └── 📂 frame_interpolation/
│       └── rife_v4.26_heavy.safetensors
\`\`\`

Approximate storage for this profile: **129 GiB**.

H3 Relay's GPL-3.0 code license does not grant rights to these separately licensed model weights. Review and accept every upstream model license and obtain any required authorization before downloading or use.

## Report Issue

Note: Please update ComfyUI first ([guide](https://docs.comfy.org/installation/update_comfyui)) and prepare required models.

- Cannot run / runtime errors: [ComfyUI/issues](https://github.com/comfyanonymous/ComfyUI/issues)
- UI / frontend issues: [ComfyUI_frontend/issues](https://github.com/Comfy-Org/ComfyUI_frontend/issues)
- Workflow issues: [workflow_templates/issues](https://github.com/Comfy-Org/workflow_templates/issues)`,
    [-2650, 1840],
    [540, 1040],
  ),
  markdownNote(
    77,
    "Note: Size Settings Reference",
    `| megapixels | Aspect | Output (multiple=32) |
|---|---|---|
| 0.2 | 16:9 | 608 x 352 |
| 0.3 | 16:9 | 736 x 416 |
| 0.4 | 16:9 | 864 x 480 |
| 0.5 | 16:9 | 960 x 544 |
| 0.6 | 16:9 | 1056 x 608 |
| 0.7 | 16:9 | 1152 x 640 |
| 0.8 | 16:9 | 1216 x 672 |
| 0.9 | 16:9 | 1280 x 736 |
| 0.98 | 16:9 | 1344 x 768 |
| 1.0 | 16:9 | 1376 x 768 |
| 1.2 | 16:9 | 1504 x 832 |
| 1.5 | 16:9 | 1664 x 928 |
| 1.8 | 16:9 | 1824 x 1024 |
| 2.0 | 16:9 | 1920 x 1088 |`,
    [-1540, 1840],
    [300, 520],
  ),
  {
    id: 78,
    type: "H3RelayCacheManager",
    pos: [-1540, 2420],
    size: [420, 170],
    flags: {},
    order: 78,
    mode: 0,
    inputs: [
      widgetInput("action", "COMBO"),
      widgetInput("keep_revisions_per_shot", "INT"),
      widgetInput("budget_gb", "FLOAT"),
    ],
    outputs: [
      output("cache_path", "STRING"),
      output("status", "STRING"),
    ],
    properties: nodeProperties("H3RelayCacheManager"),
    widgets_values: ["inspect", 2, 100.0],
    title: "H3 RELAY · CACHE MANAGER",
  },
];

const links = [];
let nextLink = 1;

function connect(originId, originSlot, targetId, targetSlot, type) {
  const id = nextLink++;
  links.push([id, originId, originSlot, targetId, targetSlot, type]);
  const origin = nodes.find((node) => node.id === originId);
  const target = nodes.find((node) => node.id === targetId);
  origin.outputs[originSlot].links.push(id);
  target.inputs[targetSlot].link = id;
}

connect(3, 0, 4, 0, "H3_RELAY_MODEL");
connect(5, 0, 6, 0, "H3_RELAY_MODEL");
connect(1, 0, 20, 1, "H3_RELAY_SEQUENCE");
for (let index = 0; index < 4; index += 1) {
  connect(4, 0, 20 + index, 0, "H3_RELAY_MODEL");
  connect(10 + index, 0, 20 + index, 2, "STRING");
  if (index > 0) connect(19 + index, 0, 20 + index, 1, "H3_RELAY_SEQUENCE");
  connect(6, 0, 30 + index, 0, "H3_RELAY_MODEL");
  connect(20 + index, 0, 30 + index, 1, "H3_RELAY_SEQUENCE");
  connect(2, 0, 30 + index, 2, "STRING");
  if (index > 0) connect(39 + index, 0, 30 + index, 8, "H3_RELAY_ENHANCED");
  connect(7, 0, 40 + index, 0, "H3_RELAY_INTERPOLATION");
  connect(30 + index, 0, 40 + index, 1, "H3_RELAY_ENHANCED");
}
connect(43, 0, 50, 0, "H3_RELAY_ENHANCED");

const workflow = {
  id: crypto.randomUUID(),
  revision: 0,
  last_node_id: 78,
  last_link_id: nextLink - 1,
  version: 0.4,
  config: {},
  nodes,
  links,
  groups: [
    { id: 1, title: "H3 MODEL + GLOBAL SETTINGS", bounding: [-1530, -1170, 1230, 2150], color: "#315b7d", flags: {} },
    { id: 2, title: "RAW H3 · REVIEW BEFORE FINISHING", bounding: [-750, -820, 2860, 1120], color: "#78503b", flags: {} },
    { id: 3, title: "LTX MODEL STACK", bounding: [-1530, 1030, 1230, 620], color: "#51366f", flags: {} },
    { id: 4, title: "OPTIONAL LTX 2× ENHANCEMENT SEQUENCE", bounding: [-750, 390, 2860, 380], color: "#6d4ca3", flags: {} },
    { id: 5, title: "OPTIONAL CORE INTERPOLATION SEQUENCE", bounding: [-750, 780, 2860, 410], color: "#376d63", flags: {} },
    { id: 6, title: "FINAL ASSEMBLY", bounding: [2190, 780, 640, 800], color: "#2f6b3f", flags: {} },
  ],
  definitions: { subgraphs: [] },
  extra: {
    frontendVersion: "1.49.6",
    ds: { scale: 0.38, offset: [1570, 670] },
    h3_relay: {
      version: 4,
      source_runtime_seconds: 1566.69,
      h3_context_frames: 18,
      ltx_context_frames: 17,
      ltx_context_steps: 3,
      interpolation_multiplier: 2,
      assembly: "frame_exact_direct_join",
    },
  },
};

await fs.mkdir(path.dirname(OUTPUT), { recursive: true });
await fs.writeFile(OUTPUT, `${JSON.stringify(workflow, null, 2)}\n`);
process.stdout.write(`${OUTPUT}\n`);
