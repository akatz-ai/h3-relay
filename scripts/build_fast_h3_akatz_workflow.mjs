#!/usr/bin/env node

import fs from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const output = path.join(
  root,
  "example_workflows/H3-Relay-FastH3-Akatz-3x10-Ultimate.json",
);
const simpleOutput = path.join(
  root,
  "example_workflows/H3-Relay-FastH3-VSA-One-Shot.json",
);
const ultimateOutput = path.join(
  root,
  "example_workflows/H3-Relay-FastH3-VSA-One-Shot-Ultimate-2x.json",
);

const nodes = [];
const links = [];
let nextNodeId = 1;
let nextLinkId = 1;

const coreProps = (name) => ({
  cnr_id: "comfy-core",
  ver: "0.34.0",
  "Node name for S&R": name,
});
const relayProps = (name) => ({
  cnr_id: "h3-relay",
  ver: "1.0.2",
  "Node name for S&R": name,
});
const widget = (name, type, extra = {}) => ({
  name,
  type,
  link: null,
  widget: { name },
  ...extra,
});
const socket = (name, type, extra = {}) => ({ name, type, link: null, ...extra });
const outputSocket = (name, type) => ({ name, type, links: null });

function addNode({
  type,
  title,
  pos,
  size,
  inputs = [],
  outputs = [],
  widgets = [],
  named = {},
  properties,
  mode = 0,
  color,
  bgcolor,
}) {
  const node = {
    id: nextNodeId++,
    type,
    pos,
    size,
    flags: {},
    order: nodes.length,
    mode,
    inputs,
    outputs,
    properties: properties || relayProps(type),
    widgets_values: widgets,
    widgets_values_named: named,
    title,
  };
  if (color) node.color = color;
  if (bgcolor) node.bgcolor = bgcolor;
  nodes.push(node);
  return node;
}

function connect(origin, originName, target, targetName, type) {
  const originSlot = origin.outputs.findIndex((item) => item.name === originName);
  const targetSlot = target.inputs.findIndex((item) => item.name === targetName);
  if (originSlot < 0 || targetSlot < 0) {
    throw new Error(`invalid link ${origin.title}:${originName} -> ${target.title}:${targetName}`);
  }
  const id = nextLinkId++;
  links.push([id, origin.id, originSlot, target.id, targetSlot, type]);
  const outputLinks = origin.outputs[originSlot].links;
  origin.outputs[originSlot].links = Array.isArray(outputLinks)
    ? [...outputLinks, id]
    : [id];
  target.inputs[targetSlot].link = id;
}

function note(title, text, pos, size) {
  return addNode({
    type: "MarkdownNote",
    title,
    pos,
    size,
    widgets: [text],
    named: { value: text },
    properties: {},
    color: "#222",
    bgcolor: "#000",
  });
}

function textNode(title, value, pos, size = [640, 360]) {
  return addNode({
    type: "PrimitiveStringMultiline",
    title,
    pos,
    size,
    inputs: [widget("value", "STRING")],
    outputs: [outputSocket("STRING", "STRING")],
    widgets: [value],
    named: { value },
    properties: coreProps("PrimitiveStringMultiline"),
  });
}

function loadImage(title, filename, pos) {
  return addNode({
    type: "LoadImage",
    title,
    pos,
    size: [300, 330],
    outputs: [outputSocket("IMAGE", "IMAGE"), outputSocket("MASK", "MASK")],
    widgets: [filename, "image"],
    named: { image: filename, upload: "image" },
    properties: coreProps("LoadImage"),
  });
}

function generateNode(index, pos, seed) {
  return addNode({
    type: "H3RelayGenerateShot",
    title: index === 1
      ? "SHOT 1 · GENERATE / REVIEW / REROLL RAW FASTH3"
      : `SHOT ${index} · CONTINUE RAW FASTH3 FROM SHOT ${index - 1}`,
    pos,
    size: [620, 760],
    inputs: [
      socket("h3_model", "H3_RELAY_MODEL"),
      socket("sequence", "H3_RELAY_SEQUENCE"),
      socket("prompt", "STRING"),
      socket("first_frame", "IMAGE", { shape: 7 }),
      socket("last_frame", "IMAGE", { shape: 7 }),
      socket("reference_image_1", "IMAGE", { shape: 7 }),
      socket("reference_image_2", "IMAGE", { shape: 7 }),
      socket("reference_image_3", "IMAGE", { shape: 7 }),
      socket("additional_reference_images.reference_image_4", "IMAGE", {
        shape: 7,
        label: "reference_image_4",
      }),
      socket("additional_reference_images.reference_image_5", "IMAGE", {
        shape: 7,
        label: "reference_image_5",
      }),
      socket("reference_video", "IMAGE", { shape: 7 }),
      socket("reference_video_audio", "AUDIO", { shape: 7 }),
      socket("reference_audio", "AUDIO", { shape: 7 }),
      widget("seed", "INT"),
      widget("duration_seconds", "FLOAT"),
      widget("h3_steps", "INT"),
      widget("output_crf", "INT"),
      widget("ref_image_size", "COMBO"),
      widget("shot_id", "STRING", { shape: 7 }),
    ],
    outputs: [
      outputSocket("sequence", "H3_RELAY_SEQUENCE"),
      outputSocket("video", "VIDEO"),
      outputSocket("video_path", "STRING"),
      outputSocket("status", "STRING"),
    ],
    widgets: [seed, "fixed", 10.0, 4, 18, "match", `akatz_chase_${index}`],
    named: {
      seed,
      control_after_generate: "fixed",
      duration_seconds: 10.0,
      h3_steps: 4,
      output_crf: 18,
      ref_image_size: "match",
      shot_id: `akatz_chase_${index}`,
    },
  });
}

function ultimateNode(index, pos, seed) {
  return addNode({
    type: "H3RelayUltimateEnhanceShot",
    title: `SHOT ${index} · OPTIONAL H3 ULTIMATE 2× ENHANCE`,
    pos,
    size: [640, 700],
    inputs: [
      socket("h3_model", "H3_RELAY_MODEL"),
      socket("sequence", "H3_RELAY_SEQUENCE"),
      socket("enhancement_prompt", "STRING"),
      socket("previous_enhanced", "H3_RELAY_ENHANCED", { shape: 7 }),
      socket("reference_image_1", "IMAGE", { shape: 7 }),
      socket("reference_image_2", "IMAGE", { shape: 7 }),
      socket("reference_image_3", "IMAGE", { shape: 7 }),
      socket("additional_reference_images.reference_image_4", "IMAGE", {
        shape: 7,
        label: "reference_image_4",
      }),
      socket("additional_reference_images.reference_image_5", "IMAGE", {
        shape: 7,
        label: "reference_image_5",
      }),
      widget("refinement_seed", "INT"),
      widget("output_crf", "INT"),
      widget("ref_image_size", "COMBO"),
      widget("temporal_chunk_frames", "INT"),
      widget("temporal_overlap_frames", "INT"),
      widget("anchor_strength", "FLOAT"),
      widget("tile_width", "INT"),
      widget("tile_height", "INT"),
      widget("spatial_overlap", "INT"),
    ],
    outputs: [
      outputSocket("enhanced", "H3_RELAY_ENHANCED"),
      outputSocket("video", "VIDEO"),
      outputSocket("video_path", "STRING"),
      outputSocket("status", "STRING"),
    ],
    widgets: [seed, "fixed", 18, "match", 136, 17, 0.999, 1024, 1024, 128],
    named: {
      refinement_seed: seed,
      control_after_generate: "fixed",
      output_crf: 18,
      ref_image_size: "match",
      temporal_chunk_frames: 136,
      temporal_overlap_frames: 17,
      anchor_strength: 0.999,
      tile_width: 1024,
      tile_height: 1024,
      spatial_overlap: 128,
    },
  });
}

function interpolateNode(index, pos) {
  return addNode({
    type: "H3RelayInterpolateShot",
    title: `SHOT ${index} · OPTIONAL RIFE 24 → 48 FPS (BYPASSED)`,
    pos,
    size: [640, 430],
    mode: 4,
    inputs: [
      socket("interpolation", "H3_RELAY_INTERPOLATION"),
      socket("enhanced", "H3_RELAY_ENHANCED"),
      widget("multiplier", "INT"),
      widget("output_crf", "INT"),
      widget("chunk_frames", "INT"),
    ],
    outputs: [
      outputSocket("enhanced", "H3_RELAY_ENHANCED"),
      outputSocket("video", "VIDEO"),
      outputSocket("video_path", "STRING"),
      outputSocket("status", "STRING"),
    ],
    widgets: [2, 18, 48],
    named: { multiplier: 2, output_crf: 18, chunk_frames: 48 },
  });
}

const globalPrompt = `subject_definitions:
<Subject 1> is Akatz, the young female cyberpunk katana fighter defined by <Picture 1>, <Picture 2>, and <Picture 3>: preserve the same pale face, amber-orange eyes, long flowing blue-black hair with heavy bangs, slim athletic proportions, glossy dark-blue leather jacket, fitted black clothing, orange harness straps, fingerless gloves, utility holsters, buckled black combat boots, and black-sheathed katana with a cyan-to-magenta reflective edge.
<Subject 2> is the rain-soaked neon alley and street defined by <Picture 4>: dark towers, graffiti-covered walls, cyan and magenta signs, parked futuristic cars, steam vents, and deep puddle reflections.
<Subject 3> is the steep cyberpunk stairwell and connected service architecture defined by <Picture 5>: layered landings, black railings, cyan directional markings, dense vertical signs, exposed rock and industrial structure, and deep descending perspective.

continuity_rules:
Create one continuous premium live-action anime action film. Preserve <Subject 1>'s exact face, amber eyes, hair, bangs, body proportions, jacket, black clothing, orange straps, gloves, holsters, boots, and katana. Preserve travel direction, rain, illumination, weapon state, guard design, drone design, and geography across incoming H3 history. Every new segment begins from the exact incoming visual and audio moment without a reset, recap, fade, flash, title, or establishing cut. Keep action fast but physically readable. No blood, gore, dismemberment, duplicated protagonist, costume changes, or unexplained teleportation.

visual_style:
Cinematic semi-realistic cyberpunk anime rendered like a high-budget live-action film: crisp facial anatomy, deep blacks, cobalt-blue shadows, magenta and crimson highlights, controlled bloom, wet reflective materials, energetic camera movement, coherent motion blur, and readable choreography.

audio_rules:
Continuous heavy rain and neon-city ambience carry across every segment. Katana rings, laser passes, boots, fabric, impacts, drone rotors, doors, and room tone synchronize precisely with visible action. No dialogue unless explicitly written in the shot prompt.`;

const shotPrompts = [
  `summary:
Akatz escapes armored guards through the neon alley, deflects laser fire, defeats two pursuers without gore, destroys a drone, and keeps running around the corner toward the stairwell.

detailed_description:
[Shot 1] Continue from a low rear three-quarter tracking view inside <Subject 2>. <Subject 1> sprints through heavy rain while four faceless black-armored guards pursue twenty meters behind and fire narrow red laser pulses. Her boots splash through puddles, her blue-black hair and orange straps stream naturally, and the camera tracks at waist height without losing her face in profile.
[Shot 2] At 00:03.000, the camera whip-pans to a frontal medium-wide view as <Subject 1> pivots while still moving, draws the katana, deflects one laser pulse into a wall showering blue-white sparks, ducks beneath a second pulse, and uses the flat and hilt to knock two guards off balance. They fall safely into puddles; no blood or gore.
[Shot 3] At 00:06.500, she turns the alley corner. A compact black pursuit drone dives from above. The camera arcs beside her as she makes one clean rising cut through the drone, its two sparking halves tumble behind her, and she accelerates toward the cyan-lit entrance of <Subject 3>. End with her left boot crossing the stairwell threshold at full running speed while the remaining guards round the wet corner behind her.

overall_soundscape:
Continuous heavy rain, neon transformer hum, distant traffic, running boot splashes, three sharp laser passes, katana draw and metallic ring, armor impacts, drone rotors, one synchronized blade strike, falling metal, and fast breathing.

non_diegetic_music:
An original dark synthwave chase pulse at approximately 150 BPM begins with tight electronic drums and low distorted bass; it continues without resolving at the segment boundary.`,
  `summary:
Continue the exact chase through the cyberpunk stairwell as Akatz uses acrobatics to evade guards and destroy more drones, then reaches a narrow service passage and races toward a half-open garage door.

detailed_description:
[Shot 1] Continue from the exact incoming frame and sound as <Subject 1>'s left boot lands inside <Subject 3>. The camera follows over her shoulder down the first cyan-marked flight. Red laser pulses strike railings behind her while the same surviving guards enter at the upper landing. She never reverses direction.
[Shot 2] At 00:02.500, use one continuous descending crane move as <Subject 1> plants a boot on the railing, vaults across the stairwell void, catches the opposite rail with one gloved hand, swings beneath two laser pulses, and lands on the lower flight. Two black drones descend beside the vertical signs. She cuts the first drone during the landing and kicks the second into an empty wall, producing sparks but no explosion.
[Shot 3] At 00:06.000, the camera runs backward at chest height while she exits the stairwell into a narrow blue-lit service passage. One guard blocks the route with an electrified baton. She parries once, rotates under his arm, sweeps him safely to the floor with the katana sheath, and continues running. A segmented metal garage door ahead is already descending, leaving a narrowing gap at floor level. End as she lowers her center of gravity and commits to the slide, still several meters from the door.

overall_soundscape:
The same rain becomes muffled inside the stairwell, with rapid footsteps, rail vibrations, laser ricochets, clothing movement, drone rotors, blade rings, electrical crackle, one armored fall, service-vent hum, and the garage motor growing louder.

non_diegetic_music:
The same synthwave chase cue adds syncopated metallic percussion and rises toward the closing door without a cadence.`,
  `summary:
Continue the exact slide beneath the closing garage door, reach safety in a quiet cyberpunk cafe-bar, sheath the katana, sit down, and calmly order a cold ginger soda.

detailed_description:
[Shot 1] Continue from the exact incoming frame and audio. <Subject 1> drops into a controlled feet-first slide across the wet service floor as the segmented garage door closes. The low side-tracking camera stays with her. She passes beneath the door with only centimeters of clearance; her hair, orange straps, katana sheath, hands, and boots clear naturally. The door strikes the floor behind her with a heavy final clang, cutting off the guards and laser fire.
[Shot 2] At 00:03.000, she rises in one fluid motion inside a small warm cyberpunk cafe-bar connected to the service bay. Amber pendant lamps and quiet cyan menu panels reflect on dark wood and brushed steel. The camera settles into a steady medium profile as she exhales, checks the closed door, wipes rain from her cheek, and slides the katana fully into its black sheath with one precise click.
[Shot 3] At 00:06.500, <Subject 1> walks to the nearest empty stool, sits facing the bartender and camera in a clear three-quarter medium close-up, places both relaxed gloved hands on the counter, and says with a tired controlled voice: <d>[English] Ginger soda. Lots of ice.</d> Her face remains visible while speaking. She gives one small relieved breath as a cold glass is set before her. Hold on her exact identity, outfit, sheathed katana, and the quiet bar; no fade.

overall_soundscape:
Garage motor, sliding fabric, one heavy door clang, then sharply reduced chase noise; warm ventilation, quiet refrigerator hum, distant rain on metal, soft boot steps, katana sheath click, a single spoken line, glass on wood, fizz, and ice settling.

non_diegetic_music:
The same synthwave cue loses its drums after the door closes and resolves into a soft low sustained chord beneath the final drink order.`,
];

const refinementDirective = `Preserve the accepted raw shot's exact subjects, action, frame timing, composition, camera motion, geometry, lighting, color palette, audio timing, and scene identity. Use the same five connected references to restore stable facial features, costume details, weapon geometry, environment texture, and clean wet-surface reflections at exactly 2x spatial resolution. Do not redesign the character or environment; do not add cuts, objects, people, text, motion, or reframing.`;

const simpleGlobalPrompt = `visual_style:
Premium cinematic cyberpunk anime rendered like a high-budget live-action film, with crisp anatomy, deep cobalt shadows, restrained magenta and cyan highlights, controlled bloom, coherent motion blur, and physically readable movement.

continuity_rules:
Maintain one coherent subject, wardrobe, environment, direction of travel, illumination, camera language, and soundscape for the complete generation. No text, titles, fades, duplicated subjects, or unexplained cuts.`;

const simpleShotPrompt = `summary:
A lone cyberpunk courier races across a rain-slick rooftop, vaults a ventilation duct, turns to deflect one red energy pulse with a short luminous blade, and escapes through a maintenance doorway.

detailed_description:
[Shot 1] A low side-tracking medium-wide shot follows a lone courier sprinting across a rain-soaked rooftop at night. Dark towers and restrained cyan-magenta signs reflect in shallow puddles. Their coat and hair react naturally to speed and wind while the camera preserves a readable silhouette.
[Shot 2] At 00:02.000, the courier plants one hand on a waist-high ventilation duct, vaults it cleanly, lands without losing momentum, and glances over one shoulder as a narrow red energy pulse approaches.
[Shot 3] At 00:03.500, the camera arcs to a frontal three-quarter view. The courier draws a short cyan blade, deflects the pulse into an empty metal panel with synchronized sparks, then turns and disappears through a maintenance doorway. Hold briefly on the closing door and settling rain; no fade.

overall_soundscape:
Continuous rooftop rain, distant traffic, boot splashes, fabric movement, one energy pulse, a bright blade ring, synchronized metal sparks, a pneumatic doorway, and rooftop wind.

non_diegetic_music:
An original restrained dark synth pulse supports the action and resolves beneath the closing doorway.`;

const sizeSettingsReference = `| megapixels | Aspect | Native H3 (multiple=32) | H3 Ultimate 2x |
|---|---|---|---|
| 0.2 | 16:9 | 608 x 352 | 1216 x 704 |
| 0.3 | 16:9 | 736 x 416 | 1472 x 832 |
| 0.4 | 16:9 | 864 x 480 | 1728 x 960 |
| 0.5 | 16:9 | 960 x 544 | 1920 x 1088 |
| 0.6 | 16:9 | 1056 x 608 | 2112 x 1216 |
| 0.7 | 16:9 | 1152 x 640 | 2304 x 1280 |
| 0.8 | 16:9 | 1216 x 672 | 2432 x 1344 |
| 0.9 | 16:9 | 1280 x 736 | 2560 x 1472 |
| 0.98 | 16:9 | 1344 x 768 | 2688 x 1536 |

The bundled examples use **832 x 480**, which becomes **1664 x 960** after Ultimate. Sequence width and height must remain multiples of 32. Ultimate derives the exact 2x canvas and safely clamps tile maxima to it.`;

function modelLinksText({ includeUltimate = false, includeRife = false } = {}) {
  const ultimateModels = includeUltimate ? `

**latent_upscale_models**

- [minimax_h3_latent_upscaler_3d_fp16.safetensors](https://huggingface.co/LBH-123-AI/Minimax_h3_latent_Upscaler/resolve/main/minimax_h3_latent_upscaler_3d_fp16.safetensors)` : "";
  const rifeModels = includeRife ? `

**frame_interpolation (optional)**

- [rife_v4.26_heavy.safetensors](https://huggingface.co/Comfy-Org/frame_interpolation/resolve/main/frame_interpolation/rife_v4.26_heavy.safetensors)` : "";
  const ultimateTree = includeUltimate
    ? `\n    ${includeRife ? "├" : "└"}── 📂 latent_upscale_models/\n    ${includeRife ? "│" : " "}   └── minimax_h3_latent_upscaler_3d_fp16.safetensors`
    : "";
  const rifeTree = includeRife
    ? `\n    └── 📂 frame_interpolation/\n        └── rife_v4.26_heavy.safetensors`
    : "";
  const vaeBranch = includeUltimate || includeRife ? "├" : "└";
  return `## Model Links

**diffusion_models**

- [minimax_h3_fastvideo_vsa_datafree_1300step_4step_int8_convrot.safetensors](https://huggingface.co/Kijai/MiniMax-H3-experimental/resolve/main/minimax_h3_fastvideo_vsa_datafree_1300step_4step_int8_convrot.safetensors)

**text_encoders**

- [qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors](https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main/text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors)

**vae**

- [minimax_h3_video_vae_fp16.safetensors](https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main/vae/minimax_h3_video_vae_fp16.safetensors)
- [minimax_h3_audio_vae_fp32.safetensors](https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main/vae/minimax_h3_audio_vae_fp32.safetensors)${ultimateModels}${rifeModels}

## Model Storage Location

\`\`\`
📂 ComfyUI/
└── 📂 models/
    ├── 📂 diffusion_models/
    │   └── minimax_h3_fastvideo_vsa_datafree_1300step_4step_int8_convrot.safetensors
    ├── 📂 text_encoders/
    │   └── qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors
    ${vaeBranch}── 📂 vae/
    │   ├── minimax_h3_video_vae_fp16.safetensors
    │   └── minimax_h3_audio_vae_fp32.safetensors${ultimateTree}${rifeTree}
\`\`\`

H3 Relay expands its public nodes into native conditioning/VAE stages, so the text encoder and both VAEs are required even though they are not visible as separate canvas nodes.`;
}

function relayGuideText({
  includeUltimate = false,
  includeSequence = false,
  includeRife = false,
} = {}) {
  const finishing = includeUltimate
    ? `\n3. Run **H3 Ultimate 2x Enhance** only after accepting the raw result. It reuses the accepted AV latent and complete shot prompt.\n4. Ultimate emits 24fps video at exactly 2x width and height.${includeRife ? " The bundled RIFE nodes are bypassed by default; enable all of them for 48fps." : " This focused example intentionally stops at 24fps."}`
    : "";
  const sequencing = includeSequence
    ? `\n\n## Sequence behavior\n\n- Shots 2 and 3 continue from 18 frames of accepted visual/audio history. Three 10-second generation windows deliver about 28.9 seconds after repeated history is trimmed.\n- Run or reroll each Generate Shot independently. Final assemblers restore cached predecessors and run only missing or stale stages.`
    : "";
  const managedStages = includeUltimate
    ? "Generate, Ultimate, and interpolation intermediates"
    : "Generate intermediates";
  return `## H3 Relay · FastH3 VSA

This workflow requires the experimental FastH3 VSA runtime. It does **not** run on an ordinary stock ComfyUI checkout yet.

## Exact tested runtime

- **ComfyUI:** Kijai VSA commit [10febb01d7be73d1491cf5e5347b5ab8b6c2c09e](https://github.com/kijai/ComfyUI/commit/10febb01d7be73d1491cf5e5347b5ab8b6c2c09e) from draft [PR #15958](https://github.com/Comfy-Org/ComfyUI/pull/15958)
- **comfy-kitchen:** official PyPI \`comfy-kitchen==0.2.33\` CUDA wheel, with CUDA \`sol_attn\` and \`sol_attn_chunked\`
- **VSA adapter:** included in H3 Relay for both generation and Ultimate refinement; no temporary custom node is required
- **H3 Relay:** [github.com/akatz-ai/h3-relay](https://github.com/akatz-ai/h3-relay); use the release containing **FastH3 VSA Profile**${includeUltimate ? " and **H3 Ultimate 2x Enhance**" : ""}
${includeUltimate ? "- **MMH3 Ultimate Upscale:** [bbaudio-2025/Comfyui-MMH3-UltimateUpscale](https://github.com/bbaudio-2025/Comfyui-MMH3-UltimateUpscale), tested at commit [6db8fa5a4e4ca0718d2ea8d08002ea899fe27721](https://github.com/bbaudio-2025/Comfyui-MMH3-UltimateUpscale/commit/6db8fa5a4e4ca0718d2ea8d08002ea899fe27721)\n- **H3 latent upscaler:** [LBH-123-AI/Minimax_h3_latent_Upscaler](https://huggingface.co/LBH-123-AI/Minimax_h3_latent_Upscaler)" : ""}

Until ComfyUI's VSA model support is available in a normal release, treat this as an advanced experimental workflow. The reference runtime uses Python 3.11, PyTorch 2.13.0+cu130, and an NVIDIA driver compatible with CUDA 13. See VALIDATION.md for the official-wheel test results and limitations.

## Run order

1. Keep the FastH3 profile at Euler/simple, four forwards, shifts 12/3, Spectrum off, and VSA 10 percent. H3 Relay owns these values even if a stale visible step widget says otherwise.
2. Run **Generate Shot**, review its native 24fps video and synchronized H3 audio, and reroll before accepting downstream work.${finishing}${sequencing}

## Managed outputs

${managedStages} are integrity-checked managed cache artifacts. Their previews are playable, but they are not published files. Use a final Assemble node or connect a native Save Video node when you want a durable output in ComfyUI's output directory.`;
}

note(
  "READ FIRST · AKATZ 3×10 FASTH3 RELAY",
  `# Akatz · three-stage cyberpunk escape

This canvas uses three independently reviewable **10-second FastH3 VSA generation windows**. The 18-frame AV history carried into shots 2 and 3 makes the accepted result one continuous sequence (about 28.9 delivered seconds after repeated history is trimmed).

## Recommended use

1. Run **Shot 1 Generate**, review/reroll it, then accept its sequence output.
2. Run Shot 2, then Shot 3. Each downstream shot is invalidated only if its accepted predecessor changes.
3. Run **Assemble Raw** for the native 832×480/24fps sequence.
4. Optionally run each **H3 Ultimate 2× Enhance** in order. It reuses the connected FastH3 model, the accepted raw latent, the complete original shot prompt, and all five references; output is 1664×960/24fps.
5. RIFE nodes are bypassed by default. Enable all three to produce a coherent 48fps finishing stream, or leave all bypassed for 24fps.

Selecting the final enhanced assembler restores cached raw/enhanced work and runs only missing or stale stages. The learned H3 latent upscaler and MMH3 Ultimate custom nodes must be installed.`,
  [-2480, -1540],
  [1130, 660],
);

note(
  "DIRECTOR MAP · REFERENCE ROLES",
  `## Reference roles

- **Pictures 1–3:** Akatz identity, full costume, face, hair, amber eyes, orange harness, boots, and katana.
- **Picture 4:** rain-soaked neon alley used primarily in shot 1 and the chase transition.
- **Picture 5:** cyan stairwell and service architecture used primarily in shot 2.
- **Shot 3:** continues the accepted service passage into a warm cafe-bar; the prompt owns this new space while Pictures 1–3 keep Akatz stable.

The Ultimate node automatically reuses each accepted shot's complete global + scene prompt. Its connected text input is only an additional conservative refinement directive.`,
  [-1320, -1540],
  [920, 660],
);

let modelLinksNote;
let relayGuideNote;
let sizeNote;

const model = addNode({
  type: "H3RelayFastH3VSAModelLoader",
  title: "FASTH3 VSA · SHARED GENERATION + ULTIMATE REFINEMENT MODEL",
  pos: [-2460, -760],
  size: [650, 300],
  inputs: [
    widget("model_name", "COMBO"),
    widget("weight_dtype", "COMBO"),
    widget("manual_cache_revision", "STRING"),
  ],
  outputs: [outputSocket("h3_model", "H3_RELAY_MODEL")],
  widgets: [
    "minimax_h3_fastvideo_vsa_datafree_1300step_4step_int8_convrot.safetensors",
    "default",
    "akatz-3x10-ultimate-v1",
  ],
  named: {
    model_name: "minimax_h3_fastvideo_vsa_datafree_1300step_4step_int8_convrot.safetensors",
    weight_dtype: "default",
    manual_cache_revision: "akatz-3x10-ultimate-v1",
  },
});

const sequence = addNode({
  type: "H3RelaySequenceStart",
  title: "SEQUENCE START · 832×480 · 18-FRAME AV HISTORY",
  pos: [-2460, -400],
  size: [650, 470],
  inputs: [
    widget("run_name", "STRING"),
    widget("global_prompt", "STRING"),
    widget("width", "INT"),
    widget("height", "INT"),
    widget("h3_overlap_frames", "COMBO"),
    widget("sampler", "COMBO"),
    widget("scheduler", "COMBO"),
    widget("spectrum_enabled", "BOOLEAN"),
  ],
  outputs: [
    outputSocket("sequence", "H3_RELAY_SEQUENCE"),
    outputSocket("status", "STRING"),
  ],
  widgets: [
    "fast_h3_akatz_cyberpunk_escape_3x10",
    globalPrompt,
    832,
    480,
    18,
    "euler",
    "simple",
    false,
  ],
  named: {
    run_name: "fast_h3_akatz_cyberpunk_escape_3x10",
    global_prompt: globalPrompt,
    width: 832,
    height: 480,
    h3_overlap_frames: 18,
    sampler: "euler",
    scheduler: "simple",
    spectrum_enabled: false,
  },
});

const interpolation = addNode({
  type: "H3RelayInterpolationModelLoader",
  title: "OPTIONAL RIFE MODEL · SHARED BY ALL SHOTS",
  pos: [-2460, 140],
  size: [650, 220],
  inputs: [widget("model_name", "COMBO"), widget("manual_cache_revision", "STRING")],
  outputs: [outputSocket("interpolation", "H3_RELAY_INTERPOLATION")],
  widgets: ["rife_v4.26_heavy.safetensors", "v1"],
  named: { model_name: "rife_v4.26_heavy.safetensors", manual_cache_revision: "v1" },
});

const imageFiles = [
  ["Picture 1 · Akatz full reference", "akatz-general-reference.png"],
  ["Picture 2 · Akatz character sheet", "akatz-character-sheet.png"],
  ["Picture 3 · Akatz face close-up", "akatz-face-closeup.png"],
  ["Picture 4 · neon alley", "cyberpunk-neon-street.jpg"],
  ["Picture 5 · cyberpunk stairwell", "cyberpunk-stairwell.jpg"],
];
const images = imageFiles.map(([title, filename], index) =>
  loadImage(title, filename, [-1660 + (index % 3) * 330, -760 + Math.floor(index / 3) * 390]),
);

const promptNodes = shotPrompts.map((prompt, index) =>
  textNode(`SHOT ${index + 1} · DIRECTOR PROMPT`, prompt, [-820 + index * 760, -760]),
);
const finishPrompt = textNode(
  "ULTIMATE · CONSERVATIVE REFINEMENT DIRECTIVE",
  refinementDirective,
  [-820, 2530],
  [1400, 280],
);
const generators = [
  generateNode(1, [-820, -320], 424246),
  generateNode(2, [-60, -320], 424247),
  generateNode(3, [700, -320], 424248),
];
const rawAssemble = addNode({
  type: "H3RelayAssembleRaw",
  title: "ASSEMBLE ACCEPTED RAW FASTH3 SEQUENCE",
  pos: [1450, -60],
  size: [600, 410],
  inputs: [
    socket("sequence", "H3_RELAY_SEQUENCE"),
    widget("filename", "STRING"),
    widget("audio_bitrate", "INT"),
  ],
  outputs: [
    outputSocket("video", "VIDEO"),
    outputSocket("video_path", "STRING"),
    outputSocket("status", "STRING"),
  ],
  widgets: ["fast_h3_akatz_escape_raw", 256],
  named: { filename: "fast_h3_akatz_escape_raw", audio_bitrate: 256 },
});

const ultimate = [
  ultimateNode(1, [-820, 760], 424264),
  ultimateNode(2, [-60, 760], 424265),
  ultimateNode(3, [700, 760], 424266),
];
const rife = [
  interpolateNode(1, [-820, 1540]),
  interpolateNode(2, [-60, 1540]),
  interpolateNode(3, [700, 1540]),
];
const enhancedAssemble = addNode({
  type: "H3RelayAssemble",
  title: "ASSEMBLE H3 ULTIMATE 2× / OPTIONAL 48 FPS SEQUENCE",
  pos: [1450, 1120],
  size: [630, 460],
  inputs: [
    socket("enhanced", "H3_RELAY_ENHANCED"),
    widget("output_stage", "COMBO"),
    widget("filename", "STRING"),
    widget("audio_bitrate", "INT"),
  ],
  outputs: [
    outputSocket("video", "VIDEO"),
    outputSocket("video_path", "STRING"),
    outputSocket("status", "STRING"),
  ],
  widgets: ["auto", "fast_h3_akatz_escape_ultimate_2x", 256],
  named: {
    output_stage: "auto",
    filename: "fast_h3_akatz_escape_ultimate_2x",
    audio_bitrate: 256,
  },
});

modelLinksNote = note(
  "Note: Model Links",
  modelLinksText({ includeUltimate: true, includeRife: true }),
  [-3012, -762],
  [500, 805],
);

relayGuideNote = note(
  "Note: H3 Relay",
  `${relayGuideText({ includeUltimate: true, includeSequence: true, includeRife: true })}

## Bundled reference assets

Copy the five files from \`example_workflows/assets/\` into \`ComfyUI/input/\` before loading this workflow. The saved Load Image nodes expect the exact bundled filenames. Pictures 1-3 define Akatz; Pictures 4-5 define the alley and stairwell.`,
  [-3015, 90],
  [500, 760],
);

sizeNote = note(
  "Note: Size Settings Reference",
  sizeSettingsReference,
  [-2456, 460],
  [360, 440],
);

connect(sequence, "sequence", generators[0], "sequence", "H3_RELAY_SEQUENCE");
for (let index = 0; index < generators.length; index += 1) {
  const generator = generators[index];
  connect(model, "h3_model", generator, "h3_model", "H3_RELAY_MODEL");
  connect(promptNodes[index], "STRING", generator, "prompt", "STRING");
  if (index > 0) {
    connect(generators[index - 1], "sequence", generator, "sequence", "H3_RELAY_SEQUENCE");
  }
  connect(model, "h3_model", ultimate[index], "h3_model", "H3_RELAY_MODEL");
  connect(generator, "sequence", ultimate[index], "sequence", "H3_RELAY_SEQUENCE");
  connect(finishPrompt, "STRING", ultimate[index], "enhancement_prompt", "STRING");
  connect(interpolation, "interpolation", rife[index], "interpolation", "H3_RELAY_INTERPOLATION");
  connect(ultimate[index], "enhanced", rife[index], "enhanced", "H3_RELAY_ENHANCED");
  if (index > 0) {
    connect(rife[index - 1], "enhanced", ultimate[index], "previous_enhanced", "H3_RELAY_ENHANCED");
  }
  images.forEach((image, imageIndex) => {
    const socketName = imageIndex < 3
      ? `reference_image_${imageIndex + 1}`
      : `additional_reference_images.reference_image_${imageIndex + 1}`;
    connect(image, "IMAGE", generator, socketName, "IMAGE");
    connect(image, "IMAGE", ultimate[index], socketName, "IMAGE");
  });
}
connect(generators[2], "sequence", rawAssemble, "sequence", "H3_RELAY_SEQUENCE");
connect(rife[2], "enhanced", enhancedAssemble, "enhanced", "H3_RELAY_ENHANCED");

const workflow = {
  id: "d712e7c2-89ab-4d3e-87c0-fast-h3-akatz-3x10",
  revision: 0,
  last_node_id: nextNodeId - 1,
  last_link_id: nextLinkId - 1,
  nodes,
  links,
  groups: [],
  config: {},
  extra: {
    frontendVersion: "1.49.6",
    ds: { scale: 0.56, offset: [1320, 850] },
    h3_relay_example: {
      profile: "fast_h3_vsa",
      shots: 3,
      generation_window_seconds: 10,
      native_resolution: [832, 480],
      ultimate_resolution: [1664, 960],
      references: imageFiles.map(([, filename]) => filename),
    },
  },
  version: 0.4,
  definitions: { subgraphs: [] },
};

function cloneWorkflow(source) {
  return JSON.parse(JSON.stringify(source));
}

function rebuildSubset(source, keepIds) {
  const subset = cloneWorkflow(source);
  const keep = new Set([...keepIds].map(Number));
  subset.nodes = subset.nodes.filter((node) => keep.has(Number(node.id)));
  subset.links = subset.links.filter(
    (link) => keep.has(Number(link[1])) && keep.has(Number(link[3])),
  );
  for (const [order, node] of subset.nodes.entries()) {
    node.order = order;
    for (const input of node.inputs || []) input.link = null;
    for (const outputItem of node.outputs || []) outputItem.links = null;
  }
  const byId = new Map(subset.nodes.map((node) => [Number(node.id), node]));
  for (const link of subset.links) {
    const [linkId, originId, originSlot, targetId, targetSlot] = link;
    const origin = byId.get(Number(originId));
    const target = byId.get(Number(targetId));
    const outputItem = origin.outputs[Number(originSlot)];
    outputItem.links = Array.isArray(outputItem.links)
      ? [...outputItem.links, Number(linkId)]
      : [Number(linkId)];
    target.inputs[Number(targetSlot)].link = Number(linkId);
  }
  subset.last_node_id = Math.max(...subset.nodes.map((node) => Number(node.id)));
  subset.last_link_id = Math.max(0, ...subset.links.map((link) => Number(link[0])));
  return subset;
}

function nodeById(target, id) {
  const found = target.nodes.find((node) => Number(node.id) === Number(id));
  if (!found) throw new Error(`workflow subset is missing node ${id}`);
  return found;
}

function setTextNode(target, id, title, value, pos, size) {
  const targetNode = nodeById(target, id);
  targetNode.title = title;
  targetNode.widgets_values = [value];
  targetNode.widgets_values_named = { value };
  targetNode.pos = pos;
  targetNode.size = size;
}

function setNote(target, id, title, value, pos, size) {
  setTextNode(target, id, title, value, pos, size);
}

function configureSimpleCore(target, suffix) {
  const targetModel = nodeById(target, model.id);
  targetModel.title = "FASTH3 VSA PROFILE · LOCKED FOUR-FORWARD MODEL";
  targetModel.pos = [-820, -760];
  targetModel.size = [650, 300];
  targetModel.widgets_values = [
    "minimax_h3_fastvideo_vsa_datafree_1300step_4step_int8_convrot.safetensors",
    "default",
    `fast-h3-vsa-${suffix}-v1`,
  ];
  targetModel.widgets_values_named = {
    model_name: "minimax_h3_fastvideo_vsa_datafree_1300step_4step_int8_convrot.safetensors",
    weight_dtype: "default",
    manual_cache_revision: `fast-h3-vsa-${suffix}-v1`,
  };

  const targetSequence = nodeById(target, sequence.id);
  targetSequence.title = "SEQUENCE START · 832×480 · FASTH3 VSA";
  targetSequence.pos = [-820, -400];
  targetSequence.size = [650, 470];
  targetSequence.widgets_values = [
    `fast_h3_vsa_${suffix}`,
    simpleGlobalPrompt,
    832,
    480,
    18,
    "euler",
    "simple",
    false,
  ];
  targetSequence.widgets_values_named = {
    run_name: `fast_h3_vsa_${suffix}`,
    global_prompt: simpleGlobalPrompt,
    width: 832,
    height: 480,
    h3_overlap_frames: 18,
    sampler: "euler",
    scheduler: "simple",
    spectrum_enabled: false,
  };

  setTextNode(
    target,
    promptNodes[0].id,
    "ONE-SHOT · H3 DIRECTOR PROMPT",
    simpleShotPrompt,
    [-80, -760],
    [640, 360],
  );
  const targetGenerate = nodeById(target, generators[0].id);
  targetGenerate.title = "GENERATE / REVIEW / REROLL · RAW FASTH3 VSA";
  targetGenerate.pos = [-80, -320];
  targetGenerate.size = [620, 760];
  targetGenerate.widgets_values = [
    424246,
    "fixed",
    5.0,
    4,
    18,
    "match",
    `fast_h3_vsa_${suffix}`,
  ];
  targetGenerate.widgets_values_named = {
    seed: 424246,
    control_after_generate: "fixed",
    duration_seconds: 5.0,
    h3_steps: 4,
    output_crf: 18,
    ref_image_size: "match",
    shot_id: `fast_h3_vsa_${suffix}`,
  };
}

const instructionalIds = [modelLinksNote.id, relayGuideNote.id, sizeNote.id];
const simpleWorkflow = rebuildSubset(workflow, [
  ...instructionalIds,
  model.id,
  sequence.id,
  promptNodes[0].id,
  generators[0].id,
]);
configureSimpleCore(simpleWorkflow, "one_shot");
setNote(
  simpleWorkflow,
  modelLinksNote.id,
  "Note: Model Links",
  modelLinksText(),
  [-1420, -760],
  [520, 800],
);
setNote(
  simpleWorkflow,
  relayGuideNote.id,
  "Note: H3 Relay",
  relayGuideText(),
  [-1420, 80],
  [520, 760],
);
setNote(
  simpleWorkflow,
  sizeNote.id,
  "Note: Size Settings Reference",
  sizeSettingsReference,
  [-820, 120],
  [380, 440],
);
simpleWorkflow.id = "89d09e50-8c5c-4c46-a710-fast-h3-vsa-one-shot";
simpleWorkflow.extra = {
  frontendVersion: "1.49.6",
  ds: { scale: 0.72, offset: [1120, 700] },
  h3_relay_example: {
    tier: "01-simple-vsa",
    profile: "fast_h3_vsa",
    shots: 1,
    generation_window_seconds: 5,
    native_resolution: [832, 480],
    required_models: 4,
  },
};

const ultimateWorkflow = rebuildSubset(workflow, [
  ...instructionalIds,
  model.id,
  sequence.id,
  promptNodes[0].id,
  finishPrompt.id,
  generators[0].id,
  ultimate[0].id,
]);
configureSimpleCore(ultimateWorkflow, "one_shot_ultimate_2x");
setTextNode(
  ultimateWorkflow,
  finishPrompt.id,
  "ULTIMATE · CONSERVATIVE REFINEMENT DIRECTIVE",
  "Preserve the accepted raw shot's exact subject, action, frame timing, composition, camera motion, geometry, lighting, color palette, and audio timing. Restore stable fine detail, clean edges, material texture, and wet-surface reflections at exactly 2x spatial resolution. Do not redesign the subject or environment; do not add cuts, objects, people, text, motion, or reframing.",
  [650, -760],
  [640, 300],
);
const oneShotUltimate = nodeById(ultimateWorkflow, ultimate[0].id);
oneShotUltimate.title = "H3 ULTIMATE 2× ENHANCE · ACCEPTED RAW LATENT";
oneShotUltimate.pos = [650, -320];
oneShotUltimate.size = [640, 700];
setNote(
  ultimateWorkflow,
  modelLinksNote.id,
  "Note: Model Links",
  modelLinksText({ includeUltimate: true }),
  [-1420, -760],
  [520, 850],
);
setNote(
  ultimateWorkflow,
  relayGuideNote.id,
  "Note: H3 Relay",
  relayGuideText({ includeUltimate: true }),
  [-1420, 130],
  [520, 800],
);
setNote(
  ultimateWorkflow,
  sizeNote.id,
  "Note: Size Settings Reference",
  sizeSettingsReference,
  [-820, 500],
  [400, 440],
);
ultimateWorkflow.id = "5de37eb4-5073-48f9-a1c0-fast-h3-vsa-ultimate-2x";
ultimateWorkflow.extra = {
  frontendVersion: "1.49.6",
  ds: { scale: 0.62, offset: [1050, 700] },
  h3_relay_example: {
    tier: "02-vsa-ultimate-2x",
    profile: "fast_h3_vsa",
    shots: 1,
    generation_window_seconds: 5,
    native_resolution: [832, 480],
    ultimate_resolution: [1664, 960],
    required_models: 5,
  },
};

await Promise.all([
  fs.writeFile(output, `${JSON.stringify(workflow, null, 2)}\n`, "utf8"),
  fs.writeFile(simpleOutput, `${JSON.stringify(simpleWorkflow, null, 2)}\n`, "utf8"),
  fs.writeFile(ultimateOutput, `${JSON.stringify(ultimateWorkflow, null, 2)}\n`, "utf8"),
]);
for (const filename of [simpleOutput, ultimateOutput, output]) console.log(filename);
