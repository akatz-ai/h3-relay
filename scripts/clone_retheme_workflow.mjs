#!/usr/bin/env node

import crypto from "node:crypto";
import fs from "node:fs/promises";
import path from "node:path";

const WIDGET_INPUTS = {
  PrimitiveStringMultiline: ["value"],
  MarkdownNote: ["value"],
  LoadImage: ["image"],
  H3RelayH3HybridModelLoader: [
    "base_model", "overlay_model", "overlay_preset",
    "manual_cache_revision", "block_range_start", "block_range_end",
    "final_adaln_from_overlay", "custom_overlays", "custom_base",
    "weight_dtype",
  ],
  H3RelayH3ModelLoader: [
    "model_name", "text_encoder_name", "video_vae_name",
    "audio_vae_name", "weight_dtype", "manual_cache_revision",
  ],
  H3RelayModelLoRA: ["lora_name", "strength"],
  H3RelayAttention: ["attention"],
  H3RelaySequenceStart: [
    "run_name", "global_prompt", "width", "height",
    "h3_overlap_frames", "sampler", "scheduler", "spectrum_enabled",
  ],
  H3RelayGenerateShot: [
    "seed", "control_after_generate", "duration_seconds", "h3_steps",
    "output_crf", "ref_image_size", "shot_id",
  ],
  H3RelayLTXModelLoader: [
    "model_name", "vae_name", "latent_2x_model_name",
    "text_encoder_name", "distilled_lora", "distilled_strength",
    "pixel_upscale_ic_lora", "pixel_upscale_ic_strength", "weight_dtype",
    "manual_cache_revision",
  ],
  H3RelayEnhanceShot: [
    "output_crf", "context_window_frames", "context_overlap_frames",
    "vae_temporal_tile_frames", "vae_temporal_overlap_frames",
  ],
  H3RelayInterpolationModelLoader: [
    "model_name", "manual_cache_revision",
  ],
  H3RelayInterpolateShot: ["multiplier", "output_crf", "chunk_frames"],
  H3RelayAssemble: ["output_stage", "filename", "audio_bitrate"],
  H3RelayCacheManager: ["action", "keep_revisions_per_shot", "budget_gb"],
};

const CANONICAL_INPUT_ORDER = {
  H3RelayGenerateShot: [
    "h3_model", "sequence", "prompt",
    "seed", "duration_seconds", "h3_steps", "output_crf",
    "ref_image_size", "shot_id",
    "first_frame", "last_frame",
    "reference_image_1", "reference_image_2", "reference_image_3",
    "reference_video", "reference_video_audio", "reference_audio",
  ],
  H3RelayEnhanceShot: [
    "ltx_model", "sequence", "enhancement_prompt",
    "output_crf", "context_window_frames", "context_overlap_frames",
    "vae_temporal_tile_frames", "vae_temporal_overlap_frames",
    "previous_enhanced",
  ],
};

function usage() {
  return [
    "usage: node scripts/clone_retheme_workflow.mjs \\",
    "  --source <saved-workflow.json> \\",
    "  --template <canonical-workflow.json> \\",
    "  --spec <retheme-spec.json> \\",
    "  --output <new-workflow.json> [--force]",
  ].join("\n");
}

function parseArgs(argv) {
  const result = { force: false };
  for (let index = 0; index < argv.length; index += 1) {
    const token = argv[index];
    if (token === "--force") {
      result.force = true;
      continue;
    }
    if (!token.startsWith("--")) throw new Error(`unexpected argument ${token}`);
    const key = token.slice(2);
    const value = argv[index + 1];
    if (!value || value.startsWith("--")) {
      throw new Error(`missing value for ${token}`);
    }
    result[key] = value;
    index += 1;
  }
  for (const key of ["source", "template", "spec", "output"]) {
    if (!result[key]) throw new Error(`${usage()}\n\nmissing --${key}`);
  }
  return result;
}

function clone(value) {
  return structuredClone(value);
}

function widgetState(node) {
  const names = WIDGET_INPUTS[node.type] || [];
  const values = Array.isArray(node.widgets_values) ? node.widgets_values : [];
  const hasNamedState = node.widgets_values_named
    && typeof node.widgets_values_named === "object";
  const offset = !hasNamedState
    && values.length === names.length + 1
    && values[0] === ""
    ? 1
    : 0;
  const named = {
    ...(node.widgets_values_named && typeof node.widgets_values_named === "object"
      ? node.widgets_values_named
      : {}),
  };
  names.forEach((name, index) => {
    if (!(name in named) && index + offset < values.length) {
      named[name] = values[index + offset];
    }
  });
  return named;
}

function applyWidgetState(node, state) {
  const names = WIDGET_INPUTS[node.type];
  if (!names) return;
  const defaults = widgetState(node);
  const merged = { ...defaults, ...state };
  node.widgets_values = names.map((name) => merged[name]);
  node.widgets_values_named = Object.fromEntries(
    names.map((name) => [name, merged[name]]),
  );
}

function nodeKey(node) {
  return `${node.type}\u0000${node.title || ""}`;
}

function preserveNodeState(template, source) {
  for (const key of ["pos", "size", "flags", "mode", "color", "bgcolor"]) {
    if (source[key] !== undefined) template[key] = clone(source[key]);
  }
  if (source.properties !== undefined) template.properties = clone(source.properties);
  const names = WIDGET_INPUTS[template.type];
  if (names) applyWidgetState(template, widgetState(source));
  else if (source.widgets_values !== undefined) {
    template.widgets_values = clone(source.widgets_values);
  }
}

function canonicalizeInputOrder(workflow) {
  for (const node of workflow.nodes) {
    const orderedNames = CANONICAL_INPUT_ORDER[node.type];
    if (!orderedNames) continue;
    const oldInputs = node.inputs || [];
    const rank = new Map(orderedNames.map((name, index) => [name, index]));
    const indexed = oldInputs.map((input, oldIndex) => ({ input, oldIndex }));
    indexed.sort((left, right) => {
      const leftRank = rank.has(left.input.name)
        ? rank.get(left.input.name)
        : orderedNames.length + left.oldIndex;
      const rightRank = rank.has(right.input.name)
        ? rank.get(right.input.name)
        : orderedNames.length + right.oldIndex;
      return leftRank - rightRank;
    });
    const oldToNew = new Map(
      indexed.map((item, newIndex) => [item.oldIndex, newIndex]),
    );
    for (const link of workflow.links) {
      if (Number(link[3]) !== Number(node.id)) continue;
      const newSlot = oldToNew.get(Number(link[4]));
      if (newSlot === undefined) {
        throw new Error(`cannot remap link ${link[0]} into ${node.title}`);
      }
      link[4] = newSlot;
    }
    node.inputs = indexed.map((item) => item.input);
  }
}

function findUnique(nodes, predicate, label) {
  const matches = nodes.filter(predicate);
  if (matches.length !== 1) {
    throw new Error(`${label}: expected one node, found ${matches.length}`);
  }
  return matches[0];
}

function setNamedWidget(node, name, value) {
  const names = WIDGET_INPUTS[node.type];
  if (!names || !names.includes(name)) {
    throw new Error(`${node.title || node.type} has no widget named ${name}`);
  }
  const state = widgetState(node);
  state[name] = value;
  applyWidgetState(node, state);
}

function loadImageNode(id, reference, outputLinks) {
  const position = reference.position || [0, 0];
  const size = reference.size || [560, 520];
  const node = {
    id,
    type: "LoadImage",
    pos: position,
    size,
    flags: {},
    order: id,
    mode: 0,
    inputs: [{
      name: "image",
      type: "COMBO",
      widget: { name: "image" },
      link: null,
    }],
    outputs: [
      { name: "IMAGE", type: "IMAGE", links: outputLinks },
      { name: "MASK", type: "MASK", links: [] },
    ],
    properties: { "Node name for S&R": "LoadImage" },
    widgets_values: [reference.image, "image"],
    widgets_values_named: { image: reference.image },
    title: reference.title,
  };
  return node;
}

function validateLinks(workflow) {
  const nodes = new Map(workflow.nodes.map((node) => [Number(node.id), node]));
  const links = new Map();
  for (const link of workflow.links) {
    const [id, originId, originSlot, targetId, targetSlot, type] = link;
    if (links.has(Number(id))) throw new Error(`duplicate link id ${id}`);
    const origin = nodes.get(Number(originId));
    const target = nodes.get(Number(targetId));
    if (!origin || !target) throw new Error(`link ${id} has a missing endpoint`);
    if (!origin.outputs?.[originSlot]) throw new Error(`link ${id} has bad origin slot`);
    if (!target.inputs?.[targetSlot]) throw new Error(`link ${id} has bad target slot`);
    if (target.inputs[targetSlot].link !== id) {
      throw new Error(`link ${id} does not match target input state`);
    }
    if (!origin.outputs[originSlot].links?.includes(id)) {
      throw new Error(`link ${id} does not match origin output state`);
    }
    if (origin.outputs[originSlot].type !== type || target.inputs[targetSlot].type !== type) {
      throw new Error(`link ${id} type ${type} does not match its sockets`);
    }
    links.set(Number(id), link);
  }
}

const args = parseArgs(process.argv.slice(2));
const resolved = Object.fromEntries(
  ["source", "template", "spec", "output"].map((key) => [
    key,
    path.resolve(args[key]),
  ]),
);
const [source, workflow, spec] = await Promise.all([
  fs.readFile(resolved.source, "utf8").then(JSON.parse),
  fs.readFile(resolved.template, "utf8").then(JSON.parse),
  fs.readFile(resolved.spec, "utf8").then(JSON.parse),
]);

if (spec.format !== "h3_relay_retheme_v1") {
  throw new Error(`unsupported retheme spec format ${spec.format}`);
}
if (!Array.isArray(spec.shot_prompts) || spec.shot_prompts.length !== 4) {
  throw new Error("retheme spec requires exactly four shot_prompts");
}
if (!Array.isArray(spec.references)) throw new Error("retheme spec requires references");

if (!args.force) {
  try {
    await fs.access(resolved.output);
    throw new Error(`refusing to overwrite ${resolved.output}; pass --force intentionally`);
  } catch (error) {
    if (error?.code !== "ENOENT") throw error;
  }
}

const sourceByKey = new Map();
for (const node of source.nodes || []) {
  const key = nodeKey(node);
  if (!sourceByKey.has(key)) sourceByKey.set(key, []);
  sourceByKey.get(key).push(node);
}
for (const node of workflow.nodes || []) {
  const candidates = sourceByKey.get(nodeKey(node)) || [];
  const prior = candidates.length === 1
    ? candidates[0]
    : source.nodes.find((candidate) =>
      candidate.id === node.id && candidate.type === node.type);
  if (prior) preserveNodeState(node, prior);
}
canonicalizeInputOrder(workflow);

workflow.id = crypto.randomUUID();
workflow.revision = 0;
workflow.groups = clone(source.groups || workflow.groups || []);
workflow.config = clone(source.config || workflow.config || {});
workflow.extra = {
  ...(workflow.extra || {}),
  ...(source.extra || {}),
};

const sequence = findUnique(
  workflow.nodes,
  (node) => node.type === "H3RelaySequenceStart",
  "sequence start",
);
setNamedWidget(sequence, "run_name", spec.run_name);
setNamedWidget(sequence, "global_prompt", spec.global_prompt);

const enhancement = findUnique(
  workflow.nodes,
  (node) => node.type === "PrimitiveStringMultiline"
    && node.title === "GLOBAL LTX ENHANCEMENT PROMPT",
  "global enhancement prompt",
);
setNamedWidget(enhancement, "value", spec.enhancement_prompt);

for (let index = 0; index < 4; index += 1) {
  const prompt = findUnique(
    workflow.nodes,
    (node) => node.type === "PrimitiveStringMultiline"
      && node.title === `SHOT ${index + 1} PROMPT`,
    `shot ${index + 1} prompt`,
  );
  setNamedWidget(prompt, "value", spec.shot_prompts[index]);
}

const assemble = findUnique(
  workflow.nodes,
  (node) => node.type === "H3RelayAssemble",
  "assembler",
);
setNamedWidget(assemble, "filename", spec.output_filename);

let nextNodeId = Math.max(0, ...workflow.nodes.map((node) => Number(node.id))) + 1;
let nextLinkId = Math.max(0, ...workflow.links.map((link) => Number(link[0]))) + 1;
const generators = workflow.nodes
  .filter((node) => node.type === "H3RelayGenerateShot")
  .sort((a, b) => String(a.title).localeCompare(String(b.title)));
if (generators.length !== 4) throw new Error(`expected four generators, found ${generators.length}`);

for (const reference of spec.references) {
  if (!reference.image || !reference.title || !reference.target_input) {
    throw new Error("each reference requires image, title, and target_input");
  }
  const nodeId = nextNodeId++;
  const outputLinks = [];
  const referenceNode = loadImageNode(nodeId, reference, outputLinks);
  workflow.nodes.push(referenceNode);
  for (const generator of generators) {
    const targetSlot = generator.inputs.findIndex(
      (input) => input.name === reference.target_input,
    );
    if (targetSlot < 0) {
      throw new Error(`${generator.title} has no input ${reference.target_input}`);
    }
    if (generator.inputs[targetSlot].link !== null) {
      throw new Error(`${generator.title} input ${reference.target_input} is already linked`);
    }
    const linkId = nextLinkId++;
    generator.inputs[targetSlot].link = linkId;
    outputLinks.push(linkId);
    workflow.links.push([
      linkId,
      nodeId,
      0,
      generator.id,
      targetSlot,
      "IMAGE",
    ]);
  }
}

workflow.last_node_id = Math.max(...workflow.nodes.map((node) => Number(node.id)));
workflow.last_link_id = Math.max(...workflow.links.map((link) => Number(link[0])));
validateLinks(workflow);

const outputDirectory = path.dirname(resolved.output);
await fs.mkdir(outputDirectory, { recursive: true });
const temporary = path.join(
  outputDirectory,
  `.${path.basename(resolved.output)}.${crypto.randomUUID()}.tmp`,
);
try {
  await fs.writeFile(temporary, `${JSON.stringify(workflow, null, 2)}\n`);
  await fs.rename(temporary, resolved.output);
} finally {
  await fs.rm(temporary, { force: true });
}

process.stdout.write([
  `created ${resolved.output}`,
  `source ${resolved.source}`,
  `template ${resolved.template}`,
  `settings restored by name; ${spec.references.length} references connected`,
].join("\n") + "\n");
