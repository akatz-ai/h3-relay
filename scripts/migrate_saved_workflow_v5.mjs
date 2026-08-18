#!/usr/bin/env node

import fs from "node:fs/promises";
import path from "node:path";

const ROOT = path.resolve(import.meta.dirname, "..");
const sourcePath = process.argv[2];
if (!sourcePath) {
  throw new Error("usage: node scripts/migrate_saved_workflow_v3.mjs <workflow.json>");
}

const target = path.resolve(sourcePath);
const templatePath = path.join(
  ROOT,
  "example_workflows/H3-Relay-Orbital-Storm-Spectrum16-58s.json",
);
const current = JSON.parse(await fs.readFile(target, "utf8"));
const template = JSON.parse(await fs.readFile(templatePath, "utf8"));

const hasModelNote = (current.nodes || []).some((node) => node.id === 61);
if (!hasModelNote) {
  const legacyModelNote = (current.nodes || []).find((node) =>
    node.type === "MarkdownNote"
    && String(node.widgets_values?.[0] || "").startsWith("# H3 Relay model installation"));
  if (legacyModelNote) legacyModelNote.id = 61;
}

const canonicalShots = new Map([
  ["SHOT 1 · GENERATE / REVIEW / REROLL RAW H3", 20],
  ["SHOT 2 · CONTINUE RAW H3 FROM SHOT 1", 21],
  ["SHOT 3 · CONTINUE RAW H3 FROM SHOT 2", 22],
  ["SHOT 4 · CONTINUE RAW H3 FROM SHOT 3", 23],
]);
for (const [title, canonicalId] of canonicalShots) {
  const candidates = (current.nodes || []).filter((node) =>
    node.type === "H3RelayGenerateShot" && node.title === title);
  const preferred = candidates.find((node) => node.id !== canonicalId)
    || candidates.find((node) => node.id === canonicalId);
  if (!preferred || preferred.id === canonicalId) continue;
  const oldId = preferred.id;
  current.nodes = current.nodes.filter((node) =>
    node === preferred || node.id !== canonicalId);
  preferred.id = canonicalId;
  for (const link of current.links || []) {
    if (link[1] === oldId) link[1] = canonicalId;
    if (link[3] === oldId) link[3] = canonicalId;
  }
}

function renameInput(node, from, to) {
  const input = (node.inputs || []).find((candidate) => candidate.name === from);
  if (!input) return;
  input.name = to;
  input.localized_name = to;
  if (input.widget?.name === from) input.widget.name = to;
}

for (const node of current.nodes || []) {
  if ([
    "H3RelayH3HybridModelLoader",
    "H3RelayH3ModelLoader",
    "H3RelayLTXModelLoader",
    "H3RelayInterpolationModelLoader",
  ].includes(node.type)) {
    renameInput(node, "cache_revision", "manual_cache_revision");
  }
  if (node.type === "H3RelayLTXModelLoader") {
    renameInput(node, "upscale_model_name", "latent_2x_model_name");
    renameInput(node, "upscale_lora", "pixel_upscale_ic_lora");
    renameInput(node, "upscale_strength", "pixel_upscale_ic_strength");
  }
  if (node.type === "H3RelayEnhanceShot") {
    renameInput(node, "enhanced_crf", "output_crf");
  }
  if (node.type === "H3RelayGenerateShot") {
    renameInput(node, "shot_name", "shot_id");
  }
}
const currentNodes = new Map((current.nodes || []).map((node) => [node.id, node]));
const templateIds = new Set(template.nodes.map((node) => node.id));

const customH3ModelPath = (current.links || []).some((link) =>
  link[3] === 4 && link[4] === 0 && link[1] !== 3);
if (customH3ModelPath) {
  const removed = new Set(template.links
    .filter((link) => link[1] === 3 && link[3] === 4)
    .map((link) => link[0]));
  template.links = template.links.filter((link) => !removed.has(link[0]));
  const origin = template.nodes.find((node) => node.id === 3);
  const targetNode = template.nodes.find((node) => node.id === 4);
  origin.outputs[0].links = (origin.outputs[0].links || [])
    .filter((id) => !removed.has(id));
  targetNode.inputs[0].link = null;
}

for (const node of template.nodes) {
  const prior = currentNodes.get(node.id);
  if (!prior) continue;
  node.pos = prior.pos ?? node.pos;
  node.flags = prior.flags ?? node.flags;
  if (node.type !== "MarkdownNote") {
    node.size = prior.size ?? node.size;
    if (prior.color !== undefined) node.color = prior.color;
    if (prior.bgcolor !== undefined) node.bgcolor = prior.bgcolor;
  }

  if (node.type === "PrimitiveStringMultiline") {
    node.widgets_values = prior.widgets_values ?? node.widgets_values;
  } else if (node.type === "H3RelaySequenceStart") {
    const values = prior.widgets_values || [];
    const hasOverlap = (prior.inputs || []).some((input) =>
      input.name === "h3_overlap_frames");
    if (values.length === 8 && hasOverlap) {
      node.widgets_values = values.slice(0, 8);
    } else if (values.length >= 8) {
      node.widgets_values = [
        values[0], values[1], values[2], values[3],
        18, values[5], values[6], values[7],
      ];
    } else if (values.length === 7) {
      node.widgets_values = [
        values[0], values[1], values[2], values[3],
        18, values[4], values[5], values[6],
      ];
    } else if (values.length >= 6) {
      const profile = values.at(-1);
      const sampler = profile === "res_multistep" ? "res_multistep" : "euler";
      const scheduler = profile === "native_spectrum_euler_beta57"
        ? "beta57"
        : (profile === "native_euler_beta" || profile === "turbo_auto"
          ? "beta"
          : "simple");
      node.widgets_values = [
        values[0], values[1], values[2], values[3],
        18, sampler, scheduler, profile === "native_spectrum_euler_beta57",
      ];
    }
  } else if (node.type === "H3RelayGenerateShot") {
    const values = prior.widgets_values || [];
    const priorInputNames = (prior.inputs || []).map((input) => input.name);
    const priorUsesShotId = priorInputNames.includes("shot_id")
      && !priorInputNames.includes("shot_name");
    const hasOutputCrf = (prior.inputs || []).some((input) =>
      input.name === "output_crf");
    const plausibleNew = (
      values.length === 7
      && Number.isInteger(values[0])
      && typeof values[1] === "string"
      && typeof values[2] === "number"
      && Number.isInteger(values[3])
      && Number.isInteger(values[4])
      && typeof values[5] === "string"
      && typeof values[6] === "string"
    );
    if (plausibleNew && hasOutputCrf && priorUsesShotId) {
      node.widgets_values = [...values];
    } else {
      const legacyId = values[0] ?? "";
      const plausibleLegacy = (
        [6, 7].includes(values.length)
        && typeof legacyId === "string"
        && Number.isInteger(values[1])
        && typeof values[2] === "string"
        && typeof values[3] === "number"
        && Number.isInteger(values[4])
        && (values.length === 6 || Number.isInteger(values[5]))
        && typeof values.at(-1) === "string"
      );
      if (plausibleLegacy) {
        node.widgets_values = values.length === 7
          ? [...values.slice(1), legacyId]
          : [values[1], values[2], values[3], values[4], 18, values[5], legacyId];
      }
    }
  } else if (node.type === "H3RelayEnhanceShot") {
    const values = prior.widgets_values || [];
    if (values.length >= 6) node.widgets_values = values.slice(1, 6);
    else if (values.length === 5) node.widgets_values = [...values];
  } else if (node.type === "H3RelayInterpolateShot") {
    const values = prior.widgets_values || [];
    if (values.length >= 3) {
      node.widgets_values = values.slice(-3);
    } else if (values.length >= 2) {
      node.widgets_values = [values.at(-2), values.at(-1), 48];
    }
  } else if (node.type === "H3RelayAssemble") {
    node.widgets_values = prior.widgets_values ?? node.widgets_values;
  }
}

const extraNodes = (current.nodes || [])
  .filter((node) => !templateIds.has(node.id))
  .map((node) => structuredClone(node));
for (const node of extraNodes) {
  for (const input of node.inputs || []) input.link = null;
  for (const output of node.outputs || []) output.links = [];
}
template.nodes.push(...extraNodes);

const allNodes = new Map(template.nodes.map((node) => [node.id, node]));
let nextLink = Math.max(0, ...template.links.map((link) => link[0])) + 1;
for (const link of current.links || []) {
  const [, originId, originSlot, targetId, targetSlot, type] = link;
  if (templateIds.has(originId) && templateIds.has(targetId)) continue;
  if (!allNodes.has(originId) || !allNodes.has(targetId)) continue;
  const id = nextLink++;
  template.links.push([id, originId, originSlot, targetId, targetSlot, type]);
  const origin = allNodes.get(originId);
  const targetNode = allNodes.get(targetId);
  if (origin.outputs?.[originSlot]) {
    origin.outputs[originSlot].links = origin.outputs[originSlot].links || [];
    origin.outputs[originSlot].links.push(id);
  }
  if (targetNode.inputs?.[targetSlot]) targetNode.inputs[targetSlot].link = id;
}

template.id = current.id || template.id;
template.revision = current.revision || 0;
template.last_node_id = Math.max(...template.nodes.map((node) => Number(node.id)));
template.last_link_id = nextLink - 1;
template.extra = {
  ...template.extra,
  ds: current.extra?.ds || template.extra?.ds,
};

await fs.writeFile(target, `${JSON.stringify(template, null, 2)}\n`);
process.stdout.write(
  `migrated ${target}; preserved ${extraNodes.length} user-added nodes\n`,
);
