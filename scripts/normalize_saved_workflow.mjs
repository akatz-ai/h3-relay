#!/usr/bin/env node

import fs from "node:fs/promises";
import path from "node:path";

const target = process.argv[2];
if (!target) {
  throw new Error("usage: node scripts/normalize_saved_workflow.mjs <workflow.json>");
}

const resolved = path.resolve(target);
const workflow = JSON.parse(await fs.readFile(resolved, "utf8"));
let changed = 0;

for (const node of workflow.nodes || []) {
  if (node.type === "H3RelaySequenceStart") {
    const values = Array.isArray(node.widgets_values) ? node.widgets_values : [];
    const inputs = Array.isArray(node.inputs) ? node.inputs : [];
    if (!inputs.some((input) => input.name === "h3_overlap_frames")) {
      const samplerIndex = inputs.findIndex((input) => input.name === "sampler");
      const insertAt = samplerIndex >= 0 ? samplerIndex : Math.min(4, inputs.length);
      inputs.splice(insertAt, 0, {
        localized_name: "h3_overlap_frames",
        name: "h3_overlap_frames",
        type: "COMBO",
        widget: { name: "h3_overlap_frames" },
        link: null,
      });
      node.widgets_values = [...values.slice(0, 4), 18, ...values.slice(4)];
      node.widgets_values_named = {
        ...(node.widgets_values_named || {}),
        h3_overlap_frames: 18,
      };
      changed += 1;
    }
    continue;
  }

  if (node.type === "H3RelayGenerateShot") {
    const values = Array.isArray(node.widgets_values) ? node.widgets_values : [];
    const named = node.widgets_values_named || {};
    const inputs = Array.isArray(node.inputs) ? node.inputs : [];
    const hasLegacyShotName = inputs.some((input) => input.name === "shot_name");
    const hasOutputCrf = (node.inputs || []).some((input) =>
      input.name === "output_crf");
    const hasShotId = inputs.some((input) => input.name === "shot_id");
    if (values.length === 7 && hasOutputCrf && hasShotId && !hasLegacyShotName) {
      continue;
    }
    if (![6, 7].includes(values.length)) {
      throw new Error(`H3 Relay shot node ${node.id} has incomplete widget values`);
    }
    const legacyId = values[0] ?? "";
    const normalized = values.length === 7
      ? [...values.slice(1), legacyId]
      : [values[1], values[2], values[3], values[4], 18, values[5], legacyId];
    for (const input of inputs) {
      if (input.name !== "shot_name") continue;
      input.name = "shot_id";
      input.localized_name = "shot_id";
      if (input.widget?.name === "shot_name") input.widget.name = "shot_id";
    }
    node.widgets_values = normalized;
    node.widgets_values_named = {
      ...named,
      seed: normalized[0],
      control_after_generate: normalized[1],
      duration_seconds: normalized[2],
      h3_steps: normalized[3],
      output_crf: normalized[4],
      ref_image_size: normalized[5],
      shot_id: normalized[6],
    };
    delete node.widgets_values_named.shot_name;
    changed += 1;
    continue;
  }

  if (node.type === "H3RelayInterpolateShot") {
    const values = Array.isArray(node.widgets_values) ? node.widgets_values : [];
    const inputs = Array.isArray(node.inputs) ? node.inputs : [];
    if (!inputs.some((input) => input.name === "chunk_frames")) {
      inputs.push({
        localized_name: "chunk_frames",
        name: "chunk_frames",
        type: "INT",
        widget: { name: "chunk_frames" },
        link: null,
      });
      node.widgets_values = [...values.slice(0, 2), 48];
      node.widgets_values_named = {
        ...(node.widgets_values_named || {}),
        chunk_frames: 48,
      };
      changed += 1;
    }
    continue;
  }

  if (node.type !== "H3RelayEnhanceShot") continue;
  const fields = [
    ["context_window_frames", 193],
    ["context_overlap_frames", 64],
    ["vae_temporal_tile_frames", 128],
    ["vae_temporal_overlap_frames", 16],
  ];
  const inputs = Array.isArray(node.inputs) ? node.inputs : [];
  const legacyCrf = inputs.find((input) => input.name === "enhanced_crf");
  if (legacyCrf) {
    legacyCrf.name = "output_crf";
    legacyCrf.localized_name = "output_crf";
    if (legacyCrf.widget?.name === "enhanced_crf") {
      legacyCrf.widget.name = "output_crf";
    }
  }
  for (const [name] of fields) {
    if (inputs.some((input) => input.name === name)) continue;
    inputs.push({
      localized_name: name,
      name,
      type: "INT",
      widget: { name },
      link: null,
    });
  }
  node.inputs = inputs;
  const named = node.widgets_values_named || {};
  const crf = Number.isInteger(named.output_crf)
    ? named.output_crf
    : (Number.isInteger(named.enhanced_crf)
      ? named.enhanced_crf
    : (node.widgets_values?.length >= 6
      ? node.widgets_values[1]
      : (node.widgets_values?.[0] ?? 18)));
  node.widgets_values = [crf, ...fields.map(([, value]) => value)];
  node.widgets_values_named = {
    ...named,
    output_crf: crf,
    ...Object.fromEntries(fields),
  };
  changed += 1;
}

await fs.writeFile(resolved, `${JSON.stringify(workflow, null, 2)}\n`);
process.stdout.write(`normalized ${changed} H3 Relay nodes in ${resolved}\n`);
