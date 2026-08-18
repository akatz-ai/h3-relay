#!/usr/bin/env node

import crypto from "node:crypto";
import fs from "node:fs/promises";
import path from "node:path";

const [targetValue, expectedSha] = process.argv.slice(2);
if (!targetValue || !expectedSha) {
  throw new Error(
    "usage: node scripts/upgrade_interpolation_chunks.mjs "
    + "<workflow.json> <expected-sha256>",
  );
}

const target = path.resolve(targetValue);
const source = await fs.readFile(target);
const actualSha = crypto.createHash("sha256").update(source).digest("hex");
if (actualSha !== expectedSha) {
  throw new Error(
    `refusing to update changed workflow: expected ${expectedSha}, found ${actualSha}`,
  );
}

const workflow = JSON.parse(source.toString("utf8"));
let changed = 0;
for (const node of workflow.nodes || []) {
  if (node.type !== "H3RelayInterpolateShot") continue;
  const inputs = Array.isArray(node.inputs) ? node.inputs : [];
  if (!inputs.some((input) => input.name === "chunk_frames")) {
    const outputCrf = inputs.findIndex((input) => input.name === "output_crf");
    const insertAt = outputCrf >= 0 ? outputCrf + 1 : inputs.length;
    inputs.splice(insertAt, 0, {
      localized_name: "chunk_frames",
      name: "chunk_frames",
      type: "INT",
      widget: { name: "chunk_frames" },
      link: null,
    });
    node.inputs = inputs;
  }
  const values = Array.isArray(node.widgets_values) ? node.widgets_values : [];
  if (values.length < 3) node.widgets_values = [...values.slice(0, 2), 48];
  node.widgets_values_named = {
    ...(node.widgets_values_named || {}),
    chunk_frames: node.widgets_values[2] ?? 48,
  };
  changed += 1;
}

await fs.writeFile(target, `${JSON.stringify(workflow, null, 2)}\n`);
const result = await fs.readFile(target);
const resultSha = crypto.createHash("sha256").update(result).digest("hex");
process.stdout.write(
  `updated ${changed} interpolation nodes; ${actualSha} -> ${resultSha}\n`,
);
