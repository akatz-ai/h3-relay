import hashlib
import importlib.util
import json
import pathlib
import subprocess
import tempfile
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / "example_workflows/H3-Relay-Orbital-Storm-Spectrum16-58s.json"
SCRIPT = ROOT / "scripts/clone_retheme_workflow.mjs"
BENCHMARK = ROOT / "benchmark/run_saved_workflow_benchmark.py"


def digest(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def benchmark_module():
    spec = importlib.util.spec_from_file_location("h3_relay_benchmark", BENCHMARK)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class CloneRethemeWorkflowTest(unittest.TestCase):
    def test_stale_frontend_layout_restores_named_settings_and_references(self):
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            source = root / "source.json"
            output = root / "output.json"
            spec_path = root / "spec.json"

            workflow = json.loads(TEMPLATE.read_text(encoding="utf-8"))
            generators = [
                node for node in workflow["nodes"]
                if node["type"] == "H3RelayGenerateShot"
            ]
            for index, node in enumerate(generators):
                node["pos"] = [1000 + index * 100, 2000]
                widgets = [9000 + index, "fixed", 15.0, 16, 18, "match", ""]
                node["widgets_values"] = widgets
                node["widgets_values_named"] = {
                    "seed": widgets[0],
                    "control_after_generate": "fixed",
                    "duration_seconds": 15.0,
                    "h3_steps": 16,
                    "output_crf": 18,
                    "ref_image_size": "match",
                    "shot_id": "",
                }
                sockets = [item for item in node["inputs"] if "widget" not in item]
                widget_inputs = [item for item in node["inputs"] if "widget" in item]
                node["inputs"] = sockets + widget_inputs

            ltx_attention = next(
                node for node in workflow["nodes"]
                if node.get("title") == "LTX ATTENTION · PYTORCH (SWAP OR BYPASS TO TEST)"
            )
            ltx_attention["widgets_values"] = ["comfy kitchen attention"]
            ltx_attention["widgets_values_named"] = {
                "attention": "comfy kitchen attention",
            }
            source.write_text(json.dumps(workflow), encoding="utf-8")
            source_hash = digest(source)

            retheme = {
                "format": "h3_relay_retheme_v1",
                "run_name": "new_story",
                "output_filename": "new_story_output",
                "global_prompt": "new global prompt",
                "enhancement_prompt": "new enhancement prompt",
                "shot_prompts": [f"new shot {index}" for index in range(1, 5)],
                "references": [
                    {
                        "title": "CHARACTERS",
                        "image": "characters.png",
                        "target_input": "reference_image_1",
                        "position": [-1000, 1700],
                    },
                    {
                        "title": "ENVIRONMENT",
                        "image": "environment.png",
                        "target_input": "reference_image_2",
                        "position": [-400, 1700],
                    },
                ],
            }
            spec_path.write_text(json.dumps(retheme), encoding="utf-8")

            subprocess.run([
                "node", str(SCRIPT),
                "--source", str(source),
                "--template", str(TEMPLATE),
                "--spec", str(spec_path),
                "--output", str(output),
            ], check=True, capture_output=True, text=True)

            self.assertEqual(digest(source), source_hash)
            result = json.loads(output.read_text(encoding="utf-8"))
            result_generators = sorted(
                (node for node in result["nodes"]
                 if node["type"] == "H3RelayGenerateShot"),
                key=lambda node: node["title"],
            )
            for index, node in enumerate(result_generators):
                self.assertEqual(node["pos"], [1000 + index * 100, 2000])
                self.assertEqual(
                    [item["name"] for item in node["inputs"][:9]],
                    [
                        "h3_model", "sequence", "prompt", "seed",
                        "duration_seconds", "h3_steps", "output_crf",
                        "ref_image_size", "shot_id",
                    ],
                )
                self.assertEqual(
                    node["widgets_values"],
                    [9000 + index, "fixed", 15.0, 16, 18, "match", ""],
                )
                self.assertEqual(
                    node["widgets_values_named"]["control_after_generate"],
                    "fixed",
                )
                self.assertEqual(
                    node["widgets_values_named"]["duration_seconds"],
                    15.0,
                )
                self.assertIsNotNone(next(
                    item for item in node["inputs"]
                    if item["name"] == "reference_image_1"
                )["link"])
                self.assertIsNotNone(next(
                    item for item in node["inputs"]
                    if item["name"] == "reference_image_2"
                )["link"])

            sequence = next(
                node for node in result["nodes"]
                if node["type"] == "H3RelaySequenceStart"
            )
            self.assertEqual(sequence["widgets_values"][0], "new_story")
            self.assertEqual(sequence["widgets_values_named"]["run_name"], "new_story")
            prompts = sorted(
                (node for node in result["nodes"]
                 if node.get("title", "").startswith("SHOT ")
                 and node["type"] == "PrimitiveStringMultiline"),
                key=lambda node: node["title"],
            )
            self.assertEqual(
                [node["widgets_values_named"]["value"] for node in prompts],
                retheme["shot_prompts"],
            )
            self.assertEqual(
                ltx_attention["widgets_values_named"]["attention"],
                "comfy kitchen attention",
            )
            result_ltx_attention = next(
                node for node in result["nodes"]
                if node.get("title") == "LTX ATTENTION · PYTORCH (SWAP OR BYPASS TO TEST)"
            )
            self.assertEqual(
                result_ltx_attention["widgets_values"],
                ["comfy kitchen attention"],
            )

            prompt = benchmark_module().api_prompt_from_workflow(
                result, "validation_run")
            first = prompt[str(result_generators[0]["id"])]["inputs"]
            self.assertEqual(first["duration_seconds"], 15.0)
            self.assertEqual(first["h3_steps"], 16)
            self.assertEqual(first["output_crf"], 18)
            self.assertEqual(first["ref_image_size"], "match")
            self.assertNotIn("control_after_generate", first)
            character_loader = next(
                node for node in result["nodes"] if node.get("title") == "CHARACTERS")
            self.assertEqual(
                first["reference_image_1"],
                [str(character_loader["id"]), 0],
            )
            self.assertEqual(
                prompt[str(character_loader["id"])]["inputs"]["image"],
                "characters.png",
            )


if __name__ == "__main__":
    unittest.main()
