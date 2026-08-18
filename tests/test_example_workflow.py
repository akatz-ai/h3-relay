import json
import pathlib
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / "example_workflows" / "H3-Relay-Orbital-Storm-Spectrum16-58s.json"


class ExampleWorkflowTest(unittest.TestCase):
    def setUp(self):
        self.workflow = json.loads(WORKFLOW.read_text(encoding="utf-8"))

    def test_direct_nodes_replace_subgraphs(self):
        self.assertEqual(
            self.workflow.get("definitions", {}).get("subgraphs", []), [])
        types = [node["type"] for node in self.workflow["nodes"]]
        self.assertEqual(types.count("H3RelaySequenceStart"), 1)
        self.assertEqual(types.count("H3RelayGenerateShot"), 4)
        self.assertEqual(types.count("H3RelayEnhanceShot"), 4)
        self.assertEqual(types.count("H3RelayInterpolateShot"), 4)
        self.assertEqual(types.count("H3RelayAssemble"), 1)
        self.assertEqual(types.count("H3RelayH3HybridModelLoader"), 1)
        self.assertEqual(types.count("H3RelayLTXModelLoader"), 1)
        self.assertEqual(types.count("H3RelayAttention"), 2)
        self.assertEqual(types.count("H3RelayInterpolationModelLoader"), 1)
        self.assertEqual(types.count("H3RelayCacheManager"), 1)

    def test_sequence_names_and_bypass_contract(self):
        self.assertFalse(any(link[5] == "H3_RELAY_FINISH" for link in self.workflow["links"]))
        enhanced = [
            link for link in self.workflow["links"]
            if link[5] == "H3_RELAY_ENHANCED"
        ]
        self.assertEqual(len(enhanced), 8)
        for node in self.workflow["nodes"]:
            names = [item["name"] for item in node.get("inputs", [])]
            self.assertNotIn("raw_sequence", names)
            self.assertNotIn("finish", names)
            self.assertNotIn("enhanced_sequence", names)
            self.assertNotIn("previous_enhanced_sequence", names)
            if node["type"] == "H3RelayInterpolateShot":
                self.assertEqual(node["inputs"][1]["name"], "enhanced")
                self.assertEqual(node["outputs"][0]["name"], "enhanced")
            if node["type"] == "H3RelayEnhanceShot":
                self.assertEqual(node["inputs"][0]["name"], "ltx_model")

    def test_linked_prompts_are_labeled_sockets(self):
        for node in self.workflow["nodes"]:
            prompt_name = {
                "H3RelayGenerateShot": "prompt",
                "H3RelayEnhanceShot": "enhancement_prompt",
            }.get(node["type"])
            if prompt_name is None:
                continue
            prompt = next(item for item in node["inputs"] if item["name"] == prompt_name)
            self.assertIsNotNone(prompt["link"])
            self.assertNotIn("widget", prompt)

    def test_reference_settings_match_published_spectrum_run(self):
        nodes = {node["id"]: node for node in self.workflow["nodes"]}
        self.assertEqual(
            nodes[3]["widgets_values"][:2],
            [
                "minimax_h3_fl2va_int8_convrot.safetensors",
                "minimax_h3_ref2va_int8_convrot.safetensors",
            ],
        )
        self.assertEqual(
            nodes[5]["widgets_values"][6],
            "ltx-2.5-22b-ic-lora-pixel-spatial-upscaler-x2-1.0.safetensors",
        )
        self.assertEqual(
            nodes[1]["widgets_values"],
            [
                "orbital_storm_h3_relay_spectrum16",
                nodes[1]["widgets_values"][1],
                832,
                480,
                18,
                "euler",
                "beta57",
                True,
            ],
        )
        generators = sorted(
            (node for node in self.workflow["nodes"]
             if node["type"] == "H3RelayGenerateShot"),
            key=lambda node: node["title"],
        )
        self.assertEqual(len(generators), 4)
        for node in generators:
            widgets = node["widgets_values"]
            self.assertEqual(
                widgets, [424243, "fixed", 15.0, 16, 18, "match", ""]
            )
            names = [item["name"] for item in node["inputs"]]
            self.assertIn("output_crf", names)
            self.assertIn("shot_id", names)
            self.assertNotIn("shot_name", names)
        ltx_attention = next(
            node for node in self.workflow["nodes"]
            if node.get("title") ==
            "LTX ATTENTION · PYTORCH (SWAP OR BYPASS TO TEST)"
        )
        self.assertEqual(
            ltx_attention["widgets_values"], ["comfy kitchen attention"])
        for node_id in range(40, 44):
            self.assertEqual(
                nodes[node_id]["widgets_values"],
                [2, 18, 48],
            )
        for node_id in range(30, 34):
            self.assertEqual(
                nodes[node_id]["widgets_values"],
                [18, 193, 64, 128, 16],
            )

        loader_inputs = [item["name"] for item in nodes[5]["inputs"]]
        self.assertIn("latent_2x_model_name", loader_inputs)
        self.assertIn("pixel_upscale_ic_lora", loader_inputs)
        self.assertIn("manual_cache_revision", loader_inputs)
        self.assertNotIn("cache_revision", loader_inputs)
        for node_id in range(30, 34):
            names = [item["name"] for item in nodes[node_id]["inputs"]]
            self.assertIn("output_crf", names)
            self.assertNotIn("enhanced_crf", names)

    def test_widget_arrays_match_node_contract(self):
        expected = {
            "H3RelaySequenceStart": 8,
            "H3RelayGenerateShot": 7,
            "H3RelayEnhanceShot": 5,
            "H3RelayInterpolateShot": 3,
            "H3RelayAssemble": 3,
            "H3RelayCacheManager": 3,
        }
        for node in self.workflow["nodes"]:
            if node["type"] in expected:
                self.assertEqual(
                    len(node.get("widgets_values", [])), expected[node["type"]]
                )

    def test_reference_markdown_note_format(self):
        nodes = {node["id"]: node for node in self.workflow["nodes"]}
        expected = {
            60: "Note: H3 Relay",
            61: "Note: Model Links",
            77: "Note: Size Settings Reference",
        }
        for node_id, title in expected.items():
            note = nodes[node_id]
            self.assertEqual(note["type"], "MarkdownNote")
            self.assertEqual(note["title"], title)
            self.assertEqual(note["color"], "#222")
            self.assertEqual(note["bgcolor"], "#000")
            self.assertEqual(note["properties"], {})
        self.assertNotIn("Note", [node["type"] for node in self.workflow["nodes"]])

    def test_model_links_and_storage_note_is_complete(self):
        note = next(node for node in self.workflow["nodes"] if node["id"] == 61)
        text = note["widgets_values"][0]
        for value in (
            "## Model Links",
            "## Model Storage Location",
            "## Report Issue",
            "https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main/",
            "https://huggingface.co/Lightricks/LTX-2.5/resolve/main/",
            "minimax_h3_fl2va_int8_convrot.safetensors",
            "minimax_h3_ref2va_int8_convrot.safetensors",
            "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors",
            "gemma4-12b-with-proj-ltx-2.5-comfy-int8-convrot.safetensors",
            "ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors",
            "ltx-2.5-22b-ic-lora-pixel-spatial-upscaler-x2-1.0.safetensors",
            "rife_v4.26_heavy.safetensors",
            "📂 ComfyUI/",
            "📂 latent_upscale_models/",
            "📂 frame_interpolation/",
        ):
            self.assertIn(value, text)

    def test_public_example_uses_only_original_story_material(self):
        serialized = json.dumps(self.workflow).lower()
        self.assertIn("mara voss", serialized)
        self.assertIn("keon rell", serialized)
        self.assertIn("asteria", serialized)
        self.assertIn("orbital_storm_h3_relay_spectrum16", serialized)

    def test_overview_note_documents_sampling_controls(self):
        note = next(node for node in self.workflow["nodes"] if node["id"] == 60)
        text = note["widgets_values"][0]
        for value in (
            "**sampler**",
            "**scheduler**",
            "**spectrum_enabled**",
            "KSamplerSelect",
            "BasicScheduler",
            "Euler + beta57 + Spectrum enabled",
        ):
            self.assertIn(value, text)


if __name__ == "__main__":
    unittest.main()
