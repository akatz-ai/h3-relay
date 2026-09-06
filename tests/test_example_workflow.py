import json
import importlib.util
import hashlib
import pathlib
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / "example_workflows" / "H3-Relay-Orbital-Storm-Spectrum16-58s.json"
FAST_WORKFLOW = (
    ROOT
    / "example_workflows"
    / "H3-Relay-FastH3-Akatz-3x10-Ultimate.json"
)
FAST_SIMPLE_WORKFLOW = (
    ROOT / "example_workflows" / "H3-Relay-FastH3-VSA-One-Shot.json"
)
FAST_ULTIMATE_WORKFLOW = (
    ROOT
    / "example_workflows"
    / "H3-Relay-FastH3-VSA-One-Shot-Ultimate-2x.json"
)
FAST_ASSETS = ROOT / "example_workflows" / "assets"
BENCHMARK = ROOT / "benchmark" / "run_saved_workflow_benchmark.py"


def benchmark_module():
    spec = importlib.util.spec_from_file_location("h3_relay_benchmark_fast", BENCHMARK)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


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

    def test_fast_h3_akatz_sequence_uses_ultimate_finish_stream(self):
        workflow = json.loads(FAST_WORKFLOW.read_text(encoding="utf-8"))
        types = [node["type"] for node in workflow["nodes"]]
        self.assertEqual(types.count("H3RelayFastH3VSAModelLoader"), 1)
        self.assertEqual(types.count("H3RelayGenerateShot"), 3)
        self.assertEqual(types.count("H3RelayUltimateEnhanceShot"), 3)
        self.assertEqual(types.count("H3RelayEnhanceShot"), 0)
        self.assertEqual(types.count("H3RelayInterpolateShot"), 3)
        self.assertEqual(types.count("H3RelayAssembleRaw"), 1)
        self.assertEqual(types.count("H3RelayAssemble"), 1)
        self.assertEqual(types.count("H3RelayH3HybridModelLoader"), 0)
        self.assertEqual(types.count("H3RelayLTXModelLoader"), 0)
        self.assertEqual(types.count("H3RelayAttention"), 0)
        self.assertEqual(types.count("LoadImage"), 5)

        nodes = {node["id"]: node for node in workflow["nodes"]}
        fast_profile = next(
            node for node in workflow["nodes"]
            if node["type"] == "H3RelayFastH3VSAModelLoader"
        )
        self.assertEqual(
            fast_profile["widgets_values"][0],
            "minimax_h3_fastvideo_vsa_datafree_1300step_4step_int8_convrot.safetensors",
        )
        sequence = next(
            node for node in workflow["nodes"]
            if node["type"] == "H3RelaySequenceStart"
        )
        self.assertEqual(sequence["widgets_values"][2:4], [832, 480])
        self.assertEqual(sequence["widgets_values"][6:8], ["simple", False])
        for node in workflow["nodes"]:
            if node["type"] == "H3RelayGenerateShot":
                self.assertEqual(node["widgets_values"][2:6], [10.0, 4, 18, "match"])
                input_names = [item["name"] for item in node["inputs"]]
                self.assertIn(
                    "additional_reference_images.reference_image_5",
                    input_names,
                )
            if node["type"] == "H3RelayUltimateEnhanceShot":
                self.assertEqual(
                    node["widgets_values"][2:],
                    [18, "match", 136, 17, 0.999, 1024, 1024, 128],
                )
                input_names = [item["name"] for item in node["inputs"]]
                self.assertEqual(input_names[:3], [
                    "h3_model", "sequence", "enhancement_prompt",
                ])
                self.assertIn("previous_enhanced", input_names)
                self.assertIn(
                    "additional_reference_images.reference_image_5",
                    input_names,
                )
        raw = next(
            node for node in workflow["nodes"]
            if node["type"] == "H3RelayAssembleRaw"
        )
        self.assertIsNotNone(raw["inputs"][0]["link"])
        serialized = json.dumps(workflow).lower()
        for value in (
            "akatz", "katana", "cyberpunk", "ginger soda", "ultimate",
            "akatz-character-sheet.png", "cyberpunk-stairwell.jpg",
        ):
            self.assertIn(value, serialized)
        for value in (
            "mara voss", "keon rell", "orbital storm", "walter white",
            "jesse pinkman", "breaking bad",
        ):
            self.assertNotIn(value, serialized)

        notes = {
            node["title"]: node["widgets_values"][0]
            for node in workflow["nodes"]
            if node["type"] == "MarkdownNote"
        }
        self.assertEqual(
            set(notes),
            {
                "READ FIRST · AKATZ 3×10 FASTH3 RELAY",
                "DIRECTOR MAP · REFERENCE ROLES",
                "Note: Model Links",
                "Note: H3 Relay",
                "Note: Size Settings Reference",
            },
        )
        for value in (
            "10febb01d7be73d1491cf5e5347b5ab8b6c2c09e",
            "comfy-kitchen==0.2.33",
            "6db8fa5a4e4ca0718d2ea8d08002ea899fe27721",
            "minimax_h3_latent_upscaler_3d_fp16.safetensors",
            "rife_v4.26_heavy.safetensors",
            "example_workflows/assets/",
        ):
            self.assertIn(value, json.dumps(notes))
        self.assertNotIn("LTX 2.5", json.dumps(notes))

    def test_fast_h3_examples_form_three_documented_tiers(self):
        simple = json.loads(FAST_SIMPLE_WORKFLOW.read_text(encoding="utf-8"))
        ultimate = json.loads(FAST_ULTIMATE_WORKFLOW.read_text(encoding="utf-8"))
        complex_workflow = json.loads(FAST_WORKFLOW.read_text(encoding="utf-8"))
        expected_counts = [
            (simple, 1, 0, 0, 0),
            (ultimate, 1, 1, 0, 0),
            (complex_workflow, 3, 3, 3, 5),
        ]
        for workflow, generators, finishers, rife, images in expected_counts:
            types = [node["type"] for node in workflow["nodes"]]
            self.assertEqual(types.count("H3RelayFastH3VSAModelLoader"), 1)
            self.assertEqual(types.count("H3RelaySequenceStart"), 1)
            self.assertEqual(types.count("H3RelayGenerateShot"), generators)
            self.assertEqual(
                types.count("H3RelayUltimateEnhanceShot"), finishers
            )
            self.assertEqual(types.count("H3RelayInterpolateShot"), rife)
            self.assertEqual(types.count("LoadImage"), images)
            self.assertGreaterEqual(types.count("MarkdownNote"), 3)
            notes = "\n".join(
                node["widgets_values"][0]
                for node in workflow["nodes"]
                if node["type"] == "MarkdownNote"
            )
            for required in (
                "10febb01d7be73d1491cf5e5347b5ab8b6c2c09e",
                "comfy-kitchen==0.2.33",
                "minimax_h3_fastvideo_vsa_datafree_1300step_4step_int8_convrot.safetensors",
                "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors",
                "minimax_h3_video_vae_fp16.safetensors",
                "minimax_h3_audio_vae_fp32.safetensors",
            ):
                self.assertIn(required, notes)
            self.assertNotIn("minimax_h3_fl2va_int8", notes)
            self.assertNotIn("ltx-2.5", notes.lower())
        simple_generator = next(
            node for node in simple["nodes"]
            if node["type"] == "H3RelayGenerateShot"
        )
        self.assertEqual(
            simple_generator["widgets_values"],
            [424246, "fixed", 5.0, 4, 18, "match", "fast_h3_vsa_one_shot"],
        )
        self.assertTrue(all(
            item["link"] is None
            for item in simple_generator["inputs"]
            if "reference_image" in item["name"]
        ))
        ultimate_notes = "\n".join(
            node["widgets_values"][0]
            for node in ultimate["nodes"]
            if node["type"] == "MarkdownNote"
        )
        self.assertIn(
            "minimax_h3_latent_upscaler_3d_fp16.safetensors",
            ultimate_notes,
        )
        self.assertNotIn("rife_v4.26_heavy.safetensors", ultimate_notes)

    def test_fast_h3_reference_assets_are_exact(self):
        expected = {
            "akatz-general-reference.png": "374b282a571c183fcfb18b9614bc1c84633a1c5bb7352d88cba4258e9ef94d5e",
            "akatz-character-sheet.png": "661984bed3ab547c30ce2a63f057363ea7d1760ef87d3ebb868e2e1cbb0bc545",
            "akatz-face-closeup.png": "078a546f3950c65430773297f6135cad33dbb27a7f2bed92a394587772066449",
            "cyberpunk-neon-street.jpg": "99e70e2f6fa2ce1eaad39fc2916f65ea5625ac939ac94912ab8a1379a498d7ae",
            "cyberpunk-stairwell.jpg": "2bc7c8fb84c757147590363920ad6a658a265f07d702b3e6d52fc4ea728d4880",
        }
        for filename, digest in expected.items():
            self.assertEqual(
                hashlib.sha256((FAST_ASSETS / filename).read_bytes()).hexdigest(),
                digest,
            )

    def test_fast_h3_akatz_workflow_converts_with_bypassed_rife(self):
        workflow = json.loads(FAST_WORKFLOW.read_text(encoding="utf-8"))
        prompt = benchmark_module().api_prompt_from_workflow(
            workflow, "fast_h3_akatz_conversion_test"
        )
        classes = [item["class_type"] for item in prompt.values()]
        self.assertEqual(classes.count("H3RelayGenerateShot"), 3)
        self.assertEqual(classes.count("H3RelayUltimateEnhanceShot"), 3)
        self.assertEqual(classes.count("H3RelayInterpolateShot"), 0)
        self.assertEqual(classes.count("LoadImage"), 5)
        for item in prompt.values():
            if item["class_type"] == "LoadImage":
                self.assertTrue(item["inputs"]["image"])
            if item["class_type"] == "H3RelayUltimateEnhanceShot":
                self.assertEqual(item["inputs"]["temporal_chunk_frames"], 136)
                self.assertEqual(item["inputs"]["tile_width"], 1024)
                self.assertIn(
                    "additional_reference_images.reference_image_5",
                    item["inputs"],
                )
        assembler = next(
            item for item in prompt.values()
            if item["class_type"] == "H3RelayAssemble"
        )
        final_ultimate = next(
            node for node in workflow["nodes"]
            if node["type"] == "H3RelayUltimateEnhanceShot"
            and node["title"].startswith("SHOT 3")
        )
        self.assertEqual(
            assembler["inputs"]["enhanced"], [str(final_ultimate["id"]), 0]
        )

    def test_fast_h3_focused_workflows_convert_without_hidden_canvas_stages(self):
        cases = [
            (FAST_SIMPLE_WORKFLOW, 4, 0),
            (FAST_ULTIMATE_WORKFLOW, 6, 1),
        ]
        for workflow_path, api_count, ultimate_count in cases:
            workflow = json.loads(workflow_path.read_text(encoding="utf-8"))
            prompt = benchmark_module().api_prompt_from_workflow(
                workflow, workflow_path.stem + "_conversion_test"
            )
            classes = [item["class_type"] for item in prompt.values()]
            self.assertEqual(len(prompt), api_count)
            self.assertEqual(classes.count("H3RelayGenerateShot"), 1)
            self.assertEqual(
                classes.count("H3RelayUltimateEnhanceShot"), ultimate_count
            )
            self.assertEqual(classes.count("LoadImage"), 0)
            self.assertEqual(classes.count("H3RelayInterpolateShot"), 0)


if __name__ == "__main__":
    unittest.main()
