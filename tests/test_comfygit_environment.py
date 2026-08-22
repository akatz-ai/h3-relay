import pathlib
import tomllib
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
ENVIRONMENT = ROOT / "comfygit_environment"
WORKFLOW_NAME = "H3-Relay-Orbital-Storm-Spectrum16-58s"


class ComfyGitEnvironmentTest(unittest.TestCase):
    def setUp(self):
        self.manifest = tomllib.loads(
            (ENVIRONMENT / "pyproject.toml").read_text(encoding="utf-8")
        )
        self.comfygit = self.manifest["tool"]["comfygit"]

    def test_pins_runtime_and_h3_relay_commits(self):
        self.assertEqual(
            self.comfygit["comfyui_commit_sha"],
            "1c6d8d45b3693bfbb32385b410d813a7fd6be216",
        )
        self.assertEqual(
            self.comfygit["nodes"]["h3-relay"]["version"],
            "50cfe4ce38726d1590247bcc49ea8edf3bbd6081",
        )

    def test_declares_twelve_required_sourced_models(self):
        workflow = self.comfygit["workflows"][WORKFLOW_NAME]
        models = workflow["models"]
        catalog = self.comfygit["models"]

        self.assertEqual(len(models), 12)
        self.assertTrue(all(model["criticality"] == "required" for model in models))
        self.assertTrue(all(model["status"] == "resolved" for model in models))
        self.assertTrue(all(model["hash"] in catalog for model in models))
        self.assertTrue(all(catalog[model["hash"]]["sources"] for model in models))

    def test_uses_official_downloaded_model_identities(self):
        catalog = self.comfygit["models"]

        self.assertEqual(
            catalog["b14d650b0a0e9d3a"]["filename"],
            "minimax_h3_video_vae_fp16.safetensors",
        )
        self.assertEqual(
            catalog["d4534c86defbf0c2"]["filename"],
            "gemma4-12b-with-proj-ltx-2.5-comfy-int8-convrot.safetensors",
        )

    def test_packaged_workflow_matches_public_example(self):
        packaged = ENVIRONMENT / "workflows" / f"{WORKFLOW_NAME}.json"
        public = ROOT / "example_workflows" / f"{WORKFLOW_NAME}.json"

        self.assertEqual(packaged.read_bytes(), public.read_bytes())


if __name__ == "__main__":
    unittest.main()
