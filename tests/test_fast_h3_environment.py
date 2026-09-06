import json
import pathlib
import re
import tomllib
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
RECIPE = ROOT / "comfygit_fast_h3_environment"


class FastH3EnvironmentTest(unittest.TestCase):
    def setUp(self):
        self.data = tomllib.loads((RECIPE / "pyproject.toml").read_text())

    def test_official_wheel_and_runtime_pins(self):
        self.assertIn("comfy-kitchen==0.2.33", self.data["project"]["dependencies"])
        overrides = self.data["tool"]["uv"]["override-dependencies"]
        self.assertIn("comfy-kitchen==0.2.33", overrides)
        for package in ("torch", "torchvision", "torchaudio"):
            self.assertIn(package, self.data["project"]["dependencies"])
        cg = self.data["tool"]["comfygit"]
        self.assertEqual(cg["comfyui_repository"], "https://github.com/kijai/ComfyUI.git")
        self.assertEqual(cg["comfyui_commit_sha"], "10febb01d7be73d1491cf5e5347b5ab8b6c2c09e")
        self.assertEqual(cg["comfyui_version_type"], "commit")
        self.assertEqual(set(cg["nodes"]), {"h3-relay", "mmh3-ultimate"})
        for node in cg["nodes"].values():
            self.assertTrue(node["repository"].startswith("https://github.com/"))
            self.assertRegex(node["version"], r"^[0-9a-f]{40}$")
        self.assertNotIn("file:", (RECIPE / "pyproject.toml").read_text())

    def test_workflows_match_packaged_examples(self):
        workflows = self.data["tool"]["comfygit"]["workflows"]
        self.assertEqual(len(workflows), 3)
        for name, config in workflows.items():
            path = RECIPE / config["path"]
            self.assertEqual(json.loads(path.read_text()), json.loads(
                (ROOT / "example_workflows" / (name + ".json")).read_text()))
            self.assertNotIn("sol-attn-minimax", config["nodes"])

    def test_models_have_immutable_public_sources(self):
        models = self.data["tool"]["comfygit"]["models"]
        self.assertEqual(len(models), 5)
        for digest, model in models.items():
            self.assertRegex(digest, r"^[0-9a-f]{16}$")
            self.assertGreater(model["size"], 0)
            self.assertEqual(pathlib.PurePosixPath(model["relative_path"]).name, model["filename"])
            for source in model["sources"]:
                self.assertTrue(re.match(r"https://huggingface.co/[^/]+/[^/]+/resolve/[0-9a-f]{40}/", source))


if __name__ == "__main__":
    unittest.main()
