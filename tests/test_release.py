import ast
import pathlib
import tomllib
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]


class ReleaseMetadataTest(unittest.TestCase):
    def test_registry_publish_runs_from_matching_version_tags(self):
        workflow = (
            ROOT / ".github" / "workflows" / "publish_action.yml"
        ).read_text(encoding="utf-8")
        self.assertIn('      - "v*.*.*"', workflow)
        self.assertIn("workflow_dispatch:", workflow)
        self.assertIn("Verify tag matches package version", workflow)
        self.assertIn('test "v${package_version}" = "${GITHUB_REF_NAME}"', workflow)

    def test_generate_shot_seed_declares_frontend_control(self):
        source = (ROOT / "h3_relay" / "nodes.py").read_text(encoding="utf-8")
        class_source = source.split("class H3RelayGenerateShot", 1)[1]
        class_source = class_source.split("class H3Relay", 1)[0]
        self.assertIn('"control_after_generate": True', class_source)

    def test_registry_identity(self):
        metadata = tomllib.loads((ROOT / "pyproject.toml").read_text())
        self.assertEqual(metadata["project"]["name"], "h3-relay")
        self.assertEqual(metadata["project"]["version"], "1.0.2")
        self.assertEqual(metadata["tool"]["comfy"]["PublisherId"], "akatz")
        self.assertEqual(
            metadata["tool"]["comfy"]["Icon"],
            "https://i.imgur.com/aiqQI7U.png",
        )
        self.assertEqual(
            metadata["tool"]["comfy"]["requires-comfyui"], ">=0.32.0"
        )

    def test_runtime_python_has_no_dynamic_execution_calls(self):
        roots = [ROOT / "h3_relay", ROOT / "__init__.py"]
        violations = []
        paths = []
        for root in roots:
            paths.extend(root.rglob("*.py") if root.is_dir() else [root])
        for path in paths:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for item in ast.walk(tree):
                if (
                    isinstance(item, ast.Call)
                    and isinstance(item.func, ast.Name)
                    and item.func.id in {"eval", "exec"}
                ):
                    violations.append("%s:%d" % (path, item.lineno))
        self.assertEqual(violations, [])


if __name__ == "__main__":
    unittest.main()
