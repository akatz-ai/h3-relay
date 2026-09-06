import importlib.util
import pathlib
import sys
import tempfile
import unittest
from unittest.mock import patch


ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "benchmark"))
spec = importlib.util.spec_from_file_location(
    "release_validation", ROOT / "benchmark" / "validate_fast_h3_release.py"
)
validation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(validation)


class ReleaseValidationTest(unittest.TestCase):
    def test_one_shot_ultimate_does_not_silently_skip_finishing(self):
        submissions = []
        workflow = ROOT / "example_workflows" / "H3-Relay-FastH3-VSA-One-Shot-Ultimate-2x.json"
        import json
        prompt = validation.api_prompt_from_workflow(json.loads(workflow.read_text()), "test")
        schema = {node["class_type"]: {} for node in prompt.values()}
        schema.update(H3RelayInternalFastH3VSA={}, H3RelayAssemble={})

        def request(endpoint, path, body=None):
            if path == "/queue":
                return {"queue_running": [], "queue_pending": []}
            if path == "/object_info":
                return schema
            if path == "/system_stats":
                return {}
            if path == "/h3_relay/staged":
                submissions.append(body)
                return {"run_id": "test-stage"}
            if path == "/h3_relay/staged/test-stage":
                return {"status": "success", "current_stage": 3,
                        "current_prompt_id": None, "error": None, "jobs": []}
            raise AssertionError("Unexpected request: " + path)

        with tempfile.TemporaryDirectory() as directory:
            args = ["validate", "--endpoint", "http://test", "--workflow", str(workflow),
                    "--result-dir", str(pathlib.Path(directory) / "result"), "--run-name", "test"]
            with patch.object(sys, "argv", args), patch.object(validation, "request", request):
                self.assertEqual(validation.main(), 0)
        self.assertEqual(len(submissions), 1)
        body = submissions[0]
        target = body["prompt"][body["assemble_node_id"]]
        self.assertEqual(target["class_type"], "H3RelayAssemble")
        finisher = body["prompt"][target["inputs"]["enhanced"][0]]
        self.assertEqual(finisher["class_type"], "H3RelayUltimateEnhanceShot")


if __name__ == "__main__":
    unittest.main()
