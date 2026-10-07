"""Prepared-mask persistence and input invalidation without executing SAM."""
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

import torch

PACKAGE = "prepared_mask_test"
DIRECTORY = Path(__file__).parents[1] / "h3_relay"
package = ModuleType(PACKAGE)
package.__path__ = [str(DIRECTORY)]


class PreparedMaskTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.directory = Path(self.temp.name)
        (self.directory / "source.mp4").write_bytes(b"video first")
        (self.directory / "sam.safetensors").write_bytes(b"model first")
        self.patch = patch.dict(sys.modules, {PACKAGE: package, "folder_paths": SimpleNamespace(
            get_output_directory=lambda: self.temp.name,
            get_annotated_filepath=lambda name: str(self.directory / name),
            get_full_path=lambda category, name: str(self.directory / name))})
        self.patch.start()
        self.cache = __import__(PACKAGE + ".removal_inputs", fromlist=["removal_inputs"])
        self.node = __import__(PACKAGE + ".person_remover", fromlist=["person_remover"]).H3RelayGreenMask()
        self.graph = {
            "1": {"class_type": "LoadVideo", "inputs": {"file": "source.mp4"}},
            "2": {"class_type": "GetVideoComponents", "inputs": {"video": ["1", 0]}},
            "3": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": "sam.safetensors"}},
            "4": {"class_type": "CLIPTextEncode", "inputs": {"text": "person", "clip": ["3", 1]}},
            "5": {"class_type": "SAM3_VideoTrack", "inputs": {"images": ["2", 0], "model": ["3", 0],
                        "conditioning": ["4", 0], "max_objects": 1}},
            "6": {"class_type": "SAM3_TrackToMask", "inputs": {"track_data": ["5", 0], "object_indices": ""}},
            "7": {"class_type": "H3RelayGreenMask", "inputs": {"images": ["2", 0], "mask": ["6", 0], "expand_pixels": 1}},
            "8": {"class_type": "H3RelayPersonRemover", "inputs": {"seed": 123, "window_controls": "{}"}},
        }
        self.hidden = dict(unique_id="7", extra_pnginfo={"workflow": {"id": "test-workflow"}}, execution_prompt=self.graph)

    def tearDown(self):
        self.patch.stop()
        self.temp.cleanup()

    def test_lazy_reroll_restores_exact_prepared_pixels_after_reload(self):
        frames = torch.rand(3, 32, 32, 3)
        mask = torch.zeros(3, 32, 32)
        mask[:, 10:20, 10:20] = 1
        self.assertEqual(self.node.check_lazy_status(None, None, 1, **self.hidden), ["images", "mask"])
        expected, = self.node.apply(frames, mask, 1, **self.hidden)
        fingerprint = self.node.IS_CHANGED(expand_pixels=1, **self.hidden)
        self.graph["8"]["inputs"] = {"seed": 999, "window_controls": "changed", "steps": 20}
        self.assertEqual(fingerprint, self.node.IS_CHANGED(expand_pixels=1, **self.hidden))
        # No source/mask tensors available: the tracker is not needed at all.
        self.assertEqual(self.node.check_lazy_status(None, None, 1, **self.hidden), [])
        restored, = self.node.apply(None, None, 1, **self.hidden)
        self.assertTrue(torch.equal(restored, expected))
        import importlib
        importlib.reload(self.cache)
        restored, = self.node.apply(None, None, 1, **self.hidden)
        self.assertTrue(torch.equal(restored, expected))

    def test_source_bytes_mask_settings_and_model_changes_require_tracking_again(self):
        signature = lambda: self.cache.signature(self.graph, "7", 1)
        original = signature()
        # A same-size replacement still changes identity.
        (self.directory / "source.mp4").write_bytes(b"video other")
        self.assertNotEqual(original, signature())
        (self.directory / "source.mp4").write_bytes(b"video first")
        self.assertEqual(original, signature())
        for node, field, value in (("4", "text", "different person"), ("5", "max_objects", 2),
                                   ("6", "object_indices", "1")):
            old = self.graph[node]["inputs"][field]
            self.graph[node]["inputs"][field] = value
            self.assertNotEqual(original, signature())
            self.graph[node]["inputs"][field] = old
        self.assertNotEqual(original, self.cache.signature(self.graph, "7", 5))
        (self.directory / "sam.safetensors").write_bytes(b"model changed")
        self.assertNotEqual(original, signature())

    def test_unsupported_upstream_and_other_workflows_cannot_reuse_prepared_masks(self):
        frames = torch.rand(3, 32, 32, 3)
        self.node.apply(frames, torch.zeros(3, 32, 32), 1, **self.hidden)
        other = dict(self.hidden, extra_pnginfo={"workflow": {"id": "other"}})
        self.assertEqual(self.node.check_lazy_status(None, None, 1, **other), ["images", "mask"])
        self.graph["1"]["class_type"] = "ExternalMutableSource"
        self.assertIsNone(self.cache.signature(self.graph, "7", 1))
        self.assertEqual(self.node.check_lazy_status(None, None, 1, **self.hidden), ["images", "mask"])

    def test_damaged_prepared_payload_is_rejected(self):
        self.node.apply(torch.zeros(3, 32, 32, 3), torch.zeros(3, 32, 32), 1, **self.hidden)
        tensor = next(self.directory.rglob("prepared/**/*.pt"))
        tensor.write_bytes(b"damaged")
        with self.assertRaisesRegex(ValueError, "damaged"):
            self.node.apply(None, None, 1, **self.hidden)


if __name__ == "__main__":
    unittest.main()
