import sys
import types
import unittest
from unittest import mock

import torch

from h3_relay import fast_h3_vsa


class FastH3VSATest(unittest.TestCase):
    def setUp(self):
        fast_h3_vsa._PLANS.clear()
        fast_h3_vsa.reset_runtime_stats()

    @staticmethod
    def _runtime_modules(*, capabilities=("sol_attn",)):
        kitchen = types.ModuleType("comfy_kitchen")
        kitchen.__path__ = []
        kitchen.sol_attn = object()
        kitchen.list_backends = lambda: {
            "cuda": {
                "available": True,
                "disabled": False,
                "unavailable_reason": None,
                "capabilities": list(capabilities),
            }
        }
        backends = types.ModuleType("comfy_kitchen.backends")
        backends.__path__ = []
        cuda = types.ModuleType("comfy_kitchen.backends.cuda")
        cuda.sol_attn_chunked = object()
        backends.cuda = cuda
        return {
            "comfy_kitchen": kitchen,
            "comfy_kitchen.backends": backends,
            "comfy_kitchen.backends.cuda": cuda,
        }, cuda

    def test_official_cuda_runtime_contract_is_accepted(self):
        modules, cuda = self._runtime_modules()
        with (
            mock.patch.dict(sys.modules, modules),
            mock.patch.object(
                fast_h3_vsa.importlib.metadata,
                "version",
                return_value="0.2.31",
            ),
            mock.patch.object(torch.cuda, "is_available", return_value=True),
            mock.patch.object(
                torch.cuda,
                "get_device_capability",
                return_value=(8, 9),
            ),
        ):
            self.assertIs(fast_h3_vsa.require_runtime(), cuda)

    def test_old_comfy_kitchen_fails_before_model_mutation(self):
        modules, _cuda = self._runtime_modules()
        with (
            mock.patch.dict(sys.modules, modules),
            mock.patch.object(
                fast_h3_vsa.importlib.metadata,
                "version",
                return_value="0.2.30",
            ),
        ):
            with self.assertRaisesRegex(RuntimeError, "requires comfy-kitchen"):
                fast_h3_vsa.require_runtime()

    def test_cuda_without_sol_attn_capability_is_rejected(self):
        modules, _cuda = self._runtime_modules(capabilities=())
        with (
            mock.patch.dict(sys.modules, modules),
            mock.patch.object(
                fast_h3_vsa.importlib.metadata,
                "version",
                return_value="0.2.31",
            ),
        ):
            with self.assertRaisesRegex(RuntimeError, "cannot use comfy-kitchen CUDA"):
                fast_h3_vsa.require_runtime()

    def test_vsa_plan_preserves_every_original_row_once(self):
        layout = types.SimpleNamespace(
            signature=(2, 2, 4, 4, 0),
            segments=[(0, 2, "text"), (2, 10, "video")],
            seq_len=10,
        )
        plan = fast_h3_vsa._vsa_plan(layout, torch.device("cpu"))
        self.assertEqual(plan["padded_rows"], 128)
        self.assertEqual(plan["prefix_blocks"], 1)
        self.assertEqual(plan["block_len"].tolist(), [2, 8])
        live = plan["source_rows"][plan["source_rows"] >= 0]
        self.assertEqual(sorted(live.tolist()), list(range(10)))
        self.assertEqual(plan["source_rows"][plan["inverse"]].tolist(), list(range(10)))

    def test_checkpoint_without_vsa_gate_is_rejected(self):
        attention = types.SimpleNamespace(qkv_proj=object(), head_dim=128)
        block = types.SimpleNamespace(attn=attention)
        diffusion = types.SimpleNamespace(
            rope_freqs=object(),
            _forward=object(),
            blocks=[block],
        )

        class Model:
            def get_model_object(self, name):
                self.requested_name = name
                return diffusion

        with mock.patch.object(fast_h3_vsa, "require_runtime", return_value=object()):
            with self.assertRaisesRegex(RuntimeError, "to_gate_compress"):
                fast_h3_vsa.apply_fast_h3_vsa(Model())


if __name__ == "__main__":
    unittest.main()
