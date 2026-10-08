"""Raw checkpoint fidelity, prefix dependency rules, and stale-lock rejection."""
import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

spec = importlib.util.spec_from_file_location("removal_cache", Path(__file__).parents[1] / "h3_relay/removal_cache.py")
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


class CacheTests(unittest.TestCase):
    def test_window_seed_derivation_is_stable_distinct_and_bounded(self):
        self.assertEqual([m.window_seed(904234, i) for i in range(3)],
                         [1384856719383077662, 2610881700782126544, 16844053075130660878])
        seeds = [m.window_seed(904234, i) for i in range(100)]
        self.assertEqual(len(set(seeds)), 100)
        self.assertTrue(all(0 <= s <= m.MAX_SEED for s in seeds))
        self.assertTrue(all(m.window_seed(904235, i) != seeds[i] for i in range(100)))

    def test_master_change_resets_overrides_while_same_master_keeps_rerolls(self):
        def key(master):
            return m.configuration_key(self.frames, self.frames[:1],
                {'master_seed': str(master), 'seed_scheme': m.SEED_SCHEME}, {}, '18')
        original = key(123)
        record = m.save(self.scope, original, 0, 0, 22, 43,
                        m.window_seed(123, 0), [], self.frames, self.audio)
        controls = json.dumps({'config_key': original, 'windows': {
            '0': {'locked': True, 'record_id': record['record_id'], 'seed': record['seed']},
            '1': {'seed': '987'}}})
        unchanged, reset = m.controls_for_configuration(controls, self.scope, key(123))
        self.assertFalse(reset)
        choices = m.plan(unchanged, self.scope, original, [0, 21], 22, 43, 123, derive_seeds=True)
        self.assertEqual(choices[0]['record_id'], record['record_id'])
        self.assertEqual(choices[1]['seed'], 987)
        fresh, reset = m.controls_for_configuration(controls, self.scope, key(124))
        self.assertTrue(reset)
        choices = m.plan(fresh, self.scope, key(124), [0, 21], 22, 43, 124, derive_seeds=True)
        self.assertEqual(choices, [{'seed': m.window_seed(124, i), 'record_id': ''} for i in range(2)])

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.directory = Path(self.temp.name)
        self.patch = patch.dict("sys.modules", {"folder_paths": SimpleNamespace(
            get_output_directory=lambda: self.temp.name, get_full_path=lambda *a: None)})
        self.patch.start()
        self.scope = m.scope_key("18", {"workflow": {"id": "one"}})
        self.frames = torch.rand(22, 32, 32, 3)
        self.audio = {"sample_rate": 32000, "waveform": torch.randn(1, 2, 29333)}

    def tearDown(self):
        self.patch.stop()
        self.temp.cleanup()

    def save(self, index=0, previous=()):
        return m.save(self.scope, "config", index, index * 21, 22, 64, 123 + index,
                      previous, self.frames, self.audio)

    def test_saved_window_is_exact_and_survives_module_reload(self):
        record = self.save()
        # Restore through a new module instance, without any in-memory cache.
        again = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(again)
        frames, audio = again.load(self.scope, record["record_id"], "config")
        self.assertTrue(torch.equal(frames, self.frames))
        self.assertTrue(torch.equal(audio["waveform"], self.audio["waveform"]))
        self.assertEqual(audio["sample_rate"], 32000)
        with self.assertRaisesRegex(ValueError, "different inputs"):
            m.load(self.scope, record["record_id"], "other-config")
        path = m._record_path(self.scope, record["record_id"], ".pt")
        path.write_bytes(b"damaged")
        with self.assertRaisesRegex(ValueError, "damaged"):
            m.load(self.scope, record["record_id"], "config")

    def test_changed_video_clears_legacy_locks_and_old_seed_overrides(self):
        record = self.save()
        old = json.dumps({'windows': {'0': {'locked': True, 'record_id': record['record_id']},
                                      '1': {'seed': '987'}}})
        normalized, reset = m.controls_for_configuration(old, self.scope, 'new-video')
        self.assertTrue(reset)
        self.assertEqual(json.loads(normalized), {'config_key': 'new-video', 'windows': {}})
        choices = m.plan(normalized, self.scope, 'new-video', [0, 21], 22, 43, 456)
        self.assertEqual(choices, [{'seed': 456, 'record_id': ''}] * 2)
        # Resetting the current set never deletes prior render artifacts.
        self.assertTrue(torch.equal(m.load(self.scope, record['record_id'], 'config')[0], self.frames))

    def test_same_video_keeps_exact_prefix_and_reroll_seed(self):
        record = self.save()
        old = json.dumps({'config_key': 'config', 'windows': {
            '0': {'locked': True, 'record_id': record['record_id']}, '1': {'seed': '987'}}})
        normalized, reset = m.controls_for_configuration(old, self.scope, 'config')
        self.assertFalse(reset)
        choices = m.plan(normalized, self.scope, 'config', [0, 21, 42], 22, 64, 456)
        self.assertEqual(choices[0]['record_id'], record['record_id'])
        self.assertEqual(choices[1]['seed'], 987)

    def test_bound_unlocked_controls_reset_after_settings_change(self):
        normalized, reset = m.controls_for_configuration(
            '{"config_key":"old-settings","windows":{"0":{"seed":"987"}}}',
            self.scope, 'new-settings')
        self.assertTrue(reset)
        self.assertEqual(json.loads(normalized)['windows'], {})
        normalized, reset = m.controls_for_configuration(
            '{"windows":{"0":{"seed":"987"}}}', self.scope, 'new-video', 'old-video')
        self.assertTrue(reset)
        self.assertEqual(json.loads(normalized)['windows'], {})

    def test_auto_reset_does_not_mask_missing_checkpoints_or_bad_controls(self):
        with self.assertRaisesRegex(ValueError, 'unavailable'):
            m.controls_for_configuration(json.dumps({'windows': {
                '0': {'locked': True, 'record_id': 'a'*32}}}), self.scope, 'new')
        for bad in ('[]', '{"windows":[]}', '{"windows":{"0":null}}'):
            with self.assertRaises(ValueError):
                m.controls_for_configuration(bad, self.scope, 'config')

    def test_prefix_locking_preserves_seed_and_rejects_changed_history(self):
        first = self.save()
        second = self.save(1, [first["record_id"]])
        data = {"windows": {"0": {"locked": True, "record_id": first["record_id"]},
                            "1": {"locked": True, "record_id": second["record_id"]},
                            "2": {"seed": "18446744073709551615"}}}
        choices = m.plan(json.dumps(data), self.scope, "config", [0, 21, 42], 22, 64, 999)
        self.assertEqual([c["seed"] for c in choices], [123, 124, m.MAX_SEED])
        self.assertEqual([c["record_id"] for c in choices], [first["record_id"], second["record_id"], ""])
        with self.assertRaisesRegex(ValueError, "different inputs"):
            m.plan(json.dumps(data), self.scope, "changed", [0, 21, 42], 22, 64, 999)
        data["windows"]["0"]["locked"] = False
        with self.assertRaisesRegex(ValueError, "earlier windows"):
            m.plan(json.dumps(data), self.scope, "config", [0, 21, 42], 22, 64, 999)

    def test_input_pixels_settings_and_model_versions_invalidate_locks(self):
        graph = {"18": {"inputs": {"model": ["1", 0]}},
                 "1": {"class_type": "UNETLoader", "inputs": {"unet_name": "model.safetensors"}}}
        model = self.directory / "model.safetensors";model.write_bytes(b"first")
        import folder_paths
        folder_paths.get_full_path = lambda *a: str(model)
        settings = {"steps": 12, "window_frames": 22}
        initial = m.configuration_key(self.frames, self.frames[:1], settings, graph, "18")
        self.assertEqual(initial, m.configuration_key(self.frames, self.frames[:1], settings, graph, "18"))
        changed = self.frames.clone();changed[5, 0, 0, 0] += .1
        self.assertNotEqual(initial, m.configuration_key(changed, self.frames[:1], settings, graph, "18"))
        self.assertNotEqual(initial, m.configuration_key(self.frames, self.frames[:1], {"steps": 20}, graph, "18"))
        model.write_bytes(b"second model")
        self.assertNotEqual(initial, m.configuration_key(self.frames, self.frames[:1], settings, graph, "18"))

    def test_other_workflows_and_path_inputs_cannot_restore_checkpoint(self):
        record = self.save()
        other = m.scope_key("18", {"workflow": {"id": "two"}})
        with self.assertRaisesRegex(ValueError, "unavailable"):
            m.load(other, record["record_id"], "config")
        for value in ("../outside", "", "x" * 32):
            with self.assertRaises(ValueError):m.metadata(self.scope, value)
        with self.assertRaises(ValueError):m.metadata("../outside", record["record_id"])


if __name__ == "__main__":
    unittest.main()
