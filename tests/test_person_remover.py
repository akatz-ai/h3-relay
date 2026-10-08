"""Timeline invariants for source-aligned removal, including awkward tails."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import torch

spec = importlib.util.spec_from_file_location('person_remover', Path(__file__).parents[1] / 'h3_relay/person_remover.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

class TimelineTests(unittest.TestCase):
    def test_expansion_keeps_history_separate_and_returns_cached_preview_identity(self):
        class Graph:
            def __init__(self):
                self.nodes = {}

            def node(self, class_type, **inputs):
                key = str(len(self.nodes))
                self.nodes[key] = {"class_type": class_type, "inputs": inputs}
                return SimpleNamespace(out=lambda slot: [key, slot])

            def finalize(self):
                return self.nodes

        cache = SimpleNamespace(SEED_SCHEME="h3-window-seed-v1",
            controls_for_configuration=lambda controls, *a: (controls, False),
            scope_key=lambda *a: "scope", configuration_key=lambda *a: "config",
            plan=lambda controls, scope, config, starts, size, count, seed, **kwargs:
                [{"seed": seed, "record_id": ""} for _ in starts])
        modules = {
            "comfy_execution.graph_utils": SimpleNamespace(GraphBuilder=Graph),
            "preview_contract.removal_previews": SimpleNamespace(begin_run=lambda *args, **kwargs: "preview-run-id"),
            "preview_contract": SimpleNamespace(removal_cache=cache),
        }
        with patch.dict("sys.modules", modules), patch.object(m, "__package__", "preview_contract"):
            result = m.H3RelayPersonRemover().generate(
                None, None, None, None, torch.zeros(77, 32, 32, 3), torch.zeros(1, 32, 32, 3),
                24., "remove", 123, 20, window_frames="39")
        nodes = list(result["expand"].values())
        self.assertEqual([n["inputs"]["length"] for n in nodes
                          if n["class_type"] == "MiniMaxH3ReferenceToVideo"], [39, 56])
        self.assertEqual(sum(n["class_type"] == "H3RelaySourceHistory" for n in nodes), 1)
        self.assertEqual([n["inputs"]["start"] for n in nodes
                          if n["class_type"] == "H3RelayRemovalAppend"], [0, 38])
        self.assertEqual(result["ui"], {"h3_removal_run": ["preview-run-id"]})
        self.assertEqual([n["inputs"]["noise_seed"] for n in nodes if n["class_type"] == "RandomNoise"], [123, 123])
        cache.plan = lambda *a, **kwargs: [{"seed": 123, "record_id": "saved-first"}, {"seed": 456, "record_id": ""}]
        with patch.dict("sys.modules", modules), patch.object(m, "__package__", "preview_contract"):
            result = m.H3RelayPersonRemover().generate(
                None, None, None, None, torch.zeros(77, 32, 32, 3), torch.zeros(1, 32, 32, 3),
                24., "remove", 123, 12, window_frames="39")
        nodes = list(result["expand"].values())
        self.assertEqual(sum(n["class_type"] == "H3RelayRemovalRestore" for n in nodes), 1)
        self.assertEqual(sum(n["class_type"] == "SamplerCustomAdvanced" for n in nodes), 1)
        self.assertEqual([n["inputs"]["length"] for n in nodes
                          if n["class_type"] == "MiniMaxH3ReferenceToVideo"], [56])
        self.assertEqual([n["inputs"]["noise_seed"] for n in nodes if n["class_type"] == "RandomNoise"], [456])

    def test_source_alignment_and_history(self):
        for rate in (32000, 48000):
            for count in (1, 21, 22, 23, 38, 39, 40, 41, 43, 44, 64, 124, 241):
                source = torch.arange(count).float().reshape(-1, 1, 1, 1).expand(-1, 1, 1, 3)
                previous = None
                for start in m.window_starts(count):
                    window, anchor, history, audio_history = m.H3RelayRemovalWindow().prepare(source, source[:1], start, previous)
                    self.assertEqual(len(window), 22)
                    if previous is not None:
                        self.assertEqual(anchor[0, 0, 0, 0], start)
                        self.assertEqual(history[:, 0, 0, 0].tolist(), list(range(start-17, start+1)))
                        self.assertEqual(audio_history['waveform'].shape[-1], round(18*rate/24))
                    audio = {'sample_rate': rate, 'waveform': torch.arange(round(start*rate/24), round((start+22)*rate/24)).reshape(1, 1, -1).float()}
                    previous, frames, joined_audio = m.H3RelayRemovalAppend().append(window, audio, start, count, previous)
                self.assertTrue(torch.equal(frames, source))
                self.assertTrue(torch.equal(joined_audio['waveform'].flatten(), torch.arange(round(count*rate/24))))
        self.assertEqual(m.window_starts(124), [0,21,42,63,84,102])

    def test_larger_windows_keep_exact_source_and_audio_positions(self):
        for size in (39, 56, 124, 362):
            for count in (size - 1, size, size + 1, size + 20, size * 2 - 1, size * 2 + 3):
                source = torch.arange(count).float().reshape(-1, 1, 1, 1).expand(-1, 1, 1, 3)
                for rate in (32000, 48000):
                    previous = None
                    for start in m.window_starts(count, size):
                        window, anchor, history, audio_history = m.H3RelayRemovalWindow().prepare(
                            source, source[:1], start, previous, window_frames=size)
                        self.assertEqual(len(window), size)
                        if previous is not None:
                            self.assertEqual(anchor[0, 0, 0, 0], start)
                            self.assertEqual(len(history), 18)
                            self.assertEqual(audio_history['waveform'].shape[-1], round(18 * rate / 24))
                        audio = {'sample_rate': rate, 'waveform': torch.arange(
                            round(start * rate / 24), round((start + size) * rate / 24)).reshape(1, 1, -1).float()}
                        previous, frames, audio = m.H3RelayRemovalAppend().append(
                            window, audio, start, count, previous, window_frames=size)
                    self.assertTrue(torch.equal(frames, source))
                    self.assertTrue(torch.equal(audio['waveform'].flatten(), torch.arange(round(count * rate / 24))))
        self.assertEqual(m.window_starts(77, 39), [0, 38])

    def test_invalid_window_sizes_are_rejected(self):
        for size in (0, 5, 21, 23, 40, 379):
            with self.assertRaises(ValueError):
                m.window_starts(124, size)

    def test_mask_preserves_unselected_pixels(self):
        source = torch.rand(2, 11, 11, 3)
        mask = torch.zeros(2, 11, 11)
        mask[:, 5, 5] = 1
        out, = m.H3RelayGreenMask().apply(source, mask, 1)
        selected = torch.zeros_like(mask, dtype=torch.bool)
        selected[:, 4:7, 4:7] = True
        self.assertTrue(torch.equal(source[~selected], out[~selected]))
        self.assertTrue(torch.equal(out[selected], torch.tensor([0.,1.,0.]).expand(18,3)))
        with self.assertRaises(ValueError):
            m.H3RelayGreenMask().apply(source, mask[:1], 1)

    def test_bad_decoded_length_rejected(self):
        with self.assertRaises(ValueError):
            m.H3RelayRemovalAppend().append(torch.zeros(21,1,1,3), {}, 0, 22)

if __name__ == '__main__':
    unittest.main()
