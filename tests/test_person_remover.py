"""Timeline invariants for source-aligned removal, including awkward tails."""
import importlib.util
from pathlib import Path
import unittest
import torch

spec = importlib.util.spec_from_file_location('person_remover', Path(__file__).parents[1] / 'h3_relay/person_remover.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

class TimelineTests(unittest.TestCase):
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
