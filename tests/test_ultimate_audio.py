import unittest

import torch

from h3_relay.nodes import _crop_audio_frames


class UltimateAudioTest(unittest.TestCase):
    def test_continuation_preserves_samples_and_pads_only_native_rounding_tail(self):
        waveform = torch.arange(74400, dtype=torch.float32).reshape(1, 1, -1)
        result = _crop_audio_frames(
            {"waveform": waveform, "sample_rate": 32000}, 1, 55, "test"
        )
        self.assertEqual(result["waveform"].shape[-1], 73333)
        self.assertTrue(torch.equal(result["waveform"][..., :-266], waveform[..., 1333:]))
        self.assertEqual(torch.count_nonzero(result["waveform"][..., -266:]).item(), 0)
        self.assertEqual(waveform.shape[-1], 74400)

    def test_truncated_native_decode_is_rejected(self):
        for samples in (74399, 73600, 1000):
            with self.subTest(samples=samples), self.assertRaisesRegex(ValueError, "expected at least"):
                _crop_audio_frames(
                    {"waveform": torch.ones(1, 2, samples), "sample_rate": 32000},
                    1, 55, "test",
                )

    def test_longer_native_decode_is_cropped_without_padding(self):
        waveform = torch.ones(1, 2, 97600)
        result = _crop_audio_frames(
            {"waveform": waveform, "sample_rate": 32000}, 0, 73, "test"
        )
        self.assertEqual(result["waveform"].shape[-1], 97333)
        self.assertTrue(torch.all(result["waveform"] == 1))


if __name__ == "__main__":
    unittest.main()
