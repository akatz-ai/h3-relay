"""Public character-swap workflow compatibility and real CPU video preparation."""
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

import torch
from comfy_api.latest._input_impl.video_types import VideoFromFile
from h3_relay import NODE_CLASS_MAPPINGS
from h3_relay.video_prep import (
    LabsH3PrepareVideo, LabsH3TrimOutput, canvas_dimensions, display_dimensions,
    requested_window,
)


class VideoPrepTests(unittest.TestCase):
    def test_distributed_workflow_contract(self):
        self.assertIs(NODE_CLASS_MAPPINGS['LabsH3PrepareVideo'], LabsH3PrepareVideo)
        self.assertIs(NODE_CLASS_MAPPINGS['LabsH3TrimOutput'], LabsH3TrimOutput)
        inputs = LabsH3PrepareVideo.INPUT_TYPES()
        self.assertEqual(list(inputs['required']), ['video', 'resolution'])
        self.assertEqual(inputs['required']['resolution'][0], ['480', '768'])
        self.assertEqual(list(inputs['optional']), ['duration_seconds'])
        self.assertEqual(LabsH3PrepareVideo.RETURN_NAMES, (
            'reference_frames', 'source_audio', 'width', 'height', 'h3_length',
            'source_frames', 'fps', 'summary', 'prepared_video'))
        self.assertEqual(LabsH3PrepareVideo.RETURN_TYPES,
                         ('IMAGE', 'AUDIO', 'INT', 'INT', 'INT', 'INT', 'FLOAT', 'STRING', 'VIDEO'))

    def test_pixel_budget_rotation_and_duration(self):
        for resolution, cap in [('480', 500000), ('768', 1000000)]:
            for width, height in [(1920, 1080), (1080, 1920), (640, 640), (320, 240)]:
                w, h = canvas_dimensions(width, height, resolution)
                self.assertEqual((w % 32, h % 32), (0, 0))
                self.assertLessEqual(w * h, cap)
                self.assertGreater(w * h, cap * .85)
        self.assertEqual(display_dimensions({'width': 720, 'height': 480,
            'sample_aspect_ratio': '4:3', 'side_data_list': [{'rotation': 90}]}), (480, 960))
        self.assertEqual([requested_window(x) for x in [0, 5, 6, 7]], [None, 124, 141, 175])
        for invalid in [-1, float('nan'), float('inf'), 150]:
            with self.assertRaises(ValueError):
                requested_window(invalid)

    def test_output_trimming_and_one_short_decode(self):
        images = torch.arange(22 * 3).reshape(22, 1, 1, 3)
        self.assertTrue(torch.equal(LabsH3TrimOutput().trim(images, 20)[0], images[:20]))
        held = LabsH3TrimOutput().trim(images, 23)[0]
        self.assertEqual(held.shape[0], 23)
        self.assertTrue(torch.equal(held[-1], images[-1]))
        with self.assertRaises(ValueError):
            LabsH3TrimOutput().trim(images, 24)

    def prepare_fixture(self, seconds, requested, audio):
        # Missing FFmpeg is a real packaging/test failure, not a skipped contract.
        self.assertIsNotNone(shutil.which('ffmpeg'))
        self.assertIsNotNone(shutil.which('ffprobe'))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'source.mkv'
            cmd = ['ffmpeg', '-v', 'error', '-nostdin', '-y', '-f', 'lavfi',
                   '-i', 'testsrc2=size=96x160:rate=24']
            if audio:
                cmd += ['-f', 'lavfi', '-i', 'sine=frequency=440:sample_rate=24000']
            cmd += ['-t', str(seconds), '-c:v', 'ffv1', '-threads', '1']
            if audio:
                cmd += ['-c:a', 'pcm_s16le']
            subprocess.run(cmd + [str(path)], check=True, capture_output=True, timeout=30)
            return LabsH3PrepareVideo().prepare(VideoFromFile(str(path)), '480', requested)

    def test_full_clip_audio_and_padding(self):
        frames, audio, w, h, length, count, fps, summary, preview = self.prepare_fixture(1, 0, True)
        self.assertEqual((count, length, fps), (24, 39, 24))
        self.assertEqual(frames.shape, (39, h, w, 3))
        self.assertTrue(torch.equal(frames[-1], frames[23]))
        self.assertEqual(preview.get_components().images.shape[0], 24)
        self.assertEqual(audio['waveform'].shape[-1], audio['sample_rate'])
        self.assertIn('full clip', summary)

    def test_requested_duration_trims_picture_and_audio(self):
        frames, audio, w, h, length, count, fps, summary, preview = self.prepare_fixture(2, 1, True)
        self.assertEqual((length, count), (22, 22))
        self.assertEqual(audio['waveform'].shape[-1], round(22 / 24 * audio['sample_rate']))
        self.assertEqual(preview.get_components().images.shape[0], 22)
        self.assertIn('Requested 1s', summary)

    def test_short_silent_source_is_not_extended(self):
        frames, audio, w, h, length, count, fps, summary, preview = self.prepare_fixture(.5, 5, False)
        self.assertEqual((count, length), (12, 22))
        self.assertIsNone(audio)
        self.assertEqual(preview.get_components().images.shape[0], 12)
        self.assertIn('Source ends earlier', summary)
