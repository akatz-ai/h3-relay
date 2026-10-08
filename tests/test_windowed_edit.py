"""Source alignment, configurable history, and shared preview/reroll contracts."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import torch

ROOT = Path(__file__).parents[1]
def load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / 'h3_relay' / (name + '.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
m = load('windowed_edit')
remover = load('person_remover')


class WindowedEditTests(unittest.TestCase):
    def test_source_and_audio_alignment_at_every_boundary(self):
        for size in (22, 39, 56, 124):
            for history in (h for h in m.HISTORY if h < size):
                for count in (1, size-1, size, size+1, size+17, size*2-1, size*2+3, 240):
                    source = torch.arange(count).float().reshape(-1, 1, 1, 1).expand(-1, 1, 1, 3)
                    rate = 240
                    source_audio = {'sample_rate': rate, 'waveform': torch.arange(count*10).reshape(1, 1, -1).float()}
                    previous = None
                    plan = m.plan_windows(count, size, history)
                    for entry in plan:
                        start = entry['source_start']
                        frames, prior, prior_audio, ref_audio = m.H3RelayEditWindow().prepare(
                            source, start, size, history, previous, source_audio)
                        self.assertEqual(frames[0, 0, 0, 0].item(), start)
                        self.assertEqual(ref_audio['waveform'][0, 0, 0].item(), start*10)
                        if previous is not None and history:
                            self.assertEqual(prior[:, 0, 0, 0].tolist(), list(range(start-history+1, start+1)))
                            self.assertEqual(prior_audio['waveform'].flatten().tolist(), list(range((start-history+1)*10, (start+1)*10)))
                        audio = {'sample_rate': rate, 'waveform': torch.arange(start*10, (start+size)*10).reshape(1, 1, -1).float()}
                        previous, assembled, audio = remover.H3RelayRemovalAppend().append(frames, audio, start, count, previous, size)
                    self.assertTrue(torch.equal(assembled, source), (size, history, count))
                    self.assertTrue(torch.equal(audio['waveform'], source_audio['waveform']))
                    self.assertEqual(sum(e['delivered_frames'] for e in plan), count)
        self.assertEqual([e['source_start'] for e in m.plan_windows(240)], [0, 116])

    def test_invalid_shapes_fail_clearly(self):
        for size, history in ((120, 18), (124, 17), (22, 35), (39, 39)):
            with self.assertRaises(ValueError): m.validate_settings(size, history)
        with self.assertRaises(ValueError): m.plan_windows(0)

    def test_expansion_preserves_reference_and_supports_preview_prefix_reroll(self):
        class Graph:
            def __init__(self): self.nodes = {}
            def node(self, kind, **inputs):
                key = str(len(self.nodes)); self.nodes[key] = {'class_type': kind, 'inputs': inputs}
                return SimpleNamespace(out=lambda slot: [key, slot])
            def finalize(self): return self.nodes
        captured = []
        def config(source, ref, settings, *args): captured.append(settings); return 'config'
        cache = SimpleNamespace(scope_key=lambda *a: 'scope', configuration_key=config,
            tensor_hash=lambda t: 'audio-hash',
            plan=lambda *a: [{'seed': 123, 'record_id': ''}, {'seed': 456, 'record_id': ''}])
        modules = {'comfy_execution.graph_utils': SimpleNamespace(GraphBuilder=Graph),
            'edit_contract': SimpleNamespace(removal_cache=cache),
            'edit_contract.removal_previews': SimpleNamespace(begin_run=lambda *a: 'preview-id'),
            'edit_contract.vendor.context_loop.sliding_context': SimpleNamespace(require_sliding_history_support=lambda: None)}
        def run():
            with patch.dict('sys.modules', modules), patch.object(m, '__package__', 'edit_contract'):
                return m.H3RelayWindowedEdit().generate('model', 'clip', 'video-vae', 'audio-vae',
                    torch.zeros(240, 32, 32, 3), torch.zeros(1, 32, 32, 3), 24., 'swap')
        result = run(); nodes = list(result['expand'].values())
        refs = [n['inputs'] for n in nodes if n['class_type'] == 'MiniMaxH3ReferenceToVideo']
        self.assertEqual([r['length'] for r in refs], [124, 141])
        self.assertTrue(all(r['vae'] == 'video-vae' and 'ref_images.ref_image_0' in r for r in refs))
        self.assertFalse(any(n['class_type'] == 'MiniMaxH3AddGuide' for n in nodes))
        self.assertEqual(result['ui'], {'h3_removal_run': ['preview-id']})
        self.assertEqual(captured[0]['history_frames'], 18)
        self.assertEqual(captured[0]['kind'], 'windowed_ref_edit_v1')
        cache.plan = lambda *a: [{'seed': 123, 'record_id': 'locked-first'}, {'seed': 789, 'record_id': ''}]
        nodes = list(run()['expand'].values())
        self.assertEqual(sum(n['class_type'] == 'H3RelayRemovalRestore' for n in nodes), 1)
        self.assertEqual([n['inputs']['noise_seed'] for n in nodes if n['class_type'] == 'RandomNoise'], [789])
        self.assertEqual([n['inputs']['start'] for n in nodes if n['class_type'] == 'H3RelayEditAppend'], [0, 116])
        self.assertTrue(all(n['inputs']['preview_run'] == 'preview-id' for n in nodes if n['class_type'] == 'H3RelayEditAppend'))

    def test_edit_append_forwards_preview_and_exact_cache_identity(self):
        seen = []
        delegate = SimpleNamespace(append=lambda *a, **kw: (seen.append(kw) or {}, a[0], a[1]))
        with patch.dict('sys.modules', {'edit_contract.person_remover': SimpleNamespace(H3RelayRemovalAppend=lambda: delegate)}), patch.object(m, '__package__', 'edit_contract'):
            _, frames, _ = m.H3RelayEditAppend().append(torch.zeros(123, 1, 1, 3), {}, 0, 124, 124,
                preview_run='run', cache_scope='scope', config_key='config', window_seed='456', reuse_record='record')
        self.assertEqual(len(frames), 124)
        self.assertEqual(seen[0]['reuse_record'], 'record')
        self.assertEqual(seen[0]['preview_run'], 'run')


if __name__ == '__main__': unittest.main()
