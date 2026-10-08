"""Latent relay timing, graph barriers and deferred assembly contracts."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import torch

ROOT = Path(__file__).parents[1] / 'h3_relay'
def load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / (name+'.py'))
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod); return mod
m, edit, cache = load('windowed_latent'), load('windowed_edit'), load('removal_cache')

class LatentTests(unittest.TestCase):
    def setUp(self):
        self.patches = patch.dict('sys.modules', {
            'latent_test.windowed_edit': edit,
            'latent_test.removal_cache': cache,
            'latent_test.vendor.context_loop.nodes': SimpleNamespace(_streams_from_latent=lambda z: z['samples']),
            'latent_test.vendor.context_loop.sliding_context': SimpleNamespace(require_sliding_history_support=lambda: None),
        })
        self.patches.start(); self.package = patch.object(m, '__package__', 'latent_test'); self.package.start()
    def tearDown(self):
        self.package.stop(); self.patches.stop()

    def test_aligned_plan_exact_coverage_and_history_for_all_supported_sizes(self):
        for window in edit.WINDOWS:
            for history in [h for h in edit.HISTORY if h < window]:
                for count in (1,window-1,window,window+1,window*2-1,window*2+1):
                    plan=m.plan_latent_windows(count,window,history)
                    self.assertEqual(sum(e['delivered_frames'] for e in plan),count)
                    assembled=[]
                    for e in plan:
                        start=e['source_start']; self.assertGreater(e['delivered_frames'],0)
                        assembled.extend(range(start+e['discarded_overlap'],e['source_end_exclusive']))
                        if e['index'] and history:
                            self.assertEqual(start%17,0)
                            self.assertEqual(e['discarded_overlap'],5)
                            self.assertEqual(e['history_start']%17,0)
                            self.assertGreaterEqual(e['history_start'],plan[e['index']-1]['source_start'])
                    self.assertEqual(assembled,list(range(count)))
        self.assertEqual([e['source_start'] for e in m.plan_latent_windows(240,124,18)],[0,119])
        self.assertEqual([e['source_start'] for e in m.plan_latent_windows(43,22,18)],[0,17,34])

    def test_direct_latent_history_selects_phase_aligned_tokens_and_audio(self):
        for history in (1,18,35,103):
            video=torch.arange(37.).reshape(1,1,37,1,1)
            audio=torch.arange(207.).reshape(1,1,1,207)
            prev=({'start':0,'latent':{'samples':(video,audio)}},)
            latent=object(); conditioning=[[object(),{'sentinel':True}]]
            out,z=m.H3RelayLatentHistory().apply(conditioning,latent,prev,124,history,119)
            self.assertIs(z,latent); self.assertNotIn('minimax_keyframes',conditioning[0][1])
            k=out[0][1]['minimax_keyframes']
            boundary=next(x['latent'] for x in k if x['anchor']=='first' and 'latent' in x)
            self.assertEqual(boundary.flatten().tolist(),[35.]) # source frame 119
            if history>1:
                hist=next(x['latent'] for x in k if x['anchor']=='history' and 'latent' in x)
                self.assertEqual(hist.flatten().tolist(),list(range(35-(history-1)//17*5,35)))
            first_audio=next(x['audio_latent'] for x in k if x['anchor']=='first' and 'audio_latent' in x)
            self.assertEqual(first_audio.flatten().tolist(),[198.,199.])
        with self.assertRaisesRegex(ValueError,'regular aligned'):
            m.H3RelayLatentHistory().apply(conditioning,latent,prev,124,18,116)

    def test_graph_precomputes_all_conditioning_and_defers_decode(self):
        class Graph:
            def __init__(self): self.nodes={}
            def node(self,kind,**inputs):
                key=str(len(self.nodes)); self.nodes[key]={'class_type':kind,'inputs':inputs}
                return SimpleNamespace(out=lambda slot:[key,slot])
            def finalize(self): return self.nodes
        with patch.dict('sys.modules',{'comfy_execution.graph_utils':SimpleNamespace(GraphBuilder=Graph)}):
            out,graph=m.build_latent_graph('model','clip','vv','av','source','image','prompt',
                m.plan_latent_windows(240,124,18),124,18,240,512,928,904234,8,1,'er_sde','simple',1,None)
        def ancestors(key):
            keys=set()
            for v in graph[key]['inputs'].values():
                if isinstance(v,list) and len(v)==2 and v[0] in graph:
                    keys.add(v[0]);keys.update(ancestors(v[0]))
            return keys
        refs={k for k,v in graph.items() if v['class_type']=='MiniMaxH3ReferenceToVideo'}
        samples={k for k,v in graph.items() if v['class_type']=='SamplerCustomAdvanced'}
        self.assertEqual(len(refs),2)
        for k in samples: self.assertTrue(refs <= ancestors(k))
        self.assertTrue(samples <= ancestors(out[0]))
        self.assertEqual(graph[out[0]]['class_type'],'H3RelayLatentDecode')
        self.assertFalse(any(v['class_type'] in ['VAEDecode','VAEDecodeAudio','H3RelayEditAppend'] for v in graph.values()))
        self.assertEqual([v['inputs']['noise_seed'] for v in graph.values() if v['class_type']=='RandomNoise'],[904234,904235])

    def test_final_decode_trims_overlap_padding_and_checks_shape(self):
        calls=[]
        def decode(z):
            calls.append(z);return z
        windows=tuple({'start':s,'latent':{'samples':(torch.arange(s,s+22).reshape(22,1,1,1),None)}} for s in (0,17,34))
        management=SimpleNamespace(throw_exception_if_processing_interrupted=lambda:None)
        comfy=SimpleNamespace(model_management=management)
        with patch.dict('sys.modules',{'comfy':comfy,'comfy.model_management':management}):
            frames,=m.H3RelayLatentDecode().decode(windows,SimpleNamespace(decode=decode),22,43)
        self.assertEqual(frames.flatten().tolist(),list(range(43)))
        self.assertEqual(len(calls),3)

if __name__=='__main__': unittest.main()
