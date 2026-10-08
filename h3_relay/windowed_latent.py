"""Experimental two-phase Ref2VA relay; sampled AV history, deferred decode.

Video history is sliced only at the start of H3's 1,4,4,4,4 cycle. This
preserves nominal token time spans, not equivalence to pixel re-encoding.
"""
from __future__ import annotations
import logging

LOG = logging.getLogger(__name__)


def plan_latent_windows(count, window, history):
    from .windowed_edit import validate_settings
    validate_settings(window, history)
    if count < 1:
        raise ValueError("Source video has no frames.")
    # W=17n+5 ends with a four-frame token. Continue at the preceding
    # one-frame token, with 17k history frames before it: both phase zero.
    stride = window - 5 if history else window
    starts = [0]
    while starts[-1] + window < count:
        starts.append(starts[-1] + stride)
    covered, result = 0, []
    for index, start in enumerate(starts):
        end = min(count, start + window)
        result.append(dict(index=index, source_start=start, source_end_exclusive=end,
            window_frames=window, padded_tail_frames=window-(end-start),
            history_start=start-history+1 if index and history else None,
            history_end_exclusive=start+1 if index and history else None,
            discarded_overlap=max(0, covered-start), delivered_frames=end-covered))
        covered = end
    return result


def latent_history_slices(window, history):
    """Video token slices in the immediately preceding, regular-stride window."""
    if not history:
        raise ValueError("Independent windows have no history slices.")
    boundary_frame = window - 5
    past_frames = history - 1
    if boundary_frame % 17 or past_frames % 17 or past_frames > boundary_frame:
        raise ValueError("Latent history must align to complete H3 temporal cycles.")
    boundary_token = boundary_frame // 17 * 5
    return slice(boundary_token - past_frames // 17 * 5, boundary_token), boundary_token


def _cpu(value):
    import torch
    if isinstance(value, torch.Tensor):
        return value.detach().to('cpu')
    if isinstance(value, dict):
        return {k: _cpu(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_cpu(v) for v in value]
    if isinstance(value, tuple):
        return tuple(_cpu(v) for v in value)
    return value


def _cpu_latent(latent):
    from comfy.nested_tensor import NestedTensor
    from .vendor.context_loop.nodes import _streams_from_latent
    return {**latent, 'samples': NestedTensor(tuple(_cpu(x) for x in _streams_from_latent(latent)))}


class H3RelayLatentPrepared:
    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {'conditioning': ('CONDITIONING',), 'latent': ('LATENT',),
                             'index': ('INT',)},
                'optional': {'previous': ('H3_EDIT_PREPARED',)}}
    RETURN_TYPES = ('H3_EDIT_PREPARED',)
    FUNCTION = 'collect'
    CATEGORY = 'H3 Relay/internal'

    def collect(self, conditioning, latent, index, previous=()):
        if index != len(previous):
            raise ValueError('Conditioning windows are out of order.')
        LOG.info('H3 latent relay prepared conditioning window %d', index+1)
        return (previous + ((_cpu(conditioning), _cpu_latent(latent)),),)


class H3RelayLatentSelect:
    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {'prepared': ('H3_EDIT_PREPARED',), 'index': ('INT',)}}
    RETURN_TYPES = ('CONDITIONING', 'LATENT')
    FUNCTION = 'select'
    CATEGORY = 'H3 Relay/internal'

    def select(self, prepared, index):
        if index == 0:
            LOG.info('H3 latent relay all %d conditioning windows ready; sampling phase', len(prepared))
        return prepared[index]


class H3RelayLatentHistory:
    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {'conditioning': ('CONDITIONING',), 'latent': ('LATENT',),
            'previous': ('H3_EDIT_LATENTS',), 'window_frames': ('INT',),
            'history_frames': ('INT',), 'start': ('INT',)}}
    RETURN_TYPES = ('CONDITIONING', 'LATENT')
    FUNCTION = 'apply'
    CATEGORY = 'H3 Relay/internal'

    def apply(self, conditioning, latent, previous, window_frames, history_frames, start):
        from .vendor.context_loop.nodes import _streams_from_latent
        from .vendor.context_loop.sliding_context import require_sliding_history_support
        past_slice, boundary_token = latent_history_slices(window_frames, history_frames)
        preceding = previous[-1]
        if start - preceding['start'] != window_frames - 5:
            raise ValueError('Latent continuation must use the regular aligned source stride.')
        video, audio = _streams_from_latent(preceding['latent'])
        if video.ndim != 5 or audio.ndim != 4 or boundary_token >= video.shape[2]:
            raise ValueError('Unexpected sampled H3 AV latent shape.')
        keyframes = []
        if history_frames > 1:
            require_sliding_history_support()
            keyframes.append({'anchor': 'history', 'latent': video[:, :, past_slice].clone()})
        keyframes.append({'anchor': 'first', 'resolved_frame_index': 0,
                          'latent': video[:, :, boundary_token:boundary_token+1].clone()})
        # Audio uses a 40 Hz timeline, video 24 Hz. Round each endpoint from
        # the same local origin rather than accumulating duration rounding.
        boundary_frame = window_frames - 5
        a0 = round((boundary_frame-history_frames+1)*40/24)
        a1 = round(boundary_frame*40/24)
        a2 = round((boundary_frame+1)*40/24)
        if a0 < 0 or a2 > audio.shape[-1]:
            raise ValueError('Sampled audio does not cover the aligned history interval.')
        if history_frames > 1:
            keyframes.append({'anchor': 'history', 'audio_latent': audio[..., a0:a1].clone()})
        keyframes.append({'anchor': 'first', 'resolved_frame_index': 0,
                          'audio_latent': audio[..., a1:a2].clone()})
        out = []
        for embedding, metadata in conditioning:
            if metadata.get('minimax_keyframes'):
                raise ValueError('Experimental latent relay requires fresh reference conditioning.')
            out.append([embedding, {**metadata, 'minimax_keyframes': keyframes}])
        LOG.info('H3 latent relay source_start=%d video_history_tokens=%d boundary_token=%d audio=[%d,%d,%d); no VAE round trip',
                 start, boundary_token-past_slice.start, boundary_token, a0, a1, a2)
        return out, latent


class H3RelayLatentStore:
    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {'latent': ('LATENT',), 'start': ('INT',), 'index': ('INT',)},
                'optional': {'previous': ('H3_EDIT_LATENTS',)}}
    RETURN_TYPES = ('H3_EDIT_LATENTS',)
    FUNCTION = 'store'
    CATEGORY = 'H3 Relay/internal'

    def store(self, latent, start, index, previous=()):
        if len(previous) != index:
            raise ValueError('Sampled latent windows are out of order.')
        LOG.info('H3 latent relay sampled window %d; deferring decode', index+1)
        return (previous + ({'start': start, 'latent': _cpu_latent(latent)},),)


class H3RelayLatentDecode:
    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {'windows': ('H3_EDIT_LATENTS',), 'vae': ('VAE',),
            'window_frames': ('INT',), 'source_count': ('INT',)}}
    RETURN_TYPES = ('IMAGE',)
    FUNCTION = 'decode'
    CATEGORY = 'H3 Relay/internal'

    def decode(self, windows, vae, window_frames, source_count):
        import torch
        import comfy.model_management
        from .vendor.context_loop.nodes import _streams_from_latent
        LOG.info('H3 latent relay all %d windows sampled; final decode phase', len(windows))
        parts, covered = [], 0
        for index, entry in enumerate(windows):
            comfy.model_management.throw_exception_if_processing_interrupted()
            video = _streams_from_latent(entry['latent'])[0]
            frames = vae.decode(video)
            if frames.ndim == 5:
                frames = frames.reshape(-1, *frames.shape[-3:])
            if len(frames) == window_frames-1:
                LOG.warning('H3 latent relay window %d decoded one frame short; holding its final frame', index+1)
                frames = torch.cat((frames, frames[-1:]))
            if len(frames) != window_frames:
                raise ValueError('Unexpected decoded frame count in latent relay.')
            start = entry['start']
            retained = covered-start
            end = min(source_count, start+window_frames)
            if not 0 <= retained < window_frames or end <= covered:
                raise ValueError('Latent relay output timeline is not contiguous.')
            parts.append(frames[retained:end-start])
            covered = end
            LOG.info('H3 latent relay decoded window %d; assembled frames [0,%d)', index+1, covered)
        if covered != source_count:
            raise ValueError('Latent relay did not cover the entire source.')
        return (torch.cat(parts, dim=0),)


def build_latent_graph(model, clip, video_vae, audio_vae, source, reference_image,
                       prompt, windows, window, history, count, width, height,
                       seed, steps, cfg, sampler_name, scheduler, denoise, source_audio):
    from comfy_execution.graph_utils import GraphBuilder
    from .removal_cache import window_seed
    graph = GraphBuilder()
    prepared = None
    # Every sampling branch depends on this final collected bundle: all source
    # encoding/text conditioning completes before the first diffusion forward.
    for entry in windows:
        index, start = entry['index'], entry['source_start']
        part = graph.node('H3RelayEditWindow', source=source, start=start,
            window_frames=window, history_frames=history,
            **({} if source_audio is None else {'source_audio': source_audio}))
        refs = {'ref_images.ref_image_0': reference_image, 'ref_videos.ref_video_0': part.out(0)}
        if source_audio is not None:
            refs['ref_video_audios.ref_video_audio_0'] = part.out(3)
        ref = graph.node('MiniMaxH3ReferenceToVideo', prompt=prompt, width=width, height=height,
            length=window, ref_image_size='match', clip=clip, vae=video_vae, audio_vae=audio_vae, **refs)
        prepared = graph.node('H3RelayLatentPrepared', conditioning=ref.out(0), latent=ref.out(1),
            index=index, **({} if prepared is None else {'previous': prepared.out(0)}))
    sampler = graph.node('KSamplerSelect', sampler_name=sampler_name)
    sigmas = graph.node('BasicScheduler', model=model, scheduler=scheduler, steps=steps, denoise=denoise)
    previous = None
    for entry in windows:
        index, start = entry['index'], entry['source_start']
        select = graph.node('H3RelayLatentSelect', prepared=prepared.out(0), index=index)
        cond, latent = select.out(0), select.out(1)
        if previous is not None and history:
            carry = graph.node('H3RelayLatentHistory', conditioning=cond, latent=latent,
                previous=previous.out(0), window_frames=window, history_frames=history, start=start)
            cond, latent = carry.out(0), carry.out(1)
        noise = graph.node('RandomNoise', noise_seed=window_seed(seed, index))
        guider = graph.node('CFGGuider', model=model, positive=cond, negative=cond, cfg=cfg)
        sample = graph.node('SamplerCustomAdvanced', noise=noise.out(0), guider=guider.out(0),
            sampler=sampler.out(0), sigmas=sigmas.out(0), latent_image=latent)
        previous = graph.node('H3RelayLatentStore', latent=sample.out(0), start=start, index=index,
            **({} if previous is None else {'previous': previous.out(0)}))
    decoded = graph.node('H3RelayLatentDecode', windows=previous.out(0), vae=video_vae,
                         window_frames=window, source_count=count)
    return decoded.out(0), graph.finalize()


NODE_CLASS_MAPPINGS = {c.__name__: c for c in (H3RelayLatentPrepared, H3RelayLatentSelect,
    H3RelayLatentHistory, H3RelayLatentStore, H3RelayLatentDecode)}
