"""Run from a ComfyUI checkout to validate H3 Relay's expanded raw graph."""

import pathlib
import os
import sys
import json
import tempfile


ROOT = pathlib.Path(__file__).resolve().parents[1]
COMFY_ROOT = os.environ.get("COMFYUI_ROOT")
if COMFY_ROOT:
    sys.path.insert(0, COMFY_ROOT)
sys.path.insert(0, str(ROOT))

from h3_relay import nodes
from h3_relay import cache as relay_cache
from h3_relay.vendor.context_loop.sliding_context import (
    require_sliding_history_support,
)


def main():
    require_sliding_history_support()
    from comfy.ldm.minimax.model import PackedLayout
    assert PackedLayout.supports_history_keyframes is True
    assert [nodes._duration_frames(value) for value in (1, 5, 10, 15)] == [
        39,
        124,
        243,
        362,
    ]
    assert nodes._validate_ltx_tiling(193, 64, 128, 16) == (193, 64, 128, 16)
    import torch
    with tempfile.TemporaryDirectory() as chunk_directory:
        chunk_video = os.path.join(chunk_directory, "chunk-contract.mp4")
        frames = torch.linspace(
            0.0, 1.0, 10 * 32 * 32 * 3, dtype=torch.float32
        ).reshape(10, 32, 32, 3)
        nodes.context._write_segment_video(frames, chunk_video, 24, 18)
        chunks = list(nodes._video_frame_chunks(chunk_video, 4))
        assert [int(chunk.shape[0]) for chunk in chunks] == [4, 4, 4]
        assert torch.equal(chunks[0][-1], chunks[1][0])
        assert torch.equal(chunks[1][-1], chunks[2][0])
        assert sum(int(chunk.shape[0]) for chunk in chunks) - 2 == 10
    adapter = nodes.H3RelayLTXModelAdapter()
    components = [object(), object(), object(), object()]
    custom = adapter.pack(*components, "contract-ltx-v1")[0]
    same = adapter.pack(*components, "contract-ltx-v1")[0]
    changed = adapter.pack(*components, "contract-ltx-v2")[0]
    assert custom["kind"] == "ltx"
    assert custom["model"] is components[0]
    assert custom["vae"] is components[1]
    assert custom["upscale_model"] is components[2]
    assert custom["clip"] is components[3]
    assert custom["cache_tag"] == same["cache_tag"]
    assert custom["cache_tag"] != changed["cache_tag"]
    assert "H3RelayLTXModelAdapter" in nodes.NODE_CLASS_MAPPINGS
    assert "H3RelayCacheManager" in nodes.NODE_CLASS_MAPPINGS
    cache_result = nodes.H3RelayCacheManager().manage(
        "inspect", 2, 100.0
    )
    cache_path, cache_status = cache_result["result"]
    assert cache_path == relay_cache.cache_root()
    assert "inspect is read-only" in cache_status
    assert cache_result["ui"]["text"] == [cache_status]

    import folder_paths
    original_user = folder_paths.get_user_directory()
    with tempfile.TemporaryDirectory() as temporary_user:
        folder_paths.set_user_directory(temporary_user)
        try:
            root = pathlib.Path(relay_cache.cache_path(
                "h3_chains", "contract", "checkpoints"
            ))
            root.mkdir(parents=True)
            revisions = ["1" * 32, "2" * 32, "3" * 32]
            for index, revision in enumerate(revisions, 1):
                path = root / ("clip_0001.%s.safetensors" % revision)
                path.write_bytes((revision * index).encode())
                os.utime(path, (index, index))
            (root / "clip_0001.json").write_text(
                json.dumps({"revision": revisions[-1]}), encoding="utf-8"
            )
            result = relay_cache.prune_superseded(keep_per_shot=1)
            assert result["removed_revisions"] == 2
            assert not (root / ("clip_0001.%s.safetensors" % revisions[0])).exists()
            assert not (root / ("clip_0001.%s.safetensors" % revisions[1])).exists()
            kept = root / ("clip_0001.%s.safetensors" % revisions[2])
            assert kept.exists()
            assert relay_cache.resolve_artifact(
                relay_cache.artifact_uri(str(kept))
            ) == str(kept)
        finally:
            folder_paths.set_user_directory(original_user)
    import comfy.samplers
    assert nodes.H3_SAMPLERS == list(comfy.samplers.SAMPLER_NAMES)
    assert nodes.H3_SCHEDULERS == ["beta57"] + list(
        comfy.samplers.SCHEDULER_NAMES
    )
    for sampler_name, scheduler_name in (
        ("euler", "simple"),
        ("res_multistep", "simple"),
    ):
        profile_sequence, _ = nodes.H3RelaySequenceStart().start(
            "h3_relay_contract_%s" % sampler_name,
            "Continuity contract.",
            832,
            480,
            18,
            sampler_name,
            scheduler_name,
            False,
        )
        profile_graph = nodes.H3RelayGenerateShot().generate_shot(
            {
                "format": nodes.MODEL_BUNDLE_FORMAT,
                "kind": "h3",
                "model": object(),
                "cache_tag": "test-h3-model-%s" % sampler_name,
            },
            profile_sequence,
            "A short sampler contract test.",
            42,
            1.0,
            2,
            18,
            "match",
            "",
        )
        sampler = next(
            item for item in profile_graph["expand"].values()
            if item["class_type"] == "KSamplerSelect"
        )
        scheduler = next(
            item for item in profile_graph["expand"].values()
            if item["class_type"] == "BasicScheduler"
        )
        assert sampler["inputs"]["sampler_name"] == sampler_name
        assert scheduler["inputs"]["scheduler"] == scheduler_name
        assert not any(
            item["class_type"] == "H3RelayInternalSpectrum"
            for item in profile_graph["expand"].values()
        )
    sequence, _ = nodes.H3RelaySequenceStart().start(
        "h3_relay_contract",
        "Continuity contract.",
        832,
        480,
        18,
        "euler",
        "beta57",
        True,
    )
    assert sequence["h3_sampler"] == "euler"
    assert sequence["h3_scheduler"] == "beta57"
    assert sequence["h3_spectrum_enabled"] is True
    assert sequence["h3_context_frames"] == 18
    assert sequence["h3_sampling_profile"] == "native_spectrum_euler_beta57"
    assert "native_spectrum_euler_beta57" in sequence["generation_fingerprint"]
    _, automatic_shot = nodes.context._steer_state(
        sequence, "", "Automatic shot ID contract.", 1, 39, 2,
    )
    assert automatic_shot["id"] == "shot_0001"
    wider_sequence, wider_status = nodes.H3RelaySequenceStart().start(
        "h3_relay_contract_overlap35",
        "Continuity contract.",
        832,
        480,
        35,
        "euler",
        "beta57",
        True,
    )
    wider_state, _ = nodes.context._steer_state(
        wider_sequence, "", "35-frame overlap contract.", 1, 56, 2,
    )
    assert wider_sequence["h3_context_frames"] == 35
    assert "sliding35" in wider_sequence["generation_fingerprint"]
    assert "35-frame H3 sliding history" in wider_status
    assert wider_state["plan"]["compatibility"]["context_length"] == 35
    _, crf_shot = nodes.context._steer_state(
        sequence, "crf_contract", "CRF contract.", 1, 39, 2,
    )
    default_key = nodes.context._steer_cache_key(sequence, crf_shot, "match")
    custom_crf_sequence = dict(sequence)
    custom_crf_sequence["relay_output_crf"] = 20
    custom_key = nodes.context._steer_cache_key(
        custom_crf_sequence, crf_shot, "match"
    )
    assert custom_key == default_key
    expanded = nodes.H3RelayGenerateShot().generate_shot(
        {
            "format": nodes.MODEL_BUNDLE_FORMAT,
            "kind": "h3",
            "model": object(),
            "cache_tag": "test-h3-model",
        },
        sequence,
        "A short contract test.",
        42,
        1.0,
        16,
        18,
        "match",
        "",
    )
    classes = {item["class_type"] for item in expanded["expand"].values()}
    required = {
        "H3RelayInternalSpectrum",
        "H3RelayInternalChainContext",
        "H3RelayInternalLoopTrim",
        "H3RelayInternalSegmentSave",
        "H3RelayInternalAcceptRaw",
        "H3RelayInternalVideoOutput",
    }
    assert required <= classes
    default_sampler = next(
        item for item in expanded["expand"].values()
        if item["class_type"] == "KSamplerSelect"
    )
    assert default_sampler["inputs"]["sampler_name"] == "euler"
    assert "ManualSigmas" in classes
    assert "BasicScheduler" not in classes
    print("H3 Relay runtime contract passed")


if __name__ == "__main__":
    main()
