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
from h3_relay.vendor.context_loop import patch_layout


def main():
    def stock_layout_init(*args, **kwargs):
        return args, kwargs

    def current_sol_wrapper():
        original_init = stock_layout_init

        def __init__(*args, **kwargs):
            return original_init(*args, **kwargs)

        return __init__

    for module_name in (
        "custom_nodes.sol_attn_minimax_v5",
        "h3_relay.fast_h3_vsa",
        "/tmp/ComfyUI/custom_nodes/sol_attn_minimax_v5",
    ):
        sol_wrapper = current_sol_wrapper()
        sol_wrapper.__module__ = module_name
        assert patch_layout._solattn_wrapped_init(sol_wrapper) is stock_layout_init
        replacement = lambda *args, **kwargs: None
        assert patch_layout._replace_solattn_wrapped_init(sol_wrapper, replacement)
        assert patch_layout._solattn_wrapped_init(sol_wrapper) is replacement
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
    assert nodes._resolve_ultimate_spatial_tiling(
        1664, 960, 1024, 1024, 128
    ) == (1024, 960, 128)
    assert nodes._resolve_ultimate_spatial_tiling(
        2688, 1536, 1024, 1024, 128
    ) == (1024, 1024, 128)
    assert nodes._resolve_ultimate_spatial_tiling(
        640, 64, 1024, 1024, 128
    ) == (640, 64, 32)
    assert nodes._resolve_ultimate_frame_contract(
        243, 243, 243, 18
    ) == (243, 0, 243)
    assert nodes._resolve_ultimate_frame_contract(
        226, 226, 225, 18
    ) == (226, 1, 225)
    try:
        nodes._resolve_ultimate_frame_contract(226, 243, 225, 18)
    except ValueError as exc:
        assert "accepted latent checkpoint contains 243" in str(exc)
    else:
        raise AssertionError("Ultimate output must match checkpoint frames")
    import torch
    generator_schema = nodes.H3RelayGenerateShot.GET_SCHEMA()
    generator_inputs = {item.id: item for item in generator_schema.inputs}
    for name in ("reference_image_1", "reference_image_2", "reference_image_3"):
        assert name in generator_inputs
    additional_images = generator_inputs["additional_reference_images"]
    assert additional_images.template.names == list(
        nodes.ADDITIONAL_REFERENCE_IMAGE_NAMES
    )
    assert additional_images.template.min == 0
    ultimate_schema = nodes.H3RelayUltimateEnhanceShot.GET_SCHEMA()
    ultimate_inputs = {item.id: item for item in ultimate_schema.inputs}
    assert "h3_model" in ultimate_inputs
    assert "sequence" in ultimate_inputs
    assert "previous_enhanced" in ultimate_inputs
    assert ultimate_inputs["additional_reference_images"].template.names == list(
        nodes.ADDITIONAL_REFERENCE_IMAGE_NAMES
    )
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
    assert "H3RelayFastH3VSAModelLoader" in nodes.NODE_CLASS_MAPPINGS
    assert "H3RelayInternalFastH3VSA" in nodes.NODE_CLASS_MAPPINGS
    assert "H3RelayAcceptedRawLatent" in nodes.NODE_CLASS_MAPPINGS
    assert "H3RelayUltimateEnhanceShot" in nodes.NODE_CLASS_MAPPINGS
    assert "H3RelayInternalAcceptUltimate" in nodes.NODE_CLASS_MAPPINGS
    assert "H3RelayAssembleRaw" in nodes.NODE_CLASS_MAPPINGS
    fast_bundle = nodes.H3RelayModelBundlePack().pack(
        "h3",
        object(),
        "contract-fast-h3-vsa",
        h3_profile=nodes.FAST_H3_VSA_PROFILE,
    )[0]
    assert fast_bundle["h3_profile"] == nodes.FAST_H3_VSA_PROFILE
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
    fast_sequence, _ = nodes.H3RelaySequenceStart().start(
        "fast_h3_vsa_contract",
        "FastH3 continuity contract.",
        832,
        480,
        18,
        "euler",
        "beta57",
        True,
    )
    fast_graph = nodes.H3RelayGenerateShot.generate_shot(
        {
            "format": nodes.MODEL_BUNDLE_FORMAT,
            "kind": "h3",
            "model": object(),
            "cache_tag": "test-fast-h3-vsa-model",
            "h3_profile": nodes.FAST_H3_VSA_PROFILE,
        },
        fast_sequence,
        "FastH3 profile contract.",
        42,
        1.0,
        16,
        18,
        "match",
        "",
    )
    fast_nodes = list(fast_graph["expand"].values())
    fast_classes = {item["class_type"] for item in fast_nodes}
    assert "H3RelayInternalFastH3VSA" in fast_classes
    assert "SolAttnMiniMax" not in fast_classes
    assert "H3RelayInternalSpectrum" not in fast_classes
    assert "LoraLoaderModelOnly" not in fast_classes
    fast_vsa = next(
        item
        for item in fast_nodes
        if item["class_type"] == "H3RelayInternalFastH3VSA"
    )
    assert set(fast_vsa["inputs"]) == {"model"}
    fast_shift = next(
        item for item in fast_nodes
        if item["class_type"] == "MiniMaxH3SigmaShift"
    )
    assert fast_shift["inputs"]["shift_video"] == 12.0
    assert fast_shift["inputs"]["shift_audio"] == 3.0
    fast_sampler = next(
        item for item in fast_nodes if item["class_type"] == "KSamplerSelect"
    )
    fast_scheduler = next(
        item for item in fast_nodes if item["class_type"] == "BasicScheduler"
    )
    assert fast_sampler["inputs"]["sampler_name"] == "euler"
    assert fast_scheduler["inputs"]["scheduler"] == "simple"
    assert fast_scheduler["inputs"]["steps"] == 4
    fast_accept = next(
        item for item in fast_nodes
        if item["class_type"] == "H3RelayInternalAcceptRaw"
    )
    accepted_sequence = fast_accept["inputs"]["sequence"]
    assert accepted_sequence["h3_model_profile"] == nodes.FAST_H3_VSA_PROFILE
    assert accepted_sequence["h3_sampling_profile"] == nodes.FAST_H3_VSA_PROFILE
    assert accepted_sequence["h3_sampler"] == "euler"
    assert accepted_sequence["h3_scheduler"] == "simple"
    assert accepted_sequence["h3_spectrum_enabled"] is False
    ultimate_sequence, _ = nodes.H3RelaySequenceStart().start(
        "fast_h3_ultimate_contract",
        "Global Akatz continuity contract.",
        832,
        480,
        18,
        "euler",
        "simple",
        False,
    )
    ultimate_bundle = {
        "format": nodes.MODEL_BUNDLE_FORMAT,
        "kind": "h3",
        "model": object(),
        "cache_tag": "test-fast-h3-ultimate-model",
        "h3_profile": nodes.FAST_H3_VSA_PROFILE,
    }
    ultimate_sequence, _ = nodes._sequence_with_h3_model(
        ultimate_sequence, ultimate_bundle
    )
    ultimate_sequence = dict(ultimate_sequence)
    ultimate_sequence["h3_sampling_profile"] = nodes.FAST_H3_VSA_PROFILE
    ultimate_sequence["h3_sampler"] = "euler"
    ultimate_sequence["h3_scheduler"] = "simple"
    ultimate_sequence["h3_spectrum_enabled"] = False
    ultimate_state, ultimate_shot = nodes.context._steer_state(
        ultimate_sequence,
        "shot_0001",
        "Akatz runs through a neon alley.",
        424246,
        243,
        4,
    )
    revision = "f" * 32
    h3_segment = {
        "index": 1,
        "id": "shot_0001",
        "revision": revision,
        "segment": "cache://contract/raw.mp4",
        "segment_sha256": "1" * 64,
        "checkpoint": "cache://contract/raw.safetensors",
        "checkpoint_sha256": "2" * 64,
        "checkpoint_frames": 243,
        "generated_audio": "cache://contract/raw.wav",
        "raw_frames": 243,
        "delivered_frames": 243,
    }
    ultimate_sequence["shots"] = nodes.context._effective_editor_plan(
        ultimate_state["plan"]
    )["shots"]
    ultimate_sequence["segments"] = [{
        "index": 1,
        "id": "shot_0001",
        "revision": revision,
        "h3_segment": h3_segment,
    }]
    original_ultimate_preflight = nodes._require_ultimate_engine
    nodes._require_ultimate_engine = lambda: original_ultimate_preflight({
        name: object() for name in nodes.ULTIMATE_ENGINE_NODE_TYPES
    })
    try:
        ultimate_graph = nodes.H3RelayUltimateEnhanceShot.enhance(
            ultimate_bundle,
            ultimate_sequence,
            "Preserve accepted motion and restore fine detail.",
            424264,
            18,
            "match",
            136,
            17,
            0.999,
            1024,
            1024,
            128,
        )
    finally:
        nodes._require_ultimate_engine = original_ultimate_preflight
    ultimate_nodes = list(ultimate_graph["expand"].values())
    ultimate_classes = {item["class_type"] for item in ultimate_nodes}
    assert "SolAttnMiniMax" not in ultimate_classes
    ultimate_vsa = next(
        item for item in ultimate_nodes
        if item["class_type"] == "H3RelayInternalFastH3VSA"
    )
    assert set(ultimate_vsa["inputs"]) == {"model"}
    assert {
        "MiniMaxH3SigmaShift",
        "H3RelayInternalFastH3VSA",
        "H3RelayAcceptedRawLatent",
        "MMH3LatentUpscaleWithModelParams",
        "MMH3TemporalSplitParams",
        "MMH3SpatialSplitParams",
        "MMH3UltimateUpscale",
        "H3RelayInternalAcceptUltimate",
    } <= ultimate_classes
    ultimate_conditioning = next(
        item for item in ultimate_nodes
        if item["class_type"] == "MiniMaxH3ReferenceToVideo"
    )
    assert ultimate_conditioning["inputs"]["width"] == 1664
    assert ultimate_conditioning["inputs"]["height"] == 960
    assert ultimate_conditioning["inputs"]["length"] == 243
    assert "Global Akatz continuity contract." in ultimate_conditioning["inputs"]["prompt"]
    assert "Akatz runs through a neon alley." in ultimate_conditioning["inputs"]["prompt"]
    assert "Preserve accepted motion" in ultimate_conditioning["inputs"]["prompt"]
    ultimate_accept = next(
        item for item in ultimate_nodes
        if item["class_type"] == "H3RelayInternalAcceptUltimate"
    )
    assert ultimate_accept["inputs"]["temporal_chunk_frames"] == 136
    assert ultimate_accept["inputs"]["temporal_overlap_frames"] == 17
    assert ultimate_accept["inputs"]["checkpoint_frames"] == 243
    ultimate_spatial = next(
        item for item in ultimate_nodes
        if item["class_type"] == "MMH3SpatialSplitParams"
    )
    assert ultimate_spatial["inputs"]["tile_width"] == 1024
    assert ultimate_spatial["inputs"]["tile_height"] == 960
    assert ultimate_spatial["inputs"]["spatial_w_overlap"] == 128
    assert ultimate_spatial["inputs"]["spatial_h_overlap"] == 128
    assert ultimate_accept["inputs"]["tile_width"] == 1024
    assert ultimate_accept["inputs"]["tile_height"] == 960
    assert ultimate_accept["inputs"]["spatial_overlap"] == 128
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
    reference_images = [
        torch.full((1, 2, 2, 3), float(index), dtype=torch.float32)
        for index in range(1, 10)
    ]
    legacy_reference_sequence, _ = nodes.H3RelaySequenceStart().start(
        "reference_contract", "", 832, 480, 18, "euler", "beta57", True,
    )
    _, reference_shot = nodes.context._steer_state(
        legacy_reference_sequence, "", "reference contract", 42, 39, 2,
    )
    assert nodes.context._steer_cache_key(
        legacy_reference_sequence,
        reference_shot,
        "match",
        reference_image_1=reference_images[0],
        reference_image_2=reference_images[1],
        reference_image_3=reference_images[2],
    ) == "28da0d8771fe4dbcd6e9459ead0e85c81fec825d1b02eebce211904fb4888914"
    try:
        nodes._reference_image_slots(
            additional_reference_images={"reference_image_4": reference_images[3]}
        )
    except ValueError as exc:
        assert "1 through 3" in str(exc)
    else:
        raise AssertionError("Additional references must follow images 1 through 3")
    reference_graph = nodes.H3RelayGenerateShot.generate_shot(
        {
            "format": nodes.MODEL_BUNDLE_FORMAT,
            "kind": "h3",
            "model": object(),
            "cache_tag": "test-h3-reference-model",
        },
        sequence,
        "Reference image contract.",
        42,
        1.0,
        2,
        18,
        "match",
        "",
        reference_image_1=reference_images[0],
        reference_image_2=reference_images[1],
        reference_image_3=reference_images[2],
        additional_reference_images={
            "reference_image_%d" % index: reference_images[index - 1]
            for index in range(4, 10)
        },
    )
    reference_conditioning = next(
        item for item in reference_graph["expand"].values()
        if item["class_type"] == "MiniMaxH3ReferenceToVideo"
    )
    for index, image in enumerate(reference_images):
        assert reference_conditioning["inputs"][
            "ref_images.ref_image_%d" % index
        ] is image
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
