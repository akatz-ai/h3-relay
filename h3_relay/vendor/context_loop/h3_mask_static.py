"""Static, reviewable MiniMax H3 compatibility functions for ComfyUI 0.32."""

from __future__ import annotations

import torch


def mask_row_values(mask, latent_t, lat_h, lat_w):
    """Convert a video denoise mask to one value per 2x2 patch row."""
    m = torch.nn.functional.pad(
        mask,
        (0, lat_w - mask.shape[-1], 0, lat_h - mask.shape[-2]),
        mode="replicate",
    )
    m = m.reshape(latent_t, lat_h // 2, 2, lat_w // 2, 2).amax(dim=(2, 4))
    values = m.reshape(-1)
    if bool((values >= 1.0 - 1e-3).all()):
        return None
    return values


def mod_row(vecs, row, dtype):
    return vecs[row].to(dtype)


def mod_scale_shift(h, shift, scale, segments):
    for a, b, row in segments:
        h[a:b].mul_(1.0 + mod_row(scale, row, h.dtype)).add_(
            mod_row(shift, row, h.dtype)
        )
    return h


def mod_gate(x, gate, other, segments):
    for a, b, row in segments:
        x[a:b].addcmul_(other[a:b], mod_row(gate, row, x.dtype))
    return x


def final_forward(self, x, t_emb, video_seg, audio_seg):
    shift, scale = self.adaln_proj(t_emb)

    def mod(seg):
        a, b, row = seg
        return (
            self.norm(x[a:b])
            * (1.0 + mod_row(scale, row, scale.dtype))
            + mod_row(shift, row, shift.dtype)
        ).to(torch.float32)

    return self.video_out(mod(video_seg)), self.audio_out(mod(audio_seg))


def h3_forward(
    self,
    x,
    timestep,
    context,
    transformer_options={},
    minimax_payload=None,
    denoise_mask=None,
    audio_denoise_mask=None,
    **kwargs,
):
    import comfy.patcher_extension
    import comfy.ldm.minimax.model as h3m

    scale = float((minimax_payload or {}).get("audio_scale", 1.0))
    audio_src = x[1]
    if scale != 1.0:
        shift_v = float(transformer_options.get(
            "minimax_h3_sigma_shift_video", self.sigma_shift_video
        ))
        shift_a = float(transformer_options.get(
            "minimax_h3_sigma_shift_audio", self.sigma_shift_audio
        ))
        sigma_v = (timestep.flatten()[0] / 1000.0).float().clamp(min=1e-6)
        sigma_a = h3m.time_shift_sigma(sigma_v, shift_v, shift_a)
        carry = (sigma_a / sigma_v).to(audio_src.dtype)
        x = [x[0], audio_src * carry]

    out = comfy.patcher_extension.WrapperExecutor.new_class_executor(
        self._forward,
        self,
        comfy.patcher_extension.get_all_wrappers(
            comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL,
            transformer_options,
        ),
    ).execute(
        x,
        timestep,
        context,
        transformer_options,
        minimax_payload=minimax_payload,
        denoise_mask=denoise_mask,
        audio_denoise_mask=audio_denoise_mask,
        **kwargs,
    )

    if scale != 1.0:
        out[1] = (
            (1.0 - scale) * (audio_src * carry)
            + (1.0 + (scale - 1.0) * sigma_a).to(out[1].dtype) * out[1]
        )
    return out


def h3_inner_forward(
    self,
    x,
    timestep,
    context,
    transformer_options={},
    minimax_payload=None,
    denoise_mask=None,
    audio_denoise_mask=None,
    **kwargs,
):
    import comfy.ldm.common_dit
    import comfy.ldm.minimax.model as h3m
    import comfy.model_management
    import comfy.model_prefetch

    video_x, audio_x = x[0], x[1]
    orig_t, orig_h, orig_w = video_x.shape[2], video_x.shape[3], video_x.shape[4]
    video_x = comfy.ldm.common_dit.pad_to_patch_size(video_x, self.patch_size)
    if video_x.shape[0] != 1:
        raise ValueError("MiniMax H3 supports batch size 1")
    payload = minimax_payload or {}
    device = video_x.device
    dtype = context.dtype

    latent_t, lat_h, lat_w = video_x.shape[2], video_x.shape[3], video_x.shape[4]
    audio_t = audio_x.shape[-1]
    text_len = context.shape[1]
    layout = payload.get("layout")
    if layout is None or layout.signature != (
        text_len, latent_t, lat_h, lat_w, audio_t
    ):
        layout = h3m.PackedLayout(
            text_len,
            latent_t,
            lat_h,
            lat_w,
            audio_t,
            keyframes=payload.get("keyframes"),
            refs=payload.get("refs"),
        )

    shift_v = float(transformer_options.get(
        "minimax_h3_sigma_shift_video", self.sigma_shift_video
    ))
    shift_a = float(transformer_options.get(
        "minimax_h3_sigma_shift_audio", self.sigma_shift_audio
    ))
    sigma_v = (timestep.flatten()[0] / 1000.0).float().clamp(min=1e-6)
    t_v = float(1.0 - sigma_v)
    t_a = float(1.0 - h3m.time_shift_sigma(sigma_v, shift_v, shift_a))

    vis_aug = float(payload.get(
        "visual_cond_noise_aug", h3m.VISUAL_COND_TIMESTEP
    ))
    aud_aug = float(payload.get(
        "audio_cond_noise_aug", h3m.AUDIO_COND_TIMESTEP
    ))
    seg_t = {
        "text": t_v,
        "video": t_v,
        "audio": t_a,
        "cond": max(t_v, vis_aug),
        "ref_img": max(t_v, vis_aug),
        "cond_audio": max(t_a, aud_aug),
        "ref_audio": max(t_a, aud_aug),
    }

    t_pin_v = max(t_v, h3m.VISUAL_COND_TIMESTEP)
    t_pin_a = max(t_a, h3m.AUDIO_COND_TIMESTEP)
    video_rows_t = None
    audio_rows_t = None
    if denoise_mask is not None:
        m = mask_row_values(
            denoise_mask[0, 0].to(torch.float32), latent_t, lat_h, lat_w
        )
        if m is not None:
            rows_t = (1.0 - m * sigma_v.to(m.device)).clamp(max=t_pin_v)
            if rows_t.unique().numel() == 1:
                seg_t["video"] = float(rows_t[0])
            else:
                video_rows_t = rows_t
    if audio_denoise_mask is not None:
        m = audio_denoise_mask[0, 0].to(torch.float32).reshape(-1)
        if not bool((m >= 1.0 - 1e-3).all()):
            sigma_a = 1.0 - t_a
            rows_t = (1.0 - m * sigma_a).clamp(max=t_pin_a)
            if rows_t.unique().numel() == 1:
                seg_t["audio"] = float(rows_t[0])
            else:
                audio_rows_t = rows_t

    unique_t = sorted(
        {t_v, t_a}
        | {seg_t[kind] for _, _, kind in layout.segments}
        | (
            set(video_rows_t.unique().tolist())
            if video_rows_t is not None else set()
        )
        | (
            set(audio_rows_t.unique().tolist())
            if audio_rows_t is not None else set()
        )
    )
    t_row = {value: index for index, value in enumerate(unique_t)}
    seg_tag = {
        "text": 1,
        "video": 0,
        "audio": 2,
        "cond": 0,
        "ref_img": 0,
        "cond_audio": 2,
        "ref_audio": 2,
    }

    def rows_to_mod_index(rows_t, tag):
        levels = rows_t.unique()
        base = torch.tensor(
            [t_row[value] * 3 + tag for value in levels.tolist()],
            dtype=torch.long,
            device=rows_t.device,
        )
        return base[torch.searchsorted(levels, rows_t)]

    text_tags = payload.get("text_token_tags")
    mod_segments = []
    for a, b, kind in layout.segments:
        row_base = t_row[seg_t[kind]] * 3
        if kind == "text" and text_tags is not None:
            tags = text_tags.view(-1).tolist()
            run_start = 0
            for i in range(1, b - a + 1):
                if i == b - a or tags[i] != tags[run_start]:
                    mod_segments.append((
                        a + run_start,
                        a + i,
                        row_base + int(tags[run_start]),
                    ))
                    run_start = i
        elif kind == "video" and video_rows_t is not None:
            mod_segments.append((
                a, b, rows_to_mod_index(video_rows_t, seg_tag[kind])
            ))
        elif kind == "audio" and audio_rows_t is not None:
            mod_segments.append((
                a, b, rows_to_mod_index(audio_rows_t, seg_tag[kind])
            ))
        else:
            mod_segments.append((a, b, row_base + seg_tag[kind]))

    img_update = layout.img_update.to(device)
    audio_update = layout.audio_update.to(device)
    video_rows = h3m.patchify_video(video_x.to(torch.float32), self.patch_size)
    audio_rows = h3m.pack_audio(audio_x.to(torch.float32))
    cond_video_rows = self._cond_video_rows(payload, device)
    cond_audio_rows = self._cond_audio_rows(payload, device)

    all_video_rows = video_rows
    if cond_video_rows is not None:
        all_video_rows = torch.empty(
            img_update.shape[0],
            video_rows.shape[1],
            dtype=torch.float32,
            device=device,
        )
        all_video_rows[~img_update] = cond_video_rows
        all_video_rows[img_update] = video_rows
    all_audio_rows = audio_rows
    if cond_audio_rows is not None:
        all_audio_rows = torch.empty(
            audio_update.shape[0],
            audio_rows.shape[1],
            dtype=torch.float32,
            device=device,
        )
        all_audio_rows[~audio_update] = cond_audio_rows
        all_audio_rows[audio_update] = audio_rows

    video_embed = self.video_patch_proj(all_video_rows).to(dtype)
    audio_embed = self.audio_patch_proj(all_audio_rows).to(dtype)
    text_states = context[0]
    if text_states.shape[-1] != self.hidden_size:
        text_states = self.token_refiner(
            self.condition_proj(text_states),
            transformer_options=transformer_options,
        )

    h = torch.empty(layout.seq_len, self.hidden_size, dtype=dtype, device=device)
    voff = aoff = 0
    for a, b, kind in layout.segments:
        count = b - a
        if kind == "text":
            h[a:b] = text_states
        elif kind in ("cond", "ref_img", "video"):
            h[a:b] = video_embed[voff:voff + count]
            voff += count
        else:
            h[a:b] = audio_embed[aoff:aoff + count]
            aoff += count

    t_vals = torch.tensor(unique_t, dtype=torch.float32, device=device)
    if self.use_adaln_curves:
        table = comfy.model_management.cast_to(
            self.adaln_t_table, device=device
        )
        pos = t_vals.clamp(0.0, 1.0) * (table.shape[0] - 1)
        i0 = pos.floor().long().clamp(max=table.shape[0] - 2)
        t_emb = torch.lerp(
            table[i0], table[i0 + 1], (pos - i0).unsqueeze(1)
        )
    else:
        t_emb = self.time_embedder(t_vals).to(dtype)

    rope_freqs = h3m.rope_rotation_table(
        self.rope_freqs(layout.position_ids, device), dtype
    )
    patches_replace = transformer_options.get("patches_replace", {})
    blocks_replace = patches_replace.get("dit", {})
    prefetch_queue = comfy.model_prefetch.make_prefetch_queue(
        list(self.blocks), device, transformer_options
    )
    for i, block in enumerate(self.blocks):
        comfy.model_prefetch.prefetch_queue_pop(
            prefetch_queue, device, block
        )
        if ("double_block", i) in blocks_replace:
            def block_wrap(args):
                return {"img": block(
                    args["img"],
                    args["t_emb"],
                    args["mod_segments"],
                    args["rope_freqs"],
                    transformer_options=args["transformer_options"],
                )}

            h = blocks_replace[("double_block", i)](
                {
                    "img": h,
                    "t_emb": t_emb,
                    "mod_segments": mod_segments,
                    "rope_freqs": rope_freqs,
                    "transformer_options": transformer_options,
                },
                {"original_block": block_wrap},
            )["img"]
        else:
            h = block(
                h,
                t_emb,
                mod_segments,
                rope_freqs,
                transformer_options=transformer_options,
            )
    if prefetch_queue is not None:
        comfy.model_prefetch.prefetch_queue_pop(prefetch_queue, device, None)

    va, vb, _ = next(
        segment for segment in layout.segments if segment[2] == "video"
    )
    aa, ab, _ = next(
        segment for segment in layout.segments if segment[2] == "audio"
    )
    video_seg = (
        (va, vb, rows_to_mod_index(video_rows_t, 0) // 3)
        if video_rows_t is not None
        else (va, vb, t_row[seg_t["video"]])
    )
    audio_seg = (
        (aa, ab, rows_to_mod_index(audio_rows_t, 0) // 3)
        if audio_rows_t is not None
        else (aa, ab, t_row[seg_t["audio"]])
    )
    video, audio = self.final_layer(h, t_emb, video_seg, audio_seg)
    video_out = h3m.unpatchify_video(
        video,
        latent_t,
        lat_h // 2,
        lat_w // 2,
        self.latents_dim,
        self.patch_size,
    )
    video_out = video_out[:, :, :orig_t, :orig_h, :orig_w]
    audio_out = h3m.unpack_audio(audio)
    return [-video_out.to(video_x.dtype), -audio_out.to(audio_x.dtype)]


def process_denoise_mask(self, denoise_masks):
    video_mask = denoise_masks[0]
    height, width = video_mask.shape[-2:]
    patch_height, patch_width = self.diffusion_model.patch_size[1:]
    lead = video_mask.shape[:-2]
    video_mask = torch.nn.functional.pad(
        video_mask.reshape((-1,) + video_mask.shape[-3:]),
        (0, -width % patch_width, 0, -height % patch_height),
        mode="replicate",
    )
    video_mask = video_mask.reshape(lead + video_mask.shape[-2:])
    video_mask = video_mask.reshape(
        video_mask.shape[:-2]
        + (
            video_mask.shape[-2] // patch_height,
            patch_height,
            video_mask.shape[-1] // patch_width,
            patch_width,
        )
    ).amax(dim=(-3, -1))
    video_mask = torch.round(video_mask * 256.0) / 256.0
    video_mask = video_mask.masked_fill(
        video_mask >= 0.995, 1.0
    ).masked_fill(video_mask <= 0.05, 0.0)
    denoise_masks[0] = video_mask.repeat_interleave(
        patch_height, dim=-2
    ).repeat_interleave(patch_width, dim=-1)[..., :height, :width]
    if len(denoise_masks) > 1:
        audio_mask = denoise_masks[1].amax(dim=1, keepdim=True)
        audio_mask = torch.round(audio_mask * 256.0) / 256.0
        audio_mask = audio_mask.masked_fill(
            audio_mask >= 0.995, 1.0
        ).masked_fill(audio_mask <= 0.05, 0.0)
        denoise_masks[1] = audio_mask.expand_as(denoise_masks[1]).contiguous()
    return denoise_masks


def scale_latent_inpaint(self, sigma, noise, latent_image, **kwargs):
    import comfy.ldm.minimax.model as h3m
    import comfy.model_base as model_base
    import comfy.utils as utils

    shapes = self.latent_shapes
    if shapes is None or len(shapes) < 2:
        return super(model_base.MiniMaxH3, self).scale_latent_inpaint(
            sigma=sigma,
            noise=noise,
            latent_image=latent_image,
            **kwargs,
        )
    cleans = utils.unpack_latents(latent_image, shapes)
    noises = utils.unpack_latents(noise, shapes)
    aug = h3m.VISUAL_COND_TIMESTEP
    cleans[0] = aug * cleans[0] + (1.0 - aug) * noises[0]
    scale = self.audio_scale()
    if scale != 1.0:
        model_sampling = self.model_sampling
        sigma_v = sigma.clamp(min=1e-6)
        sigma_a = h3m.time_shift_sigma(
            sigma_v, model_sampling.shift, model_sampling.audio_shift
        )
        factor = (sigma_v / sigma_a) / scale
        cleans[1] = cleans[1] * factor.view(
            factor.shape[:1] + (1,) * (cleans[1].ndim - 1)
        ).to(cleans[1].dtype)
    return utils.pack_latents(cleans)[0]


__all__ = [
    "final_forward",
    "h3_forward",
    "h3_inner_forward",
    "mask_row_values",
    "mod_gate",
    "mod_row",
    "mod_scale_shift",
    "process_denoise_mask",
    "scale_latent_inpaint",
]
