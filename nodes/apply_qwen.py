"""QwenImage / Qwen-Image 2.1 rope-offset patches."""

import comfy.utils
import torch
from einops import rearrange
import comfy.conds

class QwenImageEditApply_EditUtils:
    """Patch a QwenImage model to use per-reference ROPE position offsets.

    The QwenImage architecture natively supports h_offset/w_offset in process_img().
    This node patches the model forward to inject custom per-reference offsets
    from the EditUtils conditioning chain, enabling regional editing.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
            },
            "optional": {
                "ref_pos_match_target": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Stretch ref positions to cover target grid. "
                               "Set False for regional editing with rope offsets."}),
                "debug_log": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "apply_patch"
    CATEGORY = "QwenImage/edit"
    DESCRIPTION = (
        "Apply per-reference ROPE position offsets to a QwenImage model. "
        "Connect only the model wire — rope offsets flow through the "
        "EditUtils conditioning chain automatically."
    )

    def apply_patch(self, model, ref_pos_match_target=False, debug_log=False):
        # Detect QwenImage model
        is_qwen_image = (
            hasattr(model, "model")
            and hasattr(model.model, "diffusion_model")
            and hasattr(model.model.diffusion_model, "process_img")
            and hasattr(model.model.diffusion_model, "pe_embedder")
            and hasattr(model.model.diffusion_model, "default_ref_method")
        )
        if not is_qwen_image:
            return (model,)

        m = model.clone()
        dit = m.get_model_object("diffusion_model")
        setattr(dit, "_editutils_ref_pos_match_target", bool(ref_pos_match_target))
        setattr(dit, "_editutils_qwen_rope_debug", bool(debug_log))

        self._apply_qwen_image_rope_patch(m, dit)
        return (m,)

    @staticmethod
    def _apply_qwen_image_rope_patch(model_patcher, dit):
        """Install ROPE offset patch for QwenImage models."""
        base_model = model_patcher.model
        orig_extra_conds = base_model.extra_conds
        orig_forward = dit.forward

        # --- extra_conds: extract rope offsets from conditioning ---
        def extra_conds(**kwargs):
            out = orig_extra_conds(**kwargs)
            rope_offsets = kwargs.get("reference_rope_offsets", None)
            if rope_offsets is not None:
                out["ref_rope_offsets"] = comfy.conds.CONDConstant(rope_offsets)
            return out

        # --- forward: inject per-ref offsets into process_img calls ---
        def forward(x, timesteps, context, attention_mask=None,
                    ref_latents=None, additional_t_cond=None,
                    transformer_options={}, **kwargs):
            ref_rope_offsets = kwargs.get("ref_rope_offsets", None)

            # If no custom offsets or no refs, use original forward
            if ref_rope_offsets is None or ref_latents is None or len(ref_latents) == 0:
                return orig_forward(x, timesteps, context,
                                    attention_mask=attention_mask,
                                    ref_latents=ref_latents,
                                    additional_t_cond=additional_t_cond,
                                    transformer_options=transformer_options, **kwargs)

            # Custom forward with per-ref ROPE offsets
            from einops import repeat

            timestep = timesteps
            encoder_hidden_states = context
            encoder_hidden_states_mask = attention_mask

            if encoder_hidden_states_mask is not None and not torch.is_floating_point(encoder_hidden_states_mask):
                encoder_hidden_states_mask = (encoder_hidden_states_mask - 1).to(x.dtype) * torch.finfo(x.dtype).max

            hidden_states, img_ids, orig_shape = dit.process_img(x)
            num_embeds = hidden_states.shape[1]

            # Process refs with custom offsets
            ref_num_tokens = []
            index = 0
            ref_method = kwargs.get("ref_latents_method", dit.default_ref_method)
            timestep_zero = ref_method == "index_timestep_zero"

            for ref_idx, ref in enumerate(ref_latents):
                # Get custom offset for this ref
                if ref_idx < len(ref_rope_offsets):
                    x_off, y_off = ref_rope_offsets[ref_idx]
                else:
                    x_off, y_off = 0, 0

                index += 1
                h_offset = y_off  # pixels
                w_offset = x_off  # pixels

                kontext, kontext_ids, _ = dit.process_img(
                    ref, index=index, h_offset=h_offset, w_offset=w_offset
                )
                hidden_states = torch.cat([hidden_states, kontext], dim=1)
                img_ids = torch.cat([img_ids, kontext_ids], dim=1)
                ref_num_tokens.append(kontext.shape[1])

            timestep_zero_index = None
            if timestep_zero and index > 0:
                timestep = torch.cat([timestep, timestep * 0], dim=0)
                timestep_zero_index = num_embeds

            transformer_options = transformer_options.copy()
            transformer_options["reference_image_num_tokens"] = ref_num_tokens

            txt_start = round(max(((x.shape[-1] + (dit.patch_size // 2)) // dit.patch_size) // 2,
                                  ((x.shape[-2] + (dit.patch_size // 2)) // dit.patch_size) // 2))
            txt_ids = torch.arange(txt_start, txt_start + context.shape[1],
                                   device=x.device).reshape(1, -1, 1).repeat(x.shape[0], 1, 3)

            hidden_states = dit.img_in(hidden_states)
            encoder_hidden_states = dit.txt_norm(encoder_hidden_states)
            encoder_hidden_states = dit.txt_in(encoder_hidden_states)

            temb = dit.time_text_embed(timestep, hidden_states, additional_t_cond)

            ids = torch.cat((txt_ids, img_ids), dim=1)
            image_rotary_emb = dit.pe_embedder(ids).to(x.dtype).contiguous()
            del ids, txt_ids, img_ids

            # Run transformer blocks
            for i, block in enumerate(dit.transformer_blocks):
                transformer_options["block_index"] = i
                hidden_states, encoder_hidden_states = block(
                    hidden_states=hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    temb=temb,
                    image_rotary_emb=image_rotary_emb,
                    encoder_attention_mask=encoder_hidden_states_mask,
                    transformer_options=transformer_options,
                )

            # Output
            hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)
            hidden_states = dit.norm_out(hidden_states, temb)
            hidden_states = dit.proj_out(hidden_states)

            # Extract image tokens and unpatchify
            p = dit.patch_size
            output = hidden_states[:, -num_embeds:]
            h_len = orig_shape[-2] // p
            w_len = orig_shape[-1] // p
            output = rearrange(output, 'b (h w) (p1 p2 c) -> b c (h p1) (w p2)',
                               h=h_len, w=w_len, p1=p, p2=p)
            return output[:, :, :x.shape[2], :x.shape[3]]

        model_patcher.add_object_patch("extra_conds", extra_conds)
        model_patcher.add_object_patch("diffusion_model.forward", forward)


def _editutils_qwen21_build_sequence(dit, x, context, ref_latents, image_slots, rope_offsets=None):
    """Reimplementation of comfy.ldm.qwen_image21 model build_sequence with
    per-reference ROPE offsets added (offsets are in latent-token units).

    Faithful to upstream comfy/ldm/qwen_image21/model.py: text split at each
    vision slot with the reference latents spliced in, target image last;
    reference rope ids centred on the target with a half-token shift where the
    reference grid has the other parity. Only difference: hh/ww get the offset.
    """
    txt = dit.txt_in(context)
    ref_latents = list(ref_latents or [])
    slots = (list(image_slots or []) + [txt.shape[1]] * len(ref_latents))[:len(ref_latents)]
    bounds = [0] + slots + [txt.shape[1]]

    parts, ids, segments = [], [], []
    pos, length = 0, 0
    for i, ((start, end), img) in enumerate(zip(zip(bounds[:-1], bounds[1:]), ref_latents + [x])):
        n = end - start
        if n > 0:
            parts.append(txt[:, start:end])
            ids.append(torch.arange(pos, pos + n, device=x.device, dtype=torch.float32).unsqueeze(1).expand(n, 3))
            segments.append((length, length + n, torch.ones((n, length + n), dtype=torch.bool, device=x.device).tril(length)))
            pos += n
            length += n
        h, w = img.shape[-2:]
        parts.append(dit.img_in(img.flatten(2).transpose(1, 2)))
        x_off, y_off = rope_offsets[i] if rope_offsets is not None and i < len(rope_offsets) else (0, 0)
        # half a token where a reference grid has the other parity, so it centres on the target
        hh = torch.arange(h, device=x.device, dtype=torch.float32) - (h - h // 2) + 0.5 * (h % 2 - x.shape[-2] % 2) + y_off
        ww = torch.arange(w, device=x.device, dtype=torch.float32) - (w - w // 2) + 0.5 * (w % 2 - x.shape[-1] % 2) + x_off
        ids.append(torch.stack([torch.full((h, w), pos, device=x.device, dtype=torch.float32),
                                hh[:, None].expand(h, w), ww[None, :].expand(h, w)], dim=-1).flatten(0, 1))
        segments.append((length, length + h * w, None))
        pos += max(h, w)
        length += h * w

    pe = dit.pe_embedder(torch.cat(ids, dim=0).unsqueeze(0)).transpose(1, 2).contiguous()
    return torch.cat(parts, dim=1), pe, segments


class QwenImage21EditApply_EditUtils:
    """Patch a QwenImage21 model to use per-reference ROPE position offsets.

    The QwenImage 2.1 DiT natively splices reference latents into the text
    sequence at the vision slots (image_slots) and centres their rope ids on
    the target image. This node additionally shifts each reference's rope
    positions by per-reference pixel offsets from the EditUtils conditioning
    chain (rope_x_offset / rope_y_offset), enabling regional editing.
    """

    # latent tokens per 16 latent-channel VAE unit; rope offsets are given in pixels
    LATENT_DOWNSCALE = 16

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
            },
            "optional": {
                "debug_log": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "apply_patch"
    CATEGORY = "QwenImage/edit"
    DESCRIPTION = (
        "Apply per-reference ROPE position offsets to a QwenImage 2.1 model. "
        "Connect only the model wire — rope offsets flow through the "
        "EditUtils conditioning chain automatically."
    )

    def apply_patch(self, model, debug_log=False):
        # Detect QwenImage21 models via the detected unet config
        model_config = getattr(model.model, "model_config", None)
        unet_config = getattr(model_config, "unet_config", None) or {}
        dit = getattr(model.model, "diffusion_model", None)
        is_qwen_image21 = (
            isinstance(unet_config, dict)
            and unet_config.get("image_model") == "qwen_image21"
            and dit is not None
            and hasattr(dit, "build_sequence")
            and hasattr(dit, "pe_embedder")
        )
        if not is_qwen_image21:
            print("QwenImage21EditApply_EditUtils: model is not QwenImage21, skipping patch. "
                  "(Requires ComfyUI with Qwen-Image 2.1 support.)")
            return (model,)

        m = model.clone()
        dit = m.get_model_object("diffusion_model")
        setattr(dit, "_editutils_qwen21_rope_debug", bool(debug_log))

        self._apply_qwen_image21_rope_patch(m, dit)
        return (m,)

    @staticmethod
    def _apply_qwen_image21_rope_patch(model_patcher, dit):
        """Install ROPE offset patch for QwenImage21 models."""
        base_model = model_patcher.model
        orig_extra_conds = base_model.extra_conds
        orig_forward = dit.forward
        orig_build_sequence = dit.build_sequence

        # --- extra_conds: extract rope offsets from conditioning ---
        def extra_conds(**kwargs):
            out = orig_extra_conds(**kwargs)
            rope_offsets = kwargs.get("reference_rope_offsets", None)
            if rope_offsets is not None:
                out["ref_rope_offsets"] = comfy.conds.CONDConstant(rope_offsets)
            return out

        # --- forward: stash the per-ref offsets for build_sequence ---
        def forward(x, timesteps, context, ref_latents=None, image_slots=None,
                    transformer_options={}, **kwargs):
            setattr(dit, "_editutils_qwen21_rope_offsets", kwargs.get("ref_rope_offsets", None))
            return orig_forward(x, timesteps, context,
                                ref_latents=ref_latents,
                                image_slots=image_slots,
                                transformer_options=transformer_options, **kwargs)

        # --- build_sequence: apply the stashed offsets when present ---
        def build_sequence(x, context, ref_latents, image_slots):
            offsets = getattr(dit, "_editutils_qwen21_rope_offsets", None)
            if offsets is None or all((int(o[0]), int(o[1])) == (0, 0) for o in offsets):
                # zero offsets: native path, keeps upstream behavior and prefix cache
                return orig_build_sequence(x, context, ref_latents, image_slots)
            if getattr(dit, "_editutils_qwen21_rope_debug", False):
                print(f"[QwenImage21EditApply] rope offsets (px): {offsets}")
            latent_offsets = [(o[0] / QwenImage21EditApply_EditUtils.LATENT_DOWNSCALE,
                               o[1] / QwenImage21EditApply_EditUtils.LATENT_DOWNSCALE) for o in offsets]
            return _editutils_qwen21_build_sequence(dit, x, context, ref_latents, image_slots,
                                                    rope_offsets=latent_offsets)

        model_patcher.add_object_patch("extra_conds", extra_conds)
        model_patcher.add_object_patch("diffusion_model.forward", forward)
        model_patcher.add_object_patch("diffusion_model.build_sequence", build_sequence)
