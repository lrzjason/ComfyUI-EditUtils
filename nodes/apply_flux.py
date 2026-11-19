"""FLUX.2 klein rope-offset patch and the FLUX.3 model check."""

import comfy.utils
import torch
from einops import rearrange
import comfy.conds

class Flux2KleinEditApply_EditUtils:
    """Patch a Flux2Klein model to use per-reference ROPE position offsets.

    The Flux architecture natively supports h_offset/w_offset in process_img().
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
    CATEGORY = "Flux2Klein/edit"
    DESCRIPTION = (
        "Apply per-reference ROPE position offsets to a Flux2Klein model. "
        "Connect only the model wire — rope offsets flow through the "
        "EditUtils conditioning chain automatically."
    )

    def apply_patch(self, model, ref_pos_match_target=False, debug_log=False):
        # Detect Flux model by checking for process_img and pe_embedder
        is_flux = (
            hasattr(model, "model")
            and hasattr(model.model, "diffusion_model")
            and hasattr(model.model.diffusion_model, "process_img")
            and hasattr(model.model.diffusion_model, "pe_embedder")
        )
        if not is_flux:
            return (model,)

        m = model.clone()
        dit = m.get_model_object("diffusion_model")
        setattr(dit, "_editutils_ref_pos_match_target", bool(ref_pos_match_target))
        setattr(dit, "_editutils_flux_rope_debug", bool(debug_log))

        self._apply_flux_rope_patch(m, dit)
        return (m,)

    @staticmethod
    def _apply_flux_rope_patch(model_patcher, dit):
        """Install ROPE offset patch for Flux models."""
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
        def forward(x, timestep, context, y=None, guidance=None,
                    ref_latents=None, control=None, transformer_options={}, **kwargs):
            ref_rope_offsets = kwargs.get("ref_rope_offsets", None)

            # If no custom offsets or no refs, use original forward
            if ref_rope_offsets is None or ref_latents is None or len(ref_latents) == 0:
                return orig_forward(x, timestep, context, y=y, guidance=guidance,
                                    ref_latents=ref_latents, control=control,
                                    transformer_options=transformer_options, **kwargs)

            # Custom forward with per-ref ROPE offsets
            import comfy.patcher_extension
            from einops import repeat

            bs, c, h_orig, w_orig = x.shape
            patch_size = dit.patch_size
            h_len = ((h_orig + (patch_size // 2)) // patch_size)
            w_len = ((w_orig + (patch_size // 2)) // patch_size)

            img, img_ids = dit.process_img(x, transformer_options=transformer_options)
            img_tokens = img.shape[1]

            # Process refs with custom offsets
            ref_num_tokens = []
            pos_match = getattr(dit, "_editutils_ref_pos_match_target", False)
            ref_latents_method = kwargs.get("ref_latents_method", dit.params.default_ref_method)
            timestep_zero = ref_latents_method == "index_timestep_zero"
            index = 0

            for ref_idx, ref in enumerate(ref_latents):
                # Get custom offset for this ref
                if ref_idx < len(ref_rope_offsets):
                    x_off, y_off = ref_rope_offsets[ref_idx]
                else:
                    x_off, y_off = 0, 0

                # Use index method with custom spatial offsets
                index += dit.params.ref_index_scale
                h_offset = y_off  # pixels
                w_offset = x_off  # pixels

                kontext, kontext_ids = dit.process_img(
                    ref, index=index, h_offset=h_offset, w_offset=w_offset,
                    transformer_options=transformer_options
                )
                img = torch.cat([img, kontext], dim=1)
                img_ids = torch.cat([img_ids, kontext_ids], dim=1)
                ref_num_tokens.append(kontext.shape[1])

            timestep_zero_index = None
            if timestep_zero and index > 0:
                timestep = torch.cat([timestep, timestep * 0], dim=0)
                timestep_zero_index = [[img_tokens, img_ids.shape[1]]]

            transformer_options = transformer_options.copy()
            transformer_options["reference_image_num_tokens"] = ref_num_tokens

            txt_ids = torch.zeros((bs, context.shape[1], len(dit.params.axes_dim)),
                                  device=x.device, dtype=torch.float32)
            if len(dit.params.txt_ids_dims) > 0:
                for i in dit.params.txt_ids_dims:
                    txt_ids[:, :, i] = torch.linspace(0, context.shape[1] - 1,
                                                      steps=context.shape[1],
                                                      device=x.device, dtype=torch.float32)

            out = dit.forward_orig(img, img_ids, context, txt_ids, timestep, y,
                                   guidance, control,
                                   timestep_zero_index=timestep_zero_index,
                                   transformer_options=transformer_options,
                                   attn_mask=kwargs.get("attention_mask", None))
            out = out[:, :img_tokens]
            return rearrange(out, "b (h w) (c ph pw) -> b c (h ph) (w pw)",
                             h=h_len, w=w_len, ph=dit.patch_size, pw=dit.patch_size)[:, :, :h_orig, :w_orig]

        model_patcher.add_object_patch("extra_conds", extra_conds)
        model_patcher.add_object_patch("diffusion_model.forward", forward)


class Flux3KleinEditApply_EditUtils:
    """FLUX.3 klein edit patch.

    Unlike ``Flux2KleinEditApply_EditUtils`` this node does **not** need to
    rewrite the model forward. ComfyUI core's Flux3 model already consumes
    ``reference_latents`` natively (``comfy/model_base.py`` class ``Flux3``
    forwards them to ``comfy/ldm/flux3/model.py``, which derives the reference
    position ids itself via ``ref_ids``: negative ``l`` coords -1, -2, ...
    with ``image_rope_scale = 0.5``).

    What this node does provide is the one thing the EditUtils chain adds on top:
    a guard that the model actually is a FLUX.3 model, plus optional diagnostics.
    It exists so a FLUX.3 klein graph has the same shape as the other EditUtils
    edit graphs (and so a user who wires the FLUX.2 klein patch by mistake is
    told, instead of silently getting no reference conditioning).
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
            },
            "optional": {
                "debug_log": ("BOOLEAN", {"default": False}),
                "strict": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Raise instead of warning when the model is not a FLUX.3 model. "
                               "Disable to pass the model through unchanged."}),
            },
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "apply_patch"
    CATEGORY = "Flux3Klein/edit"
    DESCRIPTION = (
        "Validate a FLUX.3 model for the EditUtils reference-latent edit path. "
        "FLUX.3 consumes reference_latents natively, so no forward rewrite is "
        "performed; connect only the model wire."
    )

    @staticmethod
    def _is_flux3(model):
        """True when the patcher wraps comfy's native Flux3 diffusion model."""
        try:
            dit = model.get_model_object("diffusion_model")
        except Exception:
            return False, None
        # comfy/ldm/flux3/model.py Flux3: the give-away attributes are the
        # per-mode block dicts and the reference position-id helper.
        is_flux3 = (
            hasattr(dit, "content_mode_blocks")
            and hasattr(dit, "txt_mode_blocks")
            and hasattr(dit, "single_blocks")
            and hasattr(dit, "ref_ids")
        )
        return is_flux3, dit

    def apply_patch(self, model, debug_log=False, strict=True):
        is_flux3, dit = self._is_flux3(model)

        if not is_flux3:
            msg = (
                "Flux3KleinEditApply_EditUtils: the connected model does not look like a "
                "FLUX.3 model (no content_mode_blocks/ref_ids). It is probably a FLUX.2 klein "
                "or another architecture. Use Flux2KleinEditApply_EditUtils for FLUX.2 klein."
            )
            if strict:
                raise ValueError(msg)
            print(msg + " Passing the model through unchanged.")
            return (model,)

        m = model.clone()

        if debug_log:
            flow_map = bool(getattr(dit, "flow_map", False))
            print("[Flux3KleinEditApply] FLUX.3 model detected.")
            print(f"[Flux3KleinEditApply] flow_map (step-distilled): {flow_map}")
            if flow_map:
                print("[Flux3KleinEditApply] flow-map checkpoint: sample with ~4-16 steps and no CFG.")
            print("[Flux3KleinEditApply] reference latents are consumed natively "
                  "(model_base.Flux3 -> ref_ids, negative l coords, image_rope_scale 0.5).")

        return (m,)
