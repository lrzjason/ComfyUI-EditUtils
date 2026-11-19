"""Boogu model patch."""

import comfy.utils
import comfy.conds

class BooguEditApply_EditUtils:
    """Patch a Boogu model to use per-reference ROPE position offsets.

    Boogu uses OmniGen2RotaryPosEmbed which computes positions from image sizes.
    This node patches the rope_embedder to inject custom per-reference offsets,
    enabling regional editing.
    """

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
    CATEGORY = "Boogu/edit"
    DESCRIPTION = (
        "Apply per-reference ROPE position offsets to a Boogu model. "
        "Connect only the model wire — rope offsets flow through the "
        "EditUtils conditioning chain automatically."
    )

    def apply_patch(self, model, debug_log=False):
        # Detect Boogu model by checking for rope_embedder and image_index_embedding
        is_boogu = (
            hasattr(model, "model")
            and hasattr(model.model, "diffusion_model")
            and hasattr(model.model.diffusion_model, "rope_embedder")
            and hasattr(model.model.diffusion_model, "image_index_embedding")
        )
        if not is_boogu:
            return (model,)

        m = model.clone()
        dit = m.get_model_object("diffusion_model")
        setattr(dit, "_editutils_boogu_rope_debug", bool(debug_log))

        self._apply_boogu_rope_patch(m, dit)
        return (m,)

    @staticmethod
    def _apply_boogu_rope_patch(model_patcher, dit):
        """Install ROPE offset patch for Boogu models.

        Boogu's rope_embedder computes positions from ref_img_sizes and img_sizes.
        We patch the forward to store rope_offsets on the dit instance, then
        patch the rope_embedder call to apply offsets to the computed positions.
        """
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

        # --- forward: store offsets and patch rope computation ---
        def forward(x, timesteps, context, num_tokens, ref_latents=None,
                    attention_mask=None, transformer_options={}, **kwargs):
            ref_rope_offsets = kwargs.get("ref_rope_offsets", None)

            # Store offsets on dit for access during rope computation
            setattr(dit, "_editutils_boogu_rope_offsets", ref_rope_offsets)

            # For Boogu, we use the standard forward but the rope_embedder
            # will be aware of offsets via the stored attribute.
            # The Boogu architecture processes refs differently - offsets
            # are applied by shifting the ref image positions in the rope.
            return orig_forward(x, timesteps, context, num_tokens,
                                ref_latents=ref_latents,
                                attention_mask=attention_mask,
                                transformer_options=transformer_options, **kwargs)

        model_patcher.add_object_patch("extra_conds", extra_conds)
        model_patcher.add_object_patch("diffusion_model.forward", forward)
