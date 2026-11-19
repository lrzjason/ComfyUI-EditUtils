"""MiniMax-H3-Image model check (no forward rewrite needed)."""



class MinimaxH3ImageEditApply_EditUtils:
    """Model check for the **MiniMax-H3-Image** edit path (no forward rewrite).

    ComfyUI core already implements H3I natively: ``comfy/model_base.py`` class
    ``MiniMaxH3Image`` reads the ``minimax_refs`` conditioning key, builds the
    ``PackedLayout`` ``[text | references | target]`` and forwards to
    ``comfy/ldm/minimax/model.py`` ``MiniMaxH3ImageModel``, which re-injects the
    reference rows at ``t=0.999`` and predicts with the checkpoint's PDD head
    for the current sigma. Nothing needs to be patched, so this node is a guard
    in the shape of the other EditUtils edit nodes:

    - it fails (or warns, with ``strict=False``) when the model is not an H3
      **image** model — e.g. the H3 video/audio ``minimax_h3`` model or another
      architecture wired by mistake, which would silently ignore the refs;
    - with ``debug_log`` it prints the checkpoint's PDD sigma grid, the step
      count that grid implies (``len(grid) - 1``, 6 for the released
      checkpoint), and the CFG/negative contract.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
            },
            "optional": {
                "debug_log": ("BOOLEAN", {"default": True}),
                "strict": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Raise instead of warning when the model is not a MiniMax-H3-Image model. "
                               "Disable to pass the model through unchanged."}),
            },
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "apply_patch"
    CATEGORY = "MinimaxH3Image/edit"
    DESCRIPTION = (
        "Validate a MiniMax-H3-Image model for the EditUtils reference edit path. "
        "H3I consumes minimax_refs natively, so no forward rewrite is performed; "
        "connect only the model wire."
    )

    @staticmethod
    def _h3_image_info(model):
        """(is_h3_image, dit, sampling) for the patcher's diffusion model."""
        try:
            dit = model.get_model_object("diffusion_model")
        except Exception:
            return False, None, None
        sampling = getattr(getattr(model, "model", None), "model_sampling", None)
        # comfy/ldm/minimax/model.py MiniMaxH3ImageModel: no audio stream, and
        # the PDD head bank is driven by the checkpoint's pdd_sigmas.
        is_h3_image = (
            hasattr(dit, "pdd_sigmas")
            and getattr(dit, "audio_latents_dim", "missing") is None
            and hasattr(dit, "video_patch_proj")
        )
        return is_h3_image, dit, sampling

    def apply_patch(self, model, debug_log=True, strict=True):
        is_h3_image, dit, sampling = self._h3_image_info(model)

        if not is_h3_image:
            msg = (
                "MinimaxH3ImageEditApply_EditUtils: the connected model does not look like a "
                "MiniMax-H3-Image model (no pdd_sigmas / audio_latents_dim=None / video_patch_proj). "
                "The H3 video model (minimax_h3) and other architectures ignore `minimax_refs`, "
                "so the EditUtils references would have no effect."
            )
            if strict:
                raise ValueError(msg)
            print(msg + " Passing the model through unchanged.")
            return (model,)

        m = model.clone()

        if debug_log:
            sigmas = [float(s) for s in getattr(dit, "pdd_sigmas", [])]
            steps = max(1, len(sigmas) - 1)
            print("[MinimaxH3ImageEditApply] MiniMax-H3-Image model detected.")
            if sigmas:
                print(f"[MinimaxH3ImageEditApply] PDD sigma grid: {[round(s, 4) for s in sigmas]} "
                      f"-> sample with exactly {steps} steps (KSampler steps={steps}).")
            print("[MinimaxH3ImageEditApply] CFG is baked into the checkpoint: use cfg=1.0 and a "
                  "zeroed-out (ConditioningZeroOut) negative.")
            print("[MinimaxH3ImageEditApply] references ride in `minimax_refs` and are re-injected at "
                  "t=0.999 every step; the target is one 24ch latent frame at H/16 x W/16.")
            if sampling is not None:
                grid = getattr(sampling, "grid", None)
                if grid is not None:
                    print(f"[MinimaxH3ImageEditApply] sampler sigma grid: {len(grid) - 1} intervals.")

        return (m,)
