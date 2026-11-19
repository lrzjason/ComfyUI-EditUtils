"""Krea2 model patch (reference tokens / rope offsets / optional ref KV cache)."""

import comfy.utils
import math
import torch
from einops import rearrange
import comfy.conds

# Sentinel used internally to stash the original forward method on the instance.
_EDITUTILS_ORIGINAL_FORWARD_ATTR = "_editutils_krea2_original_forward"


def _editutils_make_ref_positions(ref_latents_list, bs, patch_size, device, dtype,
                                  scale_to_grid=None, rope_offsets=None):
    """Build position-id blocks for a list of reference latents.

    Each reference gets a positive incrementing frame index
    (matching UnifiedTrainer / ai-toolkit ``index_timestep_zero``):
    ref₁=+1, ref₂=+2, …  Returns ``(ref_tokens_list, ref_pos_list, total_reflen)``.

    ``scale_to_grid``: optional ``(th, tw)`` target token grid. When given, ref
    (h, w) ids are center-aligned to the target grid using the formula
    ``(i + 0.5) * (target / ref) - 0.5``. Each ref token sits at the center of
    the target region it represents, giving balanced spatial correspondence
    without adding tokens. Use when ref and target resolutions differ.

    ``rope_offsets``: optional list of ``(x_offset, y_offset)`` tuples in pixels,
    one per reference. Applied AFTER scale_to_grid. Converts pixel offset to
    token offset by dividing by patch_size. Use for regional editing: place
    each reference at a specific canvas location by offsetting its position IDs.
    """
    import comfy.ldm.common_dit

    ref_tokens = []
    ref_positions = []
    total = 0

    for i, ref in enumerate(ref_latents_list, start=1):
        # Squeeze any singleton temporal dimension (B,C,1,H,W) → (B,C,H,W)
        if ref.ndim == 5 and ref.shape[2] == 1:
            ref = ref.squeeze(2)
        elif ref.ndim == 5:
            raise ValueError(
                f"Krea2EditApply_EditUtils: reference latent has {ref.ndim}D shape "
                f"{tuple(ref.shape)} — expected 4D (B,C,H,W) or 5D with temporal=1"
            )

        if ref.shape[0] != bs:
            ref = ref.expand(bs, *ref.shape[1:])

        ref_pad = comfy.ldm.common_dit.pad_to_patch_size(ref, (patch_size, patch_size))
        rH, rW = ref_pad.shape[-2], ref_pad.shape[-1]
        rh, rw = rH // patch_size, rW // patch_size

        # Patchify
        rt = rearrange(ref_pad, "b c (h ph) (w pw) -> b (h w) (c ph pw)",
                       ph=patch_size, pw=patch_size)
        ref_tokens.append(rt)

        # Position IDs: positive incrementing frame index
        # (matching UnifiedTrainer prepare_position_ids: ref_i = +i)
        frame_idx = float(i)
        rids = torch.zeros(rh, rw, 3, device=device, dtype=dtype)
        rids[..., 0] = frame_idx
        h_ids = torch.arange(rh, device=device, dtype=dtype)
        w_ids = torch.arange(rw, device=device, dtype=dtype)
        if scale_to_grid is not None:
            th, tw = scale_to_grid
            # Center-aligned mapping: each ref token sits at the center of the
            # target region it represents. For 2x upscale (th=32, rh=16):
            #   old (edge-aligned): [0, 2, 4, ..., 30]  — never reaches 31
            #   new (center-aligned): [0.5, 2.5, ..., 30.5] — balanced coverage
            h_ids = (h_ids + 0.5) * (float(th) / float(rh)) - 0.5
            w_ids = (w_ids + 0.5) * (float(tw) / float(rw)) - 0.5
        # Apply per-reference spatial offset for regional editing (pixels → tokens)
        if rope_offsets is not None and (i - 1) < len(rope_offsets):
            x_off, y_off = rope_offsets[i - 1]
            if x_off != 0 or y_off != 0:
                h_ids = h_ids + (float(y_off) / float(patch_size))
                w_ids = w_ids + (float(x_off) / float(patch_size))
        rids[..., 1] = h_ids[:, None]
        rids[..., 2] = w_ids[None, :]
        ref_positions.append(rids.reshape(1, rh * rw, 3).repeat(bs, 1, 1))

        total += rh * rw

    return ref_tokens, ref_positions, total


def _editutils_krea2_edit_forward(self, x, timesteps, context, attention_mask=None,
                                   transformer_options=None, ref_latents=None, **kwargs):
    """Patched forward for SingleStreamDiT with reference latent support.

    Sequence order (matching UnifiedTrainer Krea2Adapter.prepare_model_input):
        [text | target | ref₁ | ref₂ | …]

    Position IDs: ref_n = (+n, h, w), target = (0, h, w)
    Per-token timestep: ref tokens get t=0 modulation, text+target get real timestep.
    Output sliced to return only target tokens.
    """
    from comfy.conds import CONDList

    # ── Normalize ref_latents to list of tensors ──
    if ref_latents is None:
        ref_list = []
    elif isinstance(ref_latents, torch.Tensor):
        ref_list = [ref_latents]
    elif isinstance(ref_latents, CONDList):
        ref_list = [r for r in ref_latents.cond if r is not None]
    elif isinstance(ref_latents, (list, tuple)):
        ref_list = [r for r in ref_latents if r is not None and isinstance(r, torch.Tensor)]
    else:
        ref_list = []

    mode = getattr(self, "_editutils_ref_timestep_mode", "editutils")

    if len(ref_list) == 0:
        original = getattr(self, _EDITUTILS_ORIGINAL_FORWARD_ATTR)
        return original(x, timesteps, context,
                        attention_mask=attention_mask,
                        transformer_options=transformer_options,
                        **kwargs)

    # ── Reference-aware path ──
    temporal = x.ndim == 5
    if temporal:
        b5, c5, t5, h5, w5 = x.shape
        x = x.reshape(b5 * t5, c5, h5, w5)

    bs, c, H_orig, W_orig = x.shape
    patch_size = self.patch
    import comfy.ldm.common_dit

    # Pad target to patch size
    x_pad = comfy.ldm.common_dit.pad_to_patch_size(x, (patch_size, patch_size))
    H, W = x_pad.shape[-2], x_pad.shape[-1]
    h_grid, w_grid = H // patch_size, W // patch_size

    # Build reference tokens + positions
    device = x_pad.device
    dtype = torch.float32
    pos_scale = (h_grid, w_grid) if getattr(
        self, "_editutils_ref_pos_match_target", False) else None
    ref_rope_offsets = kwargs.get("ref_rope_offsets", None)  # ROPE position offsets for regional editing
    ref_tokens_list, ref_pos_list, total_reflen = _editutils_make_ref_positions(
        ref_list, bs, patch_size, device, dtype, scale_to_grid=pos_scale,
        rope_offsets=ref_rope_offsets,
    )

    # Unpack context (ComfyUI Krea2 SingleStreamDiT convention)
    context = self._unpack_context(context)

    # Patchify + first linear for target
    img = rearrange(x_pad, "b c (h ph) (w pw) -> b (h w) (c ph pw)",
                    ph=patch_size, pw=patch_size)
    img = self.first(img)

    # First linear for each reference
    ref_imgs = [self.first(rt) for rt in ref_tokens_list]

    # ── Per-token timestep embedding (matching UnifiedTrainer reflen split) ──
    from comfy.ldm.flux.layers import timestep_embedding

    t_emb_real = self.tmlp(
        timestep_embedding(timesteps, self.tdim).unsqueeze(1).to(img.dtype))
    tvec_real = self.tproj(t_emb_real)   # (B, 1, dim)

    # Text fusion + MLP (match UnifiedTrainer's text_fusion → txt_in pipeline)
    context = self.txtfusion(context, mask=None,
                             transformer_options=transformer_options or {})
    context = self.txtmlp(context)

    txtlen = context.shape[1]
    imglen = img.shape[1]

    # ── Combined sequence: [text | target | ref₁ | ref₂ | …] ──
    # (matches UnifiedTrainer prepare_model_input concatenation order)
    combined = torch.cat([context, img] + ref_imgs, dim=1)

    # Position IDs (same order)
    txtpos = torch.zeros(bs, txtlen, 3, device=device, dtype=dtype)  # text: (0,0,0)
    tgtids = torch.zeros(h_grid, w_grid, 3, device=device, dtype=dtype)
    tgtids[..., 1] = torch.arange(h_grid, device=device, dtype=dtype)[:, None]
    tgtids[..., 2] = torch.arange(w_grid, device=device, dtype=dtype)[None, :]
    tgtpos = tgtids.reshape(1, h_grid * w_grid, 3).repeat(bs, 1, 1)
    pos = torch.cat([txtpos, tgtpos] + ref_pos_list, dim=1)
    freqs = self.pe_embedder(pos)

    # Per-token tvec: text=real, target=real. Reference tokens depend on mode:
    #   "editutils" (default) -> refs modulated as clean latents (t=0), matching
    #                            the UnifiedTrainer per-span recipe.
    #   "krea2edit"           -> refs share the real timestep (ai-toolkit
    #                            predict_velocity_edit), so a krea2edit-trained
    #                            LoRA runs on its native inference geometry.
    if mode == "krea2edit":
        tvec_ref = tvec_real.expand(-1, total_reflen, -1)
    else:
        t_emb_zero = self.tmlp(
            timestep_embedding(torch.zeros_like(timesteps), self.tdim).unsqueeze(1).to(img.dtype))
        tvec_ref = self.tproj(t_emb_zero).expand(-1, total_reflen, -1)
    tvec = torch.cat([tvec_real.expand(-1, txtlen, -1),
                      tvec_real.expand(-1, imglen, -1),
                      tvec_ref], dim=1)  # (B, total_seq, dim)

    # Transformer blocks
    for block in self.blocks:
        combined = block(combined, tvec, freqs, attention_mask,
                         transformer_options=transformer_options or {})

    # Final layer + slice target output (text:0→txtlen, target:txtlen→txtlen+imglen)
    final = self.last(combined, t_emb_real)
    out = final[:, txtlen: txtlen + imglen, :]

    # Un-patchify
    out = rearrange(out,
                    "b (h w) (c ph pw) -> b c (h ph) (w pw)",
                    h=h_grid, w=w_grid, ph=patch_size, pw=patch_size, c=self.channels)
    out = out[:, :, :H_orig, :W_orig]  # crop padding

    # NaN guard
    if not torch.isfinite(out).all():
        raise ValueError("Krea2EditApply_EditUtils: edit forward produced NaN/Inf output")

    if temporal:
        out = out.reshape(b5, t5, self.channels, H_orig, W_orig).movedim(1, 2)

    return out


class Krea2EditApply_EditUtils:
    """Patch a Krea2 model to accept reference latents via EditUtils conditioning.

    Matches UnifiedTrainer's forward logic:

    - Sequence: ``[text | target | ref₁ | ref₂ | …]``
    - Position IDs: ref_n = (+n, h, w), target = (0, h, w)
    - Per-token timestep: ref tokens = t=0 (clean latents), text+target = real timestep
    - Output sliced to return only target tokens

    Connect only the ``model`` wire — reference latents are extracted
    automatically from the EditUtils conditioning chain via an ``extra_conds``
    patch.  Patches via ``ModelPatcher.add_object_patch`` on a cloned model
    instance, properly tracked by ComfyUI's patch/unpatch lifecycle.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
            },
            "optional": {
                "mode": (["editutils", "krea2edit"], {
                    "default": "editutils",
                    "tooltip": "Reference-token timestep convention. "
                               "'editutils' modulates reference tokens as clean "
                               "latents (t=0, UnifiedTrainer recipe). 'krea2edit' "
                               "gives references the real timestep (ai-toolkit "
                               "predict_velocity_edit) — select this to run a "
                               "comfyui-krea2edit-trained LoRA on its native "
                               "inference geometry."}),
                "ref_pos_match_target": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Center-align ref position ids to cover the "
                               "target token grid. Each ref token is placed at "
                               "the center of the target region it represents. "
                               "Enable when reference and target resolutions "
                               "differ — otherwise a smaller ref only aligns to "
                               "the target's top-left region."}),
                "ref_kv_cache": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Cache reference-image attention K/V at the "
                               "first denoise step and reuse them for all "
                               "later steps and repeated generations with "
                               "identical refs (~2x faster). Frozen-reference "
                               "approximation, not bit-exact."}),
                "ref_strength": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "How long references constrain sampling. 1.0 = "
                               "refs attend all steps; 0.5 = refs attend the "
                               "first half of steps, then generation continues "
                               "as pure text-to-image; 0.0 = refs never "
                               "participate. Requires ref_kv_cache."}),
                "reset_cache": ("BOOLEAN", {"default": True}),
                "debug_log": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "apply_patch"
    CATEGORY = "Krea2/edit"
    DESCRIPTION = (
        "Apply reference-image conditioning to a Krea2 model. "
        "Connect only the model wire — reference latents flow "
        "through the EditUtils conditioning chain automatically. "
        "Supports multiple references. Patch is isolated to a "
        "cloned model instance via add_object_patch."
    )

    def apply_patch(self, model, mode="editutils", ref_pos_match_target=True,
                    ref_kv_cache=True, ref_strength=1.0, reset_cache=True,
                    debug_log=False):
        # Only patch Krea2 models (detect by _unpack_context attribute)
        is_krea2 = (
            hasattr(model, "model")
            and hasattr(model.model, "diffusion_model")
            and hasattr(model.model.diffusion_model, "_unpack_context")
        )
        if not is_krea2:
            return (model,)

        # Clone to isolate the patch (prevents shared-instance mutation)
        m = model.clone()

        dit = m.get_model_object("diffusion_model")
        setattr(dit, "_editutils_ref_pos_match_target", bool(ref_pos_match_target))
        setattr(dit, "_editutils_ref_timestep_mode", str(mode))

        self._apply_krea2_edit_patch(m)

        if ref_kv_cache:
            _install_ref_kv_cache_patch(m, dit, reset_cache=reset_cache,
                                        ref_strength=ref_strength,
                                        debug_log=debug_log)

        return (m,)

    @staticmethod
    def _apply_krea2_edit_patch(model_patcher):
        """Install reference-image forward patch via ModelPatcher.add_object_patch.

        Patches registered:
          - extra_conds: extracts reference_latents from conditioning (EditUtils pattern)
          - extra_conds_shapes: declares shape info for memory allocation
          - diffusion_model.forward: reference-aware dispatch (falls back to original when no refs)
        """
        base_model = model_patcher.model
        dit = model_patcher.get_model_object("diffusion_model")

        orig_extra_conds = base_model.extra_conds
        orig_extra_conds_shapes = base_model.extra_conds_shapes
        orig_forward = dit.forward

        # Save original forward once (idempotent across multiple apply calls)
        if not hasattr(dit, _EDITUTILS_ORIGINAL_FORWARD_ATTR):
            setattr(dit, _EDITUTILS_ORIGINAL_FORWARD_ATTR, orig_forward)

        # --- extra_conds: inject ref_latents from EditUtils conditioning ---
        def extra_conds(**kwargs):
            out = orig_extra_conds(**kwargs)
            cond_refs = kwargs.get("reference_latents", None)
            rope_offsets = kwargs.get("reference_rope_offsets", None)  # ROPE position offsets
            if cond_refs is not None and len(cond_refs) > 0:
                # EditUtils stores vae.encode() output which is a dict
                # {"samples": tensor} — extract the tensor first.
                ref_tensors = [
                    r["samples"] if isinstance(r, dict) else r
                    for r in cond_refs
                ]
                out["ref_latents"] = comfy.conds.CONDList([
                    base_model.process_latent_in(t) for t in ref_tensors
                ])
                if rope_offsets is not None:
                    # Wrap in CONDConstant so ComfyUI treats it as a model_cond
                    # (raw lists lack .process_cond and break sampling).
                    out["ref_rope_offsets"] = comfy.conds.CONDConstant(rope_offsets)
            return out

        # --- extra_conds_shapes: declare shape for memory allocation ---
        def extra_conds_shapes(**kwargs):
            out = orig_extra_conds_shapes(**kwargs)
            cond_refs = kwargs.get("reference_latents", None)
            if cond_refs is not None and len(cond_refs) > 0:
                ref_tensors = [
                    r["samples"] if isinstance(r, dict) else r
                    for r in cond_refs
                ]
                out["ref_latents"] = list(
                    [1, dit.channels,
                     sum(math.prod(t.size()[2:]) for t in ref_tensors) // dit.channels]
                )
            return out

        # --- forward: reference-aware dispatch ---
        def forward(
            x, timesteps, context,
            attention_mask=None,
            transformer_options=None,
            ref_latents=None,
            **kwargs,
        ):
            if transformer_options is None:
                transformer_options = {}

            if ref_latents is None or len(ref_latents) == 0:
                saved = getattr(dit, _EDITUTILS_ORIGINAL_FORWARD_ATTR)
                return saved(
                    x, timesteps, context,
                    attention_mask=attention_mask,
                    transformer_options=transformer_options,
                    **kwargs,
                )
            # Route through WrapperExecutor so DIFFUSION_MODEL wrappers from other
            # nodes/hooks still fire during edit generation.
            import comfy.patcher_extension
            return comfy.patcher_extension.WrapperExecutor.new_class_executor(
                _editutils_krea2_edit_forward, dit,
                comfy.patcher_extension.get_all_wrappers(
                    comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL, transformer_options),
            ).execute(dit, x, timesteps, context,
                      attention_mask=attention_mask,
                      transformer_options=transformer_options,
                      ref_latents=ref_latents,
                      **kwargs)

        model_patcher.add_object_patch("extra_conds", extra_conds)
        model_patcher.add_object_patch("extra_conds_shapes", extra_conds_shapes)
        model_patcher.add_object_patch("memory_usage_factor_conds", ("ref_latents",))
        model_patcher.add_object_patch("diffusion_model.forward", forward)


_KV_STATE_ATTR = "_editutils_ref_kv_state"       # per-call capture/reuse state


_KV_CACHE_ATTR = "_editutils_ref_kv_cache"       # persistent cache dict


_KV_CFG_ATTR = "_editutils_ref_kv_cfg"           # node-provided tuning config


def _ref_fingerprint(tensor):
    """Cheap content fingerprint for a ref latent tensor, memoized ON the
    tensor object itself (attribute dies with the tensor, so no id-reuse
    collisions) to avoid re-syncing GPU->CPU every denoise step."""
    fp = getattr(tensor, "_editutils_fp", None)
    if fp is not None:
        return fp
    t = tensor.detach()
    if t.device.type != "cpu":
        t = t.to("cpu", dtype=torch.float32)
    else:
        t = t.float()
    flat = t.flatten()
    stride = max(1, flat.numel() // 64)
    sample = tuple(round(float(v), 4) for v in flat[::stride][:64])
    fp = (
        tuple(t.shape),
        round(float(t.mean()), 6),
        round(float(t.std()), 6),
        round(float(t.abs().sum()), 3),
        sample,
    )
    try:
        tensor._editutils_fp = fp
    except Exception:
        pass
    return fp


def _build_cache_key(model_uuid, dtype, ref_list):
    return (
        model_uuid,
        str(dtype),
        tuple(_ref_fingerprint(r) for r in ref_list),
    )


def _make_ref_kv_attn_wrapper(attn_module, dit):
    """Reimplement Krea2 Attention.forward (comfy/ldm/krea2/model.py:75-89)
    with a capture/reuse hook driven by ``dit._editutils_ref_kv_state``.

    Capture: store ref-slice k/v (post qk-norm, post-RoPE, pre GQA repeat).
    Reuse:   concat cached ref k/v onto the freshly computed k/v.
    """
    from comfy.ldm.flux.math import apply_rope
    from comfy.ldm.modules.attention import optimized_attention_masked

    def forward(x, freqs=None, mask=None, transformer_options={}):
        q, k, v, gate = attn_module.wq(x), attn_module.wk(x), attn_module.wv(x), attn_module.gate(x)
        q = rearrange(q, "B L (H D) -> B H L D", H=attn_module.heads)
        k = rearrange(k, "B L (H D) -> B H L D", H=attn_module.kvheads)
        v = rearrange(v, "B L (H D) -> B H L D", H=attn_module.kvheads)
        q, k = attn_module.qknorm(q, k)
        if freqs is not None:
            q, k = apply_rope(q, k, freqs)

        state = getattr(dit, _KV_STATE_ATTR, None)
        if state is not None:
            layer = state["layer_counter"]
            state["layer_counter"] = layer + 1
            ref_toks = state["ref_toks"]
            if state["mode"] == "capture":
                # Refs are appended last -> tail slice. Stored on CPU to
                # avoid pinning GPU memory across steps/generations; each
                # layer's slice is tiny (~MBs) so reuse-time upload is cheap.
                state["store"][layer] = (
                    k[:, :, -ref_toks:].detach().to("cpu", copy=True),
                    v[:, :, -ref_toks:].detach().to("cpu", copy=True),
                )
            elif state["mode"] == "reuse" and not state.get("drop_refs"):
                kk, vv = state["store"][layer]
                if kk.device != k.device:
                    kk = kk.to(k.device, non_blocking=True)
                    vv = vv.to(v.device, non_blocking=True)
                kk = comfy.utils.repeat_to_batch_size(kk, k.shape[0])
                vv = comfy.utils.repeat_to_batch_size(vv, v.shape[0])
                k = torch.cat((k, kk), dim=2)
                v = torch.cat((v, vv), dim=2)

        if attn_module.kvheads != attn_module.heads:
            rep = attn_module.heads // attn_module.kvheads
            k = k.repeat_interleave(rep, dim=1)
            v = v.repeat_interleave(rep, dim=1)
        out = optimized_attention_masked(
            q, k, v, attn_module.heads, mask=mask, skip_reshape=True,
            transformer_options=transformer_options,
        )
        return attn_module.wo(out * torch.nn.functional.sigmoid(gate))

    return forward


class _ref_kv_attn_patch_guard:
    """Temporarily wrap every block's attention forward; always restore."""

    def __init__(self, dit):
        self.dit = dit
        self.saved = []

    def __enter__(self):
        for block in self.dit.blocks:
            orig = block.attn.forward
            self.saved.append((block, orig))
            block.attn.forward = _make_ref_kv_attn_wrapper(block.attn, self.dit)
        return self

    def __exit__(self, *exc):
        for block, orig in self.saved:
            block.attn.forward = orig
        self.saved = []
        return False


def _editutils_krea2_edit_forward_cached(
    self, x, timesteps, context, attention_mask=None,
    transformer_options=None, ref_latents=None, **kwargs,
):
    """KV-cached variant of _editutils_krea2_edit_forward.

    Sequence order identical to the original: [text | target | ref1 | ...].
    On cache miss: full forward + per-block ref K/V capture.
    On cache hit:  forward over [text | target] only, cached ref K/V
                   concatenated inside each attention call.
    Fallback: any anomaly -> original (uncached) edit forward.
    """
    import comfy.ldm.common_dit
    import comfy.patcher_extension
    from comfy.conds import CONDList
    from comfy.ldm.flux.layers import timestep_embedding

    if transformer_options is None:
        transformer_options = {}

    # -- Normalize ref_latents --
    if ref_latents is None:
        ref_list = []
    elif isinstance(ref_latents, torch.Tensor):
        ref_list = [ref_latents]
    elif isinstance(ref_latents, CONDList):
        ref_list = [r for r in ref_latents.cond if r is not None]
    elif isinstance(ref_latents, (list, tuple)):
        ref_list = [r for r in ref_latents if r is not None and isinstance(r, torch.Tensor)]
    else:
        ref_list = []

    # No refs -> chain to whatever forward was installed before us.
    if len(ref_list) == 0:
        if not getattr(self, "_editutils_ref_kv_logged_norefs", False):
            setattr(self, "_editutils_ref_kv_logged_norefs", True)
            print(f"[Krea2EditApply] forward called but NO refs received "
                  f"(type={type(ref_latents).__name__}); chaining through.")
        chained = getattr(self, "_editutils_ref_kv_chained_forward")
        return chained(x, timesteps, context, attention_mask=attention_mask,
                       transformer_options=transformer_options, **kwargs)

    # Temporal (video / image-as-1-frame-video) input: reshape like the
    # original forward (nodes.py:2103-2106) instead of falling back.
    temporal = x.ndim == 5
    if temporal:
        b5, c5, t5, h5, w5 = x.shape
        x = x.reshape(b5 * t5, c5, h5, w5)
        if not getattr(self, "_editutils_ref_kv_logged_5d", False):
            setattr(self, "_editutils_ref_kv_logged_5d", True)
            print(f"[Krea2EditApply] 5D input detected "
                  f"(B={b5}, C={c5}, T={t5}, H={h5}, W={w5}); reshaping to 4D.")

    # -- Cache lookup --
    _cfg = getattr(self, _KV_CFG_ATTR, None) or {}
    debug = bool(_cfg.get("debug_log", False))
    ref_strength = float(_cfg.get("ref_strength", 1.0))

    # ref_strength == 0: refs never participate -> pure text-to-image.
    if ref_strength <= 0.0:
        chained = getattr(self, "_editutils_ref_kv_chained_forward")
        return chained(x, timesteps, context, attention_mask=attention_mask,
                       transformer_options=transformer_options, **kwargs)

    cache = getattr(self, _KV_CACHE_ATTR, None)
    if cache is None:
        cache = {}
        setattr(self, _KV_CACHE_ATTR, cache)
    # Include ROPE offsets in cache key to prevent stale cache hits with different positions
    _rope_offsets_for_key = kwargs.get("ref_rope_offsets", None)
    key = _build_cache_key(id(self), x.dtype, ref_list) + (
        tuple(x.shape),
        bool(getattr(self, "_editutils_ref_pos_match_target", False)),
        tuple(tuple(o) for o in _rope_offsets_for_key) if _rope_offsets_for_key else None,
    )
    entry = cache.get(key)
    reuse = entry is not None
    if debug and not getattr(self, "_editutils_ref_kv_logged_call", False):
        setattr(self, "_editutils_ref_kv_logged_call", True)
        print(f"[Krea2EditApply] KV cache active: {len(ref_list)} ref(s), "
              f"mode={'reuse' if reuse else 'capture'}")

    # Sampling progress in [0, 1] from the FULL sigma schedule
    # ("sample_sigmas"; "sigmas" is only the current step's sigma).
    # Refs attend while progress < ref_strength; afterwards the cached ref
    # K/V are dropped and generation continues as pure text-to-image.
    progress = 0.0
    sig = transformer_options.get("sample_sigmas", None)
    if sig is None:
        sig = transformer_options.get("sigmas", None)
    if sig is not None:
        try:
            flat = sig.flatten().float()
            # Prefer the sampler-provided current-step sigma (same scale as
            # the schedule); `timesteps` may be scaled (e.g. sigma*1000).
            cur = transformer_options.get("sigmas", None)
            t_now = float(cur.flatten()[0]) if cur is not None \
                else float(timesteps.flatten()[0])
            idx = int((flat - t_now).abs().argmin())
            # Schedules usually end with a trailing 0 sigma; clamp index to
            # the real step count so the final step reaches progress ~1.
            n = int(flat.numel())
            if n > 1 and float(flat[-1]) == 0.0:
                n -= 1
            progress = min(idx / max(n - 1, 1), 1.0)
        except Exception:
            progress = 0.0
    drop_refs = reuse and progress >= ref_strength
    if debug:
        alloc = torch.cuda.memory_allocated() // 2**20 if torch.cuda.is_available() else 0
        resv = torch.cuda.memory_reserved() // 2**20 if torch.cuda.is_available() else 0
        n_cache = len(getattr(self, _KV_CACHE_ATTR, {}) or {})
        print(f"[Krea2EditApply] step progress={progress:.2f} "
              f"ref_strength={ref_strength} -> {'DROP ref' if drop_refs else 'ref active'} "
              f"| VRAM alloc={alloc}MB reserved={resv}MB | cache_entries={n_cache}")
    if drop_refs and debug and not getattr(self, "_editutils_ref_kv_logged_drop", False):
        setattr(self, "_editutils_ref_kv_logged_drop", True)
        print(f"[Krea2EditApply] ref dropped at progress={progress:.2f} "
              f"(ref_strength={ref_strength}); continuing as text-to-image")

    # -- Build tokens (ref part skipped on reuse) --
    bs, c, H_orig, W_orig = x.shape
    patch_size = self.patch
    x_pad = comfy.ldm.common_dit.pad_to_patch_size(x, (patch_size, patch_size))
    H, W = x_pad.shape[-2], x_pad.shape[-1]
    h_grid, w_grid = H // patch_size, W // patch_size
    device, dtype = x_pad.device, torch.float32

    pos_scale = (h_grid, w_grid) if getattr(
        self, "_editutils_ref_pos_match_target", False) else None
    ref_rope_offsets = kwargs.get("ref_rope_offsets", None)  # ROPE position offsets for regional editing
    ref_tokens_list, ref_pos_list, total_reflen = _editutils_make_ref_positions(
        ref_list, bs, patch_size, device, dtype, scale_to_grid=pos_scale,
        rope_offsets=ref_rope_offsets,
    )

    context = self._unpack_context(context)
    img = rearrange(x_pad, "b c (h ph) (w pw) -> b (h w) (c ph pw)",
                    ph=patch_size, pw=patch_size)
    img = self.first(img)

    t_emb_real = self.tmlp(
        timestep_embedding(timesteps, self.tdim).unsqueeze(1).to(img.dtype))
    tvec_real = self.tproj(t_emb_real)

    context = self.txtfusion(context, mask=None,
                             transformer_options=transformer_options)
    context = self.txtmlp(context)

    txtlen, imglen = context.shape[1], img.shape[1]

    txtpos = torch.zeros(bs, txtlen, 3, device=device, dtype=dtype)
    tgtids = torch.zeros(h_grid, w_grid, 3, device=device, dtype=dtype)
    tgtids[..., 1] = torch.arange(h_grid, device=device, dtype=dtype)[:, None]
    tgtids[..., 2] = torch.arange(w_grid, device=device, dtype=dtype)[None, :]
    tgtpos = tgtids.reshape(1, h_grid * w_grid, 3).repeat(bs, 1, 1)

    if reuse:
        # Strip ref span entirely (same trick as nodes_flux.py:286-292).
        combined = torch.cat([context, img], dim=1)
        pos = torch.cat([txtpos, tgtpos], dim=1)
        tvec = tvec_real.expand(-1, txtlen + imglen, -1)
    else:
        ref_imgs = [self.first(rt) for rt in ref_tokens_list]
        combined = torch.cat([context, img] + ref_imgs, dim=1)
        pos = torch.cat([txtpos, tgtpos] + ref_pos_list, dim=1)
        # Reference timestep mode (see uncached forward): "krea2edit" gives refs
        # the real timestep; default "editutils" modulates them as clean (t=0).
        if getattr(self, "_editutils_ref_timestep_mode", "editutils") == "krea2edit":
            tvec_ref = tvec_real.expand(-1, total_reflen, -1)
        else:
            t_emb_zero = self.tmlp(
                timestep_embedding(torch.zeros_like(timesteps), self.tdim)
                .unsqueeze(1).to(img.dtype))
            tvec_ref = self.tproj(t_emb_zero).expand(-1, total_reflen, -1)
        tvec = torch.cat([
            tvec_real.expand(-1, txtlen, -1),
            tvec_real.expand(-1, imglen, -1),
            tvec_ref,
        ], dim=1)

    freqs = self.pe_embedder(pos)

    # -- Block loop with capture/reuse attention wrappers --
    state = {
        "mode": "reuse" if reuse else "capture",
        "ref_toks": total_reflen,
        "layer_counter": 0,
        "store": entry["kv"] if reuse else {},
        "drop_refs": drop_refs,
    }
    setattr(self, _KV_STATE_ATTR, state)
    try:
        with _ref_kv_attn_patch_guard(self):
            for block in self.blocks:
                combined = block(combined, tvec, freqs, attention_mask,
                                 transformer_options=transformer_options)
    finally:
        setattr(self, _KV_STATE_ATTR, None)

    if not reuse:
        cache[key] = {"kv": state["store"], "ref_toks": total_reflen, "hits": 0}
        if debug:
            cpu_mb = sum(t.nelement() * t.element_size() for kv in state["store"].values() for t in kv) // 2**20
            dev = next(iter(state["store"].values()))[0].device if state["store"] else "?"
            print(f"[Krea2EditApply] KV cache captured: {total_reflen} ref tokens, "
                  f"{len(state['store'])} layers, {cpu_mb}MB on {dev}")
        # Keep only the most recent entries to bound VRAM.
        while len(cache) > 4:
            cache.pop(next(iter(cache)))
    elif debug and entry.get("hits", 0) == 0:
        print("[Krea2EditApply] reusing cached ref K/V")
    if reuse:
        entry["hits"] = entry.get("hits", 0) + 1

    # -- Final layer + slice target output --
    final = self.last(combined, t_emb_real)
    out = final[:, txtlen: txtlen + imglen, :]
    out = rearrange(out, "b (h w) (c ph pw) -> b c (h ph) (w pw)",
                    h=h_grid, w=w_grid, ph=patch_size, pw=patch_size,
                    c=self.channels)
    out = out[:, :, :H_orig, :W_orig]

    if temporal:
        out = out.reshape(b5, t5, self.channels, H_orig, W_orig).movedim(1, 2)

    return out


def _install_ref_kv_cache_patch(m, dit, reset_cache=False, ref_strength=1.0,
                                debug_log=False):
    """Install the KV-caching forward on a ModelPatcher already patched by
    Krea2EditApply_EditUtils._apply_krea2_edit_patch (its dispatch forward
    must already be present in m.object_patches)."""
    if reset_cache:
        setattr(dit, _KV_CACHE_ATTR, {})

    setattr(dit, _KV_CFG_ATTR, {
        "debug_log": bool(debug_log),
        "ref_strength": float(ref_strength),
    })

    # CRITICAL: chain to the EXISTING object patch (the Krea2EditApply
    # dispatch forward), NOT dit.forward -- object patches are only applied
    # to the live module at patch_model() time.
    chained = m.object_patches.get("diffusion_model.forward", None)
    if chained is None:
        chained = dit.forward
    setattr(dit, "_editutils_ref_kv_chained_forward", chained)
    if debug_log:
        print("[Krea2EditApply] ref KV cache patch installed")

    def forward(x, timesteps, context, attention_mask=None,
                transformer_options=None, ref_latents=None, **kwargs):
        if transformer_options is None:
            transformer_options = {}
        try:
            import comfy.patcher_extension
            return comfy.patcher_extension.WrapperExecutor.new_class_executor(
                _editutils_krea2_edit_forward_cached, dit,
                comfy.patcher_extension.get_all_wrappers(
                    comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL,
                    transformer_options),
            ).execute(dit, x, timesteps, context,
                      attention_mask=attention_mask,
                      transformer_options=transformer_options,
                      ref_latents=ref_latents, **kwargs)
        except Exception as e:
            # Fallback correctness guarantee: never break generation.
            # Log once per unique error so silent fallback is visible.
            seen = getattr(dit, "_editutils_ref_kv_err_seen", None)
            if seen is None:
                seen = set()
                setattr(dit, "_editutils_ref_kv_err_seen", seen)
            msg = repr(e)
            if msg not in seen:
                seen.add(msg)
                import traceback
                print(f"[Krea2EditApply] KV cache forward failed, "
                      f"falling back to uncached: {msg}")
                traceback.print_exc()
            saved = getattr(dit, "_editutils_ref_kv_chained_forward")
            return saved(x, timesteps, context,
                         attention_mask=attention_mask,
                         transformer_options=transformer_options,
                         ref_latents=ref_latents, **kwargs)

    m.add_object_patch("diffusion_model.forward", forward)
