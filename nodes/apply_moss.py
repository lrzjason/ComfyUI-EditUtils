"""MOSS-Image model patch (reference tokens on the 4-axis RoPE r channel)."""

import comfy.utils
import math
import torch
import comfy.conds

from .apply_krea2 import _build_cache_key

_MOSS_ORIGINAL_FORWARD_ATTR = "_editutils_moss_original_forward"


_MOSS_KV_CACHE_ATTR = "_editutils_moss_ref_kv_cache"


_MOSS_KV_CFG_ATTR = "_editutils_moss_ref_kv_cfg"


try:  # lazy-safe: older ComfyUI builds without MOSS-Image still import the pack
    from comfy.ldm.moss_image.model import (
        modulated_rms_norm as _moss_modulated_rms_norm,
        rope_split_half as _moss_rope_split_half,
        situ as _moss_situ,
    )
except Exception:
    _moss_modulated_rms_norm = None
    _moss_rope_split_half = None
    _moss_situ = None


def _editutils_moss_ops():
    """Resolve the MOSS primitives (again if the top-level import was skipped)."""
    global _moss_modulated_rms_norm, _moss_rope_split_half, _moss_situ
    if _moss_modulated_rms_norm is None:
        from comfy.ldm.moss_image.model import (
            modulated_rms_norm as _n,
            rope_split_half as _r,
            situ as _s,
        )
        _moss_modulated_rms_norm, _moss_rope_split_half, _moss_situ = _n, _r, _s
    return _moss_modulated_rms_norm, _moss_rope_split_half, _moss_situ


def _editutils_moss_normalize_refs(ref_latents):
    """Normalize the conditioning payload to a list of latent tensors."""
    from comfy.conds import CONDList

    if ref_latents is None:
        return []
    if isinstance(ref_latents, torch.Tensor):
        return [ref_latents]
    if isinstance(ref_latents, CONDList):
        items = list(ref_latents.cond)
    elif isinstance(ref_latents, dict):
        items = [ref_latents]
    elif isinstance(ref_latents, (list, tuple)):
        items = list(ref_latents)
    else:
        return []

    out = []
    for r in items:
        if isinstance(r, dict):
            r = r.get("samples", None)
        if isinstance(r, torch.Tensor):
            out.append(r)
    return out


def _editutils_moss_image_ids(index, frame, h, w, device, canvas=None):
    """(h*w, 4) RoPE coordinates (r, f, h, w) for one token grid.

    Mirrors comfy/ldm/moss_image/model.py MOSSImage.rope for the target image
    (canvas=None -> the grid's own centred coordinates).  For references,
    ``canvas=(H, W)`` follows UnifiedTrainer's ``grid_rope`` exactly: the
    reference's rows/columns are **scaled over the whole target canvas** with
    ``round`` (several reference tokens may land on one canvas cell; the ``r``
    axis keeps sources apart).  A reference with the same grid as the target
    therefore reduces to the plain centred grid.
    """
    rows = torch.arange(h, device=device, dtype=torch.float32)
    cols = torch.arange(w, device=device, dtype=torch.float32)
    if canvas is None:
        ch, cw = h, w
        h_ids = rows - (h - 1) / 2.0
        w_ids = cols - (w - 1) / 2.0
    else:
        ch, cw = int(canvas[0]), int(canvas[1])
        if h > 1:
            rows = torch.round(rows * (ch - 1) / (h - 1))
        if w > 1:
            cols = torch.round(cols * (cw - 1) / (w - 1))
        h_ids = rows - (ch - 1) / 2.0
        w_ids = cols - (cw - 1) / 2.0

    ids = torch.zeros(h * w, 4, device=device, dtype=torch.float32)
    ids[:, 0] = float(index)
    ids[:, 1] = float(frame)
    ids[:, 2] = h_ids.repeat_interleave(w)
    ids[:, 3] = w_ids.repeat(h)
    return ids


def _editutils_moss_text_ids(text_len, device):
    ids = torch.zeros(text_len, 4, device=device, dtype=torch.float32)
    ids[:, 1] = torch.arange(text_len, device=device, dtype=torch.float32)
    return ids


def _editutils_moss_pack_refs(refs, bs, text_len, device, dtype, canvas=None):
    """Reference latents -> (B, Lr, C) tokens + (Lr, 4) RoPE ids."""
    tokens, ids = [], []
    for i, ref in enumerate(refs, start=1):
        if ref.ndim == 5 and ref.shape[2] == 1:
            ref = ref.squeeze(2)
        if ref.ndim != 4:
            raise ValueError(
                f"MossImageEditApply_EditUtils: reference latent has shape "
                f"{tuple(ref.shape)} — expected 4D (B,C,H,W), or 5D with temporal=1"
            )
        ref = ref.to(device=device, dtype=dtype)
        if ref.shape[0] != bs:
            ref = comfy.utils.repeat_to_batch_size(ref, bs)
        _, c, h, w = ref.shape
        tokens.append(ref.flatten(2).transpose(1, 2))
        ids.append(_editutils_moss_image_ids(i, text_len, h, w, device, canvas=canvas))
    return torch.cat(tokens, dim=1), torch.cat(ids, dim=0)


def _editutils_moss_time_vectors(dit, timestep, dtype):
    """Native MOSS time pipeline: vec = time_embedding(t_emb), mod = 6-way adaLN."""
    from comfy.ldm.flux.layers import timestep_embedding

    vec = dit.time_embedding(timestep_embedding(timestep, 256).to(dtype))
    mod = dit.shared_block_modulation(vec).unflatten(-1, (6, -1)).unsqueeze(2).unbind(1)
    return vec, mod


def _editutils_moss_attention(block, q, k, v, pe, transformer_options,
                              kv_capture=None, kv_cache=None):
    """Native MOSS attention, optionally capturing / injecting reference K/V.

    Without capture or cache this is exactly the model's own ``attention()``
    (including the fused RMSNorm+RoPE kernel).  The cache path needs the
    post-RoPE K/V as tensors, so it uses the split-half RoPE used by the
    training implementation; both paths are mathematically identical.
    """
    B, N, _ = q.shape
    heads = block.num_heads

    if kv_capture is None and kv_cache is None:
        from comfy.ldm.moss_image.model import attention as moss_attention
        return moss_attention(q, k, v, block.q_norm, block.k_norm, pe, heads,
                              block.comfy_attention, transformer_options)

    from comfy.ldm.modules.attention import optimized_attention
    _, rope_split_half, _ = _editutils_moss_ops()

    q = rope_split_half(block.q_norm(q.view(B, N, heads, -1)), pe)
    k = rope_split_half(block.k_norm(k.view(B, N, heads, -1)), pe)
    v = v.view(B, N, heads, -1)

    if kv_capture is not None:
        kv_capture.append((k.detach(), v.detach()))

    if kv_cache is not None:
        kk, vv = kv_cache
        if kk.device != k.device:
            kk = kk.to(k.device, non_blocking=True)
            vv = vv.to(v.device, non_blocking=True)
        kk = comfy.utils.repeat_to_batch_size(kk, B)
        vv = comfy.utils.repeat_to_batch_size(vv, B)
        k = torch.cat((k, kk.to(k.dtype)), dim=1)
        v = torch.cat((v, vv.to(v.dtype)), dim=1)

    q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
    return optimized_attention(
        q, k, v, heads, skip_reshape=True,
        preferred_attention=block.comfy_attention,
        transformer_options=transformer_options,
    )


def _editutils_moss_gated_residual(x, y, split, gate, rgate):
    """x + y * gate for tokens [:split] and x + y * rgate for tokens [split:]."""
    if split <= 0:
        return x + y * rgate
    if split >= x.shape[1]:
        return x + y * gate
    return torch.cat((x[:, :split] + y[:, :split] * gate,
                      x[:, split:] + y[:, split:] * rgate), dim=1)


def _editutils_moss_block_forward(block, x, mod_real, mod_ref, split, pe,
                                  transformer_options, kv_capture=None,
                                  kv_cache=None):
    """MOSS Block.forward with per-span adaLN modulation.

    Tokens [:split] (target image + text) use the real timestep modulation;
    tokens [split:] (clean reference tokens) use the t=0 one.  ``split=0`` is
    the reference-only precompute pass, ``split=x.shape[1]`` the cached live
    pass with no references in the sequence.
    """
    shift_a, scale_a, gate_a, shift_f, scale_f, gate_f = mod_real
    rshift_a, rscale_a, rgate_a, rshift_f, rscale_f, rgate_f = mod_ref
    modulated_rms_norm, _, situ = _editutils_moss_ops()

    def norm_span(h, norm, scale, shift, rscale, rshift):
        if split <= 0:
            return modulated_rms_norm(h, norm, rscale, rshift)
        if split >= h.shape[1]:
            return modulated_rms_norm(h, norm, scale, shift)
        return torch.cat((
            modulated_rms_norm(h[:, :split], norm, scale, shift),
            modulated_rms_norm(h[:, split:], norm, rscale, rshift),
        ), dim=1)

    y = norm_span(x, block.norm1, scale_a, shift_a, rscale_a, rshift_a)
    if block.fused_qkv:
        q, k, v = block.proj["qkv"](y).chunk(3, dim=-1)
    else:
        q, k, v = block.proj["q"](y), block.proj["k"](y), block.proj["v"](y)

    out = _editutils_moss_attention(block, q, k, v, pe, transformer_options,
                                    kv_capture=kv_capture, kv_cache=kv_cache)
    x = _editutils_moss_gated_residual(x, block.proj["o"](out), split,
                                       gate_a, rgate_a)

    y = norm_span(x, block.norm2, scale_f, shift_f, rscale_f, rshift_f)
    hidden, gate = block.proj["ffn_up_gate"](y).chunk(2, dim=-1)
    return _editutils_moss_gated_residual(
        x, block.proj["ffn_down"](situ(hidden, gate)), split, gate_f, rgate_f)


def _editutils_moss_run_blocks(dit, h, block_fn, vec, mod, pe, transformer_options):
    """MOSS block loop, honouring patches_replace (SLG) and weight prefetch.

    The aimdo malloc graph is intentionally NOT started here.  Its allocation
    graph is recorded per thread and expects a fixed allocation sequence, but
    this forward runs passes with different token counts (reference precompute
    vs. live denoising, and cond vs. uncond text lengths); replaying it across
    those shapes raises "aimdo memory compile error".  The weight prefetch
    queue below is independent of the graph and still works.
    """
    import comfy.model_prefetch

    blocks_replace = transformer_options.get("patches_replace", {}).get("dit", {})
    transformer_options["total_blocks"] = len(dit.layers)
    transformer_options["block_type"] = "single"
    device = h.device
    prefetch_queue = comfy.model_prefetch.make_prefetch_queue(
        list(dit.layers), device, transformer_options)
    for i, block in enumerate(dit.layers):
        # no malloc_scope: keeps the aimdo memory compiler out of this loop
        comfy.model_prefetch.prefetch_queue_pop(
            prefetch_queue, device, block, h.dtype)
        transformer_options["block_index"] = i
        if ("single_block", i) in blocks_replace:
            def block_wrap(args, block=block, i=i):
                return {"img": block_fn(block, args["img"], i)}

            h = blocks_replace[("single_block", i)](
                {"img": h, "vec": vec, "mod": mod, "pe": pe,
                 "transformer_options": transformer_options},
                {"original_block": block_wrap},
            )["img"]
        else:
            h = block_fn(block, h, i)
    comfy.model_prefetch.prefetch_queue_pop(prefetch_queue, device, None)
    return h


def _editutils_moss_finish(dit, h, vec, n_img, B, H, W):
    """Native MOSS output head, restricted to the target image tokens."""
    modulated_rms_norm, _, _ = _editutils_moss_ops()
    head_shift, head_scale = dit.video_head_modulation(vec).unflatten(
        -1, (2, -1)).unsqueeze(2).unbind(1)
    out = dit.final_linear_video(modulated_rms_norm(
        h[:, :n_img], dit.final_norm_video, head_scale, head_shift))
    return out.transpose(1, 2).reshape(B, dit.latent_channels, H, W)


def _editutils_moss_forward_with_refs(dit, x, timestep, context, refs,
                                      transformer_options):
    """Full reference pass: [target image | text | refs], refs at t=0.

    ``refs == []`` is the graph-free native-equivalent pass (used when
    ``ref_strength`` drops the references), so a reference-enabled generation
    never falls back into the aimdo malloc graph.
    """
    B, _, H, W = x.shape
    dtype, device = x.dtype, x.device
    transformer_options = transformer_options.copy()
    text_len = context.shape[1]
    n_img = H * W

    reftok = refids = None
    if refs:
        reftok, refids = _editutils_moss_pack_refs(
            refs, B, text_len, device, dtype, canvas=(H, W))
        if reftok.shape[-1] != dit.latent_channels:
            raise ValueError(
                f"MossImageEditApply_EditUtils: reference latent has {reftok.shape[-1]} "
                f"channels but this model expects {dit.latent_channels}. Encode the "
                f"reference with the MOSS-Image / FLUX.2 VAE."
            )

    ids_parts = [
        _editutils_moss_image_ids(0, text_len, H, W, device),
        _editutils_moss_text_ids(text_len, device),
    ]
    if refs:
        ids_parts.append(refids)
    ids = torch.cat(ids_parts, dim=0)
    pe = dit.pe_embedder(ids.unsqueeze(0)).transpose(1, 2).contiguous()

    vec, mod = _editutils_moss_time_vectors(dit, timestep, dtype)
    _, mod0 = _editutils_moss_time_vectors(dit, torch.zeros_like(timestep), dtype)

    txt = dit.text_embedder(context, pe[:, n_img:n_img + text_len],
                            transformer_options)
    h = torch.cat((dit.video_embedder(x.flatten(2).transpose(1, 2)), txt), dim=1)
    if refs:
        h = torch.cat((h, dit.video_embedder(reftok)), dim=1)
    split = n_img + text_len

    def block_fn(block, h, i):
        return _editutils_moss_block_forward(
            block, h, mod, mod0, split, pe, transformer_options)

    h = _editutils_moss_run_blocks(dit, h, block_fn, vec, mod, pe,
                                   transformer_options)
    return _editutils_moss_finish(dit, h, vec, n_img, B, H, W)


def _editutils_moss_precompute_ref_kv(dit, x, timestep, text_len, refs,
                                      transformer_options):
    """One t=0 pass over the reference tokens only; returns per-block K/V."""
    B = x.shape[0]
    dtype, device = x.dtype, x.device
    transformer_options = transformer_options.copy()

    reftok, refids = _editutils_moss_pack_refs(
        refs, B, text_len, device, dtype, canvas=(x.shape[-2], x.shape[-1]))
    pe = dit.pe_embedder(refids.unsqueeze(0)).transpose(1, 2).contiguous()
    vec0, mod0 = _editutils_moss_time_vectors(dit, torch.zeros_like(timestep), dtype)
    h = dit.video_embedder(reftok)
    store = {}

    def block_fn(block, h, i):
        capture = []
        h = _editutils_moss_block_forward(
            block, h, mod0, mod0, 0, pe, transformer_options, kv_capture=capture)
        k, v = capture[0]
        # CPU keeps repeated runs from pinning gigabytes of VRAM.
        store[i] = (k.to("cpu", copy=True), v.to("cpu", copy=True))
        return h

    _editutils_moss_run_blocks(dit, h, block_fn, vec0, mod0, pe,
                               transformer_options)
    return store, reftok.shape[1]


def _editutils_moss_forward_with_cached_refs(dit, x, timestep, context, ref_kv,
                                             transformer_options):
    """Live pass: [target image | text], cached reference K/V appended in attention."""
    B, _, H, W = x.shape
    dtype, device = x.dtype, x.device
    transformer_options = transformer_options.copy()
    text_len = context.shape[1]
    n_img = H * W

    ids = torch.cat((
        _editutils_moss_image_ids(0, text_len, H, W, device),
        _editutils_moss_text_ids(text_len, device),
    ), dim=0)
    pe = dit.pe_embedder(ids.unsqueeze(0)).transpose(1, 2).contiguous()
    vec, mod = _editutils_moss_time_vectors(dit, timestep, dtype)

    txt = dit.text_embedder(context, pe[:, n_img:], transformer_options)
    h = torch.cat((dit.video_embedder(x.flatten(2).transpose(1, 2)), txt), dim=1)
    split = h.shape[1]

    def block_fn(block, h, i):
        return _editutils_moss_block_forward(
            block, h, mod, mod, split, pe, transformer_options,
            kv_cache=ref_kv.get(i))

    h = _editutils_moss_run_blocks(dit, h, block_fn, vec, mod, pe,
                                   transformer_options)
    return _editutils_moss_finish(dit, h, vec, n_img, B, H, W)


def _editutils_moss_skipped_blocks(transformer_options):
    replace = transformer_options.get("patches_replace", {}).get("dit", {})
    return tuple(sorted(k[1] for k in replace
                        if isinstance(k, tuple) and k[0] == "single_block"))


def _editutils_moss_sampling_progress(transformer_options, timestep):
    """Sampling progress in [0, 1] from the full sigma schedule (Krea2 logic)."""
    sig = transformer_options.get("sample_sigmas", None)
    if sig is None:
        sig = transformer_options.get("sigmas", None)
    if sig is None:
        return 0.0
    try:
        flat = sig.flatten().float()
        cur = transformer_options.get("sigmas", None)
        t_now = float(cur.flatten()[0]) if cur is not None \
            else float(timestep.flatten()[0])
        idx = int((flat - t_now).abs().argmin())
        n = int(flat.numel())
        if n > 1 and float(flat[-1]) == 0.0:
            n -= 1
        return min(idx / max(n - 1, 1), 1.0)
    except Exception:
        return 0.0


def _editutils_moss_edit_forward(self, x, timestep, context,
                                 transformer_options=None, ref_latents=None,
                                 **kwargs):
    """Patched MOSS-Image forward with reference-latent support."""
    if transformer_options is None:
        transformer_options = {}

    refs = _editutils_moss_normalize_refs(ref_latents)
    orig = getattr(self, _MOSS_ORIGINAL_FORWARD_ATTR, None)
    if orig is None:  # defensive: patch installed without the saved forward
        raise RuntimeError(
            "MossImageEditApply_EditUtils: original forward missing; re-apply the patch node")
    if len(refs) == 0:
        return orig(x, timestep, context,
                    transformer_options=transformer_options, **kwargs)

    if x.ndim != 4:
        raise ValueError(
            f"MossImageEditApply_EditUtils: expected a 4D latent (B,C,H,W), "
            f"got shape {tuple(x.shape)}"
        )

    cfg = getattr(self, _MOSS_KV_CFG_ATTR, None) or {}
    use_cache = bool(cfg.get("kv_cache", True))
    ref_strength = float(cfg.get("ref_strength", 1.0))
    debug = bool(cfg.get("debug_log", False))

    # ref_strength == 0 -> references never participate.  Use the graph-free
    # native-equivalent pass instead of the stock forward: the reference loop
    # intentionally avoids the aimdo malloc graph (see _run_blocks), and a
    # fallback into it could hit the same "memory compile error" when the
    # cond/uncond token counts differ.
    if ref_strength <= 0.0:
        return _editutils_moss_forward_with_refs(
            self, x, timestep, context, [], transformer_options)

    progress = _editutils_moss_sampling_progress(transformer_options, timestep)
    # Strict '>': ref_strength=1.0 must keep references on EVERY step (the
    # final step reports progress 1.0).
    if progress > ref_strength:
        if debug and not getattr(self, "_editutils_moss_logged_drop", False):
            setattr(self, "_editutils_moss_logged_drop", True)
            print(f"[MossImageEditApply] ref dropped at progress={progress:.2f} "
                  f"(ref_strength={ref_strength}); continuing as text-to-image")
        return _editutils_moss_forward_with_refs(
            self, x, timestep, context, [], transformer_options)

    text_len = int(context.shape[1])

    if not use_cache:
        if debug and not getattr(self, "_editutils_moss_logged_nocache", False):
            setattr(self, "_editutils_moss_logged_nocache", True)
            print("[MossImageEditApply] ref_kv_cache=False: references ride in "
                  "the sequence and attend to text + noisy image")
        return _editutils_moss_forward_with_refs(
            self, x, timestep, context, refs, transformer_options)

    cache = getattr(self, _MOSS_KV_CACHE_ATTR, None)
    if cache is None:
        cache = {}
        setattr(self, _MOSS_KV_CACHE_ATTR, cache)
    key = _build_cache_key(id(self), x.dtype, refs) + (
        tuple(x.shape), text_len, _editutils_moss_skipped_blocks(transformer_options),
    )
    entry = cache.get(key)
    if entry is None:
        entry_kv, ref_tokens = _editutils_moss_precompute_ref_kv(
            self, x, timestep, text_len, refs, transformer_options)
        entry = {"kv": entry_kv, "ref_tokens": ref_tokens, "hits": 0}
        cache[key] = entry
        if debug:
            cpu_mb = sum(t.nelement() * t.element_size()
                         for kv in entry_kv.values() for t in kv) // 2**20
            print(f"[MossImageEditApply] ref K/V captured: {ref_tokens} ref "
                  f"token(s), {len(entry_kv)} layers, {cpu_mb}MB on cpu")
        while len(cache) > 4:  # bound CPU memory across prompts
            cache.pop(next(iter(cache)))
    elif debug and entry.get("hits", 0) == 0:
        print(f"[MossImageEditApply] reusing cached ref K/V "
              f"({entry.get('ref_tokens', 0)} tokens)")
    entry["hits"] = entry.get("hits", 0) + 1

    return _editutils_moss_forward_with_cached_refs(
        self, x, timestep, context, entry["kv"], transformer_options)


class MossImageEditApply_EditUtils:
    """Patch a MOSS-Image model to accept reference latents via EditUtils conditioning.

    Matches the Krea2 edit shape (and ai-toolkit's MOSS-Image edit recipe):

    - Sequence: ``[target image | text | ref1 | ref2 | ...]``
    - 4-axis RoPE: reference n uses ``(r=n, f=text_len, centred h/w)``;
      target image uses ``(r=0, f=text_len, ...)``.
    - Per-token timestep: reference tokens use t=0 (clean latents), the target
      image and text use the real timestep.
    - Output is sliced back to the target image tokens only.
    - ``ref_kv_cache`` (default on) precomputes reference attention K/V once,
      matching the ai-toolkit ``kv_cache`` training mode.  Turn it off for a
      LoRA trained without kv_cache, where the references attend to the text
      and the noisy image too.

    Connect only the ``model`` wire — reference latents are extracted from the
    EditUtils conditioning chain via an ``extra_conds`` patch.  The patch is
    isolated to a cloned model instance via ``add_object_patch``.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
            },
            "optional": {
                "ref_kv_cache": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "ON (default) is ai-toolkit's kv_cache reference mode: "
                               "the references attend only to each other and their "
                               "K/V are precomputed once at t=0, then injected as "
                               "extra keys on every step. Turn it OFF for a LoRA "
                               "trained without kv_cache (e.g. UnifiedTrainer's "
                               "reference-token forward), where references attend to "
                               "the text and noisy image too."}),
                "ref_strength": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "How long references constrain sampling. 1.0 = "
                               "all steps; 0.5 = the first half, then pure "
                               "text-to-image; 0.0 = references never participate."}),
                "reset_cache": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Clear cached reference K/V when this node runs. "
                               "Leave on unless you deliberately reuse the same "
                               "references across prompts."}),
                "debug_log": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "apply_patch"
    CATEGORY = "MossImage/edit"
    DESCRIPTION = (
        "Apply reference-image conditioning to a MOSS-Image model. "
        "Connect only the model wire — reference latents flow through the "
        "EditUtils conditioning chain automatically. Supports multiple "
        "references and the ai-toolkit kv_cache reference mode."
    )

    @staticmethod
    def _is_moss(model):
        try:
            dit = model.get_model_object("diffusion_model")
        except Exception:
            return False, None
        ok = (
            hasattr(dit, "video_embedder")
            and hasattr(dit, "text_embedder")
            and hasattr(dit, "shared_block_modulation")
            and hasattr(dit, "video_head_modulation")
            and hasattr(dit, "pe_embedder")
            and hasattr(dit, "layers")
            and hasattr(dit, "latent_channels")
        )
        return ok, dit

    def apply_patch(self, model, ref_kv_cache=True, ref_strength=1.0,
                    reset_cache=True, debug_log=False):
        is_moss, dit = self._is_moss(model)
        if not is_moss:
            return (model,)

        m = model.clone()
        setattr(dit, _MOSS_KV_CFG_ATTR, {
            "kv_cache": bool(ref_kv_cache),
            "ref_strength": float(ref_strength),
            "debug_log": bool(debug_log),
        })
        if reset_cache:
            setattr(dit, _MOSS_KV_CACHE_ATTR, {})

        self._apply_moss_edit_patch(m, dit)

        if debug_log:
            print("[MossImageEditApply] MOSS-Image model detected "
                  f"(latent_channels={dit.latent_channels}, "
                  f"blocks={len(dit.layers)}, kv_cache={bool(ref_kv_cache)}, "
                  f"ref_strength={float(ref_strength)})")
        return (m,)

    @staticmethod
    def _apply_moss_edit_patch(model_patcher, dit):
        """Install reference-latent patches via ModelPatcher.add_object_patch."""
        base_model = model_patcher.model
        orig_extra_conds = base_model.extra_conds
        orig_extra_conds_shapes = base_model.extra_conds_shapes

        # Save the native forward once (idempotent across repeated apply calls).
        if not hasattr(dit, _MOSS_ORIGINAL_FORWARD_ATTR):
            setattr(dit, _MOSS_ORIGINAL_FORWARD_ATTR, dit.forward)
        orig_forward = getattr(dit, _MOSS_ORIGINAL_FORWARD_ATTR)

        def extra_conds(**kwargs):
            out = orig_extra_conds(**kwargs)
            # Current ComfyUI BaseModel.extra_conds already converts
            # reference_latents -> out["ref_latents"]; only add it on builds
            # that do not (keeps the patch idempotent).
            if out.get("ref_latents", None) is not None:
                return out
            ref_tensors = _editutils_moss_normalize_refs(
                kwargs.get("reference_latents", None))
            if ref_tensors:
                out["ref_latents"] = comfy.conds.CONDList([
                    base_model.process_latent_in(t) for t in ref_tensors
                ])
            return out

        def extra_conds_shapes(**kwargs):
            out = orig_extra_conds_shapes(**kwargs)
            ref_tensors = _editutils_moss_normalize_refs(
                kwargs.get("reference_latents", None))
            if ref_tensors:
                # MOSS references are 128-channel FLUX.2 latents: [1, C, sum(H*W)].
                # Override the generic BaseModel hint ([1, 16, ...]) so the
                # sampler's memory-fit estimate matches the real tensors.
                total = sum(math.prod(t.shape[1:]) for t in ref_tensors)
                if total > 0:
                    out["ref_latents"] = [1, dit.latent_channels,
                                          total // dit.latent_channels]
            return out

        def forward(x, timestep, context, transformer_options=None,
                    ref_latents=None, **kwargs):
            if transformer_options is None:
                transformer_options = {}
            # CONDList/CONDRegular has no __len__; normalize first (the edit
            # forward accepts a plain list just as well).
            refs = _editutils_moss_normalize_refs(ref_latents)
            if len(refs) == 0:
                return orig_forward(x, timestep, context,
                                    transformer_options=transformer_options,
                                    **kwargs)
            # Route through WrapperExecutor so DIFFUSION_MODEL wrappers from
            # other nodes/hooks still fire during reference generation.
            import comfy.patcher_extension
            return comfy.patcher_extension.WrapperExecutor.new_class_executor(
                _editutils_moss_edit_forward, dit,
                comfy.patcher_extension.get_all_wrappers(
                    comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL,
                    transformer_options),
            ).execute(dit, x, timestep, context,
                      transformer_options=transformer_options,
                      ref_latents=refs, **kwargs)

        model_patcher.add_object_patch("extra_conds", extra_conds)
        model_patcher.add_object_patch("extra_conds_shapes", extra_conds_shapes)
        model_patcher.add_object_patch("memory_usage_factor_conds", ("ref_latents",))
        model_patcher.add_object_patch("diffusion_model.forward", forward)
