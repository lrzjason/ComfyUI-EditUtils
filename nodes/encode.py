"""EditTextEncode_EditUtils: the single encode path all models route through."""

from sympy import false
import node_helpers
import comfy.utils
import math
import torch
import comfy.conds

from ._shared import MINIMAX_H3_IMAGE_DEFAULT_SIZE, MINIMAX_H3_IMAGE_LATENT_CHANNELS, MINIMAX_H3_IMAGE_REF_MAX_SHORT_EDGE, MINIMAX_H3_IMAGE_REF_SHORT_EDGE, build_pad_canvas, minimax_h3_image_ref_size
from .flux3_bbox import build_flux3_bbox_json

class EditTextEncode_EditUtils:
    # upscale_methods = ["lanczos", "bicubic", "area"]
    # crop_methods = ["pad", "center", "disabled"]
    # example_config = {
    #     "image": None,
    #     # ref part
    #     "to_ref": True,
    #     "ref_main_image": True,
    #     "ref_longest_edge": 1024,
    #     "ref_crop": "center", #"pad" for main image, "center", "disabled"
    #     "ref_upscale": "lanczos",
    #     # vl part
    #     "to_vl": True,
    #     "vl_resize": True,
    #     "vl_target_size": 384,
    #     "vl_crop": "center",
    #     "vl_upscale": "bicubic", #to scale image down, "bicubic", "area" might better than "lanczos"
    # }
    # example_output = {
    #     "pad_info": pad_info,
    #     "noise_mask": noise_mask,
    #     "full_refs_cond": conditioning,
    #     "main_ref_cond": conditioning_only_with_main_ref,
    #     "main_image": main_image,
    #     "vae_images": vae_images,
    #     "ref_latents": ref_latents,
    #     "vl_images": vl_images,
    #     "full_prompt": full_prompt,
    #     "llama_template": llama_template
    # }
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": 
            {
                "clip": ("CLIP", ),
                "vae": ("VAE", ),
                "prompt": ("STRING", {"multiline": True, "dynamicPrompts": True}),
                "model_config": ("DICT", {"default": None}),
            },
            "optional": 
            {
                "configs": ("LIST", {"default": None, "tooltip": "List of image configuration dictionaries. When not provided, performs text-only encoding."}),
            },
            # "optional": 
            # {
            #     # "return_full_refs_cond": ("BOOLEAN", {"default": True}),
            #     # "set_noise_mask": ("BOOLEAN", {"default": False, "tooltip": "Only useful when using ref_crop == pad. It would automatically mask out the padding area."}),
            #     "instruction": ("STRING", {"multiline": True, "default": "Describe the key features of the input image (color, shape, size, texture, objects, background), then explain how the user's text instruction should alter or modify the image. Generate a new image that meets the user's requirements while maintaining consistency with the original input where appropriate."}),   
            # }
        }
    # `prompt` is appended LAST on purpose: ComfyUI matches saved links by slot
    # index, so adding it at the end keeps every existing workflow's outputs wired
    # to the same sockets.
    RETURN_TYPES = ("CONDITIONING", "LATENT", "ANY", "IMAGE", "MASK", "ANY", "STRING")
    RETURN_NAMES = ("conditioning", "latent", "custom_output", "main_image", "mask", "pad_info", "prompt")
    FUNCTION = "encode"

    CATEGORY = "advanced/conditioning"
    def encode(self, clip, vae, prompt, 
               model_config=None,
               configs=None,
            #    return_full_refs_cond=True,
            #    set_noise_mask=False,
            #    instruction="",
        ):
        # print("len(configs)")
        # print(len(configs))
        # llama_template = get_system_prompt(instruction)
        model_name = model_config["model_name"] if "model_name" in model_config else None
        is_qwen = model_name == "qwen"
        is_qwen21 = model_name == "qwen_image21"
        is_boogu = model_name == "boogu"
        # FLUX.3 klein: the reference latents go through the standard
        # `reference_latents` conditioning key (ComfyUI core class Flux3 handles
        # them), so routing mostly matters for the VL/text side and the latent
        # shape. Its text encoder is Qwen3-VL and it has no `<|vision_start|>`
        # image-prompt convention: references are pure VAE latents.
        is_flux3 = model_name in ("flux3klein", "flux3")
        # MOSS-Image: Qwen3.5-9B text encoder WITH the vision tower (so the
        # reference image can be grounded in the prompt, like Krea2) and the
        # FLUX.2 packed VAE (128ch / 16 px per token).  Its tokenizer owns the
        # chat template, so llama_template stays empty.
        is_moss = model_name in ("moss_image", "moss")
        # MiniMax-H3-Image (the IMAGE model, not the H3 video model). ComfyUI
        # core already knows how to consume it:
        #   - text: Qwen3-VL-32B (CLIP type `minimax`) with the references
        #     spliced in as "Picture n: " + <vision block> before the prompt,
        #     emitted by the tokenizer when `images=` is passed;
        #   - references: a `minimax_refs` conditioning key holding
        #     [{"kind": "image", "latent_h", "latent_w", "latent"}], which
        #     model_base.MiniMaxH3Image packs into PackedLayout and re-injects
        #     every step at t=0.999;
        #   - target: one 24-channel latent frame, [1, 24, 1, H/16, W/16].
        # So this route only has to build those values, no model patch.
        is_minimax = model_name in ("minimax_h3_image", "minimax")
        vae_unit = model_config["vae_unit"] if "vae_unit" in model_config else 8
        llama_template = model_config["llama_template"] if "llama_template" in model_config else ""
        image_prompt = ""
        
        pad_info = {
            "x": 0,
            "y": 0,
            "width": 0,
            "height": 0,
            "scale_by": 1.0,
        }
        # print("len(configs)", len(configs))
        # handle missing or empty configs: fall back to text-only encoding
        if configs is None:
            configs = []
        if len(configs) == 0:
            print("EditTextEncode_EditUtils: No image configs provided, performing text-only encoding.")

        # A config can carry only an element table (the bbox node's table-only
        # entry, used by generation mode with no image and no config to attach
        # to). It has no image to encode, so keep it out of the image loops while
        # still letting it contribute its rows to the prompt table below.
        image_configs = [
            c for c in configs
            if isinstance(c, dict) and not c.get("bbox_table_only")
        ]

        # Boogu: completely different processing - early return
        if is_boogu:
            negative_prompt = model_config.get("negative_prompt", "")
            ref_latents = []
            images_vl = []
            vae_images_boogu = []
            for i, image_obj in enumerate(configs):
                assert "image" in image_obj, "Image is missing"
                image = image_obj["image"]
                to_ref = image_obj.get("to_ref", True)
                to_vl = image_obj.get("to_vl", True)
                ref_longest_edge = image_obj.get("ref_longest_edge", 1024)
                ref_resize_mode = image_obj.get("ref_resize_mode", "longest_edge")
                ref_crop = image_obj.get("ref_crop", "pad")
                ref_upscale = image_obj.get("ref_upscale", "lanczos")
                ref_main_image = image_obj.get("ref_main_image", i == 0)
                vl_target_size = image_obj.get("vl_target_size", 384)
                vl_crop = image_obj.get("vl_crop", "center")
                vl_upscale = image_obj.get("vl_upscale", "lanczos")

                if not to_ref and not to_vl:
                    continue

                samples = image.movedim(-1, 1)

                # Vision tower input: area-based scaling to vl_target_size
                if to_vl:
                    total = int(vl_target_size * vl_target_size)
                    scale_by = math.sqrt(total / (samples.shape[3] * samples.shape[2]))
                    width = round(samples.shape[3] * scale_by)
                    height = round(samples.shape[2] * scale_by)
                    s = comfy.utils.common_upscale(samples, width, height, vl_upscale, vl_crop)
                    images_vl.append(s.movedim(1, -1)[:, :, :, :3])

                # Reference latent: aligned to vae_unit
                if to_ref:
                    if ref_resize_mode == "area":
                        total = int(ref_longest_edge * ref_longest_edge)
                        scale_by = math.sqrt(total / (samples.shape[3] * samples.shape[2]))
                        scaled_width = int(round(samples.shape[3] * scale_by))
                        scaled_height = int(round(samples.shape[2] * scale_by))
                    else:  # longest_edge
                        ori_longest_edge = max(samples.shape[2], samples.shape[3])
                        scale_by = ori_longest_edge / ref_longest_edge
                        scaled_height = int(round(samples.shape[2] / scale_by))
                        scaled_width = int(round(samples.shape[3] / scale_by))

                    if ref_crop == "pad":
                        crop = "center"
                        canvas_width = math.ceil(scaled_width / vae_unit) * vae_unit
                        canvas_height = math.ceil(scaled_height / vae_unit) * vae_unit
                        canvas = torch.zeros(
                            (samples.shape[0], samples.shape[1], canvas_height, canvas_width),
                            dtype=samples.dtype,
                            device=samples.device
                        )
                        resized_samples = comfy.utils.common_upscale(samples, scaled_width, scaled_height, ref_upscale, crop)
                        resized_width = resized_samples.shape[3]
                        resized_height = resized_samples.shape[2]
                        canvas[:, :, :resized_height, :resized_width] = resized_samples
                        if ref_main_image:
                            current_total = (samples.shape[3] * samples.shape[2])
                            total_px = int(resized_width * resized_height)
                            scale_by_val = math.sqrt(total_px / current_total)
                            pad_info = {
                                "x": 0,
                                "y": 0,
                                "width": canvas_width - resized_width,
                                "height": canvas_height - resized_height,
                                "scale_by": round(1 / scale_by_val, 3)
                            }
                        s = canvas
                    else:
                        crop = ref_crop
                        width = round(scaled_width / vae_unit) * vae_unit
                        height = round(scaled_height / vae_unit) * vae_unit
                        s = comfy.utils.common_upscale(samples, width, height, ref_upscale, crop)

                    vae_img = s.movedim(1, -1)[:, :, :, :3]
                    vae_images_boogu.append(vae_img)
                    ref_latents.append(vae.encode(vae_img))

            # Tokenize: prompt directly with images (boogu tokenizer auto-selects system prompt)
            positive = clip.encode_from_tokens_scheduled(clip.tokenize(prompt, images=images_vl))
            negative = clip.encode_from_tokens_scheduled(clip.tokenize(negative_prompt))

            # Reference latents go on BOTH positive and negative (cancels under CFG)
            if len(ref_latents) > 0:
                positive = node_helpers.conditioning_set_values(positive, {"reference_latents": ref_latents}, append=True)
                negative = node_helpers.conditioning_set_values(negative, {"reference_latents": ref_latents}, append=True)

            samples_out = ref_latents[0] if len(ref_latents) > 0 else torch.zeros(1, 4, 128, 128)
            latent_out = {"samples": samples_out}

            main_image = vae_images_boogu[0] if len(vae_images_boogu) > 0 else None

            custom_output = {
                "negative_cond": negative,
                "ref_latents": ref_latents,
                "vl_images": images_vl,
                "vae_images": vae_images_boogu,
                "full_prompt": prompt,
            }

            # Boogu tokenizes the prompt as-is (its tokenizer picks the system
            # prompt itself), so that is also the `prompt` output.
            return (positive, latent_out, custom_output, main_image, None, pad_info, prompt)

        main_image_index = -1
        for i, image_obj in enumerate(image_configs):
            if image_obj["to_ref"]:
                if main_image_index == -1 and image_obj["ref_main_image"]:
                    main_image_index = i
                    continue
                # ensure only one main image
                if main_image_index != -1:
                    image_obj["ref_main_image"] = False
        if main_image_index == -1:
            if len(image_configs) > 0:
                print("\n Auto fixing main_image_index to the first image index")
                main_image_index = 0
        
        ref_latents = []
        rope_offsets = []  # parallel list to ref_latents: (x_offset, y_offset) tuples
        vae_images = []
        vl_images = []
        
        noise_mask = None
        for i, image_obj in enumerate(image_configs):
            assert "image" in image_obj, "Image is missing"
            image = image_obj["image"]
            to_ref = image_obj["to_ref"]
            ref_main_image = image_obj["ref_main_image"]
            # H3I configs size by short edge instead, so ref_longest_edge is
            # optional (hand-written / ConfigJsonParser configs keep it).
            ref_longest_edge = image_obj.get("ref_longest_edge", 1024)
            ref_crop = image_obj["ref_crop"]
            ref_upscale = image_obj["ref_upscale"]
            # pad appearance config, only used when ref_crop == "pad".
            # default: white edge for qwen models (black edge amplifies during
            # generation, both qwen 1.0 and qwen image 2.1), black edge
            # otherwise (matches training data).
            pad_mode = image_obj.get("pad_mode", "color")
            pad_default_color = "white" if (is_qwen or is_qwen21) else "black"
            pad_color = image_obj.get("pad_color", pad_default_color)
            # '-' (the widget default) is not a valid hex, so it has no effect
            pad_color_rgb = image_obj.get("pad_color_rgb", "-")
            pad_noise = image_obj.get("pad_noise", 0.0)
            ref_resize_mode = image_obj.get("ref_resize_mode", "longest_edge")
            
            if is_qwen:
                to_vl = image_obj["to_vl"]
                vl_resize = image_obj["vl_resize"]
                vl_target_size = image_obj["vl_target_size"]
                vl_crop = image_obj["vl_crop"]
                vl_upscale = image_obj["vl_upscale"]
            elif is_moss:
                # MOSS-Image follows the Qwen edit shape: a vision-tower image
                # for the Qwen3.5 prompt grounding and a separate FLUX.2-VAE
                # latent for the model patch.  .get() keeps hand-written config
                # dicts working when they omit an optional VL field.
                to_vl = image_obj.get("to_vl", True)
                vl_resize = image_obj.get("vl_resize", True)
                vl_target_size = image_obj.get("vl_target_size", 384)
                vl_crop = image_obj.get("vl_crop", "center")
                vl_upscale = image_obj.get("vl_upscale", "bicubic")
            elif is_flux3:
                # FLUX.3 has no image tokens in the text sequence: its text
                # encoder is text-only (Qwen3-VL with the visual tower unused,
                # 8-layer tap -> 20480) and references are pure VAE latents on
                # their own token mode. So to_vl is meaningless here and is
                # forced off, which also avoids emitting a "Picture n:" prefix.
                if image_obj.get("to_vl", False):
                    print("Flux3Klein: references are pure VAE latents (no vision slots); ignoring to_vl")
                to_vl = False
            elif is_qwen21:
                # Qwen-Image 2.1: the vision-tower input is the resized reference
                # itself, so there is no separate vl resize/crop pipeline.
                to_vl = image_obj.get("to_vl", True)
                if to_vl and not to_ref:
                    print("QwenImage21: to_vl without to_ref would break vision-slot alignment; forcing to_ref=True")
                    to_ref = True
                elif to_ref and not to_vl:
                    print("QwenImage21: reference without vision slot will splice after the text sequence (untrained path)")
            elif is_minimax:
                # H3I keeps BOTH halves per reference: the Qwen3-VL vision copy
                # (the tokenizer turns it into "Picture n: " + <vision block>)
                # and the 24ch VAE latent that rides in `minimax_refs`.  There is
                # no per-reference rope offset and no `<|vision_start|>` text
                # convention — the tokenizer owns the presentation.
                to_vl = image_obj.get("to_vl", True)
                vl_resize = True
                # unused by H3I: its vision copy is sized by the short-edge rule
                vl_target_size = MINIMAX_H3_IMAGE_REF_SHORT_EDGE
                vl_crop = image_obj.get("vl_crop", "disabled")
                vl_upscale = image_obj.get("vl_upscale", "bicubic")
            else:
                to_vl = false
            
            mask = None
            if "mask" in image_obj:
                mask = image_obj["mask"]
            
            samples = image.movedim(-1, 1)
            if mask is not None:
                _, c, _, _ = samples.shape
                sample_masks = mask.unsqueeze(1).repeat(1, c, 1, 1)  # same shape
                # sample_masks = mask.movedim(-1, 1)
                # check samples and sample_masks should match
                print('samples.shape',samples.shape)
                print('sample_masks.shape',sample_masks.shape)
                assert samples.shape == sample_masks.shape, "Image and mask shape mismatch"
            
            if not to_ref and not to_vl:
                continue
            if to_ref:
                # print("ori_image.shape",samples.shape)
                # ori_height, ori_width = samples.shape[2:]
                ori_longest_edge = max(samples.shape[2], samples.shape[3])

                if is_minimax:
                    # ComfyUI's edit-reference rule (nodes_minimax_h3.py
                    # `_size_short_edge`): the SHORT edge is clamped into
                    # [ref_short_edge, ref_max_short_edge] and both sides snap
                    # to a 32 px grid, so the 24ch latent stays even and the
                    # DiT's 2x2 patch covers it exactly.
                    scaled_width, scaled_height = minimax_h3_image_ref_size(
                        samples.shape[3], samples.shape[2],
                        short_edge=image_obj.get("ref_short_edge", MINIMAX_H3_IMAGE_REF_SHORT_EDGE),
                        max_short_edge=image_obj.get("ref_max_short_edge", MINIMAX_H3_IMAGE_REF_MAX_SHORT_EDGE),
                        unit=vae_unit,
                    )
                    scale_by = ori_longest_edge / max(1, max(scaled_width, scaled_height))
                elif ref_resize_mode == "area":
                    total = int(ref_longest_edge * ref_longest_edge)
                    scale_by = math.sqrt(total / (samples.shape[3] * samples.shape[2]))
                else:  # longest_edge
                    scale_by = ori_longest_edge / ref_longest_edge
                if not is_minimax:
                    scaled_height = int(round(samples.shape[2] / scale_by))
                    scaled_width = int(round(samples.shape[3] / scale_by))
                
                # pad only apply to main image
                if ref_crop == "pad":
                    # print("In pad mode")
                    # print("scaled_width", scaled_width)
                    # print("scaled_height", scaled_height)
                    crop = "center"
                    
                    width_ceil = math.ceil(scaled_width / vae_unit)
                    # if scaled_width % vae_unit != 0:
                    #     width_ceil +=1
                    canvas_width = width_ceil * vae_unit
                    
                    height_ceil = math.ceil(scaled_height / vae_unit)
                    # if scaled_height % vae_unit != 0:
                    #     height_ceil +=1
                    canvas_height = height_ceil * vae_unit
                    
                    resized_samples = comfy.utils.common_upscale(samples, scaled_width, scaled_height, ref_upscale, crop)
                    
                    resized_width = resized_samples.shape[3]
                    resized_height = resized_samples.shape[2]
                    # pad image to canvas size with configurable pad mode/color
                    canvas = build_pad_canvas(
                        resized_samples, canvas_height, canvas_width,
                        pad_mode=pad_mode, pad_color=pad_color, pad_noise=pad_noise,
                        pad_color_rgb=pad_color_rgb, pad_color_fallback=pad_default_color
                    )
                    
                    
                    # if set_noise_mask:
                        # noise_mask = torch.zeros(canvas.shape, dtype=torch.bool, device=canvas.device)
                        # noise_mask[:, :, x_offset:resized_height, y_offset:resized_width] = 1.0
                        # print("noise_mask.shape", noise_mask.shape)
                        # noise_mask = noise_mask.movedim(1, -1)
                        # print("movedim noise_mask.shape", noise_mask.shape)
                    
                    # only return main image pad info
                    
                    current_total = (samples.shape[3] * samples.shape[2])
                    total = int(resized_width * resized_height)
                    scale_by = math.sqrt(total / current_total)
                    if ref_main_image:
                        pad_info = {
                            "x": 0,
                            "y": 0,
                            "width": canvas_width - resized_width,
                            "height": canvas_height - resized_height,
                            "scale_by": round(1 / scale_by, 3)
                        }
                    
                    # print("pad_info", pad_info)
                    s = canvas
                    
                    if mask is not None and ref_main_image:
                        mask_canvas = torch.zeros(
                            (samples.shape[0], samples.shape[1], canvas_height, canvas_width),
                            dtype=samples.dtype,
                            device=samples.device
                        )
                        
                        resized_sample_masks = comfy.utils.common_upscale(sample_masks, scaled_width, scaled_height, ref_upscale, crop)
                        mask_canvas[:, :, :resized_height, :resized_width] = resized_sample_masks
                        m =  mask_canvas
                        
                        # remove noise mask channel
                        noise_mask = m[:, :1, :, :].squeeze(1)
                        print("noise_mask.shape", noise_mask.shape)
                else:
                    crop = ref_crop
                    # handle pad method when not main image
                    if ref_crop == "pad":
                        crop = "center"
                    width = round(scaled_width / vae_unit) * vae_unit
                    height = round(scaled_height / vae_unit) * vae_unit
                    # print("width",width)
                    # print("height",height)
                    s = comfy.utils.common_upscale(samples, width, height, ref_upscale, crop)
                    
                    if mask is not None and ref_main_image:
                        m = comfy.utils.common_upscale(sample_masks, width, height, ref_upscale, crop)
                        # remove noise mask channel
                        noise_mask = m[:, :1, :, :].squeeze(1)
                        print("noise_mask.shape", noise_mask.shape)
                image = s.movedim(1, -1)
                if is_qwen21:
                    # Qwen-Image 2.1's VAE is RGBA (64ch / 16x downscale): keep all
                    # four channels — plain RGB gets an opaque alpha padded in by
                    # ComfyUI. The vision tower consumes the same resize composited
                    # over white, so every vision slot maps to a 2x2 latent group.
                    ref_latents.append(vae.encode(image))
                else:
                    # RGB encode. FLUX.3 (and FLUX.2 klein) reuses the FLUX.2
                    # packed autoencoder: 32 latent channels, 8x downscale + 2x2
                    # patchify -> 128ch at 16 px per token. Its reference
                    # normalization is the VAE's batch-norm, which ComfyUI's
                    # Flux2 latent format already applies (no scalar
                    # scaling/shift factor), so a plain RGB encode is correct.
                    ref_latents.append(vae.encode(image[:, :, :, :3]))
                rope_offsets.append((
                    image_obj.get("rope_x_offset", 0),
                    image_obj.get("rope_y_offset", 0)
                ))
                vae_images.append(image)
                if is_qwen21 and to_vl:
                    rgb = image[:, :, :, :3]
                    if image.shape[-1] > 3:
                        # the vision tower sees alpha composited over white, the vae keeps all four
                        rgb = rgb * image[:, :, :, 3:] + (1.0 - image[:, :, :, 3:])
                    vl_images.append(rgb)
            if to_vl and not is_qwen21:
                if is_minimax:
                    # ComfyUI's vision copy for H3I: short edge exactly
                    # ref_short_edge (1024 by default), no crop — the Qwen
                    # processor then smart-resizes it to its own 32 px grid.
                    vl_width, vl_height = minimax_h3_image_ref_size(
                        samples.shape[3], samples.shape[2],
                        short_edge=image_obj.get("ref_short_edge", MINIMAX_H3_IMAGE_REF_SHORT_EDGE),
                        max_short_edge=image_obj.get("ref_short_edge", MINIMAX_H3_IMAGE_REF_SHORT_EDGE),
                        unit=vae_unit,
                    )
                    scale_by = min(samples.shape[2], samples.shape[3]) / float(
                        max(1, min(vl_width, vl_height)))
                elif vl_resize:
                    # print("vl_resize")
                    total = int(vl_target_size * vl_target_size)
                    scale_by = math.sqrt(total / (samples.shape[3] * samples.shape[2]))
                else:
                    total = int(samples.shape[3] * samples.shape[2])
                    if total > 2048 * 2048:
                        print("vl_target_size too large, clipping to 2048")
                        total = 2048 * 2048
                    scale_by = math.sqrt(total / (samples.shape[3] * samples.shape[2]))

                width = round(samples.shape[3] * scale_by)
                height = round(samples.shape[2] * scale_by)
                if is_minimax:
                    # already snapped to the 32 px grid by minimax_h3_image_ref_size
                    width, height = vl_width, vl_height
                s = comfy.utils.common_upscale(samples, width, height, vl_upscale, vl_crop)
                
                image = s.movedim(1, -1)
                # handle non resize vl images
                if not is_minimax:
                    # H3I's tokenizer inserts `"Picture n: " + <vision block>`
                    # itself when the images are handed to clip.tokenize(); a
                    # text prefix here would be counted twice.
                    image_prompt += "Picture {}: <|vision_start|><|image_pad|><|vision_end|>".format(i + 1)
                vl_images.append(image)
                
        full_prompt = image_prompt + prompt
        # print("full_prompt", full_prompt)
        # print("llama_template", llama_template)
        bbox_tables = []
        if is_flux3:
            # FLUX.3 layout/edit: the element table is appended to the prompt as
            # a JSON array, "<caption> <array>", with no separate parameter. Each
            # image config may carry its own table; they are concatenated into a
            # single array so a multi-reference graph still sends one table.
            merged_rows = []
            merged_mode = "generation"
            seen_modes = set()
            for image_obj in configs:
                rows = image_obj.get("bbox_rows_parsed") or []
                if not rows:
                    continue
                merged_mode = image_obj.get("bbox_mode_parsed", "generation")
                seen_modes.add(merged_mode)
                merged_rows.extend(rows)
            if len(seen_modes) > 1:
                # Each row is still emitted in its own schema, but FLUX.3 is
                # trained on one consistent table per prompt, so flag it.
                print(f"Flux3Klein bbox: the chain mixes table modes {sorted(seen_modes)}; "
                      "each row keeps its own schema, but a single consistent mode is safer")
            if merged_rows:
                # "ref_image_N" must name a reference that exists, otherwise the
                # row points at nothing and the model has no source to keep/move.
                for row in merged_rows:
                    from_ref = row.get("from")
                    if not isinstance(from_ref, str) or not from_ref.startswith("ref_image_"):
                        continue
                    try:
                        ref_index = int(from_ref.split("_")[-1])
                    except ValueError:
                        ref_index = -1
                    if ref_index < 0 or ref_index >= len(ref_latents):
                        print(f"Flux3Klein bbox: row {row.get('id')!r} names {from_ref!r}, but this graph has "
                              f"{len(ref_latents)} reference(s); that row has no resolvable source")
                bbox_tables = merged_rows
                table = build_flux3_bbox_json(merged_rows, merged_mode)
                full_prompt = f"{full_prompt.strip()} {table}".strip()
                print(f"Flux3Klein bbox: appended {len(merged_rows)} row(s) to the prompt "
                      f"({len(table)} chars)")

        if is_qwen21:
            # QwenImage21 tokenizer auto-builds the T2I template ("Comprehend and
            # analyze the provided prompt.") and inserts the vision blocks before
            # the prompt; with a custom template they are spliced in at {}.
            # keep_vision keeps the vision embeddings as plain text conditioning
            # when no reference latents were encoded (vl-only / text-only mode).
            if llama_template:
                if len(vl_images) > 0:
                    # same splice the QwenImage21 tokenizer uses for its auto template
                    refs = " ".join("<image{}><|vision_start|><|image_pad|><|vision_end|>".format(j + 1) for j in range(len(vl_images)))
                    llama_template = llama_template.replace("{}", refs + "{}", 1)
                tokens = clip.tokenize(prompt, images=vl_images, llama_template=llama_template,
                                       keep_vision=(len(ref_latents) == 0), prevent_empty_text=True)
            else:
                tokens = clip.tokenize(prompt, images=vl_images,
                                       keep_vision=(len(ref_latents) == 0), prevent_empty_text=True)
        elif is_minimax:
            # The MiniMax H3 tokenizer owns the presentation: for every image it
            # emits `"Picture n: " + <vision block>` before the text (exactly what
            # ComfyUI's MiniMaxH3ImageConditioning does with picture_label), so
            # the prompt refers to the references as <Picture 1>, <Picture 2>, ...
            # No llama_template: a system prompt would be encoded as prompt text.
            tokens = clip.tokenize(full_prompt, images=vl_images, picture_label="Picture %d: ")
        elif llama_template == "" or llama_template is None:
            tokens = clip.tokenize(full_prompt, images=vl_images)
        else:
            tokens = clip.tokenize(full_prompt, images=vl_images, llama_template=llama_template)
        # print("editutils tokens", tokens)
        conditioning = clip.encode_from_tokens_scheduled(tokens)
        minimax_refs = []
        if is_minimax and len(ref_latents) > 0:
            # ComfyUI core's MiniMaxH3Image model_base reads `minimax_refs`
            # (comfy/model_base.py -> comfy/ldm/minimax/model.py PackedLayout):
            # a list of {"kind": "image", "latent_h", "latent_w", "latent"} where
            # the latent is the VAE-encoded [1, 24, 1, h, w] reference. The rows
            # are packed between the text and the target and re-injected at
            # t=0.999 every step (never denoised, noise-augmented by 0.001).
            minimax_refs = [{
                "kind": "image",
                "latent_h": int(z.shape[-2]),
                "latent_w": int(z.shape[-1]),
                "latent": z,
            } for z in ref_latents]
            conditioning = node_helpers.conditioning_set_values(conditioning, {"minimax_refs": minimax_refs})
        if is_qwen21:
            # 64 latent channels @ 16x spatial downscale
            samples = torch.zeros(1, 64, 128, 128)
        elif is_moss:
            # FLUX.2 packed VAE: 128 latent channels @ 16x spatial downscale
            samples = torch.zeros(1, 128, 128, 128)
        elif is_flux3:
            # FLUX.3's packed FLUX.2 VAE: 128 latent channels @ 16x
            samples = torch.zeros(1, 128, 128, 128)
        elif is_minimax:
            # 24ch @ 16x, ONE frame: [1, 24, 1, 128, 128] = 2048 x 2048 px
            samples = torch.zeros(1, MINIMAX_H3_IMAGE_LATENT_CHANNELS, 1,
                                  MINIMAX_H3_IMAGE_DEFAULT_SIZE // 16,
                                  MINIMAX_H3_IMAGE_DEFAULT_SIZE // 16)
        else:
            samples = torch.zeros(1, 4, 128, 128)
        # conditioning_only_with_main_ref = None
        conditioning_full_refs = conditioning
        if len(ref_latents) > 0:
            # conditioning_only_with_main_ref = node_helpers.conditioning_set_values(conditioning, {"reference_latents": [ref_latents[main_image_index]]}, append=True)
            # print("editutils conditioning_full_refs before", conditioning_full_refs)
            samples = ref_latents[main_image_index]
            if is_minimax:
                # Already on `minimax_refs` above; H3I has no consumer for
                # `reference_latents` or the per-ref rope offsets (PackedLayout
                # derives reference time positions from block order), so do not
                # duplicate the latents on a second key.
                pass
            else:
                extra_cond = {"reference_latents": ref_latents}
                if is_flux3:
                    # ComfyUI core's Flux3 model_base reads `reference_latents` only:
                    # it derives ref position ids itself (image_ids with the negative
                    # L index and image_rope_scale 0.5), so the EditUtils per-ref rope
                    # offsets have no consumer on this path. Emitting them anyway is
                    # harmless (unknown keys are ignored) but misleading, so skip them.
                    pass
                elif is_moss:
                    # MOSS-Image references are placed on the 4-axis RoPE r channel
                    # (r = 1, 2, ...) by MossImageEditApply_EditUtils; the per-ref
                    # x/y offsets used by the Krea2/Qwen regional-edit path have no
                    # consumer here.
                    pass
                else:
                    extra_cond["reference_rope_offsets"] = rope_offsets
                conditioning_full_refs = node_helpers.conditioning_set_values(conditioning, extra_cond, append=True)
            # conditioning_full_refs = node_helpers.conditioning_set_values(conditioning_full_refs, {"concat_latent_image": samples}, append=True)
            # print("editutils conditioning_full_refs after", conditioning_full_refs)
        if is_minimax:
            # Target grid = the main image's reference size x `scale_up`.
            # H3I denoises one 24ch frame and was trained on 4-20 MP (best
            # ~16 MP); 1 MP produces poster-like artifacts, while edit
            # references are happiest around 1 MP. Keeping the reference at
            # ~1 MP and growing only the sampled latent is what `scale_up` is
            # for (e.g. ref_short_edge=1024 + scale_up=2 -> 4 MP square).
            main_cfg = image_configs[main_image_index] if 0 <= main_image_index < len(image_configs) else {}
            scale_up = max(1.0, float(main_cfg.get("scale_up", 1.0) or 1.0))
            if scale_up > 1.0 and samples.dim() == 5:
                # stay on the 32 px grid: keep both latent sides even
                up_w = max(2, int(round(samples.shape[-1] * scale_up)))
                up_h = max(2, int(round(samples.shape[-2] * scale_up)))
                up_w -= up_w % 2
                up_h -= up_h % 2
                if (up_w, up_h) != (int(samples.shape[-1]), int(samples.shape[-2])):
                    # the sampler starts from pure noise at denoise=1, so the
                    # content is irrelevant — only the shape matters
                    samples = torch.zeros(
                        (1, MINIMAX_H3_IMAGE_LATENT_CHANNELS, 1, up_h, up_w),
                        dtype=torch.float32,
                    )
                    print(f"MinimaxH3Image: scale_up={scale_up:g} -> target latent "
                          f"{up_w * 16}x{up_h * 16} px (~{up_w * up_h * 256 / 1e6:.1f} MP)")
            target_pixels = int(samples.shape[-2]) * 16 * int(samples.shape[-1]) * 16
            if target_pixels < 3.5 * 1e6:
                print(f"MinimaxH3Image: target {int(samples.shape[-1]) * 16}x{int(samples.shape[-2]) * 16} "
                      f"(~{target_pixels / 1e6:.1f} MP) is below the 4-20 MP range H3I is trained for "
                      "(1 MP gives poster-like artifacts). Raise `scale_up` on the main image "
                      "(2.0 turns a 1 MP reference into a 4 MP target).")
        if is_minimax and noise_mask is not None:
            # H3I denoises one whole latent frame; there is no per-pixel denoise
            # mask in its packed layout (the mask path belongs to the H3 video
            # model), so a 4D mask on a 5D latent would only break the sampler.
            print("MinimaxH3Image: pixel masks are not supported by the H3 image model; "
                  "ignoring the noise mask.")
            noise_mask = None
        latent_out = {"samples": samples}
        
        if noise_mask is not None:
            latent_out["noise_mask"] = noise_mask
        
        conditioning_output = conditioning_full_refs
        main_image = None
        if len(vae_images)>0:
            main_image = vae_images[main_image_index]
            
        
        
        custom_output = {
            "pad_info": pad_info,
            "full_refs_cond": conditioning_full_refs,
            # "main_ref_cond": conditioning_only_with_main_ref,
            "main_image": main_image,
            "vae_images": vae_images,
            "ref_latents": ref_latents,
            # "llama_template": llama_template,
            "no_refs_cond": conditioning,
            "mask": noise_mask,
        }
        if is_flux3:
            # FLUX.3's text conditioning is the 20480-wide stacked Qwen3-VL
            # embedding; there is no image-prompt prefix, so full_prompt is just
            # the user prompt plus (optionally) the bounding-box element table.
            # vl_images stays empty by design.
            custom_output["full_prompt"] = prompt
            custom_output["composed_prompt"] = full_prompt
            custom_output["bbox_rows"] = bbox_tables
            custom_output["flow_map"] = model_config.get("flow_map", False)
        elif is_qwen or is_qwen21 or is_moss:
            custom_output["vl_images"] = vl_images
            custom_output["full_prompt"] = full_prompt
        elif is_minimax:
            # H3I consumes: the Qwen3-VL text conditioning (with the tokenizer's
            # "Picture n: " labels), `minimax_refs` (the 24ch reference latents)
            # and a 24ch target latent. There is no image-prompt prefix in the
            # text handed in by the user, so full_prompt is the user prompt.
            custom_output["vl_images"] = vl_images
            custom_output["minimax_refs"] = minimax_refs
            custom_output["full_prompt"] = prompt
            custom_output["composed_prompt"] = prompt
            custom_output["target_size"] = [
                int(samples.shape[-1]) * 16,
                int(samples.shape[-2]) * 16,
            ]
        
        
        # print("editutils conditioning_output", conditioning_output)
        # `prompt` output: exactly the text that was handed to the tokenizer.
        #   - flux3klein: the user prompt + the bounding-box element table
        #     (identical to custom_output["composed_prompt"]);
        #   - minimax_h3_image: the user prompt as typed — the "Picture n: "
        #     labels are inserted by the tokenizer, not by the text;
        #   - qwen / qwen_image21 / flux2klein: the prompt with the "Picture n:"
        #     vision prefix;
        #   - boogu returns earlier and never reaches here.
        # Wired straight into a text preview / SaveText node, this is the quickest
        # way to confirm what the model actually received.
        encoded_prompt = full_prompt
        return (conditioning_output, latent_out, custom_output, main_image, noise_mask, pad_info,
                encoded_prompt)
