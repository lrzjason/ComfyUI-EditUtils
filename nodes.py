from sympy import false
import node_helpers
import comfy.utils
import math
import torch
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import json
import os
import copy
import folder_paths
import hashlib

from PIL import Image, ImageOps, ImageSequence

class CropWithPadInfo_EditUtils:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "pad_info": ("ANY", ),  # pad_info dictionary containing x, y, width, height and scale
                "image": ("IMAGE", ),
            },
        }

    RETURN_TYPES = ("IMAGE", "FLOAT",)
    RETURN_NAMES = ("cropped_image", "scale_by", )
    FUNCTION = "crop_image"

    CATEGORY = "image"

    def crop_image(self, image, pad_info):
        # Extract pad information from the original padding process:
        # In the original function:
        # - resized_samples are placed at canvas[:, :, :resized_height, :resized_width]
        # - pad_info = {"x": 0, "y": 0, "width": canvas_width - resized_width, "height": canvas_height - resized_height}
        x = pad_info.get("x", 0)  # This is always 0 in the original function
        y = pad_info.get("y", 0)  # This is always 0 in the original function
        width_padding = pad_info.get("width", 0)  # Right/bottom padding added
        height_padding = pad_info.get("height", 0)  # Right/bottom padding added
        scale_by = pad_info.get("scale_by", 1.0)
        
        img = image.movedim(-1, 1)  # Convert from (H, W, C) to (C, H, W)
        
        # Calculate the original content dimensions before padding was added
        original_content_width = img.shape[3] - width_padding
        original_content_height = img.shape[2] - height_padding
        
        # Crop to get just the original content area (which was placed at position (0,0))
        cropped_img = img[:, :, x:original_content_height, y:original_content_width]
        
        # Convert back to (H, W, C) format
        cropped_image = cropped_img.movedim(1, -1)
        
        return (cropped_image, scale_by)


def get_nearest_resolution(image, resolution=1024):
    height, width, _ = image.shape
    
    # get ratio
    image_ratio = width / height

    # Calculate target dimensions that:
    # 1. Maintain the aspect ratio
    # 2. Have an area of approximately resolution^2 (1024*1024 = 1048576)
    # 3. Are divisible by 8
    target_area = resolution * resolution
    
    # width = height * image_ratio
    # width * height = target_area
    # height * image_ratio * height = target_area
    # height^2 = target_area / image_ratio
    height_optimal = math.sqrt(target_area / image_ratio)
    width_optimal = height_optimal * image_ratio
    
    # Round to nearest multiples of 8
    height_8 = round(height_optimal / 8) * 8
    width_8 = round(width_optimal / 8) * 8
    
    # Ensure minimum size of 64x64
    height_8 = max(64, height_8)
    width_8 = max(64, width_8)
    
    closest_resolution = (width_8, height_8)
    closest_ratio = width_8 / height_8

    return closest_ratio, closest_resolution


def crop_image(image,resolution):
    height, width, _ = image.shape
    closest_ratio,closest_resolution = get_nearest_resolution(image,resolution=resolution)
    image_ratio = width / height
    
    # Determine which dimension to scale by to minimize cropping
    scale_with_height = True
    if image_ratio < closest_ratio: 
        scale_with_height = False
    
    try:
        image,crop_x,crop_y = simple_center_crop(image,scale_with_height,closest_resolution)
    except Exception as e:
        print(e)
        raise e
    return image

def convert_float_unit8(image):
    image = image.astype(np.float32) * 255
    return image.astype(np.uint8)

def convert_unit8_float(image):
    image = image.astype(np.float32)
    image = image / 255.
    return image
def simple_center_crop(image,scale_with_height,closest_resolution):
    height, width, _ = image.shape
    # print("ori size:",height,width)
    if scale_with_height: 
        # Scale based on height, then crop width if needed
        up_scale = height / closest_resolution[1]
    else:
        # Scale based on width, then crop height if needed
        up_scale = width / closest_resolution[0]

    expanded_closest_size = (int(closest_resolution[0] * up_scale + 0.5), int(closest_resolution[1] * up_scale + 0.5))
    
    diff_x = expanded_closest_size[0] - width
    diff_y = expanded_closest_size[1] - height

    crop_x = 0
    crop_y = 0
    # crop extra part of the resized images
    if diff_x > 0:
        # Need to crop width (image is wider than needed)
        crop_x = diff_x // 2
        cropped_image = image[:, crop_x:width - diff_x + crop_x, :]
    elif diff_y > 0:
        # Need to crop height (image is taller than needed)
        crop_y = diff_y // 2
        cropped_image = image[crop_y:height - diff_y + crop_y, :, :]
    else:
        # No cropping needed
        cropped_image = image

    height, width, _ = cropped_image.shape  
    f_width, f_height = closest_resolution
    cropped_image = convert_float_unit8(cropped_image)
    # print("cropped_image:",cropped_image)
    img_pil = Image.fromarray(cropped_image)
    resized_img = img_pil.resize((f_width, f_height), Image.LANCZOS)
    resized_img = np.array(resized_img)
    resized_img = convert_unit8_float(resized_img)
    return resized_img, crop_x, crop_y


def get_system_prompt(instruction):
    template_prefix = "<|im_start|>system\n"
    template_suffix = "<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"
    instruction_content = ""
    if instruction == "":
        instruction_content = "Describe the key features of the input image (color, shape, size, texture, objects, background), then explain how the user's text instruction should alter or modify the image. Generate a new image that meets the user's requirements while maintaining consistency with the original input where appropriate."
    else:
        # for handling mis use of instruction
        if template_prefix in instruction:
            # remove prefix from instruction
            instruction = instruction.split(template_prefix)[1]
        if template_suffix in instruction:
            # remove suffix from instruction
            instruction = instruction.split(template_suffix)[0]
        if "{}" in instruction:
            # remove {} from instruction
            instruction = instruction.replace("{}", "")
        instruction_content = instruction
    llama_template = template_prefix + instruction_content + template_suffix
    
    return llama_template

def validate_vl_resize_indexs(vl_resize_indexs_str, valid_length):
    try:
        indexes = [int(i)-1 for i in vl_resize_indexs_str.split(",")]
        # remove duplicates
        indexes = list(set(indexes))
    except ValueError as e:
        raise ValueError(f"Invalid format for vl_resize_indexs: {e}")

    if not indexes:
        raise ValueError("vl_resize_indexs must not be empty")

    indexes = [idx for idx in indexes if 0 <= idx < valid_length]

    return indexes

class ModelConfig_EditUtils:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model_choice": (["qwen", "flux2klein", "boogu"], {
                    "default": "qwen",
                }),
                "model_name": ("STRING", {"default": ""}),
                "vae_unit": ("INT", {"default": 8, "min": 8, "max": 64, "step": 8})
            },
            "optional": {
                "instruction": ("STRING", {"multiline": True, "default": ""}),
            }
        }

    RETURN_TYPES = ("DICT",)
    RETURN_NAMES = ("model_config",)
    FUNCTION = "configure_model"

    CATEGORY = "advanced/conditioning"

    def configure_model(self, model_choice, model_name, vae_unit, instruction=""):
        
        # if model_name is not empty string, use model name else use model_choice
        model_name = model_name if model_name else model_choice
        config = {
            "model_name": model_name,
            "vae_unit": vae_unit
        }
        
        # Boogu tokenizer auto-selects system prompt, no llama_template needed
        if model_choice == "boogu":
            config["llama_template"] = ""
        else:
            # Add model-specific configurations
            config["llama_template"] = get_system_prompt(instruction)
        
        return (config,)

class QwenModelConfig_EditUtils:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "optional": {
                "instruction": ("STRING", {"multiline": True, "default": ""}),
            }
        }

    RETURN_TYPES = ("DICT",)
    RETURN_NAMES = ("model_config",)
    FUNCTION = "configure_model"

    CATEGORY = "advanced/conditioning"

    def configure_model(self, instruction=""):
        
        config = {
            "model_name": "qwen",
            "vae_unit": 8
        }
        config["llama_template"] = get_system_prompt(instruction)
        return (config,)

class Flux2KleinModelConfig_EditUtils:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "optional": {
                "instruction": ("STRING", {"multiline": True, "default": ""}),
            }
        }

    RETURN_TYPES = ("DICT",)
    RETURN_NAMES = ("model_config",)
    FUNCTION = "configure_model"

    CATEGORY = "advanced/conditioning"

    def configure_model(self, instruction=""):
        
        config = {
            "model_name": "flux2klein",
            "vae_unit": 16
        }
        config["llama_template"] = ""
        # only when instruction is not empty, set instruction for klein. It is not recommanded but allowed for customization
        if instruction != "":
            config["llama_template"] = get_system_prompt(instruction)
        return (config,)
    
class BooguModelConfig_EditUtils:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "optional": {
                "instruction": ("STRING", {"multiline": True, "default": ""}),
            }
        }

    RETURN_TYPES = ("DICT",)
    RETURN_NAMES = ("model_config",)
    FUNCTION = "configure_model"

    CATEGORY = "advanced/conditioning"

    def configure_model(self, instruction=""):
        config = {
            "model_name": "boogu",
            "vae_unit": 8,
            "llama_template": ""
        }
        # Boogu tokenizer auto-selects system prompt, instruction is ignored
        return (config,)

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
    RETURN_TYPES = ("CONDITIONING", "LATENT", "ANY", "IMAGE", "MASK", "ANY")
    RETURN_NAMES = ("conditioning", "latent", "custom_output", "main_image", "mask", "pad_info")
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
        is_boogu = model_name == "boogu"
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
            }

            return (positive, latent_out, custom_output, main_image, None, pad_info)

        main_image_index = -1
        for i, image_obj in enumerate(configs):
            if image_obj["to_ref"]:
                if main_image_index == -1 and image_obj["ref_main_image"]:
                    main_image_index = i
                    continue
                # ensure only one main image
                if main_image_index != -1:
                    image_obj["ref_main_image"] = False
        if main_image_index == -1:
            print("\n Auto fixing main_image_index to the first image index")
            main_image_index = 0
        
        ref_latents = []
        vae_images = []
        vl_images = []
        
        noise_mask = None
        for i, image_obj in enumerate(configs):
            assert "image" in image_obj, "Image is missing"
            image = image_obj["image"]
            to_ref = image_obj["to_ref"]
            ref_main_image = image_obj["ref_main_image"]
            ref_longest_edge = image_obj["ref_longest_edge"]
            ref_crop = image_obj["ref_crop"]
            ref_upscale = image_obj["ref_upscale"]
            ref_resize_mode = image_obj.get("ref_resize_mode", "longest_edge")
            
            if is_qwen:
                to_vl = image_obj["to_vl"]
                vl_resize = image_obj["vl_resize"]
                vl_target_size = image_obj["vl_target_size"]
                vl_crop = image_obj["vl_crop"]
                vl_upscale = image_obj["vl_upscale"]
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
                
                if ref_resize_mode == "area":
                    total = int(ref_longest_edge * ref_longest_edge)
                    scale_by = math.sqrt(total / (samples.shape[3] * samples.shape[2]))
                else:  # longest_edge
                    scale_by = ori_longest_edge / ref_longest_edge
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
                    # pad image to canvas size
                    canvas = torch.zeros(
                        (samples.shape[0], samples.shape[1], canvas_height, canvas_width),
                        dtype=samples.dtype,
                        device=samples.device
                    )
                    
                    resized_samples = comfy.utils.common_upscale(samples, scaled_width, scaled_height, ref_upscale, crop)
                    
                    resized_width = resized_samples.shape[3]
                    resized_height = resized_samples.shape[2]
                    # set resized samples to canvas
                    # canvas[:, :, x_offset:resized_height, y_offset:resized_width] = resized_samples
                    canvas[:, :, :resized_height, :resized_width] = resized_samples
                    
                    
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
                ref_latents.append(vae.encode(image[:, :, :, :3]))
                vae_images.append(image)
            if to_vl:
                if vl_resize:
                    # print("vl_resize")
                    total = int(vl_target_size * vl_target_size)
                else:
                    total = int(samples.shape[3] * samples.shape[2])
                    if total > 2048 * 2048:
                        print("vl_target_size too large, clipping to 2048")
                        total = 2048 * 2048
                current_total = (samples.shape[3] * samples.shape[2])
                scale_by = math.sqrt(total / current_total)
            
                width = round(samples.shape[3] * scale_by)
                height = round(samples.shape[2] * scale_by)
                s = comfy.utils.common_upscale(samples, width, height, vl_upscale, vl_crop)
                
                image = s.movedim(1, -1)
                # handle non resize vl images
                image_prompt += "Picture {}: <|vision_start|><|image_pad|><|vision_end|>".format(i + 1)
                vl_images.append(image)
                
        full_prompt = image_prompt + prompt
        # print("full_prompt", full_prompt)
        # print("llama_template", llama_template)
        # if is_qwen:
        #     tokens = clip.tokenize(full_prompt, images=vl_images, llama_template=llama_template)
        # else:
        # print("editutils image_prompt", image_prompt)
        # print("editutils prompt", prompt)
        # print("editutils llama_template", llama_template)
        if llama_template == "" or llama_template is None:
            tokens = clip.tokenize(full_prompt, images=vl_images)
        else:
            tokens = clip.tokenize(full_prompt, images=vl_images, llama_template=llama_template)
        # print("editutils tokens", tokens)
        conditioning = clip.encode_from_tokens_scheduled(tokens)
        samples = torch.zeros(1, 4, 128, 128)
        # conditioning_only_with_main_ref = None
        conditioning_full_refs = conditioning
        if len(ref_latents) > 0:
            # conditioning_only_with_main_ref = node_helpers.conditioning_set_values(conditioning, {"reference_latents": [ref_latents[main_image_index]]}, append=True)
            conditioning_full_refs = node_helpers.conditioning_set_values(conditioning, {"reference_latents": ref_latents}, append=True)
            # print("editutils conditioning_full_refs before", conditioning_full_refs)
            samples = ref_latents[main_image_index]
            # conditioning_full_refs = node_helpers.conditioning_set_values(conditioning_full_refs, {"concat_latent_image": samples}, append=True)
            # print("editutils conditioning_full_refs after", conditioning_full_refs)
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
        if is_qwen:
            custom_output["vl_images"] = vl_images
            custom_output["full_prompt"] = full_prompt
        
        
        # print("editutils conditioning_output", conditioning_output)
        return (conditioning_output, latent_out, custom_output, main_image, noise_mask, pad_info)


class ConfigJsonParser_EditUtils:
    default_config = {
        "to_ref": True,
        "ref_main_image": False,
        "ref_longest_edge": 1024,
        "ref_crop": "center",
        "ref_upscale": "lanczos",
        "to_vl": True,
        "vl_resize": True,
        "vl_target_size": 384,
        "vl_crop": "center",
        "vl_upscale": "bicubic",
        "mask": None
    }
    default_config_json = json.dumps(default_config)
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": 
            {
                "image": ("IMAGE", ),
            },
            "optional": 
            {
                "configs": ("LIST", {"default": None, "tooltip": "Configs list"}),
                "config_json": ("STRING", {"default": s.default_config_json, "multiline": True, "tooltip": "Config JSON String"}),
                "mask": ("MASK", ),
            }
        }

    RETURN_TYPES = ("LIST", "ANY", )
    RETURN_NAMES = ("configs", "config", )
    FUNCTION = "prepare_config"
    CATEGORY = "advanced/conditioning"
    def prepare_config(self, image, configs=None,
                config_json="", mask=None
        ):
        if configs is None:
            configs = []
        # print("len(configs)", len(configs))
        
        config = self.default_config.copy()
        try:
            json_config = json.loads(config_json)
        except Exception as e:
            print(f"An error occurred while loading json_config")
            print(json_config)
        
        config.update(json_config)
        config["image"] = image
        
        config["mask"] = None
        
        # print("mask.shape", mask.shape)
        # print("image.shape", image.shape)
        if mask is not None:
            # check mask height,width equals image height,width
            if mask.shape[1] != image.shape[1] or mask.shape[2] != image.shape[2]:
                print("mask height,width not equals image height,width, skipping mask")
                mask = None
            config["mask"] = mask
        
        config_output = copy.deepcopy(configs)
        del configs
        
        config_output.append(config)
        # print("len(configs)", len(configs))
        return (config_output, config, )


class Flux2KleinConfigPreparer_EditUtils:
    upscale_methods = ["lanczos", "bicubic", "area"]
    crop_methods = ["pad", "center", "disabled"]
    vl_crop_methods = ["center", "disabled"]
    resize_modes = ["longest_edge", "area"]
    
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
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": 
            {
                "image": ("IMAGE", ),
            },
            "optional": 
            {
                "configs": ("LIST", {"default": None, "tooltip": "Configs list"}),
                "to_ref": ("BOOLEAN", {"default": True, "tooltip": "Add image to reference latent"}),
                "ref_main_image": ("BOOLEAN", {"default": True, "tooltip": "Set image as main image which would return the latent as output."}),
                "ref_longest_edge": ("INT", {"default": 1024, "min": 16, "max": 4096, "step": 1, "tooltip": "Longest edge of the output latent"}),
                "ref_crop": (s.crop_methods, {"default": "pad", "tooltip": "Crop method for reference image"}),
                "ref_upscale": (s.upscale_methods, {"default": "lanczos", "tooltip": "Upscale method for reference image"}),
                "mask": ("MASK", ),
                "ref_resize_mode": (s.resize_modes, {"default": "longest_edge", "tooltip": "longest_edge: scale so the longest dimension equals ref_longest_edge. area: scale so total pixels equals ref_longest_edge squared."}),
            }
        }

    RETURN_TYPES = ("LIST", "ANY", )
    RETURN_NAMES = ("configs", "config", )
    FUNCTION = "prepare_config"

    CATEGORY = "advanced/conditioning"
    def prepare_config(self, image, configs=None,
                to_ref=True, ref_main_image=True, ref_longest_edge=1024, ref_crop="center", ref_upscale="lanczos",
                mask=None, ref_resize_mode="longest_edge"
        ):
        if configs is None:
            configs = []
        # print("len(configs)", len(configs))
        # print("configs")
        # print(configs)
        config = {
            "image": image,
            "to_ref": to_ref,
            "ref_main_image": ref_main_image,
            "ref_longest_edge": ref_longest_edge,
            "ref_crop": ref_crop,
            "ref_upscale": ref_upscale,
            "ref_resize_mode": ref_resize_mode,
        }
        
        
        config_output = copy.deepcopy(configs)
        
        # print("mask.shape", mask.shape)
        # print("image.shape", image.shape)
        if mask is not None:
            # check mask height,width equals image height,width
            if mask.shape[1] != image.shape[1] or mask.shape[2] != image.shape[2]:
                print("mask height,width not equals image height,width, skipping mask")
                mask = None
            config["mask"] = mask
        
        del configs
        
        
        config_output.append(config)
        # print("len(configs)", len(configs))
        return (config_output, config, )


class BooguConfigPreparer_EditUtils:
    upscale_methods = ["lanczos", "bicubic", "area"]
    crop_methods = ["pad", "center", "disabled"]
    vl_crop_methods = ["center", "disabled"]
    resize_modes = ["longest_edge", "area"]

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required":
            {
                "image": ("IMAGE", ),
            },
            "optional":
            {
                "configs": ("LIST", {"default": None, "tooltip": "Configs list"}),
                "to_ref": ("BOOLEAN", {"default": True, "tooltip": "Add image to reference latent"}),
                "ref_main_image": ("BOOLEAN", {"default": True, "tooltip": "Set image as main image which would return the latent as output."}),
                "ref_longest_edge": ("INT", {"default": 1024, "min": 16, "max": 4096, "step": 1, "tooltip": "Longest edge of the output latent"}),
                "ref_crop": (s.crop_methods, {"default": "pad", "tooltip": "Crop method for reference image"}),
                "ref_upscale": (s.upscale_methods, {"default": "lanczos", "tooltip": "Upscale method for reference image"}),
                "to_vl": ("BOOLEAN", {"default": True, "tooltip": "Add image to boogu vision tower encode"}),
                "vl_target_size": ("INT", {"default": 384, "min": 384, "max": 2048, "tooltip": "Target size for vision tower input"}),
                "vl_crop": (s.vl_crop_methods, {"default": "center", "tooltip": "Crop method for vision tower input"}),
                "vl_upscale": (s.upscale_methods, {"default": "lanczos", "tooltip": "Upscale method for vision tower input"}),
                "mask": ("MASK", ),
                "ref_resize_mode": (s.resize_modes, {"default": "longest_edge", "tooltip": "longest_edge: scale so the longest dimension equals ref_longest_edge. area: scale so total pixels equals ref_longest_edge squared."}),
            }
        }

    RETURN_TYPES = ("LIST", "ANY", )
    RETURN_NAMES = ("configs", "config", )
    FUNCTION = "prepare_config"

    CATEGORY = "advanced/conditioning"

    def prepare_config(self, image, configs=None,
                to_ref=True, ref_main_image=True, ref_longest_edge=1024, ref_crop="pad", ref_upscale="lanczos",
                to_vl=True, vl_target_size=384, vl_crop="center", vl_upscale="lanczos",
                mask=None, ref_resize_mode="area"
        ):
        if configs is None:
            configs = []
        config = {
            "image": image,
            "to_ref": to_ref,
            "ref_main_image": ref_main_image,
            "ref_longest_edge": ref_longest_edge,
            "ref_crop": ref_crop,
            "ref_upscale": ref_upscale,
            "to_vl": to_vl,
            "vl_target_size": vl_target_size,
            "vl_crop": vl_crop,
            "vl_upscale": vl_upscale,
            "ref_resize_mode": ref_resize_mode,
        }

        config_output = copy.deepcopy(configs)

        if mask is not None:
            # check mask height,width equals image height,width
            if mask.shape[1] != image.shape[1] or mask.shape[2] != image.shape[2]:
                print("mask height,width not equals image height,width, skipping mask")
                mask = None
            config["mask"] = mask

        del configs

        config_output.append(config)
        return (config_output, config, )


class Flux2KleinOutputExtractor_EditUtils:
    preset_keys = [
        "pad_info",
        "main_image",
        "vae_images",
        "ref_latents",
        "full_prompt",
        "llama_template",
        "no_refs_cond",
        "mask"
    ]
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": 
            {
                "custom_output": ("ANY", ),
            }
        }

    RETURN_TYPES = ("ANY", "IMAGE", "LIST", "LIST", "STRING", "STRING", "CONDITIONING", "MASK")
    RETURN_NAMES = ("pad_info", "main_image", "vae_images", "ref_latents", "full_prompt", "llama_template", "no_refs_cond", "mask")
    FUNCTION = "extract"

    CATEGORY = "advanced/conditioning"
    
    def extract(self, custom_output):
        pad_info = custom_output.get('pad_info')
        main_image = custom_output.get('main_image')
        vae_images = custom_output.get('vae_images')
        ref_latents = custom_output.get('ref_latents')
        full_prompt = custom_output.get('full_prompt')
        llama_template = custom_output.get('llama_template')
        no_refs_cond = custom_output.get('no_refs_cond')
        mask = custom_output.get('mask')
        
        return (pad_info, main_image, vae_images, ref_latents, full_prompt, llama_template, no_refs_cond, mask)


class BooguOutputExtractor_EditUtils:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required":
            {
                "custom_output": ("ANY", ),
            }
        }

    RETURN_TYPES = ("CONDITIONING", "LIST", "LIST", "IMAGE")
    RETURN_NAMES = ("negative_cond", "ref_latents", "vl_images", "main_image")
    FUNCTION = "extract"

    CATEGORY = "advanced/conditioning"

    def extract(self, custom_output):
        negative_cond = custom_output.get("negative_cond")
        ref_latents = custom_output.get("ref_latents")
        vl_images = custom_output.get("vl_images")
        vae_images = custom_output.get("vae_images")
        main_image = vae_images[0] if vae_images and len(vae_images) > 0 else None
        return (negative_cond, ref_latents, vl_images, main_image)


class QwenConfigPreparer_EditUtils:
    upscale_methods = ["lanczos", "bicubic", "area"]
    crop_methods = ["pad", "center", "disabled"]
    vl_crop_methods = ["center", "disabled"]
    resize_modes = ["longest_edge", "area"]
    
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
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": 
            {
                "image": ("IMAGE", ),
            },
            "optional": 
            {
                "configs": ("LIST", {"default": None, "tooltip": "Configs list"}),
                "to_ref": ("BOOLEAN", {"default": True, "tooltip": "Add image to reference latent"}),
                "ref_main_image": ("BOOLEAN", {"default": True, "tooltip": "Set image as main image which would return the latent as output."}),
                "ref_longest_edge": ("INT", {"default": 1024, "min": 8, "max": 4096, "step": 1, "tooltip": "Longest edge of the output latent"}),
                "ref_crop": (s.crop_methods, {"default": "pad", "tooltip": "Crop method for reference image"}),
                "ref_upscale": (s.upscale_methods, {"default": "lanczos", "tooltip": "Upscale method for reference image"}),
    
                "to_vl": ("BOOLEAN", {"default": True, "tooltip": "Add image to qwenvl 2.5 encode"}),
                "vl_resize": ("BOOLEAN", {"default": True, "tooltip": "Resize image before qwenvl 2.5 encode"}),
                "vl_target_size": ("INT", {"default": 384, "min": 384, "max": 2048, "tooltip": "Target size of the qwenvl 2.5 encode"}),
                "vl_crop": (s.vl_crop_methods, {"default": "center", "tooltip": "Crop method for reference image"}),
                "vl_upscale": (s.upscale_methods, {"default": "lanczos", "tooltip": "Upscale method for reference image"}),
                "mask": ("MASK", ),
                "ref_resize_mode": (s.resize_modes, {"default": "longest_edge", "tooltip": "longest_edge: scale so the longest dimension equals ref_longest_edge. area: scale so total pixels equals ref_longest_edge squared."}),
            }
        }

    RETURN_TYPES = ("LIST", "ANY", )
    RETURN_NAMES = ("configs", "config", )
    FUNCTION = "prepare_config"

    CATEGORY = "advanced/conditioning"
    def prepare_config(self, image, configs=None,
                to_ref=True, ref_main_image=True, ref_longest_edge=1024, ref_crop="center", ref_upscale="lanczos",
                to_vl=True, vl_resize=True, vl_target_size=384, vl_crop="center", vl_upscale="bicubic",
                mask=None, ref_resize_mode="longest_edge"
        ):
        if configs is None:
            configs = []
        # print("len(configs)", len(configs))
        # print("configs")
        # print(configs)
        config = {
            "image": image,
            "to_ref": to_ref,
            "ref_main_image": ref_main_image,
            "ref_longest_edge": ref_longest_edge,
            "ref_crop": ref_crop,
            "ref_upscale": ref_upscale,
            "ref_resize_mode": ref_resize_mode,
            
            "to_vl": to_vl,
            "vl_resize": vl_resize,
            "vl_target_size": vl_target_size,
            "vl_crop": vl_crop,
            "vl_upscale": vl_upscale
        }
        
        
        config_output = copy.deepcopy(configs)
        
        # print("mask.shape", mask.shape)
        # print("image.shape", image.shape)
        if mask is not None:
            # check mask height,width equals image height,width
            if mask.shape[1] != image.shape[1] or mask.shape[2] != image.shape[2]:
                print("mask height,width not equals image height,width, skipping mask")
                mask = None
            config["mask"] = mask
        
        del configs
        
        
        config_output.append(config)
        # print("len(configs)", len(configs))
        return (config_output, config, )

class QwenEditOutputExtractor_EditUtils:
    preset_keys = [
        "pad_info",
        "full_refs_cond",
        # "main_ref_cond",
        "main_image",
        "vae_images",
        "ref_latents",
        "vl_images",
        "full_prompt",
        # "llama_template",
        "no_refs_cond",
        "mask"
    ]
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": 
            {
                "custom_output": ("ANY", ),
            }
        }

    RETURN_TYPES = ("ANY", "CONDITIONING", "CONDITIONING", "IMAGE", "LIST", "LIST", "LIST", "STRING", "STRING", "CONDITIONING", "MASK")
    RETURN_NAMES = ("pad_info", "full_refs_cond", "main_ref_cond", "main_image", "vae_images", "ref_latents", "vl_images", "full_prompt", "llama_template", "no_refs_cond", "mask")
    FUNCTION = "extract"

    CATEGORY = "advanced/conditioning"
    
    def extract(self, custom_output):
        pad_info = custom_output.get('pad_info')
        full_refs_cond = custom_output.get('full_refs_cond')
        main_ref_cond = custom_output.get('main_ref_cond')
        main_image = custom_output.get('main_image')
        vae_images = custom_output.get('vae_images')
        ref_latents = custom_output.get('ref_latents')
        vl_images = custom_output.get('vl_images')
        full_prompt = custom_output.get('full_prompt')
        llama_template = custom_output.get('llama_template')
        no_refs_cond = custom_output.get('no_refs_cond')
        mask = custom_output.get('mask')
        
        return (pad_info, full_refs_cond, main_ref_cond, main_image, vae_images, ref_latents, vl_images, full_prompt, llama_template, no_refs_cond, mask)



class ListExtractor_EditUtils:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": 
            {
                "items": ("LIST", ),
                "index": ("INT", {"default": 0, "min": 0, "max": 1000000, "step": 1, "tooltip": "Index of the image"}),
            }
        }

    RETURN_TYPES = ("ANY", )
    RETURN_NAMES = ("item", )
    FUNCTION = "extract"

    CATEGORY = "advanced/conditioning"
    def extract(self, items, index):
        assert index < len(items), f"Index out of range, len(image_list): {len(items)}"
        
        return (items[index], )


class ClearRefLatents_EditUtils:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": 
            {
                "conditioning": ("CONDITIONING", ),
            }
        }

    RETURN_TYPES = ("CONDITIONING", )
    RETURN_NAMES = ("conditioning", )
    FUNCTION = "clear_refs"

    CATEGORY = "advanced/conditioning"
    def clear_refs(self, conditioning):
        cleared = node_helpers.conditioning_set_values(conditioning, {"reference_latents": []}, append=False)
        return (cleared, )


condition_dir = os.path.join(folder_paths.models_dir, "conditions")

if "conditions" not in folder_paths.folder_names_and_paths:
    folder_paths.folder_names_and_paths["conditions"] = ([condition_dir], {'.ckpt'})


class SaveCondition_EditUtils:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "condition": ("CONDITIONING",),
                "filename": ("STRING", {"default": "condition_tensor"}),
            },
        }

    RETURN_TYPES = ()
    FUNCTION = "save_condition"
    CATEGORY = "advanced/conditioning"
    OUTPUT_NODE = True

    def save_condition(self, condition, filename):
        os.makedirs(condition_dir, exist_ok=True)

        safe_filename = os.path.basename(filename)
        if not safe_filename.endswith(".ckpt"):
            safe_filename += ".ckpt"

        save_dir = folder_paths.folder_names_and_paths["conditions"][0][0]
        save_path = os.path.join(save_dir, safe_filename)

        try:
            tensors_to_save = []
            for cond, cond_info in condition:
                tensors_to_save.append((cond.clone(), cond_info))
            torch.save(tensors_to_save, save_path)
            print(f"Condition tensor saved to {save_path}")
        except Exception as e:
            print(f"Error saving condition tensor: {e}")

        return ()


class LoadCondition_EditUtils:
    @classmethod
    def INPUT_TYPES(cls):
        os.makedirs(condition_dir, exist_ok=True)

        condition_files = []
        try:
            condition_files = folder_paths.get_filename_list("conditions")
            condition_files = [os.path.splitext(file)[0] for file in condition_files]
        except Exception as e:
            print(f"Error listing condition files: {e}")

        if not condition_files:
            condition_files = ["No files found"]

        return {
            "required": {
                "filename": (condition_files, ),
            },
        }

    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "load_condition"
    CATEGORY = "advanced/conditioning"

    def load_condition(self, filename):
        os.makedirs(condition_dir, exist_ok=True)

        safe_filename = os.path.basename(filename)
        if not safe_filename.endswith(".ckpt"):
            safe_filename += ".ckpt"

        load_dir = folder_paths.folder_names_and_paths["conditions"][0][0]
        load_path = os.path.join(load_dir, safe_filename)

        try:
            if not os.path.exists(load_path):
                raise FileNotFoundError(f"Condition file not found: {load_path}")
            loaded_tensors = torch.load(load_path)
            print(f"Condition tensor loaded from {load_path}")
            return (loaded_tensors,)
        except Exception as e:
            print(f"Error loading condition tensor: {e}")
            return ([],)


class LoadConditionFromLoras_EditUtils:
    @classmethod
    def INPUT_TYPES(cls):
        os.makedirs(condition_dir, exist_ok=True)

        condition_files = []
        try:
            condition_files = folder_paths.get_filename_list("loras")
            condition_files = [os.path.splitext(file)[0] for file in condition_files]
        except Exception as e:
            print(f"Error listing condition files: {e}")

        if not condition_files:
            condition_files = ["No files found"]

        return {
            "required": {
                "filename": (condition_files, ),
            },
        }

    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "load_condition"
    CATEGORY = "advanced/conditioning"

    def load_condition(self, filename):
        os.makedirs(condition_dir, exist_ok=True)

        safe_filename = os.path.basename(filename)
        if not safe_filename.endswith(".ckpt"):
            safe_filename += ".ckpt"

        load_dir = folder_paths.folder_names_and_paths["conditions"][0][0]
        load_path = os.path.join(load_dir, safe_filename)

        try:
            if not os.path.exists(load_path):
                raise FileNotFoundError(f"Condition file not found: {load_path}")
            loaded_tensors = torch.load(load_path)
            print(f"Condition tensor loaded from {load_path}")
            return (loaded_tensors,)
        except Exception as e:
            print(f"Error loading condition tensor: {e}")
            return ([],)


class Any2Image_EditUtils:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": 
            {
                "item": ("ANY", ),
            }
        }

    RETURN_TYPES = ("IMAGE", )
    RETURN_NAMES = ("item", )
    FUNCTION = "extract"

    CATEGORY = "advanced/conditioning"
    def extract(self, item):
        return (item, )


class Any2Latent_EditUtils:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": 
            {
                "item": ("ANY", ),
            }
        }

    RETURN_TYPES = ("LATENT", )
    RETURN_NAMES = ("item", )
    FUNCTION = "extract"

    CATEGORY = "advanced/conditioning"
    def extract(self, item):
        latent_out = {"samples": item}
        return (latent_out, )  

class AdaptiveLongestEdge_EditUtils:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": 
            {
                "image": ("IMAGE", ),
                "min_size": ("INT", {"default": 512, "min": 64, "max": 4096, "step": 1, "tooltip": "When image is smaller than min_size, it will be resized to at least the min_size."}),
                "max_size": ("INT", {"default": 2048, "min": 512, "max": 4096, "step": 1, "tooltip": "When image is larger than max_size, it will be resized to under the max_size."}),
            },
        }

    RETURN_TYPES = ("INT", )
    RETURN_NAMES = ("longest_edge", )
    FUNCTION = "calculate_longest_edge"

    CATEGORY = "advanced/conditioning"
    def calculate_longest_edge(self, image, min_size, max_size):
        output = max(image.shape[1], image.shape[2])
        
        # print("image.shape[2], image.shape[3]", image.shape[1], image.shape[2])
        # print("longest_edge", output)
        if output <= max_size:
            if output < min_size:
                output = min_size
            return (output, )
        # Find how many times m fits into n
        k = int(math.ceil(output / max_size))
        # print("k", k)
        # Scale down by that factor
        
        output = int(output / k)
        
        if output < min_size:
            output = min_size
        # print("output", output)
        return (output, )  


class LoadImageWithFilename_EditUtils:
    @classmethod
    def INPUT_TYPES(s):
        input_dir = folder_paths.get_input_directory()
        files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]
        files = folder_paths.filter_files_content_types(files, ["image"])
        return {"required":
                    {"image": (sorted(files), {"image_upload": True})},
                }

    CATEGORY = "image"

    RETURN_TYPES = ("IMAGE", "MASK", "STRING", )
    RETURN_NAMES = ("image", "mask", "filename", )
    FUNCTION = "load_image"
    def load_image(self, image):
        image_path = folder_paths.get_annotated_filepath(image)
        
        # get filename from image_path without ext
        filename = os.path.splitext(os.path.basename(image_path))[0]
        
        img = node_helpers.pillow(Image.open, image_path)

        output_images = []
        output_masks = []
        w, h = None, None

        excluded_formats = ['MPO']

        for i in ImageSequence.Iterator(img):
            i = node_helpers.pillow(ImageOps.exif_transpose, i)

            if i.mode == 'I':
                i = i.point(lambda i: i * (1 / 255))
            image = i.convert("RGB")

            if len(output_images) == 0:
                w = image.size[0]
                h = image.size[1]

            if image.size[0] != w or image.size[1] != h:
                continue

            image = np.array(image).astype(np.float32) / 255.0
            image = torch.from_numpy(image)[None,]
            if 'A' in i.getbands():
                mask = np.array(i.getchannel('A')).astype(np.float32) / 255.0
                mask = 1. - torch.from_numpy(mask)
            elif i.mode == 'P' and 'transparency' in i.info:
                mask = np.array(i.convert('RGBA').getchannel('A')).astype(np.float32) / 255.0
                mask = 1. - torch.from_numpy(mask)
            else:
                mask = torch.zeros((64,64), dtype=torch.float32, device="cpu")
            output_images.append(image)
            output_masks.append(mask.unsqueeze(0))

        if len(output_images) > 1 and img.format not in excluded_formats:
            output_image = torch.cat(output_images, dim=0)
            output_mask = torch.cat(output_masks, dim=0)
        else:
            output_image = output_images[0]
            output_mask = output_masks[0]

        return (output_image, output_mask, filename)

    @classmethod
    def IS_CHANGED(s, image):
        image_path = folder_paths.get_annotated_filepath(image)
        m = hashlib.sha256()
        with open(image_path, 'rb') as f:
            m.update(f.read())
        return m.digest().hex()

    @classmethod
    def VALIDATE_INPUTS(s, image):
        if not folder_paths.exists_annotated_filepath(image):
            return "Invalid image file: {}".format(image)

        return True


class DiffMask_EditUtils:
    """
    Enhanced difference mask generation node:
    - Supports frequency-domain high-pass filtering, extremely sensitive to high-frequency details (textures, edges, tiny displacements)
    - Can blend with original pixel difference to balance robustness and sensitivity
    - Preserves morphological opening and Gaussian blur for denoising and edge smoothing
    """
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image1": ("IMAGE", ),
                "image2": ("IMAGE", ),
                "threshold": ("FLOAT", {"default": 0.05, "min": 0.0, "max": 1.0, "step": 0.001, "tooltip": "Binarization threshold (0-1)"}),
                "blur_radius": ("INT", {"default": 5, "min": 0, "max": 50, "step": 1, "tooltip": "Gaussian blur kernel size (0 means no blur)"}),
                "erode_size": ("INT", {"default": 3, "min": 0, "max": 20, "step": 1, "tooltip": "Erosion kernel size (0 means no erosion)"}),
                "dilate_size": ("INT", {"default": 3, "min": 0, "max": 20, "step": 1, "tooltip": "Dilation kernel size (0 means no dilation)"}),
                "use_frequency": ("BOOLEAN", {"default": True, "tooltip": "Enable frequency-domain high-pass filtering (strongly recommended)"}),
                "highpass_sigma": ("FLOAT", {"default": 10.0, "min": 1.0, "max": 100.0, "step": 0.5, "tooltip": "High-pass filter cutoff frequency (sigma), smaller values emphasize higher frequencies"}),
                "freq_weight": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 1.0, "step": 0.05, "tooltip": "Weight for frequency-domain difference, remaining weight goes to original pixel difference"}),
            },
        }

    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("mask",)
    FUNCTION = "generate_diff_mask"

    CATEGORY = "image"

    def generate_diff_mask(self, image1, image2, threshold, blur_radius, erode_size, dilate_size,
                           use_frequency, highpass_sigma, freq_weight):
        # Ensure images have the same shape
        if image1.shape != image2.shape:
            raise ValueError(f"Image shapes do not match: {image1.shape} vs {image2.shape}")

        # Convert to grayscale (luminance weighted average) to simplify frequency-domain processing
        def rgb_to_luminance(img):
            # img: (B, H, W, C) range [0,1]
            luminance = 0.299 * img[..., 0] + 0.587 * img[..., 1] + 0.114 * img[..., 2]
            return luminance.unsqueeze(1)  # (B,1,H,W)

        gray1 = rgb_to_luminance(image1)
        gray2 = rgb_to_luminance(image2)

        # Original pixel difference (spatial domain)
        pixel_diff = torch.abs(gray1 - gray2)  # (B,1,H,W)

        if use_frequency:
            # Frequency-domain difference calculation (high-pass filter version)
            freq_diff = self._frequency_highpass_diff(gray1, gray2, highpass_sigma)
            # Blend frequency-domain difference with spatial-domain difference
            combined_diff = freq_weight * freq_diff + (1 - freq_weight) * pixel_diff
        else:
            combined_diff = pixel_diff

        # Binarization threshold
        mask = (combined_diff > threshold).float()

        # Morphological opening (erosion followed by dilation) to remove isolated noise
        if erode_size > 0:
            kernel_size = erode_size * 2 + 1
            padding = erode_size
            mask = -F.max_pool2d(-mask, kernel_size=kernel_size, stride=1, padding=padding)
        if dilate_size > 0:
            kernel_size = dilate_size * 2 + 1
            padding = dilate_size
            mask = F.max_pool2d(mask, kernel_size=kernel_size, stride=1, padding=padding)

        # Gaussian blur to smooth mask edges
        if blur_radius > 0:
            if blur_radius % 2 == 0:
                blur_radius += 1
            sigma = blur_radius / 3.0
            kernel_size = blur_radius
            x = torch.arange(kernel_size, dtype=torch.float32, device=mask.device) - (kernel_size - 1) / 2
            gaussian_1d = torch.exp(-x**2 / (2 * sigma**2))
            gaussian_1d = gaussian_1d / gaussian_1d.sum()
            kernel_h = gaussian_1d.view(1, 1, 1, kernel_size)
            kernel_v = gaussian_1d.view(1, 1, kernel_size, 1)
            padding = kernel_size // 2
            mask = F.conv2d(mask, kernel_h.expand(mask.size(1), 1, 1, kernel_size),
                            padding=(0, padding), groups=mask.size(1))
            mask = F.conv2d(mask, kernel_v.expand(mask.size(1), 1, kernel_size, 1),
                            padding=(padding, 0), groups=mask.size(1))
            mask = mask.clamp(0, 1)

        # Remove channel dimension, return (B, H, W)
        mask = mask.squeeze(1)
        return (mask,)

    def _frequency_highpass_diff(self, img1, img2, sigma):
        """
        Compute high-pass frequency-domain difference map between two grayscale images.
        img1, img2: (B,1,H,W) range [0,1]
        sigma: cutoff frequency for Gaussian high-pass filter (larger values preserve more low frequencies, smaller values are more high-pass)
        Returns: (B,1,H,W) difference map (range [0,1])
        """
        B, C, H, W = img1.shape
        device = img1.device

        # Real FFT (use rfft2 to save computation)
        f1 = torch.fft.rfft2(img1, norm='ortho')   # Complex tensor (B,1,H, W//2+1)
        f2 = torch.fft.rfft2(img2, norm='ortho')
        diff_f = f1 - f2

        # Build high-pass filter (Gaussian high-pass)
        # Frequency coordinates normalized to [-0.5,0.5] range, only take positive frequencies (rfft2)
        fy = torch.fft.fftfreq(H, device=device).view(-1, 1)
        fx = torch.fft.rfftfreq(W, device=device).view(1, -1)
        # Radius squared (accounting for rfft storing only half the frequencies)
        radius_sq = fy**2 + fx**2
        # Gaussian high-pass: 1 - exp(-radius_sq / (2*sigma_norm^2))
        # sigma_norm uses relative values, larger sigma is closer to all-pass, smaller sigma is more high-pass
        sigma_norm = sigma / min(H, W)  # Map sigma to normalized frequency domain
        highpass = 1 - torch.exp(-radius_sq / (2 * sigma_norm**2))
        highpass = highpass.unsqueeze(0).unsqueeze(0)  # (1,1,H, W//2+1)

        # Apply filter
        filtered_f = diff_f * highpass

        # Inverse transform back to spatial domain (real)
        filtered_spatial = torch.fft.irfft2(filtered_f, s=(H, W), norm='ortho')
        # Take absolute value as difference intensity (may have negative values)
        diff_map = torch.abs(filtered_spatial)

        # Normalize to [0,1] (across batch and spatial dimensions)
        diff_map = diff_map / (diff_map.max() + 1e-6)
        return diff_map


class QwenEditTextEncode_EditUtils:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "clip": ("CLIP", ),
                "vae": ("VAE", ),
                "prompt": ("STRING", {"multiline": True, "dynamicPrompts": True}),
            },
            "optional": {
                "image1": ("IMAGE", ),
                "image2": ("IMAGE", ),
                "image3": ("IMAGE", ),
                "ref_longest_edge": ("INT", {"default": 1024, "min": 8, "max": 4096, "step": 1, "tooltip": "Longest edge of the output latent"}),
                "mask": ("MASK", ),
            }
        }

    RETURN_TYPES = ("CONDITIONING", "LATENT", "ANY", "IMAGE", "MASK")
    RETURN_NAMES = ("conditioning", "latent", "custom_output", "main_image", "mask")
    FUNCTION = "encode"

    CATEGORY = "advanced/conditioning"
    
    def encode(self, clip, vae, prompt,
               image1=None, image2=None, image3=None,
               mask=None,  # New mask parameter
               ref_longest_edge=1024):
        # Prepare model config
        model_config = {
            "model_name": "qwen",
            "instruction": "",
            "vae_unit": 8,
            "llama_template": get_system_prompt("")
        }
        
        # Prepare configs list
        configs = []
        
        # Process each image if provided
        if image1 is not None:
            
            config1 = {
                "image": image1,
                "to_ref": True,  # Default to True
                "ref_main_image": True,  # First image is main by default
                "ref_longest_edge": ref_longest_edge,
                "ref_crop": "pad",  # Default to pad
                "ref_upscale": "lanczos",  # Default to lanczos
                "to_vl": True,  # Default to True
                "vl_resize": True,  # Default to True
                "vl_target_size": 384,  # Default to 384
                "vl_crop": "center",  # Default to center
                "vl_upscale": "lanczos"  # Default to lanczos
            }
            # Set mask to image1 if provided
            if mask is not None:
                config1["mask"] = mask
            configs.append(config1)
        
        if image2 is not None:
            config2 = {
                "image": image2,
                "to_ref": True,  # Default to True
                "ref_main_image": False,  # Only first image is main
                "ref_longest_edge": ref_longest_edge,
                "ref_crop": "pad",  # Default to pad
                "ref_upscale": "lanczos",  # Default to lanczos
                "to_vl": True,  # Default to True
                "vl_resize": True,  # Default to True
                "vl_target_size": 384,  # Default to 384
                "vl_crop": "center",  # Default to center
                "vl_upscale": "lanczos"  # Default to lanczos
            }
            configs.append(config2)
        
        if image3 is not None:
            config3 = {
                "image": image3,
                "to_ref": True,  # Default to True
                "ref_main_image": False,  # Only first image is main
                "ref_longest_edge": ref_longest_edge,
                "ref_crop": "pad",  # Default to pad
                "ref_upscale": "lanczos",  # Default to lanczos
                "to_vl": True,  # Default to True
                "vl_resize": True,  # Default to True
                "vl_target_size": 384,  # Default to 384
                "vl_crop": "center",  # Default to center
                "vl_upscale": "lanczos"  # Default to lanczos
            }
            configs.append(config3)
        
        if len(configs) == 0:
            raise ValueError("At least one image must be provided")
        
        # Call the original EditTextEncode function
        node_instance = EditTextEncode_EditUtils()
        return node_instance.encode(
            clip=clip,
            vae=vae,
            prompt=prompt,
            model_config=model_config,
            configs=configs
        )


class Flux2KleinEditTextEncode_EditUtils:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "clip": ("CLIP", ),
                "vae": ("VAE", ),
                "prompt": ("STRING", {"multiline": True, "dynamicPrompts": True}),
            },
            "optional": {
                "image1": ("IMAGE", ),
                "image2": ("IMAGE", ),
                "image3": ("IMAGE", ),
                "ref_longest_edge": ("INT", {"default": 1024, "min": 8, "max": 4096, "step": 1, "tooltip": "Longest edge of the output latent"}),
                "mask": ("MASK", ),
            }
        }

    RETURN_TYPES = ("CONDITIONING", "LATENT", "ANY", "IMAGE", "MASK")
    RETURN_NAMES = ("conditioning", "latent", "custom_output", "main_image", "mask")
    FUNCTION = "encode"

    CATEGORY = "advanced/conditioning"
    
    def encode(self, clip, vae, prompt,
               image1=None, image2=None, image3=None,
               mask=None,  # New mask parameter
               ref_longest_edge=1024):
        # Prepare model config
        model_config = {
            "model_name": "flux2klein",
            "instruction": "",
            "vae_unit": 16,
            "llama_template": get_system_prompt("")
        }
        
        # Prepare configs list
        configs = []
        
        # Process each image if provided
        if image1 is not None:
            
            config1 = {
                "image": image1,
                "to_ref": True,  # Default to True
                "ref_main_image": True,  # First image is main by default
                "ref_longest_edge": ref_longest_edge,
                "ref_crop": "pad",  # Default to pad
                "ref_upscale": "lanczos",  # Default to lanczos
            }
            # Set mask to image1 if provided
            if mask is not None:
                config1["mask"] = mask
            configs.append(config1)
        
        if image2 is not None:
            config2 = {
                "image": image2,
                "to_ref": True,  # Default to True
                "ref_main_image": False,  # Only first image is main
                "ref_longest_edge": ref_longest_edge,
                "ref_crop": "pad",  # Default to pad
                "ref_upscale": "lanczos",  # Default to lanczos
            }
            configs.append(config2)
        
        if image3 is not None:
            config3 = {
                "image": image3,
                "to_ref": True,  # Default to True
                "ref_main_image": False,  # Only first image is main
                "ref_longest_edge": ref_longest_edge,
                "ref_crop": "pad",  # Default to pad
                "ref_upscale": "lanczos",  # Default to lanczos
            }
            configs.append(config3)
        
        if len(configs) == 0:
            raise ValueError("At least one image must be provided")
        
        # Call the original EditTextEncode function
        node_instance = EditTextEncode_EditUtils()
        return node_instance.encode(
            clip=clip,
            vae=vae,
            prompt=prompt,
            model_config=model_config,
            configs=configs
        )


class BooguEditTextEncode_EditUtils:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "clip": ("CLIP", ),
                "vae": ("VAE", ),
                "prompt": ("STRING", {"multiline": True, "dynamicPrompts": True}),
            },
            "optional": {
                "negative_prompt": ("STRING", {"multiline": True, "dynamicPrompts": True, "default": ""}),
                "image1": ("IMAGE", ),
                "image2": ("IMAGE", ),
                "image3": ("IMAGE", ),
                "ref_longest_edge": ("INT", {"default": 1024, "min": 16, "max": 4096, "step": 1, "tooltip": "Longest edge of the output latent"}),
                "mask": ("MASK", ),
                "ref_resize_mode": (["longest_edge", "area"], {"default": "longest_edge", "tooltip": "longest_edge vs area-based resize"}),
            }
        }

    RETURN_TYPES = ("CONDITIONING", "CONDITIONING", "LATENT", "ANY", "IMAGE")
    RETURN_NAMES = ("positive", "negative", "latent", "custom_output", "main_image")
    FUNCTION = "encode"

    CATEGORY = "advanced/conditioning"

    def encode(self, clip, vae, prompt,
               negative_prompt="",
               image1=None, image2=None, image3=None,
               ref_longest_edge=1024, mask=None, ref_resize_mode="longest_edge"):
        # Prepare model config for boogu
        model_config = {
            "model_name": "boogu",
            "vae_unit": 8,
            "llama_template": "",
            "negative_prompt": negative_prompt
        }

        # Prepare configs list
        configs = []

        if image1 is not None:
            config1 = {
                "image": image1,
                "to_ref": True,
                "ref_main_image": True,
                "ref_longest_edge": ref_longest_edge,
                "ref_resize_mode": ref_resize_mode,
                "ref_crop": "pad",
                "ref_upscale": "lanczos",
            }
            if mask is not None:
                config1["mask"] = mask
            configs.append(config1)

        if image2 is not None:
            configs.append({
                "image": image2,
                "to_ref": True,
                "ref_main_image": False,
                "ref_longest_edge": ref_longest_edge,
                "ref_resize_mode": ref_resize_mode,
                "ref_crop": "pad",
                "ref_upscale": "lanczos",
            })

        if image3 is not None:
            configs.append({
                "image": image3,
                "to_ref": True,
                "ref_main_image": False,
                "ref_longest_edge": ref_longest_edge,
                "ref_resize_mode": ref_resize_mode,
                "ref_crop": "pad",
                "ref_upscale": "lanczos",
            })

        if len(configs) == 0:
            raise ValueError("At least one image must be provided")

        # Call the unified EditTextEncode function
        node_instance = EditTextEncode_EditUtils()
        positive, latent_out, custom_output, main_image, noise_mask, pad_info = node_instance.encode(
            clip=clip,
            vae=vae,
            prompt=prompt,
            model_config=model_config,
            configs=configs
        )

        negative = custom_output.get("negative_cond")
        return (positive, negative, latent_out, custom_output, main_image)


class LongestEdgeImageProcess_EditUtils:
    upscale_methods = ["lanczos", "bicubic", "area"]
    crop_methods = ["pad", "center", "disabled"]

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE", ),
                "ref_longest_edge": ("INT", {"default": 1024, "min": 8, "max": 4096, "step": 1, "tooltip": "Longest edge of the output image"}),
                "ref_crop": (s.crop_methods, {"default": "pad", "tooltip": "Crop method: pad adds padding, center crops to center, disabled keeps as-is"}),
                "ref_upscale": (s.upscale_methods, {"default": "lanczos", "tooltip": "Upscale method"}),
                "vae_unit": ("INT", {"default": 8, "min": 8, "max": 64, "step": 8, "tooltip": "VAE unit size for padding alignment"}),
            },
        }

    RETURN_TYPES = ("IMAGE", "ANY", "FLOAT")
    RETURN_NAMES = ("processed_image", "pad_info", "scale_by")
    FUNCTION = "process"

    CATEGORY = "image"

    def process(self, image, ref_longest_edge, ref_crop, ref_upscale, vae_unit):
        samples = image.movedim(-1, 1)

        ori_longest_edge = max(samples.shape[2], samples.shape[3])
        scale_by = ori_longest_edge / ref_longest_edge
        scaled_height = int(round(samples.shape[2] / scale_by))
        scaled_width = int(round(samples.shape[3] / scale_by))

        pad_info = {
            "x": 0,
            "y": 0,
            "width": 0,
            "height": 0,
            "scale_by": round(1 / scale_by, 3)
        }

        if ref_crop == "pad":
            crop = "center"

            width_ceil = math.ceil(scaled_width / vae_unit)
            canvas_width = width_ceil * vae_unit

            height_ceil = math.ceil(scaled_height / vae_unit)
            canvas_height = height_ceil * vae_unit

            canvas = torch.zeros(
                (samples.shape[0], samples.shape[1], canvas_height, canvas_width),
                dtype=samples.dtype,
                device=samples.device
            )

            resized_samples = comfy.utils.common_upscale(samples, scaled_width, scaled_height, ref_upscale, crop)

            resized_width = resized_samples.shape[3]
            resized_height = resized_samples.shape[2]
            canvas[:, :, :resized_height, :resized_width] = resized_samples

            current_total = (samples.shape[3] * samples.shape[2])
            total = int(resized_width * resized_height)
            scale_by_val = math.sqrt(total / current_total)

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

        processed_image = s.movedim(1, -1)

        return (processed_image, pad_info, pad_info["scale_by"])


NODE_CLASS_MAPPINGS = {
    "CropWithPadInfo_EditUtils": CropWithPadInfo_EditUtils,
    "ModelConfig_EditUtils": ModelConfig_EditUtils,
    "EditTextEncode_EditUtils": EditTextEncode_EditUtils,
    "QwenModelConfig_EditUtils": QwenModelConfig_EditUtils,
    "QwenConfigPreparer_EditUtils": QwenConfigPreparer_EditUtils,
    "QwenEditTextEncode_EditUtils": QwenEditTextEncode_EditUtils,
    "QwenEditOutputExtractor_EditUtils": QwenEditOutputExtractor_EditUtils,
    "Flux2KleinModelConfig_EditUtils": Flux2KleinModelConfig_EditUtils,
    "Flux2KleinConfigPreparer_EditUtils": Flux2KleinConfigPreparer_EditUtils,
    "Flux2KleinEditTextEncode_EditUtils": Flux2KleinEditTextEncode_EditUtils,
    "Flux2KleinOutputExtractor_EditUtils": Flux2KleinOutputExtractor_EditUtils,
    "BooguModelConfig_EditUtils": BooguModelConfig_EditUtils,
    "BooguConfigPreparer_EditUtils": BooguConfigPreparer_EditUtils,
    "BooguEditTextEncode_EditUtils": BooguEditTextEncode_EditUtils,
    "BooguOutputExtractor_EditUtils": BooguOutputExtractor_EditUtils,
    "ConfigJsonParser_EditUtils": ConfigJsonParser_EditUtils,
    "ListExtractor_EditUtils": ListExtractor_EditUtils,
    "Any2Image_EditUtils": Any2Image_EditUtils,
    "Any2Latent_EditUtils": Any2Latent_EditUtils,
    "AdaptiveLongestEdge_EditUtils": AdaptiveLongestEdge_EditUtils,
    "LoadImageWithFilename_EditUtils": LoadImageWithFilename_EditUtils,
    "LongestEdgeImageProcess_EditUtils": LongestEdgeImageProcess_EditUtils,
    "ClearRefLatents_EditUtils": ClearRefLatents_EditUtils,
    "SaveCondition_EditUtils": SaveCondition_EditUtils,
    "LoadCondition_EditUtils": LoadCondition_EditUtils,
    "LoadConditionFromLoras_EditUtils": LoadConditionFromLoras_EditUtils,
    "DiffMask_EditUtils": DiffMask_EditUtils
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "CropWithPadInfo_EditUtils": "EditUtils: Crop With Pad Info lrzjason",
    "ModelConfig_EditUtils": "EditUtils: Model Config lrzjason",
    "EditTextEncode_EditUtils": "EditUtils: EditTextEncode lrzjason",
    "QwenModelConfig_EditUtils": "EditUtils: Qwen Model Config lrzjason",
    "QwenConfigPreparer_EditUtils": "EditUtils: Qwen Config Preparer lrzjason",
    "QwenEditTextEncode_EditUtils": "EditUtils: Qwen Edit Text Encode lrzjason",
    "QwenEditConfigPreparer_EditUtils": "EditUtils: Qwen Edit Config Preparer lrzjason",
    "QwenEditOutputExtractor_EditUtils": "EditUtils: Qwen Edit Output Extractor lrzjason",
    "Flux2KleinModelConfig_EditUtils": "EditUtils: Flux2Klein Model Config lrzjason",
    "Flux2KleinConfigPreparer_EditUtils": "EditUtils: Flux2Klein Config Preparer lrzjason",
    "Flux2KleinEditTextEncode_EditUtils": "EditUtils: Flux2Klein Edit Text Encode lrzjason",
    "Flux2KleinOutputExtractor_EditUtils": "EditUtils: Flux2Klein Output Extractor lrzjason",
    "BooguModelConfig_EditUtils": "EditUtils: Boogu Model Config lrzjason",
    "BooguConfigPreparer_EditUtils": "EditUtils: Boogu Config Preparer lrzjason",
    "BooguEditTextEncode_EditUtils": "EditUtils: Boogu Edit Text Encode lrzjason",
    "BooguOutputExtractor_EditUtils": "EditUtils: Boogu Output Extractor lrzjason",
    "ConfigJsonParser_EditUtils": "EditUtils: Config Json Parser lrzjason",
    "ListExtractor_EditUtils": "EditUtils: List Extractor lrzjason",
    "Any2Image_EditUtils": "EditUtils: Any2Image lrzjason",
    "Any2Latent_EditUtils": "EditUtils: Any2Latent lrzjason",
    "AdaptiveLongestEdge_EditUtils": "EditUtils: Adaptive Longest Edge lrzjason",
    "LoadImageWithFilename_EditUtils": "EditUtils: Load Image With Filename lrzjason",
    "LongestEdgeImageProcess_EditUtils": "EditUtils: Longest Edge Image Process lrzjason",
    "ClearRefLatents_EditUtils": "EditUtils: Clear Ref Latents lrzjason",
    "SaveCondition_EditUtils": "EditUtils: Save Condition lrzjason",
    "LoadCondition_EditUtils": "EditUtils: Load Condition lrzjason",
    "LoadConditionFromLoras_EditUtils": "EditUtils: Load Condition From Loras lrzjason",
    "DiffMask_EditUtils": "EditUtils: Diff Mask lrzjason"
}
