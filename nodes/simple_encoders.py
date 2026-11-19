"""One-node wrappers (image1..image3) around EditTextEncode_EditUtils per model."""



from ._shared import MINIMAX_H3_IMAGE_REF_MAX_SHORT_EDGE, MINIMAX_H3_IMAGE_REF_SHORT_EDGE, MINIMAX_H3_IMAGE_UNIT, get_system_prompt
from .encode import EditTextEncode_EditUtils

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
                "vl_upscale": "lanczos",  # Default to lanczos
                "pad_mode": "color",  # Default to color
                "pad_color": "white",  # Default to white, black pads get amplified during qwen generation
                "pad_noise": 0.0,  # Default to no noise
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
                "vl_upscale": "lanczos",  # Default to lanczos
                "pad_mode": "color",  # Default to color
                "pad_color": "white",  # Default to white, black pads get amplified during qwen generation
                "pad_noise": 0.0,  # Default to no noise
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
                "vl_upscale": "lanczos",  # Default to lanczos
                "pad_mode": "color",  # Default to color
                "pad_color": "white",  # Default to white, black pads get amplified during qwen generation
                "pad_noise": 0.0,  # Default to no noise
            }
            configs.append(config3)
        
        if len(configs) == 0:
            raise ValueError("At least one image must be provided")
        
        # Call the original EditTextEncode function
        node_instance = EditTextEncode_EditUtils()
        # EditTextEncode also returns `pad_info` and `prompt`; these 5-output
        # wrappers deliberately expose only the first five, so slice rather
        # than forwarding the whole tuple (which relies on truncation).
        encoded = node_instance.encode(
            clip=clip,
            vae=vae,
            prompt=prompt,
            model_config=model_config,
            configs=configs
        )
        return encoded[:5]


class QwenImage21EditTextEncode_EditUtils:
    # Qwen-Image 2.1 edit text encode: up to 3 reference images through the
    # QwenImage21 branch (same resize for the Qwen3-VL vision tower and the
    # 16x VAE, 64ch latents, image_slots spliced conditioning).
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
                "ref_longest_edge": ("INT", {"default": 1024, "min": 16, "max": 4096, "step": 16, "tooltip": "Reference target size. Snapped to 16-pixel multiples."}),
                "mask": ("MASK", ),
            }
        }

    RETURN_TYPES = ("CONDITIONING", "LATENT", "ANY", "IMAGE", "MASK")
    RETURN_NAMES = ("conditioning", "latent", "custom_output", "main_image", "mask")
    FUNCTION = "encode"

    CATEGORY = "advanced/conditioning"

    def encode(self, clip, vae, prompt,
               image1=None, image2=None, image3=None,
               mask=None,
               ref_longest_edge=1024):
        # Prepare model config: default template is auto-built by the
        # QwenImage21 tokenizer (no custom system prompt).
        model_config = {
            "model_name": "qwen_image21",
            "instruction": "",
            "vae_unit": 16,
            "llama_template": ""
        }

        # Prepare configs list
        configs = []

        # Process each image if provided
        if image1 is not None:
            config1 = {
                "image": image1,
                "to_ref": True,
                "ref_main_image": True,  # First image is main by default
                "ref_longest_edge": ref_longest_edge,
                "ref_crop": "pad",
                "ref_upscale": "lanczos",
                "to_vl": True,
                "pad_mode": "color",
                "pad_color": "white",  # black pads get amplified during qwen image 2.1 generation
                "pad_noise": 0.0,
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
                "ref_crop": "pad",
                "ref_upscale": "lanczos",
                "to_vl": True,
                "pad_mode": "color",
                "pad_color": "white",  # black pads get amplified during qwen image 2.1 generation
                "pad_noise": 0.0,
            })

        if image3 is not None:
            configs.append({
                "image": image3,
                "to_ref": True,
                "ref_main_image": False,
                "ref_longest_edge": ref_longest_edge,
                "ref_crop": "pad",
                "ref_upscale": "lanczos",
                "to_vl": True,
                "pad_mode": "color",
                "pad_color": "white",  # black pads get amplified during qwen image 2.1 generation
                "pad_noise": 0.0,
            })

        if len(configs) == 0:
            raise ValueError("At least one image must be provided")

        # Call the original EditTextEncode function
        node_instance = EditTextEncode_EditUtils()
        # EditTextEncode also returns `pad_info` and `prompt`; these 5-output
        # wrappers deliberately expose only the first five, so slice rather
        # than forwarding the whole tuple (which relies on truncation).
        encoded = node_instance.encode(
            clip=clip,
            vae=vae,
            prompt=prompt,
            model_config=model_config,
            configs=configs
        )
        return encoded[:5]


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
        # EditTextEncode also returns `pad_info` and `prompt`; these 5-output
        # wrappers deliberately expose only the first five, so slice rather
        # than forwarding the whole tuple (which relies on truncation).
        encoded = node_instance.encode(
            clip=clip,
            vae=vae,
            prompt=prompt,
            model_config=model_config,
            configs=configs
        )
        return encoded[:5]


class Flux3KleinEditTextEncode_EditUtils:
    """One-node simple path for the FLUX.3 klein edit workflow (image1..image3).

    Wraps ``EditTextEncode_EditUtils`` with the flux3klein model config and the
    standard per-image config:
    - image1 is the main image (its latent is the sampling start latent and its
      pixels are returned as ``main_image``);
    - image2/image3 are plain references;
    - ``ref_crop="pad"`` so the target keeps its aspect ratio inside the 16-px
      aligned latent grid, with the pad recovered by ``CropWithPadInfo_EditUtils``.
    """

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
                "ref_longest_edge": ("INT", {"default": 1024, "min": 16, "max": 4096, "step": 1, "tooltip": "Longest edge of the main image latent grid"}),
                "ref_longest_edge2": ("INT", {"default": 1024, "min": 16, "max": 4096, "step": 1, "tooltip": "Longest edge of reference 2"}),
                "ref_longest_edge3": ("INT", {"default": 1024, "min": 16, "max": 4096, "step": 1, "tooltip": "Longest edge of reference 3"}),
                "mask": ("MASK", ),
            }
        }

    RETURN_TYPES = ("CONDITIONING", "LATENT", "ANY", "IMAGE", "MASK")
    RETURN_NAMES = ("conditioning", "latent", "custom_output", "main_image", "mask")
    FUNCTION = "encode"

    CATEGORY = "advanced/conditioning"

    def encode(self, clip, vae, prompt,
               image1=None, image2=None, image3=None,
               mask=None,
               ref_longest_edge=1024, ref_longest_edge2=1024, ref_longest_edge3=1024):
        # FLUX.3 klein: the FLUX.2 packed VAE (128ch @ 16x) and the flux3klein
        # routing key. The tokenizer builds its own instruct template and pads the
        # text window to a multiple of 80 tokens, so llama_template stays empty.
        model_config = {
            "model_name": "flux3klein",
            "instruction": "",
            "vae_unit": 16,
            "config_for": "flux3klein",
            "llama_template": "",
        }

        configs = []

        if image1 is not None:
            config1 = {
                "image": image1,
                "to_ref": True,
                "ref_main_image": True,
                "ref_longest_edge": ref_longest_edge,
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
                "ref_longest_edge": ref_longest_edge2,
                "ref_crop": "pad",
                "ref_upscale": "lanczos",
            })

        if image3 is not None:
            configs.append({
                "image": image3,
                "to_ref": True,
                "ref_main_image": False,
                "ref_longest_edge": ref_longest_edge3,
                "ref_crop": "pad",
                "ref_upscale": "lanczos",
            })

        if len(configs) == 0:
            raise ValueError("At least one image must be provided")

        node_instance = EditTextEncode_EditUtils()
        # EditTextEncode also returns `pad_info` and `prompt`; these 5-output
        # wrappers deliberately expose only the first five, so slice rather
        # than forwarding the whole tuple (which relies on truncation).
        encoded = node_instance.encode(
            clip=clip,
            vae=vae,
            prompt=prompt,
            model_config=model_config,
            configs=configs
        )
        return encoded[:5]


class MinimaxH3ImageEditTextEncode_EditUtils:
    """One-node simple path for the MiniMax-H3-Image edit workflow (image1..image3).

    Wraps ``EditTextEncode_EditUtils`` with the ``minimax_h3_image`` model config
    and the standard per-image configs:

    - image1 is the picture to edit (``ref_main_image``): its latent supplies the
      target grid and its pixels are returned as ``main_image``;
    - image2/image3 are plain references;
    - every image keeps ComfyUI's reference sizing (1024 px short edge, clamped
      to ``ref_max_short_edge``) and a matching 1024 px vision copy;
    - ``scale_up`` multiplies image1's reference grid into the sampled target
      (default 1.0). H3I wants 4-20 MP and 1 MP gives poster-like artifacts, so
      ``scale_up=2`` turns a 1 MP main reference into a 4 MP 1:1 target without
      upscaling the reference itself.

    Sampling contract (from the checkpoint, not from this node): **6 steps**,
    CFG **1.0**, any stock sampler, negative = a zeroed-out conditioning.
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "clip": ("CLIP", ),
                "vae": ("VAE", ),
                "prompt": ("STRING", {"multiline": True, "dynamicPrompts": True,
                                      "tooltip": "Describe the edited result; refer to the references as <Picture 1>, <Picture 2>, ..."}),
            },
            "optional": {
                "image1": ("IMAGE", ),
                "image2": ("IMAGE", ),
                "image3": ("IMAGE", ),
                "ref_short_edge": ("INT", {"default": 1024, "min": 32, "max": 4096, "step": 32, "tooltip": "Short edge of image1's reference latent and vision copy"}),
                "ref_short_edge2": ("INT", {"default": 1024, "min": 32, "max": 4096, "step": 32, "tooltip": "Short edge of reference 2"}),
                "ref_short_edge3": ("INT", {"default": 1024, "min": 32, "max": 4096, "step": 32, "tooltip": "Short edge of reference 3"}),
                "scale_up": ("FLOAT", {"default": 1.0, "min": 1.0, "max": 4.0, "step": 0.05, "tooltip": "Final latent size = image1's reference size x scale_up. 2.0 turns a ~1 MP main reference into a ~4 MP target."}),
                "mask": ("MASK", ),
            }
        }

    RETURN_TYPES = ("CONDITIONING", "LATENT", "ANY", "IMAGE", "MASK")
    RETURN_NAMES = ("conditioning", "latent", "custom_output", "main_image", "mask")
    FUNCTION = "encode"

    CATEGORY = "advanced/conditioning"

    def encode(self, clip, vae, prompt,
               image1=None, image2=None, image3=None,
               mask=None,
               ref_short_edge=MINIMAX_H3_IMAGE_REF_SHORT_EDGE,
               ref_short_edge2=MINIMAX_H3_IMAGE_REF_SHORT_EDGE,
               ref_short_edge3=MINIMAX_H3_IMAGE_REF_SHORT_EDGE,
               scale_up=1.0):
        # MiniMax-H3-Image: 24ch / 16 px VAE, Qwen3-VL-32B text encoder (CLIP
        # type `minimax`), references on `minimax_refs`. No template: the
        # tokenizer emits "Picture n: " + vision blocks itself.
        model_config = {
            "model_name": "minimax_h3_image",
            "vae_unit": MINIMAX_H3_IMAGE_UNIT,
            "llama_template": "",
            "config_for": "minimax_h3_image",
        }

        def image_config(image, short_edge, main):
            cfg = {
                "image": image,
                "to_ref": True,
                "ref_main_image": main,
                "ref_short_edge": short_edge,
                "ref_max_short_edge": max(short_edge, MINIMAX_H3_IMAGE_REF_MAX_SHORT_EDGE),
                "ref_crop": "disabled",
                "ref_upscale": "lanczos",
                "to_vl": True,
                "vl_crop": "disabled",
                "vl_upscale": "bicubic",
            }
            if main:
                # target grid = image1's reference size x scale_up
                cfg["scale_up"] = max(1.0, float(scale_up))
            return cfg

        configs = []
        if image1 is not None:
            config1 = image_config(image1, ref_short_edge, True)
            if mask is not None:
                config1["mask"] = mask
            configs.append(config1)
        if image2 is not None:
            configs.append(image_config(image2, ref_short_edge2, False))
        if image3 is not None:
            configs.append(image_config(image3, ref_short_edge3, False))

        node_instance = EditTextEncode_EditUtils()
        encoded = node_instance.encode(
            clip=clip,
            vae=vae,
            prompt=prompt,
            model_config=model_config,
            configs=configs
        )
        return encoded[:5]


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
        # Take the first six outputs by index rather than unpacking: EditTextEncode
        # also returns a `prompt` string at slot 7, and this node does not need it.
        encoded = node_instance.encode(
            clip=clip,
            vae=vae,
            prompt=prompt,
            model_config=model_config,
            configs=configs
        )
        positive, latent_out, custom_output, main_image, noise_mask, pad_info = encoded[:6]

        negative = custom_output.get("negative_cond")
        return (positive, negative, latent_out, custom_output, main_image)
