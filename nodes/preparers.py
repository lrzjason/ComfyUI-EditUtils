"""Per-image config nodes (the `configs` chain) plus ConfigJsonParser."""

import json
import copy

from ._shared import MINIMAX_H3_IMAGE_REF_MAX_SHORT_EDGE, MINIMAX_H3_IMAGE_REF_SHORT_EDGE, stamp_config_id

class ConfigJsonParser_EditUtils:
    # pad_mode / pad_color / pad_noise are intentionally NOT part of the default
    # template: this node is model-agnostic, so leaving them out lets the
    # EditTextEncode fallback apply (white for qwen / qwen_image21, black
    # otherwise). Add them to the JSON to override per image.
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
                "config_json": ("STRING", {"default": s.default_config_json, "multiline": True, "tooltip": "Config JSON String. Optional pad keys: pad_mode (color/edge/mirror/border_mean), pad_color (red/green/blue/white/black/custom), pad_color_rgb (6-digit hex, '#' optional, overrides pad_color when valid), pad_noise (0.0-0.2, padding only). Defaults follow the model: white for qwen / qwen_image21, black otherwise."}),
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
        
        config = ConfigJsonParser_EditUtils.default_config.copy()
        json_config = {}
        try:
            if config_json:
                json_config = json.loads(config_json)
        except Exception as e:
            print(f"An error occurred while loading json_config: {e}")
            print(config_json)
            json_config = {}

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
        
        # Stable id: lets the bbox node find this entry again after a chain
        # deepcopy instead of appending a duplicate of the same image.
        stamp_config_id(config)
        config_output.append(config)
        # print("len(configs)", len(configs))
        return (config_output, config, )


class Flux2KleinConfigPreparer_EditUtils:
    upscale_methods = ["lanczos", "bicubic", "area"]
    crop_methods = ["pad", "center", "disabled"]
    vl_crop_methods = ["center", "disabled"]
    pad_modes = ["color", "edge", "mirror", "border_mean"]
    pad_colors = ["red", "green", "blue", "white", "black", "custom"]
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
                "pad_mode": (s.pad_modes, {"default": "color", "tooltip": "How to fill the padding area when ref_crop == pad. color = solid pad_color, edge = replicate edge pixels, mirror = reflect content, border_mean = fill with the mean of the nearest edge pixels"}),
                "pad_color": (s.pad_colors, {"default": "black", "tooltip": "Pad fill color when pad_mode == color: red / green / blue / white / black / custom. Black matches the training data for flux2klein"}),
                "pad_color_rgb": ("STRING", {"default": "-", "tooltip": "Optional override: 6-digit hex color, with or without '#' (e.g. FF0000 or #FF0000). When valid it overrides pad_color. Anything else (\"-\", wrong length, non-hex, numbers, \"r,g,b\") is ignored"}),
                "pad_noise": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 0.2, "step": 0.01, "tooltip": "Gaussian jitter (std) added to the pad area to avoid perfectly flat regions; 0.02-0.05 recommended with solid colors"}),
                "mask": ("MASK", ),
                "ref_resize_mode": (s.resize_modes, {"default": "longest_edge", "tooltip": "longest_edge: scale so the longest dimension equals ref_longest_edge. area: scale so total pixels equals ref_longest_edge squared."}),
                "rope_x_offset": ("INT", {"default": 0, "min": 0, "max": 4096, "step": 8, "tooltip": "ROPE horizontal position offset in pixels (VAE-aligned). Shifts reference rightward on canvas. Requires model EditApply node."}),
                "rope_y_offset": ("INT", {"default": 0, "min": 0, "max": 4096, "step": 8, "tooltip": "ROPE vertical position offset in pixels (VAE-aligned). Shifts reference downward on canvas. Requires model EditApply node."}),
            }
        }

    RETURN_TYPES = ("LIST", "ANY", )
    RETURN_NAMES = ("configs", "config", )
    FUNCTION = "prepare_config"

    CATEGORY = "advanced/conditioning"
    def prepare_config(self, image, configs=None,
                to_ref=True, ref_main_image=True, ref_longest_edge=1024, ref_crop="center", ref_upscale="lanczos",
                pad_mode="color", pad_color="black", pad_color_rgb="-", pad_noise=0.0,
                
                mask=None, ref_resize_mode="longest_edge", rope_x_offset=0, rope_y_offset=0
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

            "pad_mode": pad_mode,
            "pad_color": pad_color,
            "pad_color_rgb": pad_color_rgb,
            "pad_noise": pad_noise,
            "ref_resize_mode": ref_resize_mode,
            "rope_x_offset": rope_x_offset,
            "rope_y_offset": rope_y_offset,
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


class Flux3KleinConfigPreparer_EditUtils:
    """Per-image config for the FLUX.3 klein edit path.

    Differences from ``Flux2KleinConfigPreparer_EditUtils`` that matter here:

    - no ``to_vl``: FLUX.3's text encoder is text-only and references are pure
      VAE latents on their own token mode, so there is no vision tower input;
    - references keep their own resolution (each ref gets its own token grid and
      its own negative ``l`` position id), so ``ref_longest_edge`` sizes that one
      reference rather than forcing a shared grid;
    - ``ref_crop == "pad"`` pads to a multiple of 16 px (the packed FLUX.2 VAE
      token size). The reference is only allowed to *grow* to that grid, never
      to be cropped below its content;
    - ``rope_x_offset / rope_y_offset`` are accepted for graph compatibility but
      have no effect: ComfyUI's Flux3 model derives reference position ids
      internally (``ref_ids``), and the EditUtils rope-offset patch is a FLUX.2
      MMDiT forward patch that does not apply to the FLUX.3 architecture.
    """

    upscale_methods = ["lanczos", "bicubic", "area"]
    crop_methods = ["pad", "center", "disabled"]
    pad_modes = ["color", "edge", "mirror", "border_mean"]
    pad_colors = ["red", "green", "blue", "white", "black", "custom"]
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
                "ref_longest_edge": ("INT", {"default": 1024, "min": 16, "max": 4096, "step": 1, "tooltip": "Longest edge of this reference's latent grid"}),
                "ref_crop": (s.crop_methods, {"default": "pad", "tooltip": "Crop method for reference image"}),
                "ref_upscale": (s.upscale_methods, {"default": "lanczos", "tooltip": "Upscale method for reference image"}),
                "pad_mode": (s.pad_modes, {"default": "color", "tooltip": "How to fill the padding area when ref_crop == pad. color = solid pad_color, edge = replicate edge pixels, mirror = reflect content, border_mean = fill with the mean of the nearest edge pixels"}),
                "pad_color": (s.pad_colors, {"default": "black", "tooltip": "Pad fill color when pad_mode == color: red / green / blue / white / black / custom"}),
                "pad_color_rgb": ("STRING", {"default": "-", "tooltip": "Optional override: 6-digit hex color, with or without '#' (e.g. FF0000 or #FF0000). When valid it overrides pad_color. Anything else (\"-\", wrong length, non-hex, numbers, \"r,g,b\") is ignored"}),
                "pad_noise": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 0.2, "step": 0.01, "tooltip": "Gaussian jitter (std) added to the pad area to avoid perfectly flat regions; 0.02-0.05 recommended with solid colors"}),
                "mask": ("MASK", ),
                "ref_resize_mode": (s.resize_modes, {"default": "longest_edge", "tooltip": "longest_edge: scale so the longest dimension equals ref_longest_edge. area: scale so total pixels equals ref_longest_edge squared."}),
                "rope_x_offset": ("INT", {"default": 0, "min": 0, "max": 4096, "step": 16, "tooltip": "Accepted for graph compatibility. FLUX.3 derives reference position ids internally, so this has no effect."}),
                "rope_y_offset": ("INT", {"default": 0, "min": 0, "max": 4096, "step": 16, "tooltip": "Accepted for graph compatibility. FLUX.3 derives reference position ids internally, so this has no effect."}),
            }
        }

    # The FLUX.3 bounding-box element table lives on its own node
    # (Flux3KleinBBox_EditUtils) so this preparer stays short; feed that node
    # this node's `config` output to attach a layout table to one image.
    RETURN_TYPES = ("LIST", "ANY", )
    RETURN_NAMES = ("configs", "config", )
    FUNCTION = "prepare_config"

    CATEGORY = "advanced/conditioning"

    def prepare_config(self, image, configs=None,
                to_ref=True, ref_main_image=True, ref_longest_edge=1024, ref_crop="pad",
                ref_upscale="lanczos",
                pad_mode="color", pad_color="black", pad_color_rgb="-", pad_noise=0.0,
                mask=None, ref_resize_mode="longest_edge", rope_x_offset=0, rope_y_offset=0
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

            "pad_mode": pad_mode,
            "pad_color": pad_color,
            "pad_color_rgb": pad_color_rgb,
            "pad_noise": pad_noise,
            "ref_resize_mode": ref_resize_mode,
            "rope_x_offset": rope_x_offset,
            "rope_y_offset": rope_y_offset,
        }
        # Stable id: the bbox node finds this entry again after a chain deepcopy.
        stamp_config_id(config)

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
                "rope_x_offset": ("INT", {"default": 0, "min": 0, "max": 4096, "step": 8, "tooltip": "ROPE horizontal position offset in pixels (VAE-aligned). Shifts reference rightward on canvas. Requires model EditApply node."}),
                "rope_y_offset": ("INT", {"default": 0, "min": 0, "max": 4096, "step": 8, "tooltip": "ROPE vertical position offset in pixels (VAE-aligned). Shifts reference downward on canvas. Requires model EditApply node."}),
            }
        }

    RETURN_TYPES = ("LIST", "ANY", )
    RETURN_NAMES = ("configs", "config", )
    FUNCTION = "prepare_config"

    CATEGORY = "advanced/conditioning"

    def prepare_config(self, image, configs=None,
                to_ref=True, ref_main_image=True, ref_longest_edge=1024, ref_crop="pad", ref_upscale="lanczos",
                to_vl=True, vl_target_size=384, vl_crop="center", vl_upscale="lanczos",
                mask=None, ref_resize_mode="area", rope_x_offset=0, rope_y_offset=0
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
            "rope_x_offset": rope_x_offset,
            "rope_y_offset": rope_y_offset,
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


class QwenConfigPreparer_EditUtils:
    upscale_methods = ["lanczos", "bicubic", "area"]
    crop_methods = ["pad", "center", "disabled"]
    vl_crop_methods = ["center", "disabled"]
    pad_modes = ["color", "edge", "mirror", "border_mean"]
    pad_colors = ["red", "green", "blue", "white", "black", "custom"]
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
                "pad_mode": (s.pad_modes, {"default": "color", "tooltip": "How to fill the padding area when ref_crop == pad. color = solid pad_color, edge = replicate edge pixels, mirror = reflect content, border_mean = fill with the mean of the nearest edge pixels"}),
                "pad_color": (s.pad_colors, {"default": "white", "tooltip": "Pad fill color when pad_mode == color: red / green / blue / white / black / custom. Default white for qwen since black pads get amplified during generation"}),
                "pad_color_rgb": ("STRING", {"default": "-", "tooltip": "Optional override: 6-digit hex color, with or without '#' (e.g. FF0000 or #FF0000). When valid it overrides pad_color. Anything else (\"-\", wrong length, non-hex, numbers, \"r,g,b\") is ignored"}),
                "pad_noise": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 0.2, "step": 0.01, "tooltip": "Gaussian jitter (std) added to the pad area to avoid perfectly flat regions; 0.02-0.05 recommended with solid colors"}),
                "mask": ("MASK", ),
                "ref_resize_mode": (s.resize_modes, {"default": "longest_edge", "tooltip": "longest_edge: scale so the longest dimension equals ref_longest_edge. area: scale so total pixels equals ref_longest_edge squared."}),
                "rope_x_offset": ("INT", {"default": 0, "min": 0, "max": 4096, "step": 8, "tooltip": "ROPE horizontal position offset in pixels (VAE-aligned, step=8). Shifts reference position IDs rightward on the canvas. Only effective with Krea2 when ref_pos_match_target=False."}),
                "rope_y_offset": ("INT", {"default": 0, "min": 0, "max": 4096, "step": 8, "tooltip": "ROPE vertical position offset in pixels (VAE-aligned, step=8). Shifts reference position IDs downward on the canvas. Only effective with Krea2 when ref_pos_match_target=False."}),
            }
        }

    RETURN_TYPES = ("LIST", "ANY", )
    RETURN_NAMES = ("configs", "config", )
    FUNCTION = "prepare_config"

    CATEGORY = "advanced/conditioning"
    def prepare_config(self, image, configs=None,
                to_ref=True, ref_main_image=True, ref_longest_edge=1024, ref_crop="center", ref_upscale="lanczos",
                to_vl=True, vl_resize=True, vl_target_size=384, vl_crop="center", vl_upscale="bicubic",
                pad_mode="color", pad_color="white", pad_color_rgb="-", pad_noise=0.0,
                
                mask=None, ref_resize_mode="longest_edge", rope_x_offset=0, rope_y_offset=0
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
            "vl_upscale": vl_upscale,

            "pad_mode": pad_mode,
            "pad_color": pad_color,
            "pad_color_rgb": pad_color_rgb,
            "pad_noise": pad_noise,
            
            "rope_x_offset": rope_x_offset,
            "rope_y_offset": rope_y_offset,
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


class MossImageConfigPreparer_EditUtils:
    """Per-image config for the MOSS-Image reference-image path.

    MOSS-Image combines two worlds:

    - its text encoder is a vision-capable **Qwen3.5-9B**, so references keep
      the Qwen-style VL half (``Picture n: <|vision_start|>…`` grounding);
    - its VAE is the **FLUX.2 packed autoencoder** (128 channels, 16 px per
      token), so references snap to 16 px and use the FLUX.2 sizing defaults.

    Defaults are chosen for the edit recipe: a plain aspect-preserving fit
    (``center``, no fabricated padding) at ~1 MP, with black padding available
    if a fixed canvas is wanted.  Unlike the Krea2/Qwen regional-edit paths,
    MOSS has no per-reference rope-offset consumer, so no ``rope_x/y_offset``
    widgets are exposed.
    """

    upscale_methods = ["lanczos", "bicubic", "area"]
    crop_methods = ["pad", "center", "disabled"]
    vl_crop_methods = ["center", "disabled"]
    pad_modes = ["color", "edge", "mirror", "border_mean"]
    pad_colors = ["red", "green", "blue", "white", "black", "custom"]
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
                "ref_longest_edge": ("INT", {"default": 1024, "min": 16, "max": 4096, "step": 16, "tooltip": "Reference target size; snapped to 16-pixel multiples (FLUX.2 VAE, 128 channels). ~1 MP is the trained range."}),
                "ref_crop": (s.crop_methods, {"default": "center", "tooltip": "How the reference fits a non-matching aspect ratio. center = aspect-preserving fit with a minimal center crop (default, closest to the edit-training fit); pad = fill to a /16 canvas; disabled = resize without cropping."}),
                "ref_upscale": (s.upscale_methods, {"default": "lanczos", "tooltip": "Resize filter for the reference image"}),
                "to_vl": ("BOOLEAN", {"default": True, "tooltip": "Feed the reference to the Qwen3.5-9B vision tower (Picture n: vision grounding). Keep on for the trained edit recipe."}),
                "vl_resize": ("BOOLEAN", {"default": True, "tooltip": "Resize before the vision encode"}),
                "vl_target_size": ("INT", {"default": 384, "min": 384, "max": 2048, "tooltip": "Target area for the vision encode (vl_target_size squared)"}),
                "vl_crop": (s.vl_crop_methods, {"default": "center", "tooltip": "Crop method for the vision encode"}),
                "vl_upscale": (s.upscale_methods, {"default": "bicubic", "tooltip": "Resize filter for the vision encode; bicubic/area are better than lanczos when downscaling"}),
                "pad_mode": (s.pad_modes, {"default": "color", "tooltip": "How to fill the padding area when ref_crop == pad"}),
                "pad_color": (s.pad_colors, {"default": "black", "tooltip": "Pad fill color when pad_mode == color; black is the safe FLUX.2/MOSS default"}),
                "pad_color_rgb": ("STRING", {"default": "-", "tooltip": "Optional override: 6-digit hex color, with or without '#' (e.g. FF0000 or #FF0000). When valid it overrides pad_color. Anything else (\"-\", wrong length, non-hex, numbers, \"r,g,b\") is ignored"}),
                "pad_noise": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 0.2, "step": 0.01, "tooltip": "Gaussian jitter (std) added to the pad area to avoid perfectly flat regions; 0.02-0.05 recommended with solid colors"}),
                "mask": ("MASK", ),
                "ref_resize_mode": (s.resize_modes, {"default": "longest_edge", "tooltip": "longest_edge: scale so the longest dimension equals ref_longest_edge. area: scale so total pixels equals ref_longest_edge squared."}),
            }
        }

    RETURN_TYPES = ("LIST", "ANY", )
    RETURN_NAMES = ("configs", "config", )
    FUNCTION = "prepare_config"

    CATEGORY = "advanced/conditioning"

    def prepare_config(self, image, configs=None,
                       to_ref=True, ref_main_image=True,
                       ref_longest_edge=1024, ref_crop="center", ref_upscale="lanczos",
                       to_vl=True, vl_resize=True, vl_target_size=384,
                       vl_crop="center", vl_upscale="bicubic",
                       pad_mode="color", pad_color="black", pad_color_rgb="-",
                       pad_noise=0.0, mask=None, ref_resize_mode="longest_edge"):
        if configs is None:
            configs = []
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
            "vl_upscale": vl_upscale,

            "pad_mode": pad_mode,
            "pad_color": pad_color,
            "pad_color_rgb": pad_color_rgb,
            "pad_noise": pad_noise,
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


class QwenImage21ConfigPreparer_EditUtils:
    # Per-image config for Qwen-Image 2.1. There is no separate vl pipeline:
    # the vision-tower input is the resized reference itself, so every image
    # is aligned to 16-pixel multiples (the 16x VAE downscale).
    upscale_methods = ["lanczos", "bicubic", "area"]
    crop_methods = ["pad", "center", "disabled"]
    resize_modes = ["longest_edge", "area"]
    pad_modes = ["color", "edge", "mirror", "border_mean"]
    pad_colors = ["red", "green", "blue", "white", "black", "custom"]

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
                "ref_longest_edge": ("INT", {"default": 1024, "min": 16, "max": 4096, "step": 16, "tooltip": "Reference target size. Snapped to 16-pixel multiples (16x VAE downscale)."}),
                "ref_crop": (s.crop_methods, {"default": "pad", "tooltip": "Crop method for reference image"}),
                "ref_upscale": (s.upscale_methods, {"default": "lanczos", "tooltip": "Upscale method for reference image"}),
                "to_vl": ("BOOLEAN", {"default": True, "tooltip": "Feed the resized reference to the Qwen3-VL text encoder and splice its latent at the vision slot. Keep enabled for trained behavior."}),
                "mask": ("MASK", ),
                "ref_resize_mode": (s.resize_modes, {"default": "longest_edge", "tooltip": "longest_edge: scale so the longest dimension equals ref_longest_edge. area: scale so total pixels equals ref_longest_edge squared."}),
                "rope_x_offset": ("INT", {"default": 0, "min": 0, "max": 4096, "step": 16, "tooltip": "ROPE horizontal position offset in pixels (16-aligned, matching the VAE downscale). Shifts reference rightward on the canvas. Requires QwenImage21EditApply_EditUtils."}),
                "rope_y_offset": ("INT", {"default": 0, "min": 0, "max": 4096, "step": 16, "tooltip": "ROPE vertical position offset in pixels (16-aligned, matching the VAE downscale). Shifts reference downward on the canvas. Requires QwenImage21EditApply_EditUtils."}),
                "pad_mode": (s.pad_modes, {"default": "color", "tooltip": "How to fill the padding area when ref_crop == pad. color = solid pad_color, edge = replicate edge pixels, mirror = reflect content, border_mean = fill with the mean of the nearest edge pixels"}),
                "pad_color": (s.pad_colors, {"default": "white", "tooltip": "Pad fill color when pad_mode == color: red / green / blue / white / black / custom. Default white for qwen image 2.1 since black pads get amplified during generation"}),
                "pad_color_rgb": ("STRING", {"default": "-", "tooltip": "Optional override: 6-digit hex color, with or without '#' (e.g. FF0000 or #FF0000). When valid it overrides pad_color. Anything else (\"-\", wrong length, non-hex, numbers, \"r,g,b\") is ignored"}),
                "pad_noise": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 0.2, "step": 0.01, "tooltip": "Gaussian jitter (std) added to the pad area to avoid perfectly flat regions; 0.02-0.05 recommended with solid colors"}),
            }
        }

    RETURN_TYPES = ("LIST", "ANY", )
    RETURN_NAMES = ("configs", "config", )
    FUNCTION = "prepare_config"

    CATEGORY = "advanced/conditioning"
    def prepare_config(self, image, configs=None,
                to_ref=True, ref_main_image=True, ref_longest_edge=1024, ref_crop="pad", ref_upscale="lanczos",
                to_vl=True, mask=None, ref_resize_mode="longest_edge", rope_x_offset=0, rope_y_offset=0,
                pad_mode="color", pad_color="white", pad_color_rgb="-", pad_noise=0.0
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
            "ref_resize_mode": ref_resize_mode,
            "to_vl": to_vl,
            "rope_x_offset": rope_x_offset,
            "rope_y_offset": rope_y_offset,
            "pad_mode": pad_mode,
            "pad_color": pad_color,
            "pad_color_rgb": pad_color_rgb,
            "pad_noise": pad_noise,
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


class MinimaxH3ImageConfigPreparer_EditUtils:
    """Per-image config for the **MiniMax-H3-Image** edit path (image model).

    One H3I reference feeds two consumers with different needs, so this node
    carries both halves of the recipe:

    - **vision copy** — Qwen3-VL sees the reference at a 1024 px short edge
      (ComfyUI: ``IMAGE_REF_SHORT_EDGE``), so the tokenizer can emit
      ``"Picture n: " + <vision block>`` before the prompt;
    - **VAE reference latent** — the DiT sees 24-channel latents on their own
      grid, with the short edge clamped into
      ``[ref_short_edge, ref_max_short_edge]`` (ComfyUI:
      1024..2048) and both sides snapped to 32 px.

    The main image (``ref_main_image=True``, first one wins) is the picture to
    edit: its reference latent supplies the target grid, multiplied by
    ``scale_up``. That keeps the two sizes independent without a second size
    API — references stay at ~1 MP (where H3I likes them) while the sampled
    latent grows to the 4-20 MP range H3I was trained for, e.g.
    ``ref_short_edge=1024`` + ``scale_up=2`` -> a 4 MP square target.

    ``ref_crop="disabled"`` (the default) is ComfyUI's behaviour: pure
    aspect-preserving resize, no crop and no fabricated padding. ``center`` and
    ``pad`` remain available for fixed-canvas graphs; the pad path snaps to
    ``vae_unit`` (32 px) and uses the same pad_mode / pad_color / pad_noise
    controls as the other preparers.
    """

    upscale_methods = ["lanczos", "bicubic", "area"]
    crop_methods = ["disabled", "center", "pad"]
    vl_crop_methods = ["disabled", "center"]
    pad_modes = ["color", "edge", "mirror", "border_mean"]
    pad_colors = ["red", "green", "blue", "white", "black", "custom"]

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
                "to_ref": ("BOOLEAN", {"default": True, "tooltip": "Encode the image as a reference latent (minimax_refs). Keep on: H3I edits are driven by these latents."}),
                "ref_main_image": ("BOOLEAN", {"default": True, "tooltip": "This image is the picture to edit: its latent supplies the target grid. Only the first one in the chain wins."}),
                "ref_short_edge": ("INT", {"default": 1024, "min": 32, "max": 4096, "step": 32, "tooltip": "Short edge of the reference latent AND of the vision copy. ComfyUI uses 1024 for references."}),
                "ref_max_short_edge": ("INT", {"default": 2048, "min": 32, "max": 8192, "step": 32, "tooltip": "Upper clamp for the reference short edge (ComfyUI: 2048). Bigger references are only downscaled to this."}),
                "scale_up": ("FLOAT", {"default": 1.0, "min": 1.0, "max": 4.0, "step": 0.05, "tooltip": "Main image only: final latent size = this image's reference size x scale_up. H3I needs 4-20 MP, references are happiest at ~1 MP, so 2.0 turns a 1 MP main reference into a ~4 MP target without upscaling the reference itself."}),
                "ref_crop": (s.crop_methods, {"default": "disabled", "tooltip": "disabled = ComfyUI's reference behaviour (aspect-preserving resize, no crop). center = minimal center crop; pad = fill to a 32 px canvas."}),
                "ref_upscale": (s.upscale_methods, {"default": "lanczos", "tooltip": "Resize filter for the reference image"}),
                "to_vl": ("BOOLEAN", {"default": True, "tooltip": "Give the reference to the Qwen3-VL text encoder as \"Picture n: \" + vision block. Keep on for the trained edit recipe."}),
                "vl_crop": (s.vl_crop_methods, {"default": "disabled", "tooltip": "Crop mode for the vision copy (disabled matches ComfyUI)"}),
                "vl_upscale": (s.upscale_methods, {"default": "bicubic", "tooltip": "Resize filter for the vision copy; bicubic/area beat lanczos when downscaling"}),
                "pad_mode": (s.pad_modes, {"default": "color", "tooltip": "How to fill the padding area when ref_crop == pad"}),
                "pad_color": (s.pad_colors, {"default": "black", "tooltip": "Pad fill color when pad_mode == color; black matches the H3I reference padding"}),
                "pad_color_rgb": ("STRING", {"default": "-", "tooltip": "Optional override: 6-digit hex color, with or without '#' (e.g. FF0000). When valid it overrides pad_color; anything else is ignored"}),
                "pad_noise": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 0.2, "step": 0.01, "tooltip": "Gaussian jitter (std) added to the pad area only; 0.02-0.05 recommended with solid colors"}),
                "mask": ("MASK", ),
            }
        }

    RETURN_TYPES = ("LIST", "ANY", )
    RETURN_NAMES = ("configs", "config", )
    FUNCTION = "prepare_config"

    CATEGORY = "advanced/conditioning"

    def prepare_config(self, image, configs=None,
                       to_ref=True, ref_main_image=True,
                       ref_short_edge=MINIMAX_H3_IMAGE_REF_SHORT_EDGE,
                       ref_max_short_edge=MINIMAX_H3_IMAGE_REF_MAX_SHORT_EDGE,
                       scale_up=1.0,
                       ref_crop="disabled", ref_upscale="lanczos",
                       to_vl=True, vl_crop="disabled", vl_upscale="bicubic",
                       pad_mode="color", pad_color="black", pad_color_rgb="-",
                       pad_noise=0.0, mask=None):
        if configs is None:
            configs = []
        if ref_max_short_edge < ref_short_edge:
            print(f"MinimaxH3Image: ref_max_short_edge ({ref_max_short_edge}) < ref_short_edge "
                  f"({ref_short_edge}); using {ref_short_edge}")
            ref_max_short_edge = ref_short_edge
        config = {
            "image": image,
            "to_ref": to_ref,
            "ref_main_image": ref_main_image,
            "ref_short_edge": ref_short_edge,
            "ref_max_short_edge": ref_max_short_edge,
            # target grid = this image's reference size x scale_up (main image only)
            "scale_up": max(1.0, float(scale_up)),
            "ref_crop": ref_crop,
            "ref_upscale": ref_upscale,

            "to_vl": to_vl,
            "vl_crop": vl_crop,
            "vl_upscale": vl_upscale,

            "pad_mode": pad_mode,
            "pad_color": pad_color,
            "pad_color_rgb": pad_color_rgb,
            "pad_noise": pad_noise,
        }
        # Stable id so a chained node can find this entry again.
        stamp_config_id(config)

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
