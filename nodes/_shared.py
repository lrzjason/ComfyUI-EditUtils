"""Shared imports, small helpers and the constants used by several node families."""

import math
import torch
import torch.nn.functional as F
import numpy as np
import uuid
from PIL import Image, ImageOps, ImageSequence

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


def stamp_config_id(config):
    """Give a config a stable id so it can be found again after a deepcopy.

    ``Flux3KleinConfigPreparer_EditUtils`` (and the other preparers) run
    ``copy.deepcopy`` on the incoming chain, so a downstream preparer holds
    *copies* of the upstream configs. Object identity therefore cannot be used to
    answer "which entry in this chain is this image?" - and getting that wrong
    makes the bbox node append a second copy of the image, which would send the
    same reference twice. The id is a plain string, so it survives the copy.
    """
    if "config_id" not in config:
        config["config_id"] = uuid.uuid4().hex
    return config


def parse_pad_color(pad_color):
    """Parse a named pad color into an (r, g, b) float tuple in [0, 1].

    Accepts the combo names ("red", "green", "blue", "white", "black") plus the
    legacy/internal names ("gray"/"grey"/"mid_gray"/"mid_grey"), "#RRGGBB" hex
    strings and "r,g,b" values so configs coming from ConfigJsonParser keep
    working. Falls back to black (0, 0, 0) for anything unrecognized.
    """
    named_colors = {
        "red": (1.0, 0.0, 0.0),
        "green": (0.0, 1.0, 0.0),
        "blue": (0.0, 0.0, 1.0),
        "black": (0.0, 0.0, 0.0),
        "white": (1.0, 1.0, 1.0),
        "gray": (0.5, 0.5, 0.5),
        "grey": (0.5, 0.5, 0.5),
        "mid_gray": (0.5, 0.5, 0.5),
        "mid_grey": (0.5, 0.5, 0.5),
    }
    fallback = named_colors["black"]

    if isinstance(pad_color, (list, tuple)) and len(pad_color) == 3:
        try:
            values = [float(v) for v in pad_color]
            if any(v > 1.0 for v in values):
                # treat as 0-255 range
                values = [v / 255.0 for v in values]
            return tuple(min(max(v, 0.0), 1.0) for v in values)
        except (TypeError, ValueError):
            print(f"[EditUtils] Invalid pad_color '{pad_color}', fallback to black")
            return fallback

    if isinstance(pad_color, str):
        name = pad_color.strip().lower()
        if name in named_colors:
            return named_colors[name]
        # support "r,g,b" strings as well
        if name.count(",") == 2:
            try:
                values = [float(v) for v in name.split(",")]
                if any(v > 1.0 for v in values):
                    values = [v / 255.0 for v in values]
                return tuple(min(max(v, 0.0), 1.0) for v in values)
            except ValueError:
                pass
        if name.startswith("#") and len(name) == 7:
            try:
                r = int(name[1:3], 16) / 255.0
                g = int(name[3:5], 16) / 255.0
                b = int(name[5:7], 16) / 255.0
                return (r, g, b)
            except ValueError:
                pass
    print(f"[EditUtils] Unknown pad_color '{pad_color}', fallback to black")
    return fallback


def parse_hex_color(value):
    """Parse a pad_color_rgb value into an (r, g, b) float tuple.

    Accepts a 6-digit hexadecimal color with OR without a leading '#'.
    Anything else ("-", "", "FF00", "FF000", "1234567", "GGGGGG", "255",
    "r,g,b", numbers) returns None, which means "not set / no effect".
    """
    if not isinstance(value, str):
        return None
    text = value.strip()
    if text.startswith("#"):
        text = text[1:]
    if len(text) != 6:
        return None
    if any(ch not in "0123456789abcdefABCDEF" for ch in text):
        return None
    return tuple(int(text[i:i + 2], 16) / 255.0 for i in (0, 2, 4))


def resolve_pad_color(pad_color, pad_color_rgb=None, fallback="black"):
    """Resolve the solid pad fill color to an (r, g, b) tuple.

    pad_color_rgb wins whenever it holds a valid 6-digit hex (with or without
    '#'), regardless of the pad_color selection. Otherwise pad_color decides:
    "red"/"green"/"blue"/"white"/"black" are named colors, and "custom" (which
    carries no color of its own) falls back to `fallback`, i.e. the model's
    default pad color - so an invalid hex simply has no effect.
    """
    color = parse_hex_color(pad_color_rgb)
    if color is not None:
        return color

    name = pad_color.strip().lower() if isinstance(pad_color, str) else None
    if name is None or name == "custom":
        print(f"[EditUtils] pad_color='{pad_color}' without a valid pad_color_rgb "
              f"hex ('{pad_color_rgb}'), using the default '{fallback}'")
        name = fallback
    return parse_pad_color(name)


def build_pad_canvas(resized_samples, canvas_height, canvas_width,
                     pad_mode="color", pad_color="black", pad_noise=0.0,
                     pad_color_rgb="-", pad_color_fallback="black"):
    """Place resized_samples at the top-left of a canvas and fill the
    remaining right/bottom padding.

    pad_mode:
        - "color": solid color fill (pad_color, overridden by a valid pad_color_rgb)
        - "edge": replicate the outermost pixel row/column
        - "mirror": reflect the content at the boundary
        - "border_mean": fill each side with the mean of its nearest pixels
    pad_color: "red" / "green" / "blue" / "white" / "black" / "custom".
    pad_color_rgb: 6-digit hex, with or without '#'; when valid it overrides pad_color.
    pad_color_fallback: color used by "custom" when pad_color_rgb is not valid.
    pad_noise: gaussian jitter (std) added to the padding area ONLY (the image
               content is never touched) to break up perfectly flat regions;
               0 disables it. 0.02-0.05 works well.
    """
    b, c, h, w = resized_samples.shape
    pad_h = max(0, canvas_height - h)
    pad_w = max(0, canvas_width - w)

    if pad_mode == "edge":
        canvas = F.pad(resized_samples, (0, pad_w, 0, pad_h), mode="replicate")
    elif pad_mode == "mirror":
        # reflect needs pad < dim; fall back to replicate otherwise
        reflect_ok = (pad_w == 0 or pad_w < w) and (pad_h == 0 or pad_h < h)
        mode = "reflect" if reflect_ok else "replicate"
        canvas = F.pad(resized_samples, (0, pad_w, 0, pad_h), mode=mode)
    elif pad_mode == "border_mean":
        k = max(1, min(8, w, h))
        # right side: mean of the nearest k columns
        right_fill = resized_samples[:, :, :, w - k:].mean(dim=3, keepdim=True)
        top_row = resized_samples if pad_w == 0 else torch.cat(
            [resized_samples, right_fill.expand(b, c, h, pad_w)], dim=3)
        # bottom side: mean of the nearest k rows, extended over the full canvas width
        bottom_fill = resized_samples[:, :, h - k:, :w].mean(dim=2, keepdim=True)
        corner_fill = resized_samples[:, :, h - k:, w - k:].mean(dim=(2, 3)).view(b, c, 1, 1)
        bottom_row = bottom_fill.expand(b, c, 1, w) if pad_w == 0 else torch.cat(
            [bottom_fill.expand(b, c, 1, w), corner_fill.expand(b, c, 1, pad_w)], dim=3)
        canvas = top_row if pad_h == 0 else torch.cat(
            [top_row, bottom_row.expand(b, c, max(pad_h, 1), canvas_width)], dim=2)
    else:
        # solid color fill: a valid pad_color_rgb hex overrides pad_color
        color = resolve_pad_color(pad_color, pad_color_rgb, fallback=pad_color_fallback)
        canvas = torch.zeros(
            (b, c, canvas_height, canvas_width),
            dtype=resized_samples.dtype,
            device=resized_samples.device
        )
        for i in range(min(c, 3)):
            canvas[:, i] = color[i]
        canvas[:, :, :h, :w] = resized_samples

    if pad_noise and float(pad_noise) > 0 and (pad_h > 0 or pad_w > 0):
        # jitter ONLY the padding, never the image content
        pad_region = torch.ones(
            (1, 1, canvas_height, canvas_width),
            dtype=canvas.dtype, device=canvas.device
        )
        pad_region[:, :, :h, :w] = 0.0
        canvas = (canvas + torch.randn_like(canvas) * float(pad_noise) * pad_region).clamp(0.0, 1.0)

    return canvas


# ---------------------------------------------------------------------------
# MiniMax-H3-Image sizing contract
# ---------------------------------------------------------------------------
# MiniMax-H3-Image (ComfyUI ``image_model="minimax_h3_image"``) is the IMAGE
# model of the MiniMax H3 family, not the H3 video/audio model. Its VAE is a
# 16 px / 24-channel autoencoder, but the DiT patches the latent 2x2, so a
# representable canvas side is a multiple of 32 px (two VAE tokens). ComfyUI's
# own node (comfy_extras/nodes_minimax_h3.py ``_size_short_edge``) sizes edit
# references the same way: the SHORT edge is clamped into
# [IMAGE_REF_SHORT_EDGE, IMAGE_REF_VAE_MAX_SHORT_EDGE] and both sides snap to
# 32 px.
MINIMAX_H3_IMAGE_UNIT = 32


MINIMAX_H3_IMAGE_REF_SHORT_EDGE = 1024


MINIMAX_H3_IMAGE_REF_MAX_SHORT_EDGE = 2048


MINIMAX_H3_IMAGE_DEFAULT_SIZE = 2048


# latent channels of the H3 image VAE (24 ch, one frame)
MINIMAX_H3_IMAGE_LATENT_CHANNELS = 24


def minimax_h3_image_ref_size(width, height, short_edge=MINIMAX_H3_IMAGE_REF_SHORT_EDGE,
                              max_short_edge=MINIMAX_H3_IMAGE_REF_MAX_SHORT_EDGE,
                              unit=MINIMAX_H3_IMAGE_UNIT):
    """Aspect-preserving reference size, ComfyUI's MiniMax-H3-Image rule.

    The short edge of the source is clamped into [short_edge, max_short_edge]
    and both sides are rounded to ``unit`` px (32), so the VAE latent grid is
    even and the model's 2x2 patch covers it exactly.

    Returns ``(scaled_width, scaled_height)``; ``(0, 0)`` for an empty image.
    """
    width, height = int(width), int(height)
    if width <= 0 or height <= 0:
        return (0, 0)
    unit = max(2, int(unit))
    low = max(unit, int(short_edge))
    high = max(low, int(max_short_edge))
    short = min(width, height)
    target = min(max(short, low), high)
    scale = target / float(short)
    scaled_width = max(unit, int(round(width * scale / unit)) * unit)
    scaled_height = max(unit, int(round(height * scale / unit)) * unit)
    return (scaled_width, scaled_height)
