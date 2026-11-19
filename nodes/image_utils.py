"""Image/latent helpers: crop with pad info, longest-edge processing, masks, alpha filter."""

import node_helpers
import comfy.utils
import math
import torch
import torch.nn.functional as F
import numpy as np
import os
import folder_paths
import hashlib
import comfy.conds
from PIL import Image, ImageOps, ImageSequence
try:
    from scipy import ndimage as _scipy_ndimage
except ImportError:
    # scipy is a ComfyUI dependency; the alpha island filter and the inward
    # feather fall back to torch-only implementations when it is unavailable.
    _scipy_ndimage = None

from ._shared import build_pad_canvas

try:
    from scipy import ndimage as _scipy_ndimage
except ImportError:
    # scipy is a ComfyUI dependency; the alpha island filter and the inward
    # feather fall back to torch-only implementations when it is unavailable.
    _scipy_ndimage = None


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
        # `reference_latents` covers the Qwen / FLUX / MOSS / Krea2 routes;
        # `minimax_refs` is where MiniMax-H3-Image keeps its 24ch reference
        # latents, so clear both (an unknown key is simply ignored by the models
        # that do not read it).
        cleared = node_helpers.conditioning_set_values(
            conditioning, {"reference_latents": [], "minimax_refs": []}, append=False)
        return (cleared, )


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
            freq_diff = DiffMask_EditUtils._frequency_highpass_diff(gray1, gray2, highpass_sigma)
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

    @staticmethod
    def _frequency_highpass_diff(img1, img2, sigma):
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


class AlphaNoiseFilter_EditUtils:
    """Clean the alpha channel of a Qwen-Image 2.1 alpha generation.

    Qwen-Image 2.1 (64ch / 16x RGBA VAE) can generate images with an alpha
    channel, but the raw output usually carries faint, low-value alpha noise
    scattered across the transparent region. It is barely visible on its own,
    yet it breaks compositing, cutouts and mask-driven editing. This node
    removes that noise and outputs a clean alpha image together with its mask.
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE", {"tooltip": "RGBA image, e.g. the VAEDecode output of the Qwen-Image 2.1 VAE"}),
            },
            "optional": {
                "mask": ("MASK", {"tooltip": "Used as the alpha when the input image has no alpha channel (RGB input)"}),
                "alpha_threshold": ("FLOAT", {"default": 0.10, "min": 0.0, "max": 1.0, "step": 0.01,
                                              "tooltip": "Alpha below this is treated as low-value noise and clipped to 0 (0 disables)"}),
                "min_area": ("INT", {"default": 100, "min": 0, "max": 65536, "step": 1,
                                     "tooltip": "Drop isolated alpha islands smaller than this many pixels (0 disables)"}),
                "feather": ("FLOAT", {"default": 2.0, "min": 0.0, "max": 32.0, "step": 0.5,
                                      "tooltip": "Width in pixels of the gradient band created inside the edge: fully transparent at the matte boundary, rising smoothly to fully opaque `feather` px inwards. The matte never grows outwards, so no pixel the generator left transparent can become visible (0 = hard edge)"}),
                "invert_mask": ("BOOLEAN", {"default": False,
                                            "tooltip": "Output 1 - alpha as the mask (white = transparent, the ComfyUI inpaint convention)"}),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK",)
    RETURN_NAMES = ("image", "mask",)
    FUNCTION = "handle_alpha"
    CATEGORY = "image"
    DESCRIPTION = (
        "Clean the alpha channel of a Qwen-Image 2.1 alpha generation: clip "
        "low-value alpha noise, drop small isolated alpha islands, then feather "
        "inwards from the edge, and output a clean RGBA image and its mask. "
        "Only the alpha channel is modified, and feathering only ever reduces "
        "alpha inside the matte, so the silhouette never grows into the "
        "transparent background."
    )

    def handle_alpha(self, image, mask=None, alpha_threshold=0.10, min_area=100,
                     feather=2.0, invert_mask=False):
        if image.ndim != 4:
            raise ValueError(f"AlphaNoiseFilter: expected an image tensor (B, H, W, C), got shape {tuple(image.shape)}")

        device = image.device
        rgb = image[..., :3]

        # --- alpha source: image alpha channel wins, fall back to the mask ---
        if image.shape[-1] >= 4:
            alpha = image[..., 3].unsqueeze(1)
        elif mask is not None:
            alpha = mask
            if alpha.ndim == 2:
                alpha = alpha.unsqueeze(0)
            alpha = alpha.unsqueeze(1)
            if alpha.shape[-2:] != rgb.shape[1:3]:
                alpha = F.interpolate(alpha, size=rgb.shape[1:3], mode="bilinear", align_corners=False)
        else:
            raise ValueError(
                "AlphaNoiseFilter: the input image has no alpha channel "
                f"({image.shape[-1]} channels) and no mask is connected."
            )

        alpha = alpha.to(dtype=torch.float32, device=device)

        # 1) cut the faint low-value noise wash
        if alpha_threshold > 0:
            alpha = alpha * (alpha >= alpha_threshold).float()

        # 2) drop small isolated alpha islands left by the cut
        if min_area > 0:
            alpha = self._remove_small_islands(alpha, min_area)

        # 3) feather: soften the edge from the inside only
        if feather > 0:
            alpha = self._feather_inward(alpha, feather)

        out_image = torch.cat([rgb, alpha.permute(0, 2, 3, 1)], dim=-1).to(device=device, dtype=image.dtype)
        out_mask = alpha[:, 0]
        if invert_mask:
            out_mask = 1.0 - out_mask
        out_mask = out_mask.to(device=device, dtype=torch.float32)
        return (out_image, out_mask)

    @staticmethod
    def _remove_small_islands(alpha, min_area):
        """Zero out connected alpha regions smaller than min_area pixels."""
        out = alpha.clone()
        for i in range(alpha.shape[0]):
            frame = alpha[i, 0].detach().cpu().numpy()
            if _scipy_ndimage is not None:
                labels, num = _scipy_ndimage.label(frame > 0)
                if num <= 0:
                    continue
                sizes = _scipy_ndimage.sum(np.ones_like(labels), labels, index=np.arange(1, num + 1))
                keep = np.zeros(num + 1, dtype=bool)
                keep[1:] = sizes >= min_area
                frame = frame * keep[labels].astype(np.float32)
            else:
                # Fallback: grayscale opening with a disk of area ~= min_area.
                r = max(1, int(math.ceil(math.sqrt(min_area / math.pi))))
                k = r * 2 + 1
                t = torch.from_numpy(frame)[None, None]
                t = -F.max_pool2d(-t, kernel_size=k, stride=1, padding=r)
                t = F.max_pool2d(t, kernel_size=k, stride=1, padding=r)
                frame = frame * (t[0, 0].numpy() > 0).astype(np.float32)
            out[i, 0] = torch.from_numpy(np.ascontiguousarray(frame)).to(alpha.device)
        return out

    @staticmethod
    def _feather_inward(alpha, width):
        """Soften the matte inwards from its own boundary as a gradient.

        `width` pixels inside the boundary the alpha fades from fully transparent
        (on the boundary itself) to fully opaque, and anything deeper is left
        untouched. Alpha is never created outside the region the matte already
        marks, so feathering cannot make a pixel visible that the generator left
        transparent -- those pixels still carry the raw background colour, which
        is what produced the coloured halo when the matte was blurred
        symmetrically.
        """
        if _scipy_ndimage is not None:
            out = alpha.clone()
            for i in range(alpha.shape[0]):
                frame = alpha[i, 0].detach().cpu().numpy()
                if not (frame > 0).any():
                    continue
                # distance_transform_edt is 1 on the outermost matte pixel, so
                # subtracting 1 makes the ramp start at 0 exactly there.
                dist = _scipy_ndimage.distance_transform_edt(frame > 0)
                t = np.clip((dist - 1.0) / width, 0.0, 1.0)
                ramp = t * t * (3.0 - 2.0 * t)          # smoothstep
                out[i, 0] = torch.from_numpy(np.ascontiguousarray(frame * ramp)).to(alpha.device)
            return out
        return AlphaNoiseFilter_EditUtils._inward_ramp_by_erosion(alpha, width)

    @staticmethod
    def _inward_ramp_by_erosion(alpha, width):
        """scipy-free fallback: the same gradient built from 1px erosions.

        Summing the matte eroded by 1..width pixels yields a linear ramp that is
        0 on the boundary, rises inwards, and stays 0 outside the matte.
        """
        rings = max(1, int(math.ceil(width)))
        acc = torch.zeros_like(alpha)
        eroded = -F.max_pool2d(-alpha, kernel_size=3, stride=1, padding=1)
        for _ in range(rings):
            acc = acc + eroded
            eroded = -F.max_pool2d(-eroded, kernel_size=3, stride=1, padding=1)
        return (acc / rings).clamp(0.0, 1.0)


class RopeOffsetFromMask_EditUtils:
    """Extract ROPE position offsets from a mask or black-and-white image.

    Finds the bounding box of the white (non-zero) region and outputs
    the top-left corner as rope_x_offset / rope_y_offset, aligned to
    the VAE unit. Also outputs region dimensions for setting ref_longest_edge.
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "mask": ("MASK", {"tooltip": "Black and white mask (B,H,W). White region indicates where the reference should be placed."}),
            },
            "optional": {
                "image": ("IMAGE", {"tooltip": "Optional black-and-white image (B,H,W,C). If provided, converted to grayscale mask. Overrides 'mask' input."}),
                "vae_unit": ("INT", {"default": 8, "min": 8, "max": 16, "step": 8, "tooltip": "VAE spatial unit (8 for Krea2/QwenImage/Boogu, 16 for Flux2Klein). Offsets are aligned to this."}),
                "threshold": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Threshold to binarize. Pixels above this value are considered 'white'."}),
            },
        }

    RETURN_TYPES = ("INT", "INT", "INT", "INT")
    RETURN_NAMES = ("rope_x_offset", "rope_y_offset", "region_width", "region_height")
    FUNCTION = "extract_offsets"
    CATEGORY = "advanced/conditioning"
    DESCRIPTION = (
        "Extract ROPE position offsets from a mask or B&W image's bounding box. "
        "White region's top-left corner becomes (rope_x_offset, rope_y_offset). "
        "Also outputs region width/height for setting ref_longest_edge."
    )

    def extract_offsets(self, mask, image=None, vae_unit=8, threshold=0.5):
        # If image is provided, convert to grayscale mask
        if image is not None:
            # image shape: (B, H, W, C) with values 0-1
            img = image[0]  # (H, W, C)
            # Convert to grayscale luminance
            if img.shape[-1] >= 3:
                gray = 0.299 * img[..., 0] + 0.587 * img[..., 1] + 0.114 * img[..., 2]
            else:
                gray = img[..., 0]
            m = gray  # (H, W)
        else:
            # mask shape: (B, H, W) with values 0-1
            if mask.ndim == 3:
                m = mask[0]  # (H, W)
            elif mask.ndim == 2:
                m = mask
            else:
                raise ValueError(f"Expected mask with 2 or 3 dimensions, got {mask.ndim}")

        # Binarize
        binary = (m > threshold).float()

        # Find non-zero pixels
        nonzero = torch.nonzero(binary, as_tuple=False)  # (N, 2) -> [row, col]

        if nonzero.shape[0] == 0:
            # All black — return full canvas dimensions
            h, w = m.shape[0], m.shape[1]
            return (0, 0, (w // vae_unit) * vae_unit, (h // vae_unit) * vae_unit)

        # Bounding box: min_row, min_col, max_row, max_col
        min_row = int(nonzero[:, 0].min().item())  # Y (height)
        min_col = int(nonzero[:, 1].min().item())  # X (width)
        max_row = int(nonzero[:, 0].max().item())
        max_col = int(nonzero[:, 1].max().item())

        # Align offsets to VAE unit (floor to nearest unit)
        rope_x_offset = (min_col // vae_unit) * vae_unit
        rope_y_offset = (min_row // vae_unit) * vae_unit

        # Region dimensions (ceil to nearest unit)
        raw_width = max_col - min_col + 1
        raw_height = max_row - min_row + 1
        region_width = ((raw_width + vae_unit - 1) // vae_unit) * vae_unit
        region_height = ((raw_height + vae_unit - 1) // vae_unit) * vae_unit

        return (rope_x_offset, rope_y_offset, region_width, region_height)


class LongestEdgeImageProcess_EditUtils:
    upscale_methods = ["lanczos", "bicubic", "area"]
    crop_methods = ["pad", "center", "disabled"]
    pad_modes = ["color", "edge", "mirror", "border_mean"]
    pad_colors = ["red", "green", "blue", "white", "black", "custom"]

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
            "optional": {
                "pad_mode": (s.pad_modes, {"default": "color", "tooltip": "How to fill the padding area when ref_crop == pad. color = solid pad_color, edge = replicate edge pixels, mirror = reflect content, border_mean = fill with the mean of the nearest edge pixels"}),
                "pad_color": (s.pad_colors, {"default": "black", "tooltip": "Pad fill color when pad_mode == color: red / green / blue / white / black / custom. Use white for qwen models since black pads get amplified during generation"}),
                "pad_color_rgb": ("STRING", {"default": "-", "tooltip": "Optional override: 6-digit hex color, with or without '#' (e.g. FF0000 or #FF0000). When valid it overrides pad_color. Anything else (\"-\", wrong length, non-hex, numbers, \"r,g,b\") is ignored"}),
                "pad_noise": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 0.2, "step": 0.01, "tooltip": "Gaussian jitter (std) added to the pad area to avoid perfectly flat regions; 0.02-0.05 recommended with solid colors"}),
            },
        }

    RETURN_TYPES = ("IMAGE", "ANY", "FLOAT")
    RETURN_NAMES = ("processed_image", "pad_info", "scale_by")
    FUNCTION = "process"

    CATEGORY = "image"

    def process(self, image, ref_longest_edge, ref_crop, ref_upscale, vae_unit,
                pad_mode="color", pad_color="black", pad_color_rgb="-", pad_noise=0.0):
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

            resized_samples = comfy.utils.common_upscale(samples, scaled_width, scaled_height, ref_upscale, crop)

            resized_width = resized_samples.shape[3]
            resized_height = resized_samples.shape[2]
            # pad image to canvas size with configurable pad mode/color
            canvas = build_pad_canvas(
                resized_samples, canvas_height, canvas_width,
                pad_mode=pad_mode, pad_color=pad_color, pad_noise=pad_noise,
                pad_color_rgb=pad_color_rgb, pad_color_fallback="black"
            )

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
