# ComfyUI-EditUtils

A collection of utility nodes for advanced image editing in ComfyUI, supporting multiple AI models including Qwen and Flux2Klein.

## Update
20260929 Added Configurable Pad Mode/Color/Noise (pad_mode, pad_color, pad_noise), White Edge Default For Qwen To Reduce Black Edge Amplification
20260407 Fixed Extra Height Unit Pad Which Introduce Color Shift

## Overview

ComfyUI-EditUtils is the follow-up version of ComfyUI-QwenEditUtils, offering enhanced capabilities for image editing workflows with support for multiple AI models. This package provides a comprehensive set of tools for advanced image editing, featuring flexible configuration options and model-specific optimizations.


## Examples
RunningHub Single Image Workflow:
https://www.runninghub.ai/post/2045207600739913729/?inviteCode=rh-v1279

## Usage Tips

For better consistency in local editing, it's recommended to use this workflow with Consistency Edit LoRA:

- Civitai Download: [Consistency Edit LoRA](https://civitai.com/models/1939453?modelVersionId=2634354)
- Huggingface Download: [Consistency Edit LoRA](https://huggingface.co/lrzjason/Consistance_Edit_Lora)

## Workflows

The plugin includes several example workflows:

- **f2k edit simple.json** - Simple workflow recommended for common usage (easy to use)
- **f2k edit single.json** - Highly customizable workflow for ComfyUI expert users
- **f2k edit multiple.json** - Highly customizable workflow for ComfyUI expert users with multiple image support

## Capabilities

- EditUtils supports direct high-resolution editing without pixel shifts, up to 2xxx ~ 3xxx resolution
- Multiple images input support:
  - Simple workflow: up to 3 images
  - Single and multiple workflows: unlimited images (connect multiple configs)


## Pad Edge Control

When `ref_crop` is set to `pad`, the image is padded to a multiple of the VAE unit size. How that padding looks is now configurable per image through `pad_mode`, `pad_color` and `pad_noise` on the config preparer nodes (`QwenConfigPreparer_EditUtils` / `Flux2KleinConfigPreparer_EditUtils`) or the `ConfigJsonParser_EditUtils` JSON:

- **pad_mode** (default: `color`):
  - `color`: solid fill using `pad_color`
  - `edge`: replicate the outermost pixel row/column
  - `mirror`: reflect the content at the boundary
  - `border_mean`: fill each side with the mean of its nearest 8 pixels
- **pad_color** (default: `white` for Qwen, `black` for Flux2Klein): named color (`black`, `white`, `gray`, `mid_gray`), hex `#RRGGBB`, or `"r,g,b"`. Any value above 1 is treated as the 0-255 range.
- **pad_noise** (default: `0.0`): Gaussian jitter (std) added to the padded area to break up perfectly flat regions. 0.02-0.05 is recommended with solid colors.

**Why this exists:** Qwen models were trained with black pads, so black padding tends to get amplified into large black edges during generation. White is therefore the default for Qwen. If a solid color is still not enough, `border_mean` or `mirror` are content-adaptive alternatives, and any mode benefits from a small `pad_noise`.

The mask/semantics of the padded area are unchanged (it is always masked out), and `CropWithPadInfo_EditUtils` still crops the padding away correctly.


## Node Categories
Documentation:

- [Node Documentation](nodes_doc.md)


## Key Features

- **Multi-Model Support**: Works with both Qwen and Flux2Klein models for versatile image editing
- **Flexible Configuration**: Per-image configuration options for reference and VL processing
- **Pad Edge Control**: Configurable pad mode/color/noise (white edge default for Qwen) to avoid black edge amplification
- **Unified Interface**: Single node EditTextEncode_EditUtils works with multiple models through configuration nodes
- **Advanced Processing**: Supports complex image editing workflows with multiple reference images
- **Comprehensive Output**: Detailed output dictionary with all processing intermediates
- **Modular Design**: Separated configuration, processing, and extraction nodes for maximum flexibility

## New Nodes

### DiffMask_EditUtils
A utility node that generates a mask highlighting the differences between two images. Useful for editing tasks where you want to identify changed regions.

**Inputs:**
- `image1`: First image
- `image2`: Second image
- `threshold`: Threshold to ignore minor differences (0.0-1.0, default: 0.05)

**Output:**
- `mask`: A mask highlighting differences between the two images

**Use Case:** Compare original and edited images to create a mask for selective editing or inpainting.

## Installation

1. Clone or download this repository into your ComfyUI's `custom_nodes` directory.
2. Restart ComfyUI.
3. The nodes will be available in the "advanced/conditioning" category.

## Changelog

### ComfyUI-EditUtils vs ComfyUI-QwenEditUtils
ComfyUI-EditUtils is the follow-up version of ComfyUI-QwenEditUtils with the following improvements:
- Multi-model support (Qwen and Flux2Klein)
- Unified node architecture with configuration nodes
- Enhanced flexibility and modularity
- Improved code organization and maintainability

## Contact
- **Twitter**: [@Lrzjason](https://twitter.com/Lrzjason)  
- **Email**: lrzjason@gmail.com  
- **QQ Group**: 866612947  
- **Wechatid**: fkdeai
- **Civitai**: [xiaozhijason](https://civitai.com/user/xiaozhijason)

## Sponsors me for more open source projects:
<div align="center">
  <table>
    <tr>
      <td align="center">
        <p>Buy me a coffee:</p>
        <img src="https://github.com/lrzjason/Comfyui-In-Context-Lora-Utils/blob/main/image/bmc_qr.png" alt="Buy Me a Coffee QR" width="200" />
      </td>
      <td align="center">
        <p>WeChat:</p>
        <img src="https://github.com/lrzjason/Comfyui-In-Context-Lora-Utils/blob/main/image/wechat.jpg" alt="WeChat QR" width="200" />
      </td>
    </tr>
  </table>
</div>
