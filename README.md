# ComfyUI-EditUtils

A collection of utility nodes for advanced image editing in ComfyUI, supporting multiple AI models including Qwen and Flux2Klein.

## Update
20260504 Added Longest Edge Image Process, Clear Ref Latents, Save/Load Condition nodes. Fixed no_refs_cond output in Output Extractors.
20260407 Fixed Extra Height Unit Pad Which Introduce Color Shift

## Overview

ComfyUI-EditUtils is the follow-up version of ComfyUI-QwenEditUtils, offering enhanced capabilities for image editing workflows with support for multiple AI models. This package provides a comprehensive set of tools for advanced image editing, featuring flexible configuration options and model-specific optimizations.


## Examples
RunningHub Single Image Workflow:
https://www.runninghub.ai/post/2045207600739913729/?inviteCode=rh-v1279

RunningHub Simple Krea2 Depth Workflow (工作流：Simple Krea Depth):
https://www.runninghub.ai/post/2082077636234313729/?inviteCode=rh-v1279

## Usage Tips

For better consistency in local editing, it's recommended to use this workflow with Consistency Edit LoRA:

- Civitai Download: [Consistency Edit LoRA](https://civitai.com/models/1939453?modelVersionId=2634354)
- Huggingface Download: [Consistency Edit LoRA](https://huggingface.co/lrzjason/Consistance_Edit_Lora)

## Workflows

Example workflows are available in the [workflows](workflows/) directory:

- **[Simple Krea2 Depth.json](workflows/Simple%20Krea2%20Depth.json)** - Simple Krea2 editing workflow with a depth LoRA. Wires `LoadImage → Krea2ModelConfig_EditUtils → EditTextEncode_EditUtils → Krea2EditApply_EditUtils → KSampler`, loading the model via `UNETLoader + LoraLoaderModelOnly` — the reference latent flows through the conditioning chain automatically.
  - Online version on RunningHub: https://www.runninghub.ai/post/2082077636234313729/?inviteCode=rh-v1279
  - Depth LoRA download: [Krea2 Depth LoRA (Civitai)](https://civitai.com/models/2815790/krea2-depth-lrzjason-20260729)

> ⚠️ **Note:** Krea2 edit support is still in development — node interfaces and behavior may change.

## Capabilities

- EditUtils supports direct high-resolution editing without pixel shifts, up to 2xxx ~ 3xxx resolution
- Multiple images input support:
  - Simple workflow: up to 3 images
  - Single and multiple workflows: unlimited images (connect multiple configs)


## Node Categories
Documentation:

- [Node Documentation](nodes_doc.md)


## Key Features

- **Multi-Model Support**: Works with both Qwen and Flux2Klein models for versatile image editing
- **Flexible Configuration**: Per-image configuration options for reference and VL processing
- **Unified Interface**: Single node EditTextEncode_EditUtils works with multiple models through configuration nodes
- **Advanced Processing**: Supports complex image editing workflows with multiple reference images
- **Comprehensive Output**: Detailed output dictionary with all processing intermediates
- **Modular Design**: Separated configuration, processing, and extraction nodes for maximum flexibility

## New Nodes

### LongestEdgeImageProcess_EditUtils
A utility node that resizes and pads an image based on a target longest edge, using the same processing logic as EditTextEncode. Useful when you need the image preprocessing without CLIP/VAE encoding.

**Inputs:**
- `image`: Input image
- `ref_longest_edge`: Target longest edge size (default: 1024)
- `ref_crop`: Crop method - "pad", "center", or "disabled" (default: "pad")
- `ref_upscale`: Upscale method (default: "lanczos")
- `vae_unit`: VAE unit size for padding alignment (default: 8)

**Outputs:**
- `processed_image`: The resized/padded image
- `pad_info`: Padding information dictionary
- `scale_by`: The scale factor

**Use Case:** Pre-process images with longest edge scaling before passing to other nodes, or reuse the same image processing pipeline outside of the encoding workflow.

### ClearRefLatents_EditUtils
A utility node that strips reference latents from a conditioning, outputting a clean conditioning without any ref latents attached.

**Inputs:**
- `conditioning`: Conditioning with reference latents

**Outputs:**
- `conditioning`: The same conditioning with reference latents cleared

**Use Case:** Remove reference latents from conditioning when you want to use the text encoding without image references.

### SaveCondition_EditUtils
Saves a conditioning tensor to a `.ckpt` file in the `models/conditions` directory.

**Inputs:**
- `condition`: The conditioning to save
- `filename`: Output filename (default: "condition_tensor")

**Use Case:** Persist conditioning tensors for later reuse without re-encoding.

### LoadCondition_EditUtils
Loads a conditioning tensor from a `.ckpt` file in the `models/conditions` directory.

**Inputs:**
- `filename`: Select from available `.ckpt` files in the conditions directory

**Outputs:**
- `conditioning`: The loaded conditioning tensor

**Use Case:** Reuse previously saved conditioning tensors in new workflows.

### LoadConditionFromLoras_EditUtils
Lists files from the loras directory and attempts to load matching `.ckpt` files from the conditions directory.

**Inputs:**
- `filename`: Select from available lora files

**Outputs:**
- `conditioning`: The loaded conditioning tensor

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
