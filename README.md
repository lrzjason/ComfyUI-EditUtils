# ComfyUI-EditUtils

A collection of utility nodes for advanced image editing in ComfyUI, supporting multiple AI models including Qwen and Flux2Klein.

## Overview

ComfyUI-EditUtils is the follow-up version of ComfyUI-QwenEditUtils, offering enhanced capabilities for image editing workflows with support for multiple AI models. This package provides a comprehensive set of tools for advanced image editing, featuring flexible configuration options and model-specific optimizations.


## Examples
<p align="center">
  <img src="simple example.png" alt="Example Workflow" width="80%" />
</p>

*Simple workflow screenshot showing f2k edit simple.json workflow*

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
