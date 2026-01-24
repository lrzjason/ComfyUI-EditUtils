# ComfyUI-EditUtils

A collection of utility nodes for advanced image editing in ComfyUI, supporting multiple AI models including Qwen and Flux2Klein.

## Overview

ComfyUI-EditUtils is the follow-up version of ComfyUI-QwenEditUtils, offering enhanced capabilities for image editing workflows with support for multiple AI models. This package provides a comprehensive set of tools for advanced image editing, featuring flexible configuration options and model-specific optimizations.

## Node Categories

### Model Configuration Nodes

#### ModelConfig_EditUtils
Universal model configuration node that supports both Qwen and Flux2Klein models.

##### Inputs
- **model_choice**: Select between "qwen" and "flux2klein" models
- **model_name**: Custom model name (optional, defaults to model choice)
- **vae_unit**: VAE unit size (default: 8 for Qwen, 16 for Flux2Klein)
- **instruction**: Custom instruction for image editing (optional)

##### Outputs
- **model_config**: Dictionary containing model configuration parameters

##### Behavior
- Configures model-specific parameters for image editing
- Sets appropriate VAE unit sizes based on model selection
- Handles Qwen-specific system prompts

#### QwenModelConfig_EditUtils
Specialized configuration node for Qwen-based models.

##### Inputs
- **instruction**: Custom instruction for image editing (optional)

##### Outputs
- **model_config**: Dictionary containing Qwen-specific configuration parameters

##### Behavior
- Configures Qwen-specific parameters with default settings
- Sets model name to "qwen" and VAE unit to 8

#### Flux2KleinModelConfig_EditUtils
Specialized configuration node for Flux2Klein models.

##### Inputs
- None required

##### Outputs
- **model_config**: Dictionary containing Flux2Klein-specific configuration parameters

##### Behavior
- Configures Flux2Klein-specific parameters with default settings
- Sets model name to "flux2klein" and VAE unit to 16

### Text Encoding Nodes

#### EditTextEncode_EditUtils
Advanced text encoding node with support for multiple reference images and model-specific configurations.

##### Inputs
- **clip**: The CLIP model to use for encoding
- **vae**: The VAE model for image encoding
- **prompt**: The text prompt to encode
- **model_config**: Model configuration from configuration nodes
- **configs**: List of image configuration dictionaries

##### Outputs
- **conditioning**: The encoded conditioning tensor
- **latent**: The encoded latent representation of the main reference image
- **custom_output**: Comprehensive output dictionary with all processing results
- **main_image**: The processed main reference image
- **mask**: The processed mask used for region of interest

##### Behavior
- Provides maximum flexibility by allowing per-image configurations for both reference and VL processing
- Supports multiple reference images with different processing requirements simultaneously
- Allows fine-grained control over scaling, cropping, and resizing for each image
- Works with both Qwen and Flux2Klein models
- Returns comprehensive output dictionary with all intermediate results

#### QwenEditTextEncode_EditUtils
Simplified text encoding node specifically for Qwen-based image editing workflows.

##### Inputs
- **clip**: The CLIP model to use for encoding
- **vae**: The VAE model for image encoding
- **prompt**: The text prompt to encode
- **image1**: First reference image for image editing (optional)
- **image2**: Second reference image for image editing (optional)
- **image3**: Third reference image for image editing (optional)
- **ref_longest_edge**: Target longest edge size for reference processing (default: 1024)

##### Outputs
- **conditioning**: The encoded conditioning tensor
- **latent**: The encoded latent representation of the main reference image
- **custom_output**: Comprehensive output dictionary with all processing results
- **main_image**: The processed main reference image
- **mask**: The processed mask used for region of interest

##### Behavior
- Simplified interface for Qwen-based image editing
- Supports up to 3 reference images for complex editing tasks
- Automatically configures Qwen-specific parameters
- Processes images separately for VAE encoding

#### Flux2KleinEditTextEncode_EditUtils
Simplified text encoding node specifically for Flux2Klein-based image editing workflows.

##### Inputs
- **clip**: The CLIP model to use for encoding
- **vae**: The VAE model for image encoding
- **prompt**: The text prompt to encode
- **image1**: First reference image for image editing (optional)
- **image2**: Second reference image for image editing (optional)
- **image3**: Third reference image for image editing (optional)
- **ref_longest_edge**: Target longest edge size for reference processing (default: 1024)

##### Outputs
- **conditioning**: The encoded conditioning tensor
- **latent**: The encoded latent representation of the main reference image
- **custom_output**: Comprehensive output dictionary with all processing results
- **main_image**: The processed main reference image
- **mask**: The processed mask used for region of interest

##### Behavior
- Simplified interface for Flux2Klein-based image editing
- Supports up to 3 reference images for complex editing tasks
- Automatically configures Flux2Klein-specific parameters with VAE unit of 16
- Processes images for reference encoding

### Configuration Preparation Nodes

#### QwenConfigPreparer_EditUtils
Helper node to create configuration objects for Qwen-based image editing.

##### Inputs
- **image**: The reference image to configure
- **configs**: Existing list of configuration objects to append to (optional)
- **to_ref**: Whether to include image in reference processing (default: True)
- **ref_main_image**: Whether this image is the main reference image (default: True for first image)
- **ref_longest_edge**: Target longest edge size for reference processing (default: 1024)
- **ref_crop**: Cropping method for reference processing (options: "pad", "center", "disabled", default: "pad")
- **ref_upscale**: Upscaling method for reference processing (options: "lanczos", "bicubic", "area", default: "lanczos")
- **to_vl**: Whether to include image in vision-language processing (default: True)
- **vl_resize**: Whether to resize image for VL processing (default: True)
- **vl_target_size**: Target size for VL processing (default: 384)
- **vl_crop**: Cropping method for VL processing (options: "center", "disabled", default: "center")
- **vl_upscale**: Upscaling method for VL processing (options: "lanczos", "bicubic", "area", default: "lanczos")
- **mask**: Optional mask for the image to define region of interest for editing

##### Outputs
- **configs**: Updated list of configuration objects
- **config**: Configuration object for the current image

##### Behavior
- Creates Qwen-specific configuration objects that define how each image should be processed
- Allows appending to existing configuration lists
- Provides default values for all configuration parameters
- Output config list can be connected directly to EditTextEncode_EditUtils

#### Flux2KleinConfigPreparer_EditUtils
Helper node to create configuration objects for Flux2Klein-based image editing.

##### Inputs
- **image**: The reference image to configure
- **configs**: Existing list of configuration objects to append to (optional)
- **to_ref**: Whether to include image in reference processing (default: True)
- **ref_main_image**: Whether this image is the main reference image (default: True for first image)
- **ref_longest_edge**: Target longest edge size for reference processing (default: 1024)
- **ref_crop**: Cropping method for reference processing (options: "pad", "center", "disabled", default: "pad")
- **ref_upscale**: Upscaling method for reference processing (options: "lanczos", "bicubic", "area", default: "lanczos")
- **mask**: Optional mask for the image to define region of interest for editing

##### Outputs
- **configs**: Updated list of configuration objects
- **config**: Configuration object for the current image

##### Behavior
- Creates Flux2Klein-specific configuration objects that define how each image should be processed
- Excludes VL-specific parameters that are not applicable to Flux2Klein
- Allows appending to existing configuration lists
- Output config list can be connected directly to EditTextEncode_EditUtils

#### ConfigJsonParser_EditUtils
Helper node to create configuration objects from JSON strings.

##### Inputs
- **image**: The reference image to configure
- **configs**: Existing list of configuration objects to append to (optional)
- **config_json**: JSON string containing configuration parameters
- **mask**: Optional mask for the image to define region of interest for editing

##### Outputs
- **configs**: Updated list of configuration objects
- **config**: Configuration object for the current image

##### Behavior
- Creates configuration objects from JSON strings
- Allows appending to existing configuration lists
- Provides a default JSON configuration template
- Output config list can be connected directly to EditTextEncode_EditUtils

### Output Extraction Nodes

#### QwenEditOutputExtractor_EditUtils
Helper node to extract specific outputs from the custom_output dictionary produced by Qwen-based nodes.

##### Inputs
- **custom_output**: The custom output dictionary from the custom node

##### Outputs
- **pad_info**: Padding information dictionary
- **full_refs_cond**: Conditioning with all reference latents
- **main_ref_cond**: Conditioning with only the main reference latent
- **main_image**: The main reference image
- **vae_images**: List of all processed VAE images
- **ref_latents**: List of all reference latents
- **vl_images**: List of all processed VL images
- **full_prompt**: The complete prompt with image descriptions
- **llama_template**: The applied system prompt template
- **no_refs_cond**: Conditioning without any reference latents
- **mask**: The processed mask used for region of interest

##### Behavior
- Extracts individual components from the complex output dictionary
- Provides access to all intermediate results from the custom node
- Enables modular processing of different output components

#### Flux2KleinOutputExtractor_EditUtils
Helper node to extract specific outputs from the custom_output dictionary produced by Flux2Klein-based nodes.

##### Inputs
- **custom_output**: The custom output dictionary from the custom node

##### Outputs
- **pad_info**: Padding information dictionary
- **main_image**: The main reference image
- **vae_images**: List of all processed VAE images
- **ref_latents**: List of all reference latents
- **full_prompt**: The complete prompt with image descriptions
- **llama_template**: The applied system prompt template
- **no_refs_cond**: Conditioning without any reference latents
- **mask**: The processed mask used for region of interest

##### Behavior
- Extracts individual components from the complex output dictionary
- Excludes VL-specific outputs not applicable to Flux2Klein
- Provides access to all relevant intermediate results from the custom node

### Utility Nodes

#### CropWithPadInfo_EditUtils
Utility node to crop an image using pad information generated by other nodes.

##### Inputs
- **image**: The image to crop
- **pad_info**: The pad information dictionary containing x, y, width, height and scale values

##### Outputs
- **cropped_image**: The cropped image with original content dimensions
- **scale_by**: The scale factor used in the original processing

##### Behavior
- Uses pad information to crop images to their original content area
- Removes padding that was added during processing
- Returns the scale factor for potential additional operations

#### ListExtractor_EditUtils
Utility node to extract a specific item from a list based on its index position.

##### Inputs
- **items**: The input list
- **index**: The index of the item to extract (default: 0)

##### Outputs
- **item**: The extracted item from the list

##### Behavior
- Extracts a single item from a list based on index
- Useful for extracting specific images from the vae_images list or other collections
- Supports any list items regardless of type

#### AdaptiveLongestEdge_EditUtils
Utility node to calculate an appropriate longest edge size for an image, ensuring it doesn't exceed specified constraints.

##### Inputs
- **image**: The input image to analyze
- **min_size**: Minimum allowed size for the longest edge (default: 512)
- **max_size**: Maximum allowed size for the longest edge (default: 2048)

##### Outputs
- **longest_edge**: The calculated longest edge size for the image that respects the size constraints

##### Behavior
- Calculates the longest edge of the input image
- If the longest edge exceeds max_size, it calculates a reduced size that fits within the constraint
- If the longest edge is smaller than min_size, it calculates an increased size
- Returns the appropriate longest edge size that can be used in other nodes for dynamic image processing

#### LoadImageWithFilename_EditUtils
Utility node to load an image with filename extraction capability.

##### Inputs
- **image**: The image file to load

##### Outputs
- **image**: The loaded image tensor
- **mask**: The extracted mask from the image
- **filename**: The filename without extension

##### Behavior
- Loads images from the input directory
- Extracts masks from images that support transparency
- Returns the filename without extension for potential use in workflows

#### Any2Image_EditUtils
Utility node to convert ANY type to IMAGE type.

##### Inputs
- **item**: The item of ANY type to convert

##### Outputs
- **item**: The converted IMAGE

##### Behavior
- Converts an ANY type input to IMAGE type
- Useful for connecting nodes that expect IMAGE type inputs

#### Any2Latent_EditUtils
Utility node to convert ANY type to LATENT type.

##### Inputs
- **item**: The item of ANY type to convert

##### Outputs
- **item**: The converted LATENT

##### Behavior
- Converts an ANY type input to LATENT type
- Creates a latent dictionary with the samples field
- Useful for connecting nodes that expect LATENT type inputs

## Key Features

- **Multi-Model Support**: Works with both Qwen and Flux2Klein models for versatile image editing
- **Flexible Configuration**: Per-image configuration options for reference and VL processing
- **Unified Interface**: Single node ([EditTextEncode_EditUtils](file:///f:/EditPlugin/ComfyUI-EditUtils/nodes.py#L276-L568)) works with multiple models through configuration nodes
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
