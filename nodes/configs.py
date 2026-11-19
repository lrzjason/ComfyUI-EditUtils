"""Model config nodes: the routing dicts consumed by EditTextEncode_EditUtils."""



from ._shared import MINIMAX_H3_IMAGE_UNIT, get_system_prompt

class ModelConfig_EditUtils:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model_choice": (["qwen", "flux2klein", "flux3klein", "boogu", "moss_image", "minimax_h3_image"], {
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
        
        # Boogu / MOSS tokenizers auto-select their prompt template, so no
        # llama_template is needed (and for MOSS a template would break the
        # automatic vision-block insertion for reference images).
        # MiniMax H3 Image has no chat template at all: its presentation is raw
        # text plus "Picture n: " + vision blocks, so a system prompt would be
        # encoded as prompt words and go off-distribution.
        if model_choice in ("boogu", "moss_image", "minimax_h3_image"):
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


class Flux3KleinModelConfig_EditUtils:
    """Model config for FLUX.3 Klein (base and step-distilled).

    FLUX.3 is NOT a rename of FLUX.2 klein — it is a different architecture:

    - per-mode transformer blocks + joint single blocks (three token modes:
      image, image_cond, text) instead of FLUX.2's double/single stream MMDiT;
    - Qwen3-VL text encoder (not Qwen3 LLM), 8 stacked hidden states
      [4, 8, 12, 16, 20, 24, 28, 32] -> joint_attention_dim 20480, text window
      right-padded to a multiple of 80 tokens;
    - references are told apart on the **L** position axis with NEGATIVE
      coordinates (-1, -2, ...) instead of FLUX.2's positive scale-10 index;
    - image H/W rope ids are scaled by image_rope_scale = 0.5;
    - the distilled checkpoint is a flow-map model (flow_map=True): 4-16 steps
      and NO classifier-free guidance.

    The VAE is the FLUX.2 packed autoencoder (32 latent channels, 8x downscale,
    2x2 patchify), so the token grid is 16 px -> vae_unit = 16, exactly as for
    FLUX.2 klein. Its normalization is a batch-norm (no scalar scaling/shift
    factor), which ComfyUI's Flux2 latent format already matches.

    ComfyUI core implements this natively (image_model "flux3", class
    Flux3 in comfy/supported_models.py), so the reference latents flow through
    the standard `reference_latents` conditioning key.
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "optional": {
                "instruction": ("STRING", {"multiline": True, "default": ""}),
                "flow_map": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "True for the step-distilled (flow-map) checkpoint: 4-16 steps, no CFG. "
                               "Informational only — ComfyUI detects flow_map from the checkpoint itself."}),
            }
        }

    RETURN_TYPES = ("DICT",)
    RETURN_NAMES = ("model_config",)
    FUNCTION = "configure_model"

    CATEGORY = "advanced/conditioning"

    def configure_model(self, instruction="", flow_map=False):
        config = {
            # routing key: the flux3 encoding path
            "model_name": "flux3klein",
            # FLUX.3 uses the FLUX.2 packed VAE -> 16 px per latent token
            "vae_unit": 16,
            "config_for": "flux3klein",
            "flow_map": bool(flow_map),
        }
        # FLUX.3's tokenizer builds its own instruct template (with the 80-token
        # right padding), so the default is an empty llama_template. A custom
        # instruction is allowed but off the trained distribution.
        config["llama_template"] = ""
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


class Krea2ModelConfig_EditUtils:
    # Krea2 uses the Qwen text encoder and the Qwen-Image VAE, so it routes
    # through the qwen encoding branch (model_name="qwen", vae_unit=8).
    # DEFAULT_INSTRUCTION = ("Describe the image by detailing the color, shape, size, texture, quantity, text, spatial relationships of the objects and background:")

    @classmethod
    def INPUT_TYPES(s):
        return {
            "optional": {
                "instruction": ("STRING", {"multiline": True,
                                           "default": "Describe the image by detailing the color, shape, size, texture, quantity, text, spatial relationships of the objects and background:"}),
            }
        }

    RETURN_TYPES = ("DICT",)
    RETURN_NAMES = ("model_config",)
    FUNCTION = "configure_model"

    CATEGORY = "advanced/conditioning"

    def configure_model(self, instruction=None):
        if not instruction:
            instruction = "Describe the image by detailing the color, shape, size, texture, quantity, text, spatial relationships of the objects and background:"
        config = {
            # routing key: qwen encoding path (Krea2 text encoder is Qwen-based)
            "model_name": "qwen",
            # Krea2's VAE is the Qwen-Image VAE -> same spatial unit as qwen
            "vae_unit": 8,
            # identifies the config origin for logging/debugging only
            "config_for": "krea2",
        }
        config["llama_template"] = get_system_prompt(instruction)
        return (config,)


class MossImageModelConfig_EditUtils:
    """Model config for MOSS-Image (the EditUtils reference-image path).

    MOSS-Image is a 5B text-to-image DiT with two components that matter here:

    - **Text encoder**: Qwen3.5-9B *with its vision tower*, so reference images
      can be grounded through the prompt the same way Krea2 does it.  The
      ``moss_image`` tokenizer owns the chat template (user turn + an open
      ``<think>`` block), so ``llama_template`` must stay empty; EditTextEncode
      then hands the ``Picture N: <|vision_start|><|image_pad|><|vision_end|>``
      prefix straight to it.
    - **VAE**: the FLUX.2 packed autoencoder (128 latent channels, 16 px per
      latent token).  Reference latents therefore snap to 16 px, exactly like
      the Flux2Klein route (``vae_unit=16``).

    This node only produces the routing dict.  The reference latents themselves
    are attached by ``EditTextEncode_EditUtils`` and consumed by
    ``MossImageEditApply_EditUtils``.
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "optional": {
                # Kept as an explicit no-op so a graph can be switched from a
                # Krea2/Qwen config node to this one without rewiring.  MOSS's
                # tokenizer builds its own prompt template; a custom template
                # would not insert the vision blocks for the reference images.
                "instruction": ("STRING", {"multiline": True, "default": "",
                                           "tooltip": "Unused: the MOSS-Image tokenizer builds its own chat template (user turn + think block). Leave empty."}),
            }
        }

    RETURN_TYPES = ("DICT",)
    RETURN_NAMES = ("model_config",)
    FUNCTION = "configure_model"

    CATEGORY = "advanced/conditioning"

    def configure_model(self, instruction=""):
        return ({
            # routing key: the MOSS-Image encode branch in EditTextEncode
            "model_name": "moss_image",
            # FLUX.2 packed VAE -> 16 px per latent token
            "vae_unit": 16,
            # no template: the moss_image tokenizer applies its own
            "llama_template": "",
            # identifies the config origin for logging/debugging only
            "config_for": "moss_image",
        },)


class QwenImage21ModelConfig_EditUtils:
    # Qwen-Image 2.1 uses a Qwen3-VL text encoder and a 16x spatial-downscale
    # VAE (64 latent channels). The reference latents are spliced into the text
    # sequence at the vision slots, so the vision-tower image and the VAE image
    # must be the SAME resized tensor: vae_unit is 16, matching the VAE
    # downscale (QwenImage21EditApply uses the same 16 px per latent token).
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
            "model_name": "qwen_image21",
            "vae_unit": 16,
            "llama_template": ""
        }
        # default: no template -> the QwenImage21 tokenizer auto-builds the
        # "Comprehend and analyze the provided prompt." T2I template with the
        # vision blocks inserted before the prompt. A custom instruction gets a
        # custom system prompt; the encode node inserts the vision blocks at {}.
        if instruction != "":
            config["llama_template"] = get_system_prompt(instruction)
        return (config,)


class MinimaxH3ImageModelConfig_EditUtils:
    """Model config for **MiniMax-H3-Image** (``image_model="minimax_h3_image"``).

    This is the *image* model of the MiniMax H3 family — not the H3
    video/audio model (``minimax_h3``), which packs a NestedTensor of video +
    audio latents and samples on a shifted schedule. H3I denoises a single
    24-channel latent frame, and its checkpoint is a 6-step PDD distill whose
    per-step heads live on a fixed sigma grid; both facts are carried by the
    checkpoint itself, so this node only produces the routing dict.

    What EditUtils adds on top of ComfyUI's ``MiniMaxH3ImageConditioning``:

    - reference images are described by per-image config nodes that can be
      chained (``MinimaxH3ImageConfigPreparer_EditUtils``), with independent
      sizing for the vision copy and the VAE reference latent;
    - the reference latents ride on ``minimax_refs`` exactly like ComfyUI's node
      builds them: ``[{"kind": "image", "latent_h": h, "latent_w": w,
      "latent": z}]``, packed between the text rows and the target rows by
      ``comfy.ldm.minimax.model.PackedLayout`` and re-injected every step at
      ``t = 0.999`` (never denoised);
    - the target latent is ``[1, 24, 1, H/16, W/16]`` — one frame, no audio.

    ``vae_unit`` is 32 here, not 16: the VAE token is 16 px but the DiT
    patches the latent 2x2, so a representable canvas side is a multiple of
    32 px (ComfyUI's own H3I node rounds edit references to 32 px for exactly
    this reason).

    Text encoder: the ``minimax`` CLIP type (Qwen3-VL-32B truncated to 50
    layers). It has no chat template — the presentation is raw text with the
    references spliced in as ``"Picture n: " + <vision block>`` before the
    prompt, which the tokenizer does when images are passed to it. Do **not**
    hand it a system prompt.
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "optional": {
                # Kept as an explicit no-op so a graph can be switched from
                # another model's config node without rewiring; the H3I
                # tokenizer has no template to override.
                "instruction": ("STRING", {"multiline": True, "default": "",
                                           "tooltip": "Unused: the MiniMax H3 tokenizer has no chat template (raw text + \"Picture n: \" labels). Leave empty."}),
            }
        }

    RETURN_TYPES = ("DICT",)
    RETURN_NAMES = ("model_config",)
    FUNCTION = "configure_model"

    CATEGORY = "advanced/conditioning"

    def configure_model(self, instruction=""):
        if instruction:
            print("MinimaxH3ImageModelConfig: the H3 image tokenizer has no chat template; "
                  "`instruction` is ignored (a system prompt would be encoded as prompt text).")
        return ({
            # routing key: the minimax_h3_image branch in EditTextEncode
            "model_name": "minimax_h3_image",
            # 32 px per latent grid step: 16 px VAE token x 2x2 DiT patch
            "vae_unit": MINIMAX_H3_IMAGE_UNIT,
            # no template: the tokenizer owns the presentation
            "llama_template": "",
            # identifies the config origin for logging/debugging only
            "config_for": "minimax_h3_image",
        },)
