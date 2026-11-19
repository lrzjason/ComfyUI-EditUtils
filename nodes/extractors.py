"""custom_output unpackers for each model."""



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


class Flux3KleinOutputExtractor_EditUtils:
    """Extract the FLUX.3 klein intermediates from the EditTextEncode custom_output."""

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required":
            {
                "custom_output": ("ANY", ),
            }
        }

    RETURN_TYPES = ("ANY", "IMAGE", "LIST", "LIST", "STRING", "CONDITIONING", "MASK", "BOOLEAN", "LIST", "STRING")
    RETURN_NAMES = ("pad_info", "main_image", "vae_images", "ref_latents", "full_prompt",
                    "no_refs_cond", "mask", "flow_map", "bbox_rows", "composed_prompt")
    FUNCTION = "extract"

    CATEGORY = "advanced/conditioning"

    def extract(self, custom_output):
        pad_info = custom_output.get('pad_info')
        main_image = custom_output.get('main_image')
        vae_images = custom_output.get('vae_images')
        ref_latents = custom_output.get('ref_latents')
        full_prompt = custom_output.get('full_prompt')
        no_refs_cond = custom_output.get('no_refs_cond')
        mask = custom_output.get('mask')
        flow_map = custom_output.get('flow_map', False)
        bbox_rows = custom_output.get('bbox_rows', [])
        composed_prompt = custom_output.get('composed_prompt', full_prompt)

        return (pad_info, main_image, vae_images, ref_latents, full_prompt,
                no_refs_cond, mask, flow_map, bbox_rows, composed_prompt)


class MinimaxH3ImageOutputExtractor_EditUtils:
    """Extract the MiniMax-H3-Image intermediates from the EditTextEncode ``custom_output``.

    ``minimax_refs`` is the exact list handed to the model (``kind``,
    ``latent_h``, ``latent_w``, ``latent`` per reference) and ``target_size`` is
    the ``[width, height]`` of the target latent in pixels — useful to confirm
    a 4 MP+ target before a 6-step sampling run.
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required":
            {
                "custom_output": ("ANY", ),
            }
        }

    RETURN_TYPES = ("ANY", "IMAGE", "LIST", "LIST", "STRING", "CONDITIONING", "MASK", "LIST", "LIST")
    RETURN_NAMES = ("pad_info", "main_image", "vae_images", "ref_latents", "full_prompt",
                    "no_refs_cond", "mask", "minimax_refs", "target_size")
    FUNCTION = "extract"

    CATEGORY = "advanced/conditioning"

    def extract(self, custom_output):
        pad_info = custom_output.get('pad_info')
        main_image = custom_output.get('main_image')
        vae_images = custom_output.get('vae_images')
        ref_latents = custom_output.get('ref_latents')
        full_prompt = custom_output.get('full_prompt')
        no_refs_cond = custom_output.get('no_refs_cond')
        mask = custom_output.get('mask')
        minimax_refs = custom_output.get('minimax_refs', [])
        target_size = custom_output.get('target_size', [])

        return (pad_info, main_image, vae_images, ref_latents, full_prompt,
                no_refs_cond, mask, minimax_refs, target_size)


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
