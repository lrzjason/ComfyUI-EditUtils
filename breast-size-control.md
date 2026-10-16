# Breast Size Control LoRA (Krea2)

**Draw the size. Don't write it.**

A Krea2 control LoRA that makes breast size a **dial you paint**: one or two filled ellipses on a black canvas, where the marker's **area** (plus its position and angle) drives the result. The caption stays a scene description, with no size words at all.

![Krea2 breast size control example](../images/k2%20breasts%20size%20control.png)

*The graph and its result, produced with [k2 breasts size control.json](../workflows/k2%20breasts%20size%20control.json). The canvas reads `832x1216 - total 2.36% (small) - marker 1/2: 1.06% @ -14.6°`, i.e. two small, slightly tilted markers.*

---

## Introduction

Most "size" attempts at this problem put the size in the **text**: `large breasts`, `huge bust`, and so on. That approach has three structural problems:

1. **The words leak into everything else.** A caption heavy on size vocabulary drags the whole composition with it: body type, clothing fit, framing, even the subject's apparent age.
2. **The scale is not calibrated.** "Large" means something different in every checkpoint and every prompt.
3. **It cannot be continuous.** Text gives you a few coarse buckets, not a value you nudge.

This LoRA moves the control out of the text and into **geometry**. The size lives in a **condition image** (a black frame with one or two `#FADBAE` markers) and the marker's **area fraction** is the dial. The caption is then free to describe only the scene, and the model reads the geometry separately from the language.

The practical result: **the size becomes a number you can measure and repeat**, without ever typing a size word.

**HF download:**

https://huggingface.co/lrzjason/Krea2-Lora-Collection/tree/main/loras

File: `breasts_control_lrzjason_20261015.safetensors`

**EditUtils (required):**

https://github.com/lrzjason/ComfyUI-EditUtils

---

## The condition image is the dial

The `_bd` condition image is not a picture of anything. It is a **measurement**, in the same spirit as a depth map or a pose skeleton: a compact, unambiguous encoding of one variable, in a form the model can read directly.

| What you change | What it controls |
|---|---|
| Marker **area** (total, as % of the frame) | **Size class**: the dial |
| Marker **position** | **Where** the breasts sit in frame |
| Marker **angle** | **Tilt** |
| Marker **count** (1 or 2) | One marker or a pair |
| Marker **shape** (circle or ellipse) | Aspect, matching the real measured box |

Because the area is a *fraction of the frame*, the same drawing means the same thing at `832x1216` and at `1536x1536`: the dial is resolution-independent.

### Six calibrated classes

The class boundaries are not guesses. They were **measured** from a labelled dataset: the per-class **median total marker area** as a fraction of the frame.

| Class | Marker area (% of frame) |
|---|---|
| `flat` | **1.49%** |
| `small` | **2.94%** |
| `medium` | **3.22%** |
| `large` | **7.18%** |
| `huge` | **13.95%** |
| `gigantic` | **16.50%** |

Those are exactly the six numbers on the canvas **preset buttons**, so one click lands you on a class the LoRA has actually seen. Intermediate values are legitimate too, because the dial is continuous, but the presets are the safe anchors.

Note the shape of the ladder: `small` to `medium` is a narrow step (2.94% to 3.22%), while `medium` to `large` to `huge` opens up fast. That is the measured distribution, not a design choice, and it is why eyeballing the size is a bad substitute for setting the area.

### Keeping the marker in-distribution

The LoRA was trained on one specific marker style. Three rules keep you inside it:

- **Background stays black.** No gradients, no textures, no other content.
- **Fill stays bright** (`#FADBAE` by default, or a similar bright colour). Do not fill with a dark tone.
- **One or two ellipses**, area = size. Overlapping markers merge as a **union**, not a sum, which is the behaviour the training data had.

The condition image is forced to the **target's exact pixel size** when you connect an `image` to the node. This matters, because the pipeline bucket-crops the reference, and a size mismatch silently displaces the markers.

---

## Training Details

> **Fill these in before publishing.** The template below is the structure; the bracketed values are the only numbers I could not verify from the repository. Everything outside the brackets is checked against the code and the workflow.

**Stage 1: [resolution] pretraining**
Trained on a [N]-pair subset of the labelled breast-size dataset, with captions generated on the scene only (no size wording) to force the size signal into the condition image rather than the text. Constant learning rate [LR] to establish the area to class mapping.

**Stage 2: [resolution] refinement**
Continued training at higher resolution, focusing on detail reconstruction at the marker boundary (edge, thin structures, fine geometry). Cosine learning rate schedule for stable convergence.

| | |
|---|---|
| Stage 1 | [res], [N] pairs (labelled dataset, scene-only captions), constant LR [LR] |
| Stage 2 | [res] refinement, cosine LR schedule |
| Total epochs | [N]+ |
| Base model | Krea2 (`krea2_raw_int8_convrot`), turbo LoRA rank 64 |
| Condition | `_bd` marker image: area = size, position = where, angle = tilt |
| Caption rule | **scene only, no size words** |

---

## Usage

> ⚠️ **This LoRA must be used together with the EditUtils plugin:**
> 👉 https://github.com/lrzjason/ComfyUI-EditUtils

1. **Install the ComfyUI-EditUtils plugin**, plus [Comfyui-LoraUtils](https://github.com/lrzjason/Comfyui-LoraUtils) for the LoRA merge nodes used in the example.
2. **Load this LoRA** in your workflow: `breasts_control_lrzjason_20261015.safetensors`, strength **1.0**, merged on top of the Krea2 turbo LoRA.
3. **Draw the control.** Add `BreastSizeControl_EditUtils`, drag one or two markers, and hit the preset button for the size you want. Connect a target image to force the exact output size.
4. **Write the scene prompt only.** Subject, clothing, framing, light. **No size words.**
5. **Sample.** `euler / simple`, **8 steps**, CFG **1.0**, denoise 1.0.

### Wiring

```
UNETLoader (krea2_raw_int8_convrot) ──► Krea2EditApply ──► [Merge LoRA: krea2_turbo] ──► [Merge LoRA: breasts_control] ──► KSampler ──► VAEDecode ──► SaveImage
CLIPLoader (qwen3vl_4b_fp8_scaled, type krea2) ┐
VAELoader  (qwen_image_vae) ───────────────────┤
PrimitiveStringMultiline (scene prompt) ───────┼──► EditTextEncode_EditUtils ──► positive + latent ──► KSampler
Krea2ModelConfig ──────────────────────────────┤                                    │
BreastSizeControl ──► QwenConfigPreparer ──────┘                                    └──► ConditioningZeroOut ──► negative
```

**Nodes:**

| Node | Purpose |
|---|---|
| `BreastSizeControl_EditUtils` | The canvas: draws the `_bd` condition image that acts as the reference. **This is the dial.** |
| `Krea2ModelConfig_EditUtils` | Krea2 model config (`model_name="qwen"` route, `vae_unit=8`, built-in instruction). |
| `QwenConfigPreparer_EditUtils` | Wraps the condition image as the reference (`ref_longest_edge=1024`, `ref_crop=pad`, `to_vl`, `vl_target_size=384`). |
| `EditTextEncode_EditUtils` | Encodes prompt + reference into the conditioning, the sampling latent and `pad_info`. |
| `Krea2EditApply_EditUtils` | Patches the model so the reference latent is injected (`ref_pos_match_target`, `ref_kv_cache=1`, `ref_strength=1.0`). |

**Model files:** `krea2_raw_int8_convrot.safetensors` (UNET), `qwen3vl_4b_fp8_scaled.safetensors` (CLIP, type `krea2`), `qwen_image_vae.safetensors` (VAE).

### Canvas tools

Drag empty space to create a marker (max two). Drag the body to move it. Drag the **N/S/E/W edge handles** for `rx`/`ry` independently (this is how a circle becomes an ellipse). Drag a **corner handle** or use the **wheel** to scale uniformly. **Shift** + edge drag for uniform. Drag the **rotation handle** to rotate, or type an exact angle. **Right click** to delete. **Preset buttons** for the six measured class areas. **Colour picker** for the fill. The **guide** checkbox dims the connected image underneath for placement, and never touches the output.

---

## Notes

- **The caption rule is not a suggestion.** If you put size words in the prompt, the text and the reference image compete for the same variable and the result drifts. Keep the text scene-only and let the geometry speak.
- **Higher-resolution inputs yield richer marker geometry**, since the marker's edge is where the extra detail lands.
- **The area is the dial, not a rendering.** The condition image is a control signal, not a target to be drawn literally.
- **Will not work as intended without the EditUtils plugin.**
- **Works best with long, descriptive scene captions**, consistent with the dataset side, where captions were written scene-only.
- **Overlapping markers merge by union**, so two crossing ellipses are not twice the size.
- ⚠️ This control LoRA and the Krea2 edit path are **still in development**, and results depend on the LoRA checkpoint and `ref_strength`.

---

## RH Workflow

**Experience link:** https://www.runninghub.ai/post/2082077636234313729/?inviteCode=rh-v1279

---

## 📬 Contact

- **𝕏 Twitter:** [@Lrzjason](https://x.com/Lrzjason)
- **📧 Email:** lrzjason@gmail.com
- **💬 QQ Group:** 866612947
- **💬 WeChat ID:** fkdeai
- **🎨 CivitAI:** [xiaozhijason](https://civitai.com/user/xiaozhijason)

---

*Intended for responsible, adult-only creative work involving adult subjects.*
