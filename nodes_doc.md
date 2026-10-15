# EditUtils 节点参考文档

> 本文档以 `nodes.py` 当前代码（`NODE_CLASS_MAPPINGS`，45 个节点）为准。
> 显示名均带 `lrzjason` 后缀；节点菜单分类多为 `advanced/conditioning`，图像类节点在 `image`，模型补丁在 `Krea2/edit` / `QwenImage/edit` / `Flux2Klein/edit` / `Flux3Klein/edit`。
> 前端脚本在 `web/js/`（`WEB_DIRECTORY = "./web"`）：`BreastSizeControl_EditUtils` 的画布编辑器，以及 `Flux3KleinBBox_EditUtils` 的 bbox 画布编辑器（`flux3_bbox_editor.js`）。
> 若文档与代码不符，以代码为准。

## 目录

- [一、模型配置节点（Model Config）](#一模型配置节点model-config)
- [二、文本/图像编码节点（Text Encode）](#二文本图像编码节点text-encode)
- [三、Config Preparer（configs 链）](#三config-preparerconfigs-链)
- [四、Output Extractor（custom_output 拆包）](#四output-extractorcustom_output-拆包)
- [五、模型补丁（Krea2 / QwenImage / QwenImage 2.1 / Flux2Klein / Flux3Klein / Boogu）](#五模型补丁krea2--qwenimage--qwenimage-21--flux2klein--flux3klein--boogu)
- [六、条件缓存（Save/Load Condition）](#六条件缓存saveload-condition)
- [七、图像处理与杂项工具](#七图像处理与杂项工具)

---

## 一、模型配置节点（Model Config）

输出均为 `model_config` (DICT)，供 `EditTextEncode_EditUtils` 的 `model_config` 输入使用。

### ModelConfig_EditUtils

**显示名**：EditUtils: Model Config lrzjason
通用模型配置节点，可选 qwen / flux2klein / flux3klein / boogu。

| 输入 | 类型 | 默认 | 说明 |
|---|---|---|---|
| model_choice | COMBO: `qwen`/`flux2klein`/`flux3klein`/`boogu` | `qwen` | 模型选择 |
| model_name | STRING | `""` | 留空时使用 model_choice |
| vae_unit | INT (8–64, step 8) | 8 | VAE 对齐单位；Qwen=8，Flux2Klein=16，Flux3Klein=16，Boogu=8 |
| instruction | STRING (optional, multiline) | `""` | 自定义 system prompt；留空用内置默认。boogu 忽略此项 |

- 输出：`model_config` (DICT) — `{model_name, vae_unit, llama_template}`。
- boogu 时 `llama_template` 强制为 `""`（Boogu tokenizer 自动选择 system prompt）。
- 注意：vae_unit 默认值是 8，选择 flux2klein / flux3klein 时需手动改为 16；建议直接用下方各模型专用 Config 节点。

### QwenModelConfig_EditUtils

**显示名**：EditUtils: Qwen Model Config lrzjason
Qwen 专用配置：固定 `model_name="qwen"`、`vae_unit=8`。

| 输入 | 类型 | 默认 | 说明 |
|---|---|---|---|
| instruction | STRING (optional, multiline) | `""` | 自定义 system prompt，留空用内置默认 |

- 输出：`model_config` (DICT)，含 `llama_template`（llama chat 模板）。

### Flux2KleinModelConfig_EditUtils

**显示名**：EditUtils: Flux2Klein Model Config lrzjason
Flux2Klein 专用配置：固定 `model_name="flux2klein"`、`vae_unit=16`。

| 输入 | 类型 | 默认 | 说明 |
|---|---|---|---|
| instruction | STRING (optional, multiline) | `""` | 留空时 `llama_template=""`（推荐）；非空时构造 llama 模板（允许自定义但不推荐） |

### Flux3KleinModelConfig_EditUtils

**显示名**：EditUtils: Flux3Klein Model Config lrzjason
FLUX.3 Klein（base 与 step-distilled）专用配置：固定 `model_name="flux3klein"`、`vae_unit=16`、`llama_template=""`。

| 输入 | 类型 | 默认 | 说明 |
|---|---|---|---|
| instruction | STRING (optional, multiline) | `""` | 留空时 `llama_template=""`（推荐，tokenizer 自带 instruct 模板并把文本窗口右填充到 80 的倍数）；非空时构造自定义模板 |
| flow_map | BOOLEAN (optional) | `False` | 标记该 checkpoint 是否为 step-distilled（flow-map）。**仅作信息记录**：ComfyUI 会自行从 checkpoint 识别 `flow_map`。为 `True` 时应使用约 4–16 步、CFG 1.0（无负向提示词） |

- 输出：`model_config` (DICT)。
- **FLUX.3 klein 不是 FLUX.2 klein 的改名**，而是另一套架构：per-mode 模块 + joint single 模块（image / image_cond / text 三种 token 模式）、Qwen3-VL 文本编码器（8 层堆叠 `[4,8,12,16,20,24,28,32]` → `joint_attention_dim=20480`）、参考图以**负的 `l` 轴坐标**（-1, -2, …）区分、图像 H/W RoPE 位置按 `image_rope_scale=0.5` 缩放；distilled 变体是 flow-map 模型（`flow_map=True`）。
- **VAE**：与 FLUX.2 相同的 packed autoencoder（32 潜通道、8× 下采样、2×2 patchify），因此每 token 对应 16 px，`vae_unit=16`。它的归一化是 VAE 自身的 batch-norm（没有标量 `scaling_factor`/`shift_factor`），与 ComfyUI 的 Flux2 latent format 一致。
- **ComfyUI 已原生支持**：`comfy/ldm/flux3/model.py`、`comfy/model_base.py class Flux3`、`comfy/supported_models.py class Flux3`（`image_model="flux3"`）。因此参考潜变量走标准 `reference_latents` 条件键，模型自己推导参考位置 id，**不需要** 重写 forward。

### BooguModelConfig_EditUtils

**显示名**：EditUtils: Boogu Model Config lrzjason
Boogu 专用配置：固定 `model_name="boogu"`、`vae_unit=8`、`llama_template=""`。

| 输入 | 类型 | 默认 | 说明 |
|---|---|---|---|
| instruction | STRING (optional, multiline) | `""` | **被忽略**（Boogu tokenizer 自动选择 system prompt） |

### Krea2ModelConfig_EditUtils

**显示名**：EditUtils: Krea2 Model Config lrzjason
Krea2 管线专用配置。固定返回 `{"model_name": "qwen", "vae_unit": 8, "config_for": "krea2", "llama_template": ...}`。

| 输入 | 类型 | 默认 | 说明 |
|---|---|---|---|
| instruction | STRING (optional, multiline) | `"Describe the image by detailing the color, shape, size, texture, quantity, text, spatial relationships of the objects and background:"` | 已内置 Krea2 推荐 instruction；留空时回退到该默认值 |

- 输出：`model_config` (DICT)。
- **为什么 model_name 是 `"qwen"`**：Krea2 的文本编码器基于 Qwen、VAE 是 Qwen-Image 的 VAE，因此 `EditTextEncode` 应走 qwen 编码分支（vae_unit=8）；`config_for: "krea2"` 仅作来源标识（日志/调试用）。
- 配合 Krea2 管线使用：本节点 → `EditTextEncode` → `Krea2EditApply` 打补丁的模型。

### QwenImage21ModelConfig_EditUtils

**显示名**：EditUtils: QwenImage 2.1 Model Config lrzjason
Qwen-Image 2.1 专用配置：固定 `model_name="qwen_image21"`、`vae_unit=16`、`llama_template=""`。

| 输入 | 类型 | 默认 | 说明 |
|---|---|---|---|
| instruction | STRING (optional, multiline) | `""` | 留空时 `llama_template=""`（推荐，tokenizer 自动构造 “Comprehend and analyze the provided prompt.” 模板）；非空时构造自定义 system prompt 模板（编码时自动在 `{}` 处插入 vision block） |

- 输出：`model_config` (DICT)。
- **为什么 vae_unit=16**：Qwen-Image 2.1 的 VAE 是 64 通道、16 倍空间下采样，因此参考图要对齐到 16 像素倍数（`QwenImage21EditApply_EditUtils` 内部同样按 16 像素 / latent token 换算 rope 偏移）。
- 与 Qwen（1.0）不同：VL 视觉塔输入就是缩放后的参考图本身（无独立 vl 缩放管线）；ref latent 会被拼接进文本序列的 vision slot（`image_slots`）。
- 需要已支持 Qwen-Image 2.1 的 ComfyUI（上游 `qwen_image21` 支持）才能加载模型。

---

## 二、文本/图像编码节点（Text Encode）

### EditTextEncode_EditUtils

**显示名**：EditUtils: EditTextEncode lrzjason
核心统一编码节点。按 `model_config.model_name` 分流到 Qwen / Flux2Klein / Flux3Klein（共用代码路径）或 Boogu（独立早返回路径）逻辑。

| 输入 | 类型 | 默认 | 说明 |
|---|---|---|---|
| clip | CLIP | — | 文本编码器 |
| vae | VAE | — | 参考图编码 |
| prompt | STRING (multiline) | — | 编辑指令 |
| model_config | DICT | None | 来自 ModelConfig 系列节点 |
| configs | LIST (optional) | None | Config Preparer 串联出的图像配置列表；不接或为空时执行纯文本编码 |

- 输出 7 个：`conditioning` (CONDITIONING) · `latent` (LATENT) · `custom_output` (ANY dict) · `main_image` (IMAGE) · `mask` (MASK) · `pad_info` (ANY dict) · **`prompt` (STRING)**。
- **`prompt` 输出**：**实际送进 tokenizer 的那段文本**，直接接线到文本预览 / SaveText 节点即可核对模型到底收到了什么。
  - `flux3klein`：用户 prompt **+ bbox 元素表**（与 `custom_output["composed_prompt"]` 完全一致）；
  - `qwen` / `qwen_image21` / `flux2klein`：带 `Picture n: <|vision_start|>…` 前缀的 prompt；
  - `boogu`：原样 prompt（Boogu tokenizer 自行选择 system prompt）。
  - 该输出**追加在最后一位**：ComfyUI 按槽位序号匹配已保存的连线，放在末尾可保证现有工作流的输出连接不受影响。

**Qwen / Flux2Klein / Flux3Klein 路径行为**
- 对每张 `to_ref` 的图：按 `ref_longest_edge` + `ref_resize_mode` 缩放，按 `vae_unit` 对齐，`ref_crop="pad"` 时补齐到 vae_unit 倍数（主图生成 `pad_info`），然后 `vae.encode` 得到 ref latent。补齐区域的填充方式由 config 的 `pad_mode` / `pad_color` / `pad_color_rgb` / `pad_noise` 控制（qwen 系默认白边，flux2klein / flux3klein 默认黑边）。
- 全部 ref latents 以 `reference_latents` 键挂到 conditioning 上（`node_helpers.conditioning_set_values`），采样模型据此读取。**这就是 EditUtils 条件链。**
- `latent` 输出 = 主图（`ref_main_image`）的 ref latent；无图时为 1×4×128×128 零 latent（qwen_image21 为 1×64×128×128，flux3klein 为 1×128×128×128）。
- Qwen 的 `to_vl` 图还会缩放到 `vl_target_size` 面积喂给 QwenVL，并在 prompt 前自动拼 `Picture n: <|vision_start|>...`；Flux2Klein 忽略 VL（代码中 `to_vl` 强制 False）。
- **Flux3Klein 额外**：`to_vl` 同样强制 False（参考图是纯 VAE 潜变量、没有 vision slot）；不写 `reference_rope_offsets`（FLUX.3 自己推导参考位置 id）；若某个 config 带 bbox 元素表，会把该表拼到 prompt 末尾，并在 `custom_output` 里给出 `composed_prompt` 与 `bbox_rows`。
- 主图带 mask 时，`latent` 附带 `noise_mask`，并从 `mask` 输出返回。
- 只会有一个主图：后续 config 的 `ref_main_image` 会被强制改为 False；都不设主图时自动取第 0 张并打印提示。

**QwenImage21 路径行为**（`model_name="qwen_image21"`）
- 每张 `to_ref` 的图按 `ref_longest_edge` + `ref_resize_mode` 缩放并对齐到 **16 像素倍数**（`vae_unit=16`，即 16 倍 VAE 下采样）。
- **VL 输入 = 缩放后的参考图本身**（无独立 vl 缩放/裁剪管线）：RGB 通道做白底 alpha 合成喂 Qwen3-VL 文本编码器，完整 RGBA（RGB 输入由 ComfyUI 自动补不透明 alpha）交给 VAE 编码。
- token 化时不拼 `Picture n:` 前缀：`llama_template` 为空时由 QwenImage21 tokenizer 自动构造 T2I 模板并插入 `<image n><|vision_start|>...`；自定义模板时在 `{}` 处自动插入 vision block。
- 无 ref latent（纯文本/to_vl only）时 `keep_vision=True`，vision embedding 作为纯文本条件保留；有 ref latent 时不传 vision token、由文本编码器记录 `image_slots`（vision slot 在文本序列中的位置）， DiT 在这些位置拼接参考 latent。
- rope 偏移随条件链（`reference_rope_offsets`）传递，配合 `QwenImage21EditApply_EditUtils` 生效。
- 需要 ComfyUI 已内置 Qwen-Image 2.1 支持（上游 `qwen_image21`）。

**Boogu 路径行为**（早返回）
- prompt 与 vl 图一起 `clip.tokenize(prompt, images=images_vl)`（Boogu tokenizer 自动选 system prompt，忽略 llama_template）。
- ref latents 同时挂到 positive 和 negative conditioning 上（CFG 下相互抵消参考）。
- negative 从 `model_config["negative_prompt"]` 编码（由 `BooguEditTextEncode` 写入；直接连 `BooguModelConfig` 时 negative 为空字符串编码）。
- `latent` 输出 = 第 0 张 ref latent；`custom_output` 含 `negative_cond/ref_latents/vl_images/vae_images`；`mask` 输出为 None。

### QwenEditTextEncode_EditUtils

**显示名**：EditUtils: Qwen Edit Text Encode lrzjason
Qwen 简易一体化编码节点（内部构造固定 config 后调用 EditTextEncode），最多 3 张图。

| 输入 | 类型 | 默认 | 说明 |
|---|---|---|---|
| clip | CLIP | — | |
| vae | VAE | — | |
| prompt | STRING (multiline) | — | |
| image1/2/3 | IMAGE (optional) | None | 至少接 image1，否则报错 |
| ref_longest_edge | INT (8–4096) | 1024 | 参考图最长边 |
| mask | MASK (optional) | None | 仅作用于 image1 |

- 内部固定参数：`ref_crop="pad"`、`ref_upscale="lanczos"`、VL 开（resize 开、vl_target_size=384、center/bicubic 中 upscale 实为 `lanczos`）、image1 为主图。
- 输出：`conditioning` / `latent` / `custom_output` / `main_image` / `mask`（与 EditTextEncode 一致，无 `pad_info` 输出端口——它在 custom_output 里）。

### QwenImage21EditTextEncode_EditUtils

**显示名**：EditUtils: QwenImage 2.1 Edit Text Encode lrzjason
Qwen-Image 2.1 简易一体化编码节点（内部构造固定 config 后调用 EditTextEncode 的 qwen_image21 分支），最多 3 张图。

| 输入 | 类型 | 默认 | 说明 |
|---|---|---|---|
| clip | CLIP | — | Qwen3-VL 文本编码器 |
| vae | VAE | — | Qwen-Image 2.1 VAE（64 通道，16 倍下采样） |
| prompt | STRING (multiline) | — | 编辑指令 |
| image1/2/3 | IMAGE (optional) | None | 至少接 image1，否则报错 |
| ref_longest_edge | INT (16–4096, step 16) | 1024 | 参考目标尺寸（自动对齐 16 像素倍数） |
| mask | MASK (optional) | None | 仅作用于 image1 |

- 内部固定参数：`ref_crop="pad"`、`ref_upscale="lanczos"`、`to_vl=True`、`vae_unit=16`、`llama_template=""`（自动 T2I 模板）、image1 为主图。
- 输出：`conditioning` / `latent` / `custom_output` / `main_image` / `mask`。
- 需要已支持 Qwen-Image 2.1 的 ComfyUI。

### Flux2KleinEditTextEncode_EditUtils

**显示名**：EditUtils: Flux2Klein Edit Text Encode lrzjason
Flux2Klein 简易一体化编码节点（vae_unit=16），无 VL 处理。

| 输入 | 类型 | 默认 | 说明 |
|---|---|---|---|
| clip | CLIP | — | |
| vae | VAE | — | |
| prompt | STRING (multiline) | — | |
| image1/2/3 | IMAGE (optional) | None | 至少接 image1 |
| ref_longest_edge | INT (8–4096) | 1024 | |
| mask | MASK (optional) | None | 仅作用于 image1 |

- 输出：`conditioning` / `latent` / `custom_output` / `main_image` / `mask`。

### Flux3KleinEditTextEncode_EditUtils

**显示名**：EditUtils: Flux3Klein Edit Text Encode lrzjason
FLUX.3 Klein 简易一体化编码节点（vae_unit=16），无 VL 处理。

| 输入 | 类型 | 默认 | 说明 |
|---|---|---|---|
| clip | CLIP | — | FLUX.3 用 Qwen3-VL 文本编码器（CLIPLoader type `flux3`） |
| vae | VAE | — | FLUX.2 packed VAE（128ch @ 16x） |
| prompt | STRING (multiline) | — | |
| image1/2/3 | IMAGE (optional) | None | 至少接 image1；image1 为主图 |
| ref_longest_edge | INT (16–4096) | 1024 | 主图 latent 网格的最长边 |
| ref_longest_edge2 | INT (16–4096) | 1024 | 参考图 2 的最长边 |
| ref_longest_edge3 | INT (16–4096) | 1024 | 参考图 3 的最长边 |
| mask | MASK (optional) | None | 仅作用于 image1 |

- 输出：`conditioning` / `latent` / `custom_output` / `main_image` / `mask`（**5 个**。内部调用 `EditTextEncode` 后截取前 5 个输出）。
- 每个参考图保留自己的分辨率（各自一个 token 网格与自己的负 `l` 位置 id），不像 Qwen-Image 2.1 那样强制统一尺寸。
- **参考图是纯 VAE 潜变量**，没有 vision slot：`to_vl` 在该路径被强制关闭。
- 需要完整 7 个输出（含 `pad_info` 与拼好元素表的 `prompt`）时，请直接使用 `EditTextEncode_EditUtils`。

### BooguEditTextEncode_EditUtils

**显示名**：EditUtils: Boogu Edit Text Encode lrzjason
Boogu 简易一体化编码节点，直接输出正负双条件。

| 输入 | 类型 | 默认 | 说明 |
|---|---|---|---|
| clip | CLIP | — | |
| vae | VAE | — | |
| prompt | STRING (multiline) | — | |
| negative_prompt | STRING (optional, multiline) | `""` | 负面提示词 |
| image1/2/3 | IMAGE (optional) | None | 至少接 image1 |
| ref_longest_edge | INT (16–4096) | 1024 | |
| mask | MASK (optional) | None | 仅作用于 image1 |
| ref_resize_mode | COMBO: `longest_edge`/`area` | `longest_edge` | 缩放模式 |

- 输出：`positive` (CONDITIONING) · `negative` (CONDITIONING) · `latent` · `custom_output` · `main_image`。
- 注意：ref latents 同时挂在 positive 和 negative 上。

---

## 三、Config Preparer（configs 链）

每个节点把一张图 + 处理参数打包成一个 config dict，append 到传入的 `configs` 列表后再输出，多个 Preparer 首尾串联即可支持任意数量参考图。最终 `configs` 接到 `EditTextEncode_EditUtils.configs`。
所有 Preparer 的 `mask` 若与 `image` 尺寸不一致会被丢弃并打印提示。

### QwenConfigPreparer_EditUtils

**显示名**：EditUtils: Qwen Config Preparer lrzjason

| 输入 | 类型 | 默认 | 说明 |
|---|---|---|---|
| image | IMAGE | — | |
| configs | LIST (optional) | None | 上游 configs 链 |
| to_ref | BOOLEAN | True | 是否进参考 latent |
| ref_main_image | BOOLEAN | True | 是否主图（其 latent 作为采样初始 latent，并生成 pad_info） |
| ref_longest_edge | INT (8–4096) | 1024 | |
| ref_crop | COMBO: `pad`/`center`/`disabled` | `pad` | pad=补齐到 vae_unit 倍数（填充方式见 pad_mode/pad_color/pad_color_rgb） |
| ref_upscale | COMBO: `lanczos`/`bicubic`/`area` | `lanczos` | |
| to_vl | BOOLEAN | True | 是否进 QwenVL 编码 |
| vl_resize | BOOLEAN | True | VL 前是否缩放 |
| vl_target_size | INT (384–2048) | 384 | VL 目标面积基准 |
| vl_crop | COMBO: `center`/`disabled` | `center` | |
| vl_upscale | COMBO: `lanczos`/`bicubic`/`area` | `lanczos` | |
| mask | MASK (optional) | None | |
| ref_resize_mode | COMBO: `longest_edge`/`area` | `longest_edge` | longest_edge=最长边对齐；area=总面积对齐 ref_longest_edge² |
| pad_mode | COMBO: `color`/`edge`/`mirror`/`border_mean` | `color` | ref_crop=pad 时填充区域的生成方式 |
| pad_color | COMBO: `red`/`green`/`blue`/`white`/`black`/`custom` | `white` | pad_mode=color 时的填充色。qwen 默认白色：黑边会在生成时被放大 |
| pad_color_rgb | STRING | `-` | 可选覆盖：6 位 hex，`#` 可有可无（如 `FF0000` 或 `#FF0000`）。合法时覆盖 pad_color；`-`/长度不对/非 hex 等无效值不生效 |
| pad_noise | FLOAT (0–0.2) | 0.0 | **只**加到填充区域的高斯抖动（std），图像内容不受影响；0.02–0.05 可打散纯平区域 |

- 输出：`configs` (LIST) · `config` (ANY，当前图的配置)。

### QwenImage21ConfigPreparer_EditUtils

**显示名**：EditUtils: QwenImage 2.1 Config Preparer lrzjason
Qwen-Image 2.1 每图配置。**无独立 VL 参数**（VL 输入就是缩放后的参考图本身）。

| 输入 | 类型 | 默认 | 说明 |
|---|---|---|---|
| image | IMAGE | — | |
| configs | LIST (optional) | None | 上游 configs 链 |
| to_ref | BOOLEAN | True | 是否进参考 latent |
| ref_main_image | BOOLEAN | True | 是否主图 |
| ref_longest_edge | INT (16–4096, step 16) | 1024 | 对齐到 16 像素倍数（16 倍 VAE 下采样） |
| ref_crop | COMBO: `pad`/`center`/`disabled` | `pad` | |
| ref_upscale | COMBO | `lanczos` | |
| to_vl | BOOLEAN | True | 是否把缩放后的参考图喂给 Qwen3-VL 并在其 vision slot 拼接 latent。**建议保持开启**（关闭走未训练路径） |
| mask | MASK (optional) | None | |
| ref_resize_mode | COMBO: `longest_edge`/`area` | `longest_edge` | |
| rope_x_offset / rope_y_offset | INT (0–4096, step 16) | 0 | ROPE 位置偏移（像素，16 对齐）；需配合 `QwenImage21EditApply_EditUtils` |
| pad_mode | COMBO: `color`/`edge`/`mirror`/`border_mean` | `color` | ref_crop=pad 时填充区域的生成方式 |
| pad_color | COMBO: `red`/`green`/`blue`/`white`/`black`/`custom` | `white` | pad_mode=color 时的填充色。2.1 默认白色：模型用黑边训练，黑边会在生成时被放大 |
| pad_color_rgb | STRING | `-` | 可选覆盖：6 位 hex，`#` 可有可无（如 `FF0000` 或 `#FF0000`）。合法时覆盖 pad_color；无效值不生效 |
| pad_noise | FLOAT (0–0.2) | 0.0 | **只**加到填充区域的高斯抖动（std），图像内容不受影响；0.02–0.05 可打散纯平区域 |

- 输出：`configs` (LIST) · `config` (ANY)。
- `to_vl=True` 但 `to_ref=False` 会在编码时被强制改回 `to_ref=True`（并打印提示），否则 vision slot 对不齐。

### Flux2KleinConfigPreparer_EditUtils

**显示名**：EditUtils: Flux2Klein Config Preparer lrzjason
与 Qwen 版相同但无 VL 参数（Flux2Klein 不走 VL）。`ref_longest_edge` 范围 16–4096。其余同 QwenConfigPreparer（to_ref / ref_main_image / ref_longest_edge=1024 / ref_crop=`pad` / ref_upscale=`lanczos` / mask / ref_resize_mode=`longest_edge`）。

### Flux3KleinConfigPreparer_EditUtils

**显示名**：EditUtils: Flux3Klein Config Preparer lrzjason
FLUX.3 Klein 的每图配置，并承载 **FLUX.3 的 bounding box（元素表）画布**。

常规参数（`ref_crop` 默认 `pad`，`ref_longest_edge` 16–4096）：

| 输入 | 类型 | 默认 | 说明 |
|---|---|---|---|
| image | IMAGE | — | |
| configs | LIST (optional) | None | 串联多个 Preparer 以使用多张参考图 |
| to_ref / ref_main_image | BOOLEAN | True / True | |
| ref_longest_edge | INT (16–4096) | 1024 | 该参考图 latent 网格的最长边 |
| ref_crop | COMBO: `pad`/`center`/`disabled` | `pad` | |
| ref_upscale | COMBO: `lanczos`/`bicubic`/`area` | `lanczos` | |
| pad_mode | COMBO: `color`/`edge`/`mirror`/`border_mean` | `color` | 仅 `ref_crop=pad` 时生效 |
| pad_color | COMBO: `red`/`green`/`blue`/`white`/`black`/`custom` | `black` | |
| pad_color_rgb | STRING | `-` | 6 位十六进制（`#` 可省），有效时覆盖 `pad_color` |
| pad_noise | FLOAT (0–0.2) | 0.0 | 仅加到填充区 |
| mask | MASK (optional) | None | |
| ref_resize_mode | COMBO: `longest_edge`/`area` | `longest_edge` | |
| rope_x_offset / rope_y_offset | INT (0–4096, step 16) | 0 | **无效果**，仅为图兼容保留（FLUX.3 自己推导参考位置 id） |

- 输出 2 个：`configs` (LIST) · `config` (ANY)。
- `to_vl` **不存在**：FLUX.3 文本编码器只吃文本，参考图是纯 VAE 潜变量。
- 本节点**不再包含 bbox 画布**：绘制与元素表已拆分到独立的 `Flux3KleinBBox_EditUtils`（见下），这样 Preparer 保持简短。
- 每个 config 会带一个稳定的 `config_id`（字符串）。因为 Preparer 会对传入链做 `copy.deepcopy`，下游链里保存的是上游 config 的**副本**，bbox 节点靠这个 id 才能认出"这张图的 config"，否则会把同一张图重复追加成第二个参考。

### Flux3KleinBBox_EditUtils

**显示名**：EditUtils: Flux3Klein BBox (Element Table) lrzjason
FLUX.3 的 **bounding box（元素表）画布节点**，独立于 Config Preparer。

| 输入 | 类型 | 默认 | 说明 |
|---|---|---|---|
| configs | LIST (optional) | None | 其它配置节点搭好的 config 链 |
| config | ANY (optional) | None | **单张图**的 config（Preparer 的 `config` 输出）；元素表挂到它上面 |
| image | IMAGE (optional) | — | **仅作绘制底图**，不参与编码 |
| bbox_mode | COMBO: `generation`/`editing` | `generation` | 选择元素表的行结构（**没有 `disabled`**：本节点就是为了生成元素表，接上就应该产出；要临时停用请用 ComfyUI 的 bypass） |
| bbox_rows | STRING（画布托管，隐藏） | `""` | 画布写入的行 JSON |
| bbox_ids | STRING（隐藏镜像） | `""` | 每行一个 id，供工作流往返恢复 |
| bbox_descs | STRING（隐藏镜像） | `""` | 行描述 JSON 数组 |

- 输出 2 个：`configs` (LIST) · **`bbox_json` (STRING)**。**没有单独的 `config` 输出**——`configs` 里已经包含"被装饰的那条 config + 元素表"，两者是一起打包的，再输出一个 config 只会造成同一份数据的第二个真相来源。
- **合并规则**：把元素表挂到 `config` 上，然后在 `configs` 中**替换**与它匹配的那一条（先按对象身份匹配，再按 `config_id` 匹配）；找不到才追加。因此把 Preparer 自己的 `configs` 输出接到这里**不会重复发送同一张图**。
- **没有 `config` 输入时**（`generation` 模式的正常用法：纯文本 + 框，完全不需要参考图）：
  - **画了框** → 追加一条**只带元素表的 config**（`bbox_table_only=True`，没有 `image`）。编码节点会跳过它的图像处理、但照常读取它的行，所以**布局依然会进 prompt**。此时 `latent` 是纯文本兜底（flux3 为 `1×128×128×128`）；如需指定尺寸，可另接空 latent 节点。
  - **没画框** → 完全透传 `configs`，`bbox_json` 为 `""`。
- 只接 `config`、不接 `configs` → 直接把装饰后的 config 追加成一条。
- **`bbox_json` 输出**：整条 config 链的元素表，**JSON 数组字符串**，可直接喂给文本预览 / SaveText / 你自己的节点：
  ```
  [{"id": "wolf_1", "from": null, "src_bbox": null, "tgt_bbox": [817, 120, 950, 167], "desc": "a cyberpunk dog"},
   {"id": "log_1", "from": null, "src_bbox": null, "tgt_bbox": [850, 600, 950, 850], "desc": "a fallen bamboo leaf made out of gold"}]
  ```
  - **没有任何行时返回空字符串 `""`**（不是 `"[]"`）：`bbox_mode=disabled`、开了模式但没画框、或行不可用（无 id / 无有效 bbox）都属于这种情况。
  - 字段严格按模式输出：`generation` 为 `{id, bbox, desc}`；`editing` 为 `{id, from, src_bbox, tgt_bbox, desc}`（`from` / `src_bbox` / `tgt_bbox` 可为 `null`，这正是 New / Remove 行的表达方式）。内部的 `_kind` 不会出现在输出里。
  - 该输出**追加在最后一位**（第 3 个接口），因此现有工作流的 `configs` / `config` 连线不受影响。
  - **链式合并**：串联多个 bbox 节点时，下游节点输出的是**整条链**的行（上游在前、本节点在后），与送进 prompt 的表格完全一致；只接一个节点时就是它自己的行。
  - **混合模式**：一条链里同时存在 `generation` 与 `editing` 表时不会崩溃——每一行都按它自己的结构输出；但 FLUX.3 是按"每次一个一致的表"训练的，所以编码节点会打印提示，建议保持一致。

#### FLUX.3 元素表（bbox）画布

在参考图上**画框**，框旁填 **id（单行）** 与 **desc（多行）**；坐标严格遵循 FLUX.3 规范：`[top, left, bottom, right]`，整数 0–1000，从左上角起算，因此与图片尺寸/宽高比无关。

| 模式 | 每行字段 | 用途 |
|---|---|---|
| `generation` | `{"id", "bbox", "desc"}` | 布局生成："画一个框，描述框里放什么" |
| `editing` | `{"id", "from", "src_bbox", "tgt_bbox", "desc"}` | 编辑：每行 保留/移动/新建/删除 一个元素 |

`editing` 的四种行（画布上的 **row** 按钮一键切换，并自动保持一致）：

| 行 | from | src_bbox vs tgt_bbox | 效果 |
|---|---|---|---|
| Keep | `ref_image_0` | 相等 | 原地不动 |
| Move | `ref_image_0` | 不同 | 移动/缩放 |
| New | `null` | `src_bbox=null` | 在 tgt 框内按 desc 生成（新增/替换/改色） |
| Remove | `ref_image_0` | `tgt_bbox=null` | 删除该元素 |

**画布操作**：空白处拖拽画框；拖框体移动；拖角点缩放；**右键**删除该行；`+ Box` 新增行；`Full frame` 新增 `0,0,1000,1000` 背景行；`Copy caption` 复制 `<id>` 说明文字；**`Refresh`** 重新测量并重绘画布与行表格（画布空白或显示陈旧时点一下即可；该操作不会破坏已有的行与选中状态）。`editing` 模式下用 **edit box**（source / target）选择当前在改哪个框：source 为实线（元素现在的位置），target 为虚线（目标位置）；`Remove` 行显示红色 `REMOVE`。`image` 复选框把连接的图像半透明垫底（**只作参考，绝不进入输出**），`labels` 控制是否在框上画 id。

#### 行布局

每个元素是一张卡片，字段各占一整行（**全宽**，长 id / 引用 / 描述都不会被截断）：

| `generation` | `editing` |
|---|---|
| `id` | `id` |
| `desc` | `source_ref` |
| `bbox` 读数 | `item desc`（位于 **`source_ref` 下方**） |
| | `src_bbox` / `tgt_bbox` 读数 |

卡片标题栏显示行号、**行类型**（`Keep` / `Move` / `New` / `Remove`，带颜色）以及右侧的操作按钮。两种模式下描述字段都标为 `item desc`：`desc` 描述的是元素本身，对**所有**行类型（Keep / Move / New / Remove）都适用，标成 `move desc` 会让人误以为只有 Move 行才有描述。`editing` 模式下它的具体含义（编辑之后该元素的样子）写在悬停提示与占位文字里。

#### 删除：两种不同的操作

两者刻意分开，因为一个会改变图像，另一个不会。

| 操作 | 位置 | 作用 |
|---|---|---|
| **Remove** | 每行按钮（`editing` 模式） | **从图像中删除该元素**。导出 `{from, src_bbox, tgt_bbox: null}`，模型会用元素背后的内容填充该框。按钮变红并显示 `Removing`；再次点击即可恢复你原先画的目标框。 |
| **Delete row** | 每行按钮 | 仅从编辑器表格中删除该行，**不导出任何内容**。用于丢弃画错的行。 |
| **右键** | 画布上 | 等同于对光标所在框执行 *Delete row*（仅编辑器）。 |

因此：要让元素**从画面中消失**，用 **Remove**；要丢掉一行画错的行，用 **Delete row**。Remove 需要一个来源，所以该行必须填写 `source_ref`——若为空会自动填入 `ref_image_0`。

**拼进 prompt**：FLUX.3 没有单独的 box 参数——元素表以 JSON 数组**追加在描述之后**，即 `"<描述> [ {...}, {...} ]"`。编码节点会自动完成拼接，并把拼好的完整 prompt 通过 `prompt` 输出（第 7 个接口）以及 `custom_output` 的 `composed_prompt` 输出，便于核对。

**提示**：id 要在描述里以 `<id>` 引用（如 `<silhouette_1>`）；框是**引导而非裁剪遮罩**，元素可以略微超出；新建元素不要给太小的框（约 40×25 px 常不出现）；`editing` 时说明与表格要一致，需要固定不动的元素就补一行 Keep。

### BooguConfigPreparer_EditUtils

**显示名**：EditUtils: Boogu Config Preparer lrzjason

| 输入 | 类型 | 默认 | 说明 |
|---|---|---|---|
| image | IMAGE | — | |
| configs | LIST (optional) | None | |
| to_ref / ref_main_image | BOOLEAN | True / True | |
| ref_longest_edge | INT (16–4096) | 1024 | |
| ref_crop | COMBO: `pad`/`center`/`disabled` | `pad` | |
| ref_upscale | COMBO | `lanczos` | |
| to_vl | BOOLEAN | True | 是否进 Boogu vision tower |
| vl_target_size | INT (384–2048) | 384 | |
| vl_crop | COMBO: `center`/`disabled` | `center` | |
| vl_upscale | COMBO | `lanczos` | |
| mask | MASK (optional) | None | |
| ref_resize_mode | COMBO: `longest_edge`/`area` | `longest_edge`（**注意：函数签名默认是 `area`，widget 默认以代码 INPUT_TYPES 为准**） | |

- 输出：`configs` (LIST) · `config` (ANY)。
- ⚠️ 代码中 `prepare_config` 的 `ref_resize_mode` 参数默认值为 `"area"`，与 widget 默认值 `"longest_edge"` 不一致；ComfyUI 会传 widget 值，实际以 widget 为准（以代码为准）。

### ConfigJsonParser_EditUtils

**显示名**：EditUtils: Config Json Parser lrzjason
用 JSON 字符串描述一张图的处理配置（高级用户向）。

| 输入 | 类型 | 默认 | 说明 |
|---|---|---|---|
| image | IMAGE | — | |
| configs | LIST (optional) | None | |
| config_json | STRING (optional, multiline) | 内置默认 JSON | 覆盖默认配置的 JSON |
| mask | MASK (optional) | None | |

- 内置默认 JSON：`{"to_ref": true, "ref_main_image": false, "ref_longest_edge": 1024, "ref_crop": "center", "ref_upscale": "lanczos", "to_vl": true, "vl_resize": true, "vl_target_size": 384, "vl_crop": "center", "vl_upscale": "bicubic", "mask": null}`（注意这里 ref_main_image 默认 **false**、ref_crop 默认 **center**，与各模型专用 Preparer 不同）。
- `pad_mode` / `pad_color` / `pad_color_rgb` / `pad_noise` **不写进内置模板**：本节点与模型无关，交由 EditTextEncode 按模型兜底（qwen / qwen_image21 白边，其余黑边）。需要覆盖时在 JSON 里显式加上这些键即可（`pad_color` 取 `red`/`green`/`blue`/`white`/`black`/`custom`，`pad_color_rgb` 为 6 位 hex、`#` 可有可无）。
- `config_json` 为空或非法 JSON 时，会打印错误并退回内置默认配置（不再抛异常）。
- 输出：`configs` (LIST) · `config` (ANY)。

---

## 四、Output Extractor（custom_output 拆包）

### QwenEditOutputExtractor_EditUtils

**显示名**：EditUtils: Qwen Edit Output Extractor lrzjason

- 输入：`custom_output` (ANY)。
- 输出 11 个：`pad_info` (ANY) · `full_refs_cond` (CONDITIONING) · `main_ref_cond` (CONDITIONING，**当前代码恒为 None**，main_ref_cond 相关逻辑已被注释掉) · `main_image` (IMAGE) · `vae_images` (LIST) · `ref_latents` (LIST) · `vl_images` (LIST) · `full_prompt` (STRING) · `llama_template` (STRING，custom_output 中已不写入，通常为 None) · `no_refs_cond` (CONDITIONING) · `mask` (MASK)。

### Flux2KleinOutputExtractor_EditUtils

**显示名**：EditUtils: Flux2Klein Output Extractor lrzjason

- 输入：`custom_output` (ANY)。
- 输出 8 个：`pad_info` (ANY) · `main_image` (IMAGE) · `vae_images` (LIST) · `ref_latents` (LIST) · `full_prompt` (STRING) · `llama_template` (STRING) · `no_refs_cond` (CONDITIONING) · `mask` (MASK)。

### Flux3KleinOutputExtractor_EditUtils

**显示名**：EditUtils: Flux3Klein Output Extractor lrzjason

- 输入：`custom_output` (ANY)。
- 输出 10 个：`pad_info` (ANY) · `main_image` (IMAGE) · `vae_images` (LIST) · `ref_latents` (LIST) · `full_prompt` (STRING) · `no_refs_cond` (CONDITIONING) · `mask` (MASK) · `flow_map` (BOOLEAN，该 checkpoint 是否 flow-map/step-distilled) · `bbox_rows` (LIST，本次解析出的元素表行) · `composed_prompt` (STRING，**已把元素表拼进去的完整 prompt**)。

### BooguOutputExtractor_EditUtils

**显示名**：EditUtils: Boogu Output Extractor lrzjason

- 输入：`custom_output` (ANY)。
- 输出 4 个：`negative_cond` (CONDITIONING) · `ref_latents` (LIST) · `vl_images` (LIST) · `main_image` (IMAGE，取 vae_images[0])。

---

## 五、模型补丁（Krea2 / QwenImage / QwenImage 2.1 / Flux2Klein / Flux3Klein / Boogu）

### Krea2EditApply_EditUtils

**显示名**：EditUtils: Krea2 Edit Apply (Model Patch) lrzjason
把 Krea2（SingleStreamDiT）模型补丁成支持参考图编辑。**用户只需连 model 线**——参考图 latent 通过 EditUtils 条件链（conditioning 上的 `reference_latents`）经 `extra_conds` 补丁自动流入。

| 输入 | 类型 | 默认 | 说明 |
|---|---|---|---|
| model | MODEL | — | Krea2 模型；非 Krea2 模型原样返回，不打补丁 |
| ref_pos_match_target | BOOLEAN (optional) | True | 将 ref 的 RoPE 位置 id 拉伸覆盖到目标 token 网格（分数坐标）。**ref 与 target 分辨率不一致时应开启**，否则小 ref 只对齐到 target 左上角区域 |
| ref_kv_cache | BOOLEAN (optional) | True | 内置参考图 KV 缓存：首轮第 1 步捕获每层 ref K/V，后续步骤复用（约 2x 加速）。冻结参考近似，非逐位精确 |
| ref_strength | FLOAT (optional, 0.0–1.0, step 0.05) | 1.0 | 控制参考图在采样过程中约束多长时间。1.0=ref 全程参与（默认，行为与旧版一致）；0.5=ref 只参与采样进度前 50%，之后丢弃缓存的 ref K/V 以纯文生图模式完成剩余步；0.0=ref 完全不参与（直接走无参考原始 forward）。**依赖 ref_kv_cache 开启才生效** |
| reset_cache | BOOLEAN (optional) | True | 每次节点执行时清空 KV 缓存（重新捕获）；同一次采样内步骤间仍复用 |
| debug_log | BOOLEAN (optional) | False | 打印 KV 缓存捕获/复用日志 |

- 输出：`model` (MODEL)。
- 补丁通过 `model.clone()` + `add_object_patch` 隔离安装，不影响原模型实例。
- 内部序列顺序 `[text | target | ref₁ | ref₂ | …]`，位置 id：target=(0,h,w)、refₙ=(+n,h,w)；ref token 使用 t=0 调制。
- Krea2 图像 latent 以 5D `[B,C,1,H,W]` 进入模型，内部自动 reshape 为 4D。
- 无参考图时自动回退原始 forward；KV 缓存路径异常时回退非缓存 forward（保证不中断生成）。
- **ref_strength 语义**：实现于缓存 forward 内，根据 `transformer_options["sigmas"]` 计算采样进度；`progress >= ref_strength` 时跳过 ref K/V 拼接（数学上等价于无 ref forward）。进度按 sigma 进度而非步数计算。`debug_log=True` 时首次丢弃会打印 `[Krea2EditApply] ref dropped at progress=...`。
- **用途**：避免 ref latent 过度约束模型——采样前段保留 ref 的结构/身份，后段释放细节与创意。**0.5~0.7 是建议的实验区间**；该调度属于训练分布外（OOD）行为，效果需肉眼验收，不保证所有 prompt/ref 组合都有正面收益。
- 原独立节点 `Krea2RefKVCache_EditUtils` 已删除，缓存功能由 `ref_kv_cache` 选项内置。

### QwenImage21EditApply_EditUtils

**显示名**：EditUtils: QwenImage 2.1 Edit Apply (ROPE Control) lrzjason
给 Qwen-Image 2.1 DiT 打 ROPE 位置偏移补丁，实现区域编辑（把参考图“放”到画布指定位置）。**用户只需连 model 线**——rope 偏移通过 EditUtils 条件链（conditioning 上的 `reference_rope_offsets`，来自 Config Preparer 的 `rope_x_offset/rope_y_offset`）自动流入。

| 输入 | 类型 | 默认 | 说明 |
|---|---|---|---|
| model | MODEL | — | QwenImage 2.1 模型；检测不通过时原样返回并打印提示 |
| debug_log | BOOLEAN (optional) | False | 打印生效的 rope 偏移 |

- 输出：`model` (MODEL)。
- 原理：Qwen-Image 2.1 原生把参考 latent 拼接进文本序列（vision slot）并把参考 RoPE 位置以目标为中心排布；本节点在 `build_sequence` 中给每张参考的 h/w 位置 id 加上 `偏移像素 / 16`（latent 单位）。
- 偏移全为 0 时直接走原生 `build_sequence`（行为与未打补丁一致，前缀 KV 缓存不受影响）。
- 补丁通过 `model.clone()` + `add_object_patch` 隔离安装：`extra_conds`（透传 `ref_rope_offsets`）、`forward`（暂存偏移）、`build_sequence`（应用偏移）。
- 需要 ComfyUI 已内置 Qwen-Image 2.1 支持（上游 `qwen_image21`），否则模型本身无法加载为该架构。

### Flux3KleinEditApply_EditUtils

**显示名**：EditUtils: Flux3Klein Edit Apply (Model Check) lrzjason
FLUX.3 Klein 编辑路径的模型校验节点。**与 Flux2KleinEditApply 不同，它不重写 forward**。

| 输入 | 类型 | 默认 | 说明 |
|---|---|---|---|
| model | MODEL | — | FLUX.3 模型 |
| debug_log | BOOLEAN (optional) | False | 打印 `flow_map`（是否 step-distilled）与原生参考处理说明 |
| strict | BOOLEAN (optional) | True | 模型不像 FLUX.3 时**抛错**；关闭则打印提示后原样返回 |

- 输出：`model` (MODEL)。
- **为什么不需要打补丁**：ComfyUI 核心的 `comfy/model_base.py class Flux3` 已原生消费 `reference_latents`，并交给 `comfy/ldm/flux3/model.py`；模型自己用 `ref_ids` 推导参考位置 id（`l` 轴负坐标 -1, -2, …，`image_rope_scale = 0.5`）。EditUtils 的 `reference_rope_offsets` 在该架构上没有消费者，因此编码节点不写这个键。
- 检测依据：模型带有 `content_mode_blocks` / `txt_mode_blocks` / `single_blocks` / `ref_ids`。
- **用途**：给 FLUX.3 图一个与其他 EditUtils 编辑图一致的外形，并在误接成 FLUX.2 klein 补丁时**明确报错**，而不是静默地拿不到参考条件。

---

## 六、条件缓存（Save/Load Condition）

缓存目录为 `ComfyUI/models/conditions/`（首次使用自动创建），文件为 `.ckpt`（torch.save 的 conditioning 列表）。典型用法：同一张原图 + 同一段 system prompt 反复编辑时，Save 一次条件，之后用 Load 直接喂采样器，跳过 CLIP/VAE 编码。

### SaveCondition_EditUtils

**显示名**：EditUtils: Save Condition lrzjason （OUTPUT_NODE，无输出）

| 输入 | 类型 | 默认 | 说明 |
|---|---|---|---|
| condition | CONDITIONING | — | 要保存的条件 |
| filename | STRING | `condition_tensor` | 自动补 `.ckpt`，只取 basename 防路径穿越 |

### LoadCondition_EditUtils

**显示名**：EditUtils: Load Condition lrzjason

| 输入 | 类型 | 默认 | 说明 |
|---|---|---|---|
| filename | COMBO | — | 列出 `models/conditions/` 中的文件 |

- 输出：`CONDITIONING`。文件不存在或加载失败时返回空列表 `[]`（仅打印错误）。

### LoadConditionFromLoras_EditUtils

**显示名**：EditUtils: Load Condition From Loras lrzjason
与 LoadCondition 相同，但下拉列表来自 `models/loras/` 目录的文件名，实际仍从 `models/conditions/` 加载同名 `.ckpt`。用于"按 LoRA 同名缓存条件"的组织方式。

- 输入：`filename` (COMBO)。输出：`CONDITIONING`。失败返回 `[]`。

### ClearRefLatents_EditUtils

**显示名**：EditUtils: Clear Ref Latents lrzjason
把 conditioning 上的 `reference_latents` 置空（其余内容保留）。

- 输入：`conditioning` (CONDITIONING)。输出：`conditioning` (CONDITIONING)。
- 用途：复用编码好的文本条件但不想要参考图（例如负面条件、或防止 ref 流向不该去的模型/采样器），也可在保存条件前清掉 ref。

---

## 七、图像处理与杂项工具

### LongestEdgeImageProcess_EditUtils

**显示名**：EditUtils: Longest Edge Image Process lrzjason （CATEGORY `image`）
把 EditTextEncode 的最长边缩放 + padding 逻辑独立出来（不做 CLIP/VAE 编码）。

| 输入 | 类型 | 默认 | 说明 |
|---|---|---|---|
| image | IMAGE | — | |
| ref_longest_edge | INT (8–4096) | 1024 | |
| ref_crop | COMBO: `pad`/`center`/`disabled` | `pad` | |
| ref_upscale | COMBO: `lanczos`/`bicubic`/`area` | `lanczos` | |
| vae_unit | INT (8–64, step 8) | 8 | |
| pad_mode | COMBO: `color`/`edge`/`mirror`/`border_mean` | `color` | ref_crop=pad 时填充区域的生成方式（optional） |
| pad_color | COMBO: `red`/`green`/`blue`/`white`/`black`/`custom` | `black` | pad_mode=color 时的填充色；qwen 系模型建议改为 white（optional） |
| pad_color_rgb | STRING | `-` | 可选覆盖：6 位 hex，`#` 可有可无，合法时覆盖 pad_color（optional） |
| pad_noise | FLOAT (0–0.2) | 0.0 | 只加到填充区域的高斯抖动，内容不受影响（optional） |

- 输出：`processed_image` (IMAGE) · `pad_info` (ANY) · `scale_by` (FLOAT)。
- `pad_info` 可喂给 `CropWithPadInfo_EditUtils` 还原出图尺寸。

### CropWithPadInfo_EditUtils

**显示名**：EditUtils: Crop With Pad Info lrzjason （CATEGORY `image`）
按 pad_info 裁掉 pad 模式添加的填充区域，还原原始内容区域。

| 输入 | 类型 | 默认 | 说明 |
|---|---|---|---|
| pad_info | ANY | — | 含 x/y/width/height/scale_by 的 dict |
| image | IMAGE | — | |

- 输出：`cropped_image` (IMAGE) · `scale_by` (FLOAT)。

### AdaptiveLongestEdge_EditUtils

**显示名**：EditUtils: Adaptive Longest Edge lrzjason
根据输入图尺寸计算合适的 ref_longest_edge：小于 min_size 提到 min_size；大于 max_size 按整数倍缩到 max_size 以内（再兜底 min_size）。

| 输入 | 类型 | 默认 | 说明 |
|---|---|---|---|
| image | IMAGE | — | |
| min_size | INT (64–4096) | 512 | |
| max_size | INT (512–4096) | 2048 | |

- 输出：`longest_edge` (INT)。典型用法：接到 Config Preparer 的 `ref_longest_edge` 做自适应分辨率。

### LoadImageWithFilename_EditUtils

**显示名**：EditUtils: Load Image With Filename lrzjason （CATEGORY `image`）
等同官方 LoadImage，额外输出不含扩展名的文件名（支持多帧图、EXIF 旋转、alpha 转 mask）。

- 输入：`image`（上传/选择 input 目录图片）。
- 输出：`image` (IMAGE) · `mask` (MASK) · `filename` (STRING)。

### Any2Image_EditUtils

**显示名**：EditUtils: Any2Image lrzjason
ANY → IMAGE 透传转换（用于把 extractor 的 ANY 输出接到 IMAGE 输入）。

- 输入：`item` (ANY)。输出：`item` (IMAGE)。无校验，接错类型会在下游报错。

### Any2Latent_EditUtils

**显示名**：EditUtils: Any2Latent lrzjason
ANY → LATENT 转换：把张量包成 `{"samples": item}`。

- 输入：`item` (ANY)。输出：`item` (LATENT)。

### ListExtractor_EditUtils

**显示名**：EditUtils: List Extractor lrzjason
按索引从 LIST 中取一项（如从 `vae_images`/`ref_latents` 取第 n 个）。

| 输入 | 类型 | 默认 | 说明 |
|---|---|---|---|
| items | LIST | — | |
| index | INT (0–1000000) | 0 | 越界会 assert 报错 |

- 输出：`item` (ANY)。

### DiffMask_EditUtils

**显示名**：EditUtils: Diff Mask lrzjason （CATEGORY `image`）
两图差异 mask：支持频域高通差异（对纹理/边缘/微小位移敏感）与像素差混合，再做腐蚀/膨胀开运算与高斯平滑。

| 输入 | 类型 | 默认 | 说明 |
|---|---|---|---|
| image1 / image2 | IMAGE | — | 形状必须一致，否则报错 |
| threshold | FLOAT (0–1) | 0.05 | 二值化阈值 |
| blur_radius | INT (0–50) | 5 | 高斯模糊核（0=不模糊，偶数自动 +1） |
| erode_size | INT (0–20) | 3 | 腐蚀核（0=不腐蚀） |
| dilate_size | INT (0–20) | 3 | 膨胀核（0=不膨胀） |
| use_frequency | BOOLEAN | True | 启用频域高通差异 |
| highpass_sigma | FLOAT (1–100) | 10.0 | 高通截止 sigma，越小越高通 |
| freq_weight | FLOAT (0–1) | 0.7 | 频域差异权重，其余为像素差权重 |

- 输出：`mask` (MASK)。

### AlphaNoiseFilter_EditUtils

**显示名**：EditUtils: Alpha Noise Filter lrzjason （CATEGORY `image`）
清理 Qwen-Image 2.1 生成图的 alpha 通道。Qwen-Image 2.1 的 64 通道 RGBA VAE 能生成带透明背景的图，但原始输出通常在透明区域散布着数值很低、肉眼几乎不可见的 alpha 噪点——单独看不明显，却会破坏抠图、合成与基于蒙版的编辑。本节点去除这类噪点，输出干净的 alpha 图与对应蒙版。

| 输入 | 类型 | 默认 | 说明 |
|---|---|---|---|
| image | IMAGE | — | RGBA 图，如 Qwen-Image 2.1 VAE 的 VAEDecode 输出 |
| mask | MASK (optional) | — | 输入图没有 alpha 通道（RGB）时作为 alpha 来源 |
| alpha_threshold | FLOAT (0–1, step 0.01) | 0.10 | 低于此值的 alpha 视为低值噪点并裁为 0（0 = 关闭） |
| min_area | INT (0–65536) | 100 | 删除像素数小于此值的独立 alpha 孤岛（0 = 关闭） |
| feather | FLOAT (0–32, step 0.5) | 2.0 | 羽化渐变带宽度（像素）。从 matte 边界处**完全透明**平滑过渡到向内 `feather` px 处**完全不透明**（0 = 硬边） |
| invert_mask | BOOLEAN | False | 蒙版输出改为 1-alpha（白=透明，ComfyUI 重绘惯例） |

- 输出：`image` (IMAGE, RGBA) · `mask` (MASK)。
- 处理顺序：**低值裁剪 → 小孤岛删除 → 向内羽化**。
- 本节点**只修改 alpha 通道**：输出的 RGB 与输入完全一致（`defringe` / `premultiply` 这类会改写 RGB 的选项已移除）。
- **羽化只向内工作，绝不外扩**：渐变带整条都落在生成器已经标记为「主体」的区域内，因此羽化不可能让任何原本透明的像素变可见。实测 `alpha noise.png`，feather=0…16 时轮廓外 alpha>0 的像素数恒为 **0**，且任何位置 alpha 都不会超过原始 matte。
- **为什么不能对称模糊**：对称高斯模糊会把渐变带推到 matte 之外，而那些像素的 RGB 仍然是生成器留下的背景色（Qwen-Image 2.1 的 VAE 是一片紫色 (152,50,196)）。实测 feather=4 时对称模糊会把 **13758** 个原本 alpha=0 的像素点亮，它们的 RGB 均值是 (92,47,102)，在深色背景上就是一圈紫晕。改成向内羽化后紫边量恒为 **0.000**。
- 羽化渐变实测（`alpha noise.png`，从 matte 边界向内的 alpha 值）：

  | feather | 0 px | 1 px | 2 px | 3 px | 4 px | 5 px | 6 px | 8 px |
  |---|---|---|---|---|---|---|---|---|
  | 0（原 matte） | 0.449 | 0.877 | 0.957 | 0.977 | 0.986 | 0.992 | 0.995 | 0.998 |
  | 2 | 0.020 | 0.493 | 0.951 | — | — | — | — | — |
  | 4 | 0.005 | 0.160 | 0.475 | 0.823 | 0.986 | — | — | — |
  | 8 | 0.001 | 0.044 | 0.149 | 0.314 | 0.510 | 0.686 | 0.842 | 0.997 |

  越靠外越透明，向内单调递增到完全不透明——是一条真正的渐变，而不是一圈固定透明度。
- **代价**：渐变带内的 alpha 被压低，所以轮廓面积会收缩（feather=2 → −2.79%，feather=4 → −5.27%，feather=8 → −10.13%），细结构（如毛发）在羽化带宽大于自身宽度时会明显变淡——羽化带内侧 2px 范围的平均 alpha 从 0.583（不羽化）降到 0.152（feather=2）。这是「不外扩」的必然代价。
- 蒙版默认是 alpha 本身（白=不透明）；需要喂给重绘类节点（如 `VAEEncodeForInpaint`）时打开 `invert_mask`。
- 小孤岛删除与向内羽化优先用 `scipy.ndimage`（ComfyUI 依赖）；scipy 不可用时分别回退到形态学开运算与「逐像素腐蚀累加」的线性渐变，行为仍然有界（不外扩）。
- `alpha_threshold=0 / min_area=0 / feather=0` 时输出与输入完全一致（透传校验用）。

### BreastSizeControl_EditUtils

**显示名**：EditUtils: Breast Size Control (Canvas) lrzjason （CATEGORY `image`）
在节点内的画布上直接手绘**胸部尺寸控制图（`_bd` 条件图）**：黑底 + 1~2 个实心椭圆标记，默认填充色 `#FADBAE`，标记的**面积**代表乳房大小、**位置**代表乳房在哪。输出与 `E:\images\breasts_size_control` 里现有 `_bd` 条件图**同一契约**（实测 60 张：全部只有 2 种颜色；标记 = 外接矩形内切椭圆，与 `box_px` 的 IoU 中位 0.988；输出尺寸与 `target_size_px` 逐张相同）。

画布是 **WYSIWYG** 的：你看到的黑底与色块就是节点渲染出的像素（浏览器 `ctx.ellipse` 与节点光栅化用的是同一个内点判定）。画布几何量以归一化坐标保存在隐藏的 `markers` 输入里，因此换分辨率不丢画：`cx`/`rx` 是宽度的比例，`cy`/`ry` 是高度的比例。相邻接线图尺寸改变时，归一化半径会自动按像素长宽比补偿，圆形不会因为画布变宽而变成椭圆。

| 输入 | 类型 | 默认 | 说明 |
|---|---|---|---|
| width / height | INT (16–16384, step 8) | 832 / 1216 | 未接 `image` 时的画布尺寸；接了 `image` 时被覆盖 |
| markers | STRING（`socketless` + `advanced`，UI 中隐藏） | 两个「medium 档」圆形 | 画布持有的 JSON 载荷（见下）。用画布编辑，不要手改 |
| antialias | BOOLEAN | False | False = 与现有 `_bd` 图完全一致的 2 色硬边；True = 4×4 超采样抗锯齿（面积不变） |
| image | IMAGE (optional) | — | 目标图：输出被强制为该图的**精确像素尺寸**（batch 取第一帧）。`_bd` 契约要求条件图与目标图同尺寸，位置才不会被 bucket 裁切弄错 |

- 输出：`image` (IMAGE)，形状 `(1, H, W, 3)`，float32，黑底 + 标记色。另随 `ui` 返回 `width`/`height`（**列表**，见下），画布据此在运行后自动对齐到实际尺寸。

**画布操作**

| 操作 | 行为 |
|---|---|
| 在空白处左键拖拽 | 新建一个标记（最多 2 个，与「一个或两个」的数据契约一致） |
| 左键拖拽标记本体 | 移动 |
| 拖拽 上/下/左/右 边中点手柄 | 单独改 `ry` / `rx`——**这是把圆拖成椭圆的工具**（现有 `_bd` 图的 bbox 长宽比中位 0.91，本来就不是正圆） |
| 按住 Shift 拖边手柄 | 等比缩放（保持正圆） |
| 拖拽 四个角手柄 / 在标记上滚轮 | 等比缩放 |
| **拖拽旋转手柄**（椭圆上方的小圆点，带一根引线） | **绕标记中心旋转**，角度实时显示；边手柄会跟着旋转，始终落在椭圆自身的轴上 |
| `↺ 15°` / `↻ 15°` / 角度输入框 / `0°` | 按 15° 微调、直接输入角度、或清零；输入框显示当前选中标记的角度 |
| 右键点标记 | 删除 |
| 预设按钮 flat…gigantic | 把**整张图**的标记总面积设到该档实测中位面积，并按两标记当前面积比例分配（保持左右相对大小与形状） |
| 取色器 | 标记填充色，默认 `#FADBAE` |
| guide 勾选框 | 把接入的目标图以 45% 透明度垫在黑底上作为对位参考；**只用于对齐，绝不会进入输出像素** |
| 状态行 | `宽x高 - total X.XX% (档位) - marker i/n: Y.YY% @ 角度°`；档位按**总面积**归类，因为实测的中位面积是两标记合计 |

> **旋转的约定**：角度单位是度，**正数 = 屏幕上顺时针**（y 轴向下，与 `krea2_geom_lab` 的 `_rotated` 约定一致），范围归一化到 `(-180, 180]`。旋转只改变朝向，**不改变面积**，所以不会影响尺寸档与预设；旋转后边手柄/角手柄/命中判定都跟着转，拖拽始终沿椭圆自身的轴生效。换画布长宽比时只补偿 `ry`（保持像素长宽比），角度不变。

**尺寸档预设**（取 `analysis.md` 附录 A 的实测中位面积占比，按总面积计）：

| 档位 | flat | small | medium | large | huge | gigantic |
|---|---|---|---|---|---|---|
| 标记总面积 / 画面 | 1.49% | 2.94% | 3.22% | 7.18% | 13.95% | 16.50% |

> 预设作用于**整张图的总面积**而不是「选中那一个」：如果按单标记给面积，两个标记时总面积会翻倍到档位的两倍，与训练分布不符。想左右大小不同，用边手柄分别拖。

**`markers` 载荷格式**（画布写入，节点读取；归一化，非像素）

```jsonc
{
  "version": 1,
  "color": "#FADBAE",              // 3/6/8 位十六进制，# 可省；解析失败回退 #FADBAE
  "circles": [                     // 1 个或 2 个（>2 会被画布限制）
    {"cx": 0.355, "cy": 0.46,      // 圆心：宽度比例 / 高度比例
     "rx": 0.086545, "ry": 0.059215,  // 半径：宽度比例 / 高度比例（默认画布 832x1216 下的正圆，72.005 px）
     "angle": 0}                   // 度，正数 = 顺时针；缺省/非法值按 0 处理，归一化到 (-180, 180]
  ]
}
```

- 面积占比 = `π · rx · ry`（与像素尺寸、与旋转角都无关，因此画布上显示的百分比就是渲染结果）。
- 正圆条件：`rx · W == ry · H`（画布在新建标记与换画布长宽比时自动满足）。
- **宽容解析**：JSON 损坏、`circles` 缺失、`rx`/`ry` ≤ 0、NaN、完全在画布外等情况都不会让工作流失败，只是该标记不画（极端情况下输出纯黑）；`angle` 非法只回退成 0，不会连累同一个标记的几何。
- 返回的 `ui` 里 `width`/`height` 是**单元素列表**（ComfyUI 在 `execution.get_output_from_returns` 里会把每个 `ui` 值展开，给裸 int 会直接抛 `TypeError: 'int' object is not iterable`）；画布读 `message.width[0]`。
- **可运行的例子**：[k2 breasts size control.json](workflows/k2%20breasts%20size%20control.json) —— Krea2 + 训练好的 `breasts_control_lrzjason_20261015` 控制 LoRA；画布画出的条件图直接当作参考图喂进 `QwenConfigPreparer_EditUtils → EditTextEncode_EditUtils → Krea2EditApply_EditUtils`。见 README 的 [Krea2 胸部尺寸控制](README_zh.md#krea2-胸部尺寸控制) 一节。

**验证**（`.tmp/test_breast_size_control.py`，全部通过）
- 六档面积实测相对误差 ≤ 0.12%（单标记与两标记拆分两种画法都测）；
- 两标记分离时总面积恰为单标记的 2 倍；重叠时是**并集**而不是相加（6.852% vs 相加 9.048%）；
- 接 `image` 时输出尺寸与目标逐像素相同（含 batch 取首帧、非 batch 输入报错）；
- 从 25 张现有 `_bd.json` 的 `box_px` 反算几何重渲染，与真实 `_bd.png` 像素的 IoU：中位 **0.9866**、最小 **0.9757**——与「内切椭圆」模型自身的上限（0.9762–0.9919）一致，即渲染器对数据集的复现精度已经到顶。

