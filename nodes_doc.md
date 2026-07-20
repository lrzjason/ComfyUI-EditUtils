# EditUtils 节点参考文档

> 本文档以 `nodes.py` 当前代码（`NODE_CLASS_MAPPINGS`，28 个节点）为准。
> 显示名均带 `lrzjason` 后缀；节点菜单分类多为 `advanced/conditioning`，图像类节点在 `image`，Krea2 补丁在 `Krea2/edit`。
> 若文档与代码不符，以代码为准。

## 目录

- [一、模型配置节点（Model Config）](#一模型配置节点model-config)
- [二、文本/图像编码节点（Text Encode）](#二文本图像编码节点text-encode)
- [三、Config Preparer（configs 链）](#三config-preparerconfigs-链)
- [四、Output Extractor（custom_output 拆包）](#四output-extractorcustom_output-拆包)
- [五、Krea2 模型补丁](#五krea2-模型补丁)
- [六、条件缓存（Save/Load Condition）](#六条件缓存saveload-condition)
- [七、图像处理与杂项工具](#七图像处理与杂项工具)

---

## 一、模型配置节点（Model Config）

输出均为 `model_config` (DICT)，供 `EditTextEncode_EditUtils` 的 `model_config` 输入使用。

### ModelConfig_EditUtils

**显示名**：EditUtils: Model Config lrzjason
通用模型配置节点，可选 qwen / flux2klein / boogu。

| 输入 | 类型 | 默认 | 说明 |
|---|---|---|---|
| model_choice | COMBO: `qwen`/`flux2klein`/`boogu` | `qwen` | 模型选择 |
| model_name | STRING | `""` | 留空时使用 model_choice |
| vae_unit | INT (8–64, step 8) | 8 | VAE 对齐单位；Qwen=8，Flux2Klein=16，Boogu=8 |
| instruction | STRING (optional, multiline) | `""` | 自定义 system prompt；留空用内置默认。boogu 忽略此项 |

- 输出：`model_config` (DICT) — `{model_name, vae_unit, llama_template}`。
- boogu 时 `llama_template` 强制为 `""`（Boogu tokenizer 自动选择 system prompt）。
- 注意：vae_unit 默认值是 8，选择 flux2klein 时需手动改为 16；建议直接用下方各模型专用 Config 节点。

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

---

## 二、文本/图像编码节点（Text Encode）

### EditTextEncode_EditUtils

**显示名**：EditUtils: EditTextEncode lrzjason
核心统一编码节点。按 `model_config.model_name` 分流到 Qwen / Flux2Klein（共用代码路径）或 Boogu（独立早返回路径）逻辑。

| 输入 | 类型 | 默认 | 说明 |
|---|---|---|---|
| clip | CLIP | — | 文本编码器 |
| vae | VAE | — | 参考图编码 |
| prompt | STRING (multiline) | — | 编辑指令 |
| model_config | DICT | None | 来自 ModelConfig 系列节点 |
| configs | LIST (optional) | None | Config Preparer 串联出的图像配置列表；不接或为空时执行纯文本编码 |

- 输出：`conditioning` (CONDITIONING) · `latent` (LATENT) · `custom_output` (ANY dict) · `main_image` (IMAGE) · `mask` (MASK) · `pad_info` (ANY dict)。

**Qwen / Flux2Klein 路径行为**
- 对每张 `to_ref` 的图：按 `ref_longest_edge` + `ref_resize_mode` 缩放，按 `vae_unit` 对齐，`ref_crop="pad"` 时黑边补齐到 vae_unit 倍数（主图生成 `pad_info`），然后 `vae.encode` 得到 ref latent。
- 全部 ref latents 以 `reference_latents` 键挂到 conditioning 上（`node_helpers.conditioning_set_values`），采样模型据此读取。**这就是 EditUtils 条件链。**
- `latent` 输出 = 主图（`ref_main_image`）的 ref latent；无图时为 1×4×128×128 零 latent。
- Qwen 的 `to_vl` 图还会缩放到 `vl_target_size` 面积喂给 QwenVL，并在 prompt 前自动拼 `Picture n: <|vision_start|>...`；Flux2Klein 忽略 VL（代码中 `to_vl` 强制 False）。
- 主图带 mask 时，`latent` 附带 `noise_mask`，并从 `mask` 输出返回。
- 只会有一个主图：后续 config 的 `ref_main_image` 会被强制改为 False；都不设主图时自动取第 0 张并打印提示。

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
| ref_crop | COMBO: `pad`/`center`/`disabled` | `pad` | pad=黑边补齐到 vae_unit 倍数 |
| ref_upscale | COMBO: `lanczos`/`bicubic`/`area` | `lanczos` | |
| to_vl | BOOLEAN | True | 是否进 QwenVL 编码 |
| vl_resize | BOOLEAN | True | VL 前是否缩放 |
| vl_target_size | INT (384–2048) | 384 | VL 目标面积基准 |
| vl_crop | COMBO: `center`/`disabled` | `center` | |
| vl_upscale | COMBO: `lanczos`/`bicubic`/`area` | `lanczos` | |
| mask | MASK (optional) | None | |
| ref_resize_mode | COMBO: `longest_edge`/`area` | `longest_edge` | longest_edge=最长边对齐；area=总面积对齐 ref_longest_edge² |

- 输出：`configs` (LIST) · `config` (ANY，当前图的配置)。

### Flux2KleinConfigPreparer_EditUtils

**显示名**：EditUtils: Flux2Klein Config Preparer lrzjason
与 Qwen 版相同但无 VL 参数（Flux2Klein 不走 VL）。`ref_longest_edge` 范围 16–4096。其余同 QwenConfigPreparer（to_ref / ref_main_image / ref_longest_edge=1024 / ref_crop=`pad` / ref_upscale=`lanczos` / mask / ref_resize_mode=`longest_edge`）。

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

### BooguOutputExtractor_EditUtils

**显示名**：EditUtils: Boogu Output Extractor lrzjason

- 输入：`custom_output` (ANY)。
- 输出 4 个：`negative_cond` (CONDITIONING) · `ref_latents` (LIST) · `vl_images` (LIST) · `main_image` (IMAGE，取 vae_images[0])。

---

## 五、Krea2 模型补丁

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

- 输出：`processed_image` (IMAGE) · `pad_info` (ANY) · `scale_by` (FLOAT)。
- `pad_info` 可喂给 `CropWithPadInfo_EditUtils` 还原出图尺寸。

### CropWithPadInfo_EditUtils

**显示名**：EditUtils: Crop With Pad Info lrzjason （CATEGORY `image`）
按 pad_info 裁掉 pad 模式添加的黑边，还原原始内容区域。

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
