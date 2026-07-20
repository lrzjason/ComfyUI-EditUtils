# EditUtils 连线手册（USAGE）

> 面向用户和 agent 的 ComfyUI 连线指南。所有行为以 `nodes.py` 代码为准。
> 配套节点参数细节见 `nodes_doc.md`。

## 0. 全局公共模式

### 0.1 条件链如何携带 reference_latents

`EditTextEncode_EditUtils` 把每张参考图 VAE 编码成 latent 后，用
`node_helpers.conditioning_set_values(cond, {"reference_latents": [...]})`
挂到输出的 CONDITIONING 上。采样器/模型补丁（如 Krea2EditApply）从 conditioning
的 `reference_latents` 键读取参考图。**因此参考图不需要单独的连线进入采样器或
Krea2 模型补丁——只连 model / conditioning 常规线即可。**

- Boogu 例外：ref latents 同时挂到 positive 和 negative 上（CFG 下抵消参考）。
- `ClearRefLatents_EditUtils` 可把某条条件上的 reference_latents 清空。

### 0.2 configs 串联方式（多参考图）

每个 `*ConfigPreparer_EditUtils` 输入一张图 + 一组处理参数，把 dict append 到
上游 `configs` 列表后再输出：

```
[LoadImage] IMAGE → QwenConfigPreparer #1 ──configs(LIST)──→ QwenConfigPreparer #2 ──configs──→ ... ──→ EditTextEncode.configs
```

- 第一个 Preparer 的 `configs` 输入留空即可（代码内部视为 `[]`）。
- 只有一张主图：把主图那个 Preparer 的 `ref_main_image=True`，其余设 False。
  主图决定采样初始 latent 和 pad_info。
- 不同类型 Preparer（Qwen/Flux2Klein/Boogu/JsonParser）产出的 config 结构兼容，
  但应按实际模型使用对应 Preparer（VL 字段不同）。
- 图片数量不限（简易一体化节点最多 3 张）。

### 0.3 主链骨架（所有管线通用）

```
LoadImage → ConfigPreparer(×N, configs 链) ─┐
ModelConfig 系列 ──model_config(DICT)──────┤
CLIP ─────────────────────────────────────┼─→ EditTextEncode ──conditioning──→ 采样器 positive
VAE ──────────────────────────────────────┘                  ──latent──────→ 采样器 latent_image
                                                             └─custom_output→ OutputExtractor(可选拆包)
采样器 → VAEDecode → (CropWithPadInfo 去黑边) → SaveImage
```

---

## 1. Qwen 管线（Qwen-Image-Edit）

### 1a. 可定制版（ConfigPreparer 链）

```
[LoadImageWithFilename] image(IMAGE) ──────────────────────────→ [QwenConfigPreparer #1] (主图 ref_main_image=True, ref_crop=pad)
[LoadImage] image(IMAGE) ──────────────────────────────────────→ [QwenConfigPreparer #2] (ref_main_image=False, ref_crop=center/disabled)
        Preparer#1.configs(LIST) → Preparer#2.configs ──configs(LIST)──┐
[QwenModelConfig] ──model_config(DICT)─────────────────────────────────┤
[CLIPLoader] ──CLIP────────────────────────────────────────────────────┼──→ [EditTextEncode]
[VAELoader] ──VAE──────────────────────────────────────────────────────┘        │
  prompt(STRING) ─────────────────────────────────────────────────────→ prompt  │
        │ conditioning(CONDITIONING) ──→ [KSampler].positive                    │
        │ latent(LATENT) ──────────────→ [KSampler].latent_image                │
        │ pad_info(ANY) ──┐                                                     │
[UNETLoader Qwen-Image-Edit] ──MODEL──→ [KSampler] ──LATENT──→ [VAEDecode] ──IMAGE──→ [CropWithPadInfo] ──→ [SaveImage]
                          pad_info ────→ [CropWithPadInfo].pad_info（去掉 pad 黑边）
```

- negative：可用官方 CLIPTextEncode 或 `EditTextEncode` 的 `no_refs_cond`（经
  `QwenEditOutputExtractor` 拆出）接 negative；也可把 `conditioning` 经
  `ClearRefLatents` 后使用（以实际工作流为准）。

### 1b. 简易版（一体化节点）

```
[LoadImage] ──IMAGE──→ [QwenEditTextEncode].image1 (.image2/.image3 可选, mask 仅作用 image1)
CLIP ──→ clip │  VAE ──→ vae │  prompt ──→ prompt
        conditioning ──→ KSampler.positive
        latent ────────→ KSampler.latent_image
        pad_info 在 custom_output 里：custom_output ──→ [QwenEditOutputExtractor].pad_info ──→ CropWithPadInfo
```

- 至少接 image1，否则运行报错 `At least one image must be provided`。

---

## 2. Flux2Klein 管线

与 Qwen 结构相同，替换为 Klein 组件：

```
[LoadImage] ──IMAGE──→ [Flux2KleinConfigPreparer #1] (主图) ──configs(LIST)──→ [#2 ...] ──configs──┐
[Flux2KleinModelConfig] ──model_config(DICT)──────────────────────────────────────────────────────┤
CLIP ──→ clip │ VAE ──→ vae │ prompt ──→ prompt ──────────────────────────────────────────────→ [EditTextEncode]
        conditioning ──→ KSampler.positive
        latent ────────→ KSampler.latent_image   （主图 ref latent）
[UNETLoader Flux2Klein] ──MODEL──→ KSampler ──→ VAEDecode ──→ CropWithPadInfo ──→ SaveImage
```

- vae_unit 固定 16（由 `Flux2KleinModelConfig` 自动设置）。
- 无 VL 编码（代码中 `to_vl` 强制 False），`vl_images` 为空。
- 简易版：`Flux2KleinEditTextEncode`（image1/2/3 + mask，同 Qwen 简易版用法）。

---

## 3. Boogu 管线

```
[LoadImage] ──IMAGE──→ [BooguConfigPreparer #1] (to_ref/to_vl, 主图) ──configs──→ [#2 ...] ──configs──┐
[BooguModelConfig] ──model_config(DICT)─────────────────────────────────────────────────────────────┤
CLIP ──→ clip │ VAE ──→ vae │ prompt ──→ prompt ────────────────────────────────────────────────→ [EditTextEncode]
        conditioning(positive) ──→ KSampler.positive
        latent ──────────────────→ KSampler.latent_image
        custom_output ──→ [BooguOutputExtractor].negative_cond(CONDITIONING) ──→ KSampler.negative
[Boogu UNet] ──MODEL──→ KSampler ──→ VAEDecode ──→ SaveImage
```

- ⚠️ negative 必须从 `custom_output` 经 `BooguOutputExtractor` 取 `negative_cond`；
  直接连 `BooguModelConfig` + `EditTextEncode` 时 negative_prompt 为空字符串。
  需要自定义负面词请用简易版 `BooguEditTextEncode`（有 `negative_prompt` 输入，
  直接输出 positive/negative 双条件）。
- Boogu 的 ref latents 同时挂在 positive 和 negative 上（CFG 抵消）。
- Boogu tokenizer 自动选择 system prompt，`instruction` 被忽略。

---

## 4. Krea2 管线

Krea2 没有专用 TextEncode/ConfigPreparer；用 **`Krea2ModelConfig_EditUtils`
（专用配置节点，已内置推荐 instruction）+ QwenConfigPreparer + EditTextEncode**
编码参考图，再用 `Krea2EditApply` 给模型打补丁。参考图经条件链自动流入补丁。

```
[LoadImage] ──IMAGE──→ [QwenConfigPreparer #1(主图)] ──configs──→ [#2 ...] ──configs──┐
[Krea2ModelConfig] ──model_config(DICT)─────────────────────────────────────────────┤
CLIP ──→ clip │ VAE ──→ vae │ prompt ──→ prompt ──────────────────────────────→ [EditTextEncode]
        conditioning(带 reference_latents) ──→ KSampler.positive
        latent ──────────────────────────────→ KSampler.latent_image
[UNETLoader Krea2] ──MODEL──→ [Krea2EditApply] ──MODEL──→ KSampler ──→ VAEDecode ──→ SaveImage
```

- `Krea2EditApply` 只需串联在 model 线上（UNETLoader → Krea2EditApply → 采样器）。
  非 Krea2 模型会原样透传，不打补丁。
- `Krea2ModelConfig` 已内置 Krea2 推荐 instruction（描述图像颜色/形状/纹理/空间关系等），
  留空即用默认；其 `model_name` 故意为 `"qwen"`——Krea2 文本编码器基于 Qwen、VAE 为
  Qwen-Image VAE，故走 qwen 编码分支（vae_unit=8）。
- Krea2 图像 latent 以 5D `[B,C,1,H,W]` 进入模型，内部自动 reshape，无需用户处理。

### 4.1 KV 缓存行为（ref_kv_cache，默认 True）

- 内置缓存（原独立节点 Krea2RefKVCache_EditUtils 已删除）。
- 第一次（缓存未命中）denoise step：完整跑 `[text|target|refs]` 序列，捕获每个
  transformer block 的 ref K/V。
- 之后所有 step（以及 ref 指纹、模型、目标形状、ref_pos_match_target 均相同的
  重复生成）：只跑 `[text|target]`，在注意力内部拼接缓存的 ref K/V，约 2x 加速。
- 这是"冻结参考"近似（与 ComfyUI 官方 FluxKVCache 相同）：ref K/V 在深层并非
  严格 step 不变，因此结果**非逐位精确**。
- 缓存按 (模型实例, dtype, ref 内容指纹, 目标 latent 形状, ref_pos_match_target)
  为 key；最多保留 4 个条目防 VRAM 膨胀。
- 缓存路径出任何异常自动回退非缓存 forward 并打印日志，不会中断生成。

### 4.2 ref_pos_match_target 何时需要（默认 True）

- ref 与 target（采样 latent）分辨率**不一致**时开启：把 ref 的 RoPE (h,w) 坐标
  拉伸覆盖整个 target token 网格（分数坐标）。否则较小的 ref 只会与 target 的
  左上角区域对齐，编辑可能只作用在局部。
- ref 与 target 分辨率一致时开不开基本无差别，默认 True 即可。

### 4.3 reset_cache 语义（默认 True）

- True：每次节点执行（每次队列运行）时把缓存字典清空——下一轮第 1 步重新捕获。
  但**同一次采样内部的步骤之间仍然复用**（捕获发生在第 1 步，第 2..N 步复用）。
- False：跨运行保留缓存；换参考图/分辨率后由于 key 含指纹和形状，会自动新建
  条目，但老条目占 VRAM（上限 4 条）。一般保持默认 True。

### 4.4 ref_strength（默认 1.0，需 ref_kv_cache 开启）

控制参考图约束采样的时长：ref 只在 `progress < ref_strength` 的采样前段参与，
之后丢弃缓存的 ref K/V，以纯文生图模式完成剩余步（数学上等价于无 ref forward）。

- 1.0 = ref 全程参与（默认，与旧版行为一致）
- 0.5 = ref 只参与前 50% 采样进度
- 0.0 = ref 完全不参与
- 用途：前段保 ref 结构/身份，后段释放细节与创意，避免 ref 过度约束。
  注意这是训练分布外调度，**建议 0.5~0.7 区间内实验并肉眼验收**。

**推荐实验方法**：固定 seed 与全部其他参数，分别用 1.0 / 0.7 / 0.5 三档各跑一次，
对比结构保持度与细节释放程度，取最满意的档位。`debug_log=True` 时首次丢弃会打印
`[Krea2EditApply] ref dropped at progress=...`，可用来确认丢弃发生的时机。

### 4.5 debug_log（默认 False）

开启后在控制台打印 `[Krea2EditApply]` 前缀日志：缓存捕获/复用、5D reshape、
无 ref 回退等。排查"参考图不生效"时先开它。

---

## 5. 条件缓存复用链路（Save/Load Condition）

适用场景：同一原图 + 同一 prompt 反复编辑/调参，跳过 CLIP+VAE 编码。

```
第一次：
  EditTextEncode.conditioning ──→ [SaveCondition] (filename=my_edit)   # 写入 ComfyUI/models/conditions/my_edit.ckpt

之后：
  [LoadCondition] (filename=my_edit) ──CONDITIONING──→ KSampler.positive
  （latent 仍需自行准备：EmptyLatent 或保留 Encode 节点的 latent 输出）
```

- `LoadConditionFromLoras`：下拉列表显示 `models/loras/` 里的文件名，但实际从
  `models/conditions/` 读同名 `.ckpt`——用于"与 LoRA 同名管理条件缓存"。
- 想保存不含参考图的纯文本条件：先过 `ClearRefLatents` 再 Save。
- 缓存文件与张量设备/dtype 相关，换模型（不同 CLIP/架构）后必须重新生成，
  否则采样器会报 shape 错误或直接串图。

---

## 6. 常见错误与排查

| 症状 | 可能原因 | 处理 |
|---|---|---|
| 节点变红：类型不匹配 | `custom_output`/`pad_info` 是 ANY，直接接 IMAGE/LATENT 端口 | 用 `Any2Image` / `Any2Latent` 转换；LIST 先经 `ListExtractor` 取单项 |
| 节点变红：`At least one image must be provided` | 简易版 Encode 节点一张图都没接 | 至少接 image1 |
| 节点变红：`Index out of range` | `ListExtractor.index` 超过列表长度 | 检查 vae_images/ref_latents 实际数量 |
| mask 不生效 | mask 与 image 尺寸不一致被静默丢弃（控制台有打印） | 把 mask resize 到与输入图一致 |
| 参考图不生效（Krea2） | conditioning 上没有 reference_latents：Encode 用了别的节点，或中途过了会重写 conditioning 的节点 | 确认链路：EditTextEncode.conditioning → 采样器 positive；开 `debug_log` 看是否打印 `NO refs received` |
| Krea2 参考图串图/换了图结果不变 | reset_cache=False 且 key 未变，或旧缓存条目命中 | 保持 reset_cache=True；换图后指纹会变，正常应自动重捕获——若仍异常重启 ComfyUI |
| 编辑只作用在图片左上区域（Krea2） | ref 分辨率小于 target 且 ref_pos_match_target=False | 打开 ref_pos_match_target |
| 出图带黑边 | ref_crop=pad 的黑边未去除 | 用 `CropWithPadInfo` + pad_info 裁剪；Flux2Klein/Qwen 需把 pad_info 从 Encode 节点（或 Extractor）连过去 |
| 出图整体偏移/变色 | 参考图处理分辨率与采样分辨率不匹配 | 用 `AdaptiveLongestEdge` 自适应 ref_longest_edge；或用 `LongestEdgeImageProcess` 显式控制 |
| LoadCondition 输出空 | 文件不存在/加载失败（控制台有错误） | 确认 `models/conditions/*.ckpt` 存在；失败时返回 `[]`，下游必然报错，先解决加载问题 |
| `BooguConfigPreparer` 缩放行为与预期不同 | widget 默认 `longest_edge` 与函数签名默认 `area` 不一致 | 显式选择 ref_resize_mode，以 widget 值为准（以代码为准） |
| 主图不对 | 多个 Preparer 都设了 ref_main_image=True | 只有第一个生效，后续被强制 False；把主图 Preparer 放第一位 |
| 采样结果与参考图无关（Boogu） | 正常现象的一部分：ref 同时挂正负条件，CFG 下抵消 | 以代码为准；调低 cfg 观察 |
