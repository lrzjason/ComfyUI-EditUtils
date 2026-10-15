# ComfyUI-EditUtils

[English](README.md) | [简体中文](README_zh.md)

一套用于 ComfyUI 高级图像编辑的工具节点集合，支持 Qwen、Qwen-Image 2.1、Flux2Klein、Flux3Klein、Krea2 等多种 AI 模型。

## 更新日志

- **20261016** 新增 **FLUX.3 Klein** 支持（`flux3klein` 路由）以及 **bbox（bounding box）画布编辑器**：新增 `Flux3Klein*` 系列节点（画布独立为 `Flux3KleinBBox_EditUtils`）。在参考图上画框、填 **id** 与 **desc**，节点会把 FLUX.3 元素表以 JSON 数组追加到 prompt —— `generation` 行（`{id, bbox, desc}`）用于布局，`editing` 行（`{id, from, src_bbox, tgt_bbox, desc}`）用于 保留/移动/新建/删除。需要 ComfyUI 已支持 FLUX.3。可运行示例：[flux3 editutils example.json](workflows/flux3%20editutils%20example.json)。
- **20261015** 新增 **BreastSizeControl_EditUtils** —— 在节点自带的画布上直接手绘胸部尺寸控制图（`_bd` 条件图）：黑底 + 1~2 个 `#FADBAE` 椭圆，可移动、可变形（圆 ⇄ 椭圆）、可**旋转**、可改色，并有一键套用实测的 flat/small/medium/large/huge/gigantic 档位面积。标记的**面积**就是尺寸旋钮。可运行的 Krea2 示例：[k2 breasts size control.json](workflows/k2%20breasts%20size%20control.json)。
- **20261007** 新增 AlphaNoiseFilter_EditUtils —— 清理 Qwen-Image 2.1 生成图的 alpha 通道（裁掉低值 alpha 噪点、删除小面积 alpha 孤岛、腐蚀/膨胀、羽化），输出干净的 RGBA 图与蒙版。羽化严格向内：渐变带建立在 matte 自身的边缘内侧，轮廓绝不向外扩展到透明区，因此不会有背景色渗出来形成彩色光晕。
- **20260920** 新增 Qwen-Image 2.1 支持（QwenImage21ModelConfig / QwenImage21ConfigPreparer / QwenImage21EditTextEncode / QwenImage21EditApply）。64 通道 16x VAE、Qwen3-VL 文本编码器、vision slot 潜变量拼接以及按参考图独立的 ROPE 偏移。示例工作流：[edit utils qwen image 2.1 example.json](workflows/edit%20utils%20qwen%20image%202.1%20example.json)。需要 ComfyUI 已支持上游的 Qwen-Image 2.1。
- **20260504** 新增 Longest Edge Image Process、Clear Ref Latents、Save/Load Condition 节点，修复 Output Extractors 中 no_refs_cond 输出问题。
- **20260407** 修复额外高度单元填充导致的偏色问题（Extra Height Unit Pad Which Introduce Color Shift）

## 项目简介

ComfyUI-EditUtils 是 ComfyUI-QwenEditUtils 的后续版本，为图像编辑工作流提供更强的能力并支持多种 AI 模型。本项目提供了一套完整的图像编辑工具，具备灵活的配置选项和针对各模型的专门优化。

## 示例

- RunningHub 单图工作流：https://www.runninghub.ai/post/2045207600739913729/?inviteCode=rh-v1279
- RunningHub Simple Krea2 Depth 工作流（工作流：Simple Krea Depth）：https://www.runninghub.ai/post/2082077636234313729/?inviteCode=rh-v1279

## 使用建议

为了在局部编辑时获得更好的一致性，建议配合 Consistency Edit LoRA 使用本工作流：

- Civitai 下载：[Consistency Edit LoRA](https://civitai.com/models/1939453?modelVersionId=2634354)
- Huggingface 下载：[Consistency Edit LoRA](https://huggingface.co/lrzjason/Consistance_Edit_Lora)

## FLUX.3 Klein

EditUtils 通过 `flux3klein` 路由支持 **FLUX.3 Klein**（base 与 step-distilled）编辑，并支持 FLUX.3 用来放置与编辑元素的 **bounding box 元素表**。

> **FLUX.3 klein 不是 FLUX.2 klein 的改名，而是另一套架构**：per-mode 模块 + joint single 模块，三种 token 模式（image、image_cond、text）；文本编码器是 **Qwen3-VL**（8 层堆叠隐状态 → `joint_attention_dim` 20480，文本窗口右填充到 80 的倍数）；参考图以 **`l` 轴负坐标**（-1, -2, …）区分；图像 H/W RoPE 按 `image_rope_scale = 0.5` 缩放；distilled 变体是 **flow-map** 模型（4–16 步、无 CFG）。VAE 与 FLUX.2 相同（32 潜通道、8× 下采样、2×2 patchify → 每 token 16 px），因此 `vae_unit = 16`。

**ComfyUI 已原生支持该架构**（`comfy/ldm/flux3/model.py`、`image_model: "flux3"`），因此参考潜变量走标准 `reference_latents` 条件键，模型自行推导参考位置 id。

**节点：**

| 节点 | 用途 |
|---|---|
| `Flux3KleinModelConfig_EditUtils` | 模型配置（`flux3klein` 路由，`vae_unit=16`）。`flow_map` 记录该 checkpoint 是否为 step-distilled |
| `Flux3KleinConfigPreparer_EditUtils` | 每图配置（`to_ref`、`ref_main_image`、`ref_longest_edge` …）。串联多个节点即可使用多张参考图 |
| `Flux3KleinBBox_EditUtils` | **bbox（元素表）画布**。输入配置链 + 单张图的 config，把元素表挂到该 config 上并返回整条链；同时以 JSON 输出该表（`bbox_json`） |
| `Flux3KleinEditTextEncode_EditUtils` | 一体化简易路径（image1–3），输出与 `EditTextEncode_EditUtils` 一致 |
| `Flux3KleinEditApply_EditUtils` | 模型校验节点。FLUX.3 **不需要**重写 forward，因此这里只校验架构（并能抓出误接 FLUX.2 klein 补丁的情况） |
| `Flux3KleinOutputExtractor_EditUtils` | 拆包 `custom_output`，额外给出 `composed_prompt`（**含元素表**的完整 prompt）与 `bbox_rows` |

**连线**（见 [flux3 editutils example.json](workflows/flux3%20editutils%20example.json)）：

```
UNETLoader (flux3_image_klein_sd_int8_convrot) ─────────────────────────────► KSampler ──► VAEDecode ──► SaveImage
CLIPLoader  (qwen3vl_4b_fp8_scaled, type flux3) ─┐
VAELoader   (flux2-vae) ─────────────────────────┤
LoadImage ──► Flux3KleinConfigPreparer ─┬─ config ──────┐
                                         └─ (configs)    │
LoadImage ──────────────────────────────── image ────────┤
                                          Flux3KleinBBox ┴─► configs ─► EditTextEncode ─► KSampler
Flux3KleinModelConfig ──────────────────────────────────────────────► EditTextEncode
                                                          EditTextEncode ─► ConditioningZeroOut ─► KSampler (负面)
```

![FLUX.3 编辑示例](images/flux3%20editutils%20example.png)

*示例图与其结果，由 [flux3 editutils example.json](workflows/flux3%20editutils%20example.json) 生成。*

**示例中的设置：**

| | |
|---|---|
| 参考图 | 缩放到 `ref_longest_edge = 1536`，`ref_crop = pad`，黑色填充 |
| 元素表 | `bbox_mode = editing`，一行（`"change the hair color to red"`） |
| 采样器 | `euler` / `simple`，**8 步**，CFG **1.0**，denoise 1.0，固定种子 |
| 负面 | 对正面条件做 `ConditioningZeroOut`（蒸馏模型不做 CFG 引导） |
| 模型 | `flux3_image_klein_sd_int8_convrot.safetensors`、`qwen3vl_4b_fp8_scaled.safetensors`（CLIP 类型 `flux3`）、`flux2-vae.safetensors` |

这里**不需要** `Flux3KleinEditApply_EditUtils`——FLUX.3 原生消费 `reference_latents`，所以该图不给模型打补丁。CFG 为 1.0、负面是置零条件，这正是 step-distilled checkpoint 的用法；图中 id 19 的便签建议试试 **4–16 步**。

> ⚠️ **运行前有两点值得注意：**
>
> 1. 该图自带的 "Notes" 便签仍是 BFL 模板的原文：写着 *"50 steps, cfg 3.5"*，并提到本图并不存在的 `Set Reference Multi Latent` 节点。上表描述的是这张图的实际行为。
> 2. `EditTextEncode_EditUtils` 的 `prompt` 控件是**空的**——编辑指令只写在行的 `desc` 里。FLUX.3 的要求（该便签自己也写了）是**先写结果图的完整 caption、把编辑融进描述**，再把元素表追加在后面，所以生成前请先填一段 caption。

画布被拆到**独立节点**，让 Config Preparer 保持简短。`Flux3KleinBBox_EditUtils` 接收 `configs`（其它配置节点搭好的链）、`config`（单张图的 config）与 `image`（**仅作绘制底图**），把元素表挂到那一个 config 上，并返回**替换过该条目**的链——因此把 Preparer 自己的 `configs` 输出接到这里也**不会重复**发送同一张图；示例中 `configs` 未接线，因此把装饰后的 config 追加进去。每张图各配一个 bbox 节点，串联的 bbox 节点会合并成同一张表，与 prompt 中一致。

### 元素表拼在哪里

`EditTextEncode_EditUtils` 会自动完成拼接：它收集链中**所有** config 的 `bbox_rows`，合并成一个 JSON 数组，追加到 prompt 末尾 —— `"<prompt> [ {...}, {...} ]"`，这正是 FLUX.3 要求的格式，没有单独的 box 参数。因此多个 bbox 节点会合并成同一张元素表。

节点用两种方式暴露结果，方便随时核对模型到底收到了什么：

- **`prompt` 输出**（第 7 个接口，`STRING`）—— **实际交给 tokenizer 的文本**，即你的 prompt **加上**元素表。接到文本预览或 SaveText 节点即可查看。
- `custom_output → composed_prompt` —— 同一个字符串，供已使用 Extractor 的图使用。

当没有画任何行时元素表会被省略，`prompt` 就是原本的 prompt。注意 `prompt` 输出**故意放在最后**：ComfyUI 按槽位序号匹配已保存的连线，放在末尾可保证现有工作流的输出连接不受影响。

### 元素表的 JSON 输出

`Flux3KleinConfigPreparer_EditUtils` 还输出 **`bbox_json`**（第 3 个接口，`STRING`）：整张元素表的 JSON 数组，可直接接文本预览、SaveText 或你自己的节点。

```json
[{"id": "wolf_1", "from": null, "src_bbox": null, "tgt_bbox": [817, 120, 950, 167], "desc": "a cyberpunk dog"},
 {"id": "log_1", "from": null, "src_bbox": null, "tgt_bbox": [850, 600, 950, 850], "desc": "a fallen bamboo leaf made out of gold"}]
```

- **没有行时返回空字符串 `""`**，而不是 `"[]"`。包括：开了模式但没画框、以及行不可用（无 id 或无有效 bbox）。
- 字段严格按当前模式输出：`generation` 为 `{id, bbox, desc}`；`editing` 为 `{id, from, src_bbox, tgt_bbox, desc}`，其中 `from` / `src_bbox` / `tgt_bbox` 可为 `null`（这正是 **New** 与 **Remove** 行的表达方式）。内部的 `_kind` 不会出现在输出里。
- 该输出在**最后一位**，因此现有的 `configs` / `config` 连线不受影响。
- **链式合并**：串联多个 bbox 节点时，下游节点输出**整条链**的行（上游在前、本节点在后），与送进 prompt 的表格完全一致；只接一个节点时就是它自己的行。
- **混合模式**：一条链里同时存在两种表不会崩溃，每行按各自结构输出；但 FLUX.3 按"每次一个一致的表"训练，编码节点会打印提示。

### Bounding boxes（元素表）

FLUX.3 把布局当作**追加在 prompt 后面的 JSON 数组**——没有单独的 box 参数。每个框是 `[top, left, bottom, right]`，整数 **0–1000**，从左上角起算，因此与图片尺寸、宽高比无关。

在 `Flux3KleinBBox_EditUtils` 的画布上画框；每行有单行的 **id** 与多行的 **desc**。

### `generation` 与 `editing` 的区别

模式决定**每一行回答的问题**，它不是强度/质量参数。按"你要对图片做什么"来选：

| | `generation` | `editing` |
|---|---|---|
| 何时用 | 用布局**生成新图** | **修改已有图** |
| 行结构 | `{"id", "bbox", "desc"}` | `{"id", "from", "src_bbox", "tgt_bbox", "desc"}` |
| 每行几个框 | **一个** —— 元素放在哪 | **两个** —— 现在在哪（source）与要放哪（target） |
| 是否指定参考图 | 不需要 | 需要，`from` = `ref_image_0`… |
| 默认 | **是** | |

- **`generation`** —— "画一个框，描述框里放什么"。每个元素给一个框，FLUX.3 围绕这些框构图。此时接入的参考图**只作绘制底图**，元素位置完全由表格决定。
- **`editing`** —— 逐元素描述源图，然后只改你关心的行。因为每行同时有 *source* 框与 *target* 框，模型才知道哪些保留、哪些移动、哪些新增、哪些删除。也只有这个模式有 **row**（行类型）按钮。

**没有 `disabled`**：本节点存在的意义就是生成元素表，接上就应该产出，而不是静默什么都不做。默认是较简单的 `generation`。要临时停用，请用 ComfyUI 的 **bypass（Ctrl+B）**。

### generation 模式下没有图像、也没有 config 输入时

`generation` 模式恰恰是**可能完全没有参考图、也没有 config** 的场景——FLUX.3 的布局 prompt 就是"描述 + 框"（API 里是 `{"prompt": caption + rows}`）。所以框必须仍然能到达模型：

- **画了框** → 节点追加一条**只带元素表的 config**（标记 `bbox_table_only=True`，没有 `image`）。编码节点跳过它的图像处理、但照常读取它的行，**布局会进 prompt**。此时 `latent` 为纯文本兜底（flux3 为 `1×128×128×128` 的零 latent，128ch @ 16x 即 2048×2048）；要指定尺寸可另接空 latent 节点。
- **没画框** → `configs` 原样透传，`bbox_json` 为 `""`。

这条"只带元素表"的条目在与图像 config 共存时同样有效：一条链里的图像 config 与该条目按顺序贡献各自的行，因此你可以在同一个 prompt 里既布局新画面、又编辑参考图。

`editing` 的四种行（画布上的 **row** 按钮会一键设好并保持一致；画布下方也直接列出这张表，"行类型 = 含义"，不必悬停）：

| 行 | 直白含义 | `from` | `src_bbox` 与 `tgt_bbox` |
|---|---|---|---|
| **Keep** | 原地不动 | 该图 | 相等 |
| **Move** | 移动 / 缩放 | 该图 | 不同 |
| **New** | 新建（新增 / 替换 / 改色） | `null` | 无来源，只有目标框 |
| **Remove** | 删除 | 该图 | 无目标框 |

### 关于 `from`、`ref`、`src` 的区别

这三个词指的是**同一件事**——元素来自哪张输入图。schema 用三种写法表达同一个概念，所以编辑行看起来像在重复自己：

- **`from`** 是**字段名**
- **`ref_image_0`** 是它的**值**（规范原文：source **reference**, such as `ref_image_0`）
- **`src`** 是同一张图的简称，用在配套字段 **`src_bbox`** 上——字面意思就是"**源**图里的那个框"，即 `from` 那张图里的框

所以一条 `editing` 行只有**三件**真实信息：

| 信息 | 字段 | 含义 |
|---|---|---|
| 它来自哪张图 | `from` | `null` = 不来自任何图，是新建的 |
| 它**在那张图里**的位置 | `src_bbox` | 现在看到的框（实线） |
| 它**在结果里**的位置 | `tgt_bbox` | 应该去到的框（虚线）；`null` = 删除 |

四种行类型只是这三件信息的不同组合——这就是它们读起来像互相变体的原因。界面上该字段标为 **`from ref`**，让字段名与取值都可见；留空表示"不来自图像"（即 **New** 行）。`generation` 模式完全没有这些字段：一行只有一个 `bbox`。

这些 JSON 键由 FLUX.3 规定，**不会**在输出表格里改名。

**画布操作**：空白处拖拽画框；拖框体移动；拖角点缩放；**右键**删除该行；`+ Box` 新增行；`Full frame` 新增 `0,0,1000,1000` 背景行；`Copy caption` 复制 `<id>` 说明文字；**`Refresh`** 重新测量并重绘画布与行表格（画布空白或显示陈旧时点一下即可；该操作不会破坏已有的行与选中状态）。`editing` 模式下用 **edit box**（source / target）选择当前改哪个框：**source** 为实线（元素现在的位置），**target** 为虚线（目标位置），`Remove` 行显示红色 `REMOVE`。`image` 复选框把连接的图像半透明垫底（**只作参考，绝不进入输出**），`labels` 控制是否在框上画 id。

**行布局**：每个元素是一张卡片，字段各占一整行（**全宽**，长 id / 引用 / 描述都不会被截断）：

| `generation` | `editing` |
|---|---|
| `id` | `id` |
| `item desc` | `source_ref` |
| `bbox` 读数 | `item desc`（位于 **`source_ref` 下方**） |
| | `src_bbox` / `tgt_bbox` 读数 |

卡片标题栏显示行号、**行类型**（`Keep` / `Move` / `New` / `Remove`，带颜色）以及操作按钮。两种模式下描述字段都标为 `item desc`：`desc` 描述的是元素本身，对**所有**行类型都适用；在 `editing` 模式下它的含义变为"编辑之后该元素的样子"，这一点写在悬停提示与占位文字里。

### 删除：两种不同的操作

两者刻意分开，因为一个会改变图像，另一个不会。

| 操作 | 位置 | 作用 |
|---|---|---|
| **Remove** | 每行按钮（`editing` 模式） | **从图像中删除该元素**。导出 `{from, src_bbox, tgt_bbox: null}`，模型会用元素背后的内容填充该框。按钮变红并显示 `Removing`；再次点击即可恢复原先画的目标框。 |
| **Delete row** | 每行按钮 | 仅从编辑器表格中删除该行，**不导出任何内容**。用于丢弃画错的行。 |
| **右键** | 画布上 | 等同于对光标所在框执行 *Delete row*（仅编辑器）。 |

因此：要让元素**从画面中消失**，用 **Remove**；要丢掉画错的行，用 **Delete row**。Remove 需要来源，所以该行必须填 `source_ref`——为空时会自动填入 `ref_image_0`。

**提示：**

- 在描述里用 `<id>` 引用元素（如 `<silhouette_1>`），让文本与表格一致。
- 框是**引导而非裁剪遮罩**，元素可以略微超出框。
- 保持设计框时所用的宽高比：0–1000 网格会随画面拉伸。
- 新建元素的框不要太小（BFL 实测约 40×25 px 的新元素常不出现）。
- 做编辑时，指令里也要说明"新增/删除"，并给必须保持不动的元素补一行 Keep。
- 文字建议每行一个框，并在 `desc` 里用引号写出准确文字。
- 采样建议：distilled checkpoint 是 flow-map 模型——约 4–16 步、CFG 1.0（不用负面提示词）；base 模型使用 CFG。

## Krea2 胸部尺寸控制

一个训练好的控制 LoRA + `BreastSizeControl_EditUtils`，把胸部尺寸变成一个**画出来的旋钮**：在画布上画 1~2 个标记，标记的**面积**（以及位置、角度）就决定结果。提示词只描述画面，**刻意不写任何尺寸词** —— 几何量应该只来自参考图。

![Krea2 胸部尺寸控制示例](images/k2%20breasts%20size%20control.png)

*示例工作流与生成结果，来自 [k2 breasts size control.json](workflows/k2%20breasts%20size%20control.json)。截图里画布的状态行显示 `832x1216 - total 2.36% (small) - marker 1/2: 1.06% @ -14.6°`，即两个偏小、略微倾斜的标记。*

**节点：**

| 节点 | 作用 |
|---|---|
| `BreastSizeControl_EditUtils` | 画布：绘制作为参考的 `_bd` 条件图（黑底 + `#FADBAE` 标记）。面积 = 尺寸，位置 = 在哪，角度 = 倾斜 |
| `Krea2ModelConfig_EditUtils` | Krea2 模型配置（`model_name="qwen"` 分支、`vae_unit=8`、内置 instruction） |
| `QwenConfigPreparer_EditUtils` | 把条件图包成参考图：`to_ref`、`ref_main_image`、`ref_longest_edge=1024`、`ref_crop=pad`、`to_vl`、`vl_target_size=384`、`ref_resize_mode=longest_edge` |
| `EditTextEncode_EditUtils` | 把提示词 + 参考图编码成 conditioning、采样 latent 与 `pad_info` |
| `Krea2EditApply_EditUtils` | 给模型打补丁以注入参考图潜变量（`ref_pos_match_target`、`ref_kv_cache=1`、`ref_strength=1.0`） |

**接线**（见 [k2 breasts size control.json](workflows/k2%20breasts%20size%20control.json)）：

```
UNETLoader (krea2_raw_int8_convrot) ──► Krea2EditApply ──► [合并 LoRA: krea2_turbo] ──► [合并 LoRA: breasts_control] ──► KSampler ──► VAEDecode ──► SaveImage
CLIPLoader (qwen3vl_4b_fp8_scaled, type krea2) ─┐
VAELoader  (qwen_image_vae) ────────────────────┤
PrimitiveStringMultiline（画面提示词）──────────┼──► EditTextEncode_EditUtils ──► positive + latent ──► KSampler
Krea2ModelConfig ───────────────────────────────┤                                    │
BreastSizeControl ──► QwenConfigPreparer ───────┘                                    └──► ConditioningZeroOut ──► negative
```

**依赖与设置：**

- `breasts_control_lrzjason_20261015.safetensors` —— 控制 LoRA（强度 1.0），叠加在 `krea2_turbo_lora_rank_64_bf16` 之上。
- 工作流用 **Comfyui-LoraUtils** 的 `LoadLoraOnly → MergeLoraToModel` 合并这两个 LoRA（[Comfyui-LoraUtils](https://github.com/lrzjason/Comfyui-LoraUtils)）；任何等价的「加载 LoRA + 给模型打补丁」节点对都可以。
- 模型文件：`krea2_raw_int8_convrot.safetensors`（UNET）、`qwen3vl_4b_fp8_scaled.safetensors`（CLIP，type `krea2`）、`qwen_image_vae.safetensors`（VAE）。
- 示例采样参数：`euler / simple`、**8 步**、CFG **1.0**、denoise 1.0、固定种子（有 turbo LoRA 才敢用这么短的步数）。
- 示例提示词**只**描述画面（人物、服装、景别、光线），不含任何尺寸措辞 —— 请保持这一点，否则文字会和参考图争夺「尺寸」这个量。这正是数据集侧的 caption 规则：**画面文字，几何条件图**。
- 图里还有一个未接出的 `CropWithPadInfo_EditUtils` 与一个 `GetImageSize` 节点，保存链路并不需要它们。

**让标记保持在训练分布内：** 这个 LoRA 就是在该标记样式上训练的，所以请保持黑底、填充 `#FADBAE`（或相近的高亮色）、1~2 个面积代表尺寸的椭圆。画布的预设按钮会把**标记总面积**设到实测的每档中位数 —— 画面占比 `flat 1.49% / small 2.94% / medium 3.22% / large 7.18% / huge 13.95% / gigantic 16.50%` —— 这是最稳妥地落在 LoRA 真正见过的档位上的方式。接了 `image` 时条件图会被强制为目标图的精确尺寸。

> ⚠️ **注意：** 该控制 LoRA 与 Krea2 编辑路径仍在开发中 —— 效果取决于 LoRA 权重与参考强度。

## Qwen-Image 2.1

EditUtils 支持 **Qwen-Image 2.1** 编辑：简易路径最多 3 张参考图，通过配置链可支持无限张。Qwen-Image 2.1 与 Qwen-Image 1.0 的工作方式不同，qwen21 系列节点已经为你处理好这些差异：

- **Qwen3-VL 文本编码器**：提示词与参考图一起编码；每张参考图的潜变量会在编码器的 vision slot（`image_slots`）处拼接进文本序列 —— 不再需要 `Picture n:` 这类提示词前缀。
- **全新 VAE**：64 通道潜变量，空间下采样 16x（支持 RGBA）。参考图对齐到 **16 像素的倍数**（与 VAE 下采样倍数一致）。
- **统一的参考图缩放**：视觉塔（vision tower）与 VAE 使用**同一张**缩放后的图像（编码器用白底合成后的 RGB，VAE 用完整 RGBA），因此不再有单独的 `vl_target_size` 流程。

![Qwen-Image 2.1 编辑示例](images/qwen%20image%2021%20example.png)

*使用 [edit utils qwen image 2.1 example.json](workflows/edit%20utils%20qwen%20image%202.1%20example.json) 生成的结果。*

**节点说明：**

| 节点 | 用途 |
|---|---|
| `QwenImage21ModelConfig_EditUtils` | 模型配置（`qwen_image21` 路径，`vae_unit=16`）。instruction 留空 = 使用内置 T2I 模板；填写自定义 instruction = 使用自定义系统提示词。 |
| `QwenImage21ConfigPreparer_EditUtils` | 单张图片的配置：`to_ref`、`ref_main_image`、`ref_longest_edge`（按 16 对齐）、`ref_crop`、`mask`、`rope_x_offset / rope_y_offset`。多张参考图时串联多个该节点。 |
| `QwenImage21EditTextEncode_EditUtils` | 单节点简易路径（image1–3），输出与 `EditTextEncode_EditUtils` 相同。 |
| `QwenImage21EditApply_EditUtils` | 可选的模型补丁，启用按参考图独立的 ROPE 偏移（区域编辑）。只需连接 model 线 —— 偏移量通过 conditioning 链路传递。 |

**连线方式**（见 [edit utils qwen image 2.1 example.json](workflows/edit%20utils%20qwen%20image%202.1%20example.json)）：

```
UNETLoader (qwen_image_2.1) ──► [QwenImage21EditApply] ──► KSampler ──► VAEDecode ──► CropWithPadInfo ──► SaveImage
CLIPLoader  (type qwen_image) ─┐
VAELoader   (qwen_image_2.1_vae) ┤
LoadImage ──► QwenImage21ConfigPreparer ─┐
QwenImage21ModelConfig ──────────────────┴──► EditTextEncode_EditUtils ──► KSampler
```

`QwenImage21EditApply_EditUtils` 是可选的 —— 只有需要 `rope_x_offset / rope_y_offset` 区域控制时，才把它接在 `UNETLoader` 与 `KSampler` 之间。

**注意事项：**

- 需要 ComfyUI 已支持上游的 Qwen-Image 2.1（`qwen_image21` 模型）；旧版本上模型无法加载，此时 EditApply 节点会原样透传模型。
- 主图填充后的潜变量即为采样的起始潜变量；解码后请用 `pad_info → CropWithPadInfo_EditUtils` 去掉填充，得到未填充的结果（与 Qwen 1.0 路径的流程一致）。
- 2.1 的 VAE 支持 RGBA，因此生成图可以带 alpha 通道。原始 alpha 通常在透明区域散布着低值噪点，建议接 `VAEDecode → AlphaNoiseFilter_EditUtils` 清理，得到干净的抠图与蒙版。
- `rope_x_offset / rope_y_offset` 只有在图中有 `QwenImage21EditApply_EditUtils` 时才生效；偏移全为 0 时模型走原生路径（不影响 prefix KV cache）。
- 请保持 Config Preparer 上的 `to_vl` 启用 —— 关闭后参考图会被拼接到文本序列之后，这是未经训练的分支。
- 建议的起步参数（与示例工作流一致）：Euler / Simple，25 步，CFG 1.0。

## 工作流

示例工作流位于 [workflows](workflows/) 目录：

- **[flux3 editutils example.json](workflows/flux3%20editutils%20example.json)** - 使用 bounding box 元素表的 FLUX.3 Klein 编辑工作流。接线 `LoadImage → Flux3KleinConfigPreparer_EditUtils → Flux3KleinBBox_EditUtils → EditTextEncode_EditUtils ← Flux3KleinModelConfig_EditUtils → KSampler → VAEDecode → SaveImage`，负面由 `ConditioningZeroOut` 提供。BBox 节点只接 Preparer 的 `config`（`configs` 输入留空），因此装饰后的 config 会被追加进链。
  - 模型：`flux3_image_klein_sd_int8_convrot.safetensors`（UNET）、`qwen3vl_4b_fp8_scaled.safetensors`（CLIP，type `flux3`）、`flux2-vae.safetensors`（VAE）。
  - 示例中的采样设置：`euler / simple`，**8 步**，CFG 1.0，固定种子（蒸馏 flow-map checkpoint 不做引导；建议区间 4–16 步）。没有 `Flux3KleinEditApply_EditUtils`——FLUX.3 原生消费 `reference_latents`。
  - 需要 ComfyUI 已支持 FLUX.3。
- **[edit utils qwen image 2.1 example.json](workflows/edit%20utils%20qwen%20image%202.1%20example.json)** - Qwen-Image 2.1 编辑工作流（单张参考图）。使用 `UNETLoader` + `CLIPLoader`（type 选 `qwen_image`）+ `VAELoader` 加载模型，然后接线 `LoadImage → QwenImage21ConfigPreparer_EditUtils → EditTextEncode_EditUtils ← QwenImage21ModelConfig_EditUtils → KSampler → VAEDecode → CropWithPadInfo_EditUtils`，用于去掉主图的填充。
  - 参考图潜变量在 vision slot（`image_slots`）处拼接进文本序列；2.1 VAE 为 64 通道、16x 空间下采样，因此参考图按 16 像素倍数对齐。
  - Config Preparer 上的 `rope_x_offset / rope_y_offset` 在该模型被 `QwenImage21EditApply_EditUtils` 打补丁后，可平移参考图在画布上的位置（区域编辑）—— 示例工作流未包含该节点。
  - 示例中的采样设置：`euler / simple`，25 步，CFG 1.0。需要 ComfyUI 已支持上游的 Qwen-Image 2.1。
- **[Simple Krea2 Depth.json](workflows/Simple%20Krea2%20Depth.json)** - 配合 depth LoRA 的简单 Krea2 编辑工作流。接线 `LoadImage → Krea2ModelConfig_EditUtils → EditTextEncode_EditUtils → Krea2EditApply_EditUtils → KSampler`，通过 `UNETLoader + LoraLoaderModelOnly` 加载模型 —— 参考图潜变量会自动沿 conditioning 链路传递。
  - RunningHub 在线版本：https://www.runninghub.ai/post/2082077636234313729/?inviteCode=rh-v1279
  - Depth LoRA 下载：[Krea2 Depth LoRA (Civitai)](https://civitai.com/models/2815790/krea2-depth-lrzjason-20260729)
- **[k2 breasts size control.json](workflows/k2%20breasts%20size%20control.json)** - Krea2 胸部尺寸控制：`BreastSizeControl_EditUtils` 在画布上画出 `_bd` 参考图，`QwenConfigPreparer_EditUtils` 把它包成参考图，`Krea2EditApply_EditUtils` 注入，两个合并的 LoRA（`krea2_turbo` + `breasts_control_lrzjason_20261015`）驱动采样。`euler / simple`，8 步，CFG 1.0。LoRA 合并节点需要 [Comfyui-LoraUtils](https://github.com/lrzjason/Comfyui-LoraUtils)。该文件是 **API 格式**导出（另外两个是 UI 图）—— 当前 ComfyUI 前端可以用常规的「加载 / 拖入」路径读取 API JSON，它与上面截图里的图完全一致。

> ⚠️ **注意：** Krea2 编辑支持仍在开发中 —— 节点接口与行为可能发生变化。

## 能力特性

- EditUtils 支持无需像素位移的直接高分辨率编辑，最高可达 2xxx ~ 3xxx 分辨率
- 支持多图输入：
  - 简易工作流：最多 3 张图
  - 单图与多图工作流：图片数量不限（串联多个配置节点）

## 填充边缘控制（Pad Edge Control）

当 `ref_crop` 设为 `pad` 时，图像会被填充到 VAE 单元尺寸的整数倍。填充区域的外观现在可以按图配置，通过配置节点上的 `pad_mode`、`pad_color`、`pad_color_rgb`、`pad_noise`（`QwenConfigPreparer_EditUtils` / `QwenImage21ConfigPreparer_EditUtils` / `Flux2KleinConfigPreparer_EditUtils` / `Flux3KleinConfigPreparer_EditUtils`）、`ConfigJsonParser_EditUtils` 的 JSON，或独立的 `LongestEdgeImageProcess_EditUtils` 节点设置：

- **pad_mode**（默认：`color`）：
  - `color`：用 `pad_color` 纯色填充
  - `edge`：复制最外侧一行／一列像素
  - `mirror`：在边界处镜像内容
  - `border_mean`：用最近 8 个边缘像素的均值填充每一侧
- **pad_color**（默认：Qwen 1.0 与 Qwen-Image 2.1 为 `white`；Flux2Klein / Flux3Klein 与独立的 `LongestEdgeImageProcess_EditUtils` 为 `black`）：下拉选项 `red`、`green`、`blue`、`white`、`black`、`custom`。
- **pad_color_rgb**（默认：`-`）：可选覆盖项，**6 位 hex 颜色，`#` 可有可无**（例如 `FF0000` 或 `#FF0000`）。只要是合法 hex 就会覆盖 `pad_color`（无论下拉选了什么）；其它值（`-`、长度不对、含非 hex 字符、纯数字、`r,g,b`）一律无效、不生效，以下拉选项为准。`custom` 本身不带颜色，hex 无效时回退到该模型默认色。
- **pad_noise**（默认：`0.0`）：**只**给填充区域加高斯抖动（标准差），图像内容完全不受影响；配合纯色填充推荐 0.02–0.05。

**为什么需要它：** Qwen 系列（1.0 与 Image 2.1）训练时使用黑边填充，因此黑色填充在生成时容易被放大成大片黑边。两条 Qwen 路径默认都改为白色填充。如果纯色仍不够，`border_mean` 或 `mirror` 是内容自适应的替代方案，且任何模式配合少量 `pad_noise` 都会更稳。

填充区域的 mask 语义保持不变（始终被掩掉），`CropWithPadInfo_EditUtils` 依旧能正确裁掉填充。

## 节点分类

文档：

- [节点文档](nodes_doc.md)

## 主要特性

- **多模型支持**：可用于 Qwen、Qwen-Image 2.1、Flux2Klein、Flux3Klein、Boogu 与 Krea2 模型，实现多样化的图像编辑
- **灵活配置**：针对每张图片单独配置参考图与 VL 处理选项
- **填充边缘控制**：可配置填充模式／颜色／噪声（Qwen 1.0 与 Qwen-Image 2.1 默认白边），避免生成时黑边被放大
- **统一接口**：通过配置节点，单个 `EditTextEncode_EditUtils` 节点即可适配多种模型
- **高级处理**：支持带多张参考图的复杂图像编辑工作流
- **完整输出**：输出字典包含所有中间处理结果
- **模块化设计**：配置、处理与提取节点相互分离，灵活度最高

## 新增节点

### LongestEdgeImageProcess_EditUtils
按目标长边对图像进行缩放与填充的工具节点，使用与 EditTextEncode 相同的处理逻辑。当你只需要图像预处理、而不需要 CLIP/VAE 编码时很有用。

**输入：**
- `image`：输入图像
- `ref_longest_edge`：目标长边尺寸（默认：1024）
- `ref_crop`：裁剪方式 - "pad"、"center" 或 "disabled"（默认："pad"）
- `ref_upscale`：放大方式（默认："lanczos"）
- `vae_unit`：用于填充对齐的 VAE 单元大小（默认：8）
- `pad_mode`（可选）：填充方式 - "color"、"edge"、"mirror" 或 "border_mean"（默认："color"）
- `pad_color`（可选）：下拉选项 - "red"、"green"、"blue"、"white"、"black" 或 "custom"（默认："black"；Qwen 系模型建议改为 "white"）
- `pad_color_rgb`（可选）：覆盖色，6 位 hex，"#" 可有可无（如 "FF0000" 或 "#FF0000"），合法时覆盖 `pad_color`（默认："-" 表示不生效）
- `pad_noise`（可选）：只加到填充区域的高斯抖动标准差（默认：0.0，可试 0.02–0.05）

**输出：**
- `processed_image`：缩放/填充后的图像
- `pad_info`：填充信息字典
- `scale_by`：缩放系数

**使用场景：** 在传给其他节点之前先做长边缩放预处理，或在编码流程之外复用同一套图像处理管线。

### Flux3KleinConfigPreparer_EditUtils

FLUX.3 Klein 的每图配置节点（`to_ref` / `ref_main_image` / `ref_longest_edge` / 填充 等）。**bbox 画布已拆分到独立节点** `Flux3KleinBBox_EditUtils`，以保持本节点简短。

**输出：** `configs` (LIST) · `config` (ANY)。

**使用场景：** 描述每张参考图如何缩放、是否作为主图等；元素表请接到 `Flux3KleinBBox_EditUtils`。

### Flux3KleinBBox_EditUtils（bbox 画布 / 元素表）

承载 FLUX.3 的 **bounding box 元素表画布**，独立于 Config Preparer，因此行编辑区域可以做得更高（行区域上限 420px，比原先高得多）。

**输入：**
- `configs`（可选）：其它配置节点搭好的 config 链
- `config`（可选）：**单张图**的 config（Preparer 的 `config` 输出）；元素表挂到它上面
- `image`（可选）：**仅作绘制底图**，不参与编码
- `bbox_mode`：`generation`（**新图布局**，行 = `{id, bbox, desc}`，一个框 = 一个元素放哪里）/ `editing`（**改已有图**，行 = `{id, from, src_bbox, tgt_bbox, desc}`，每行有"现在在哪"的 source 框与"要放哪"的 target 框，并写明来自哪张参考图）。**没有 `disabled`**：本节点就是为了生成元素表，接上就应该产出；要临时停用请用 ComfyUI 的 bypass（Ctrl+B）
- `bbox_rows`：画布托管的行 JSON（**隐藏**，请用画布编辑，不要手写）
- `bbox_ids` / `bbox_descs`：隐藏镜像字段，用于工作流往返恢复 id 与描述

**画布操作：** 空白处拖拽画框；拖框体移动；拖角点缩放；**右键**删除该行；`+ Box` 新增行；`Full frame` 新增整幅背景行；`Copy caption` 复制 `<id>` 说明文字。`editing` 模式下用 **edit box** 切换要改的是 **source**（实线，元素现在的位置）还是 **target**（虚线，目标位置）；四种行类型用 **row** 按钮一键切换。字段各占**一整行全宽**：`id` / `source_ref` / `item desc`（描述在 `source_ref` 下方）。每行有两个删除按钮：**Remove** = 从图像中删除该元素（`tgt_bbox=null`），**Delete row** = 仅从表格移除该行。`image` 复选框把连接的图像半透明垫底（仅作参考），`labels` 控制是否在框上画 id。

**输出：** `configs` (LIST) · `bbox_json` (STRING)。**没有单独的 `config` 输出**——`configs` 里已经是"被装饰的那条 config + 元素表"打包在一起，再输出一个 config 只会造成同一份数据的第二个真相来源。

- **合并规则**：把元素表挂到 `config`，然后在 `configs` 中**替换**匹配的那一条（先按对象身份，再按稳定的 `config_id`）；找不到才追加。所以把 Preparer 自己的 `configs` 输出接到这里**不会重复发送同一张图**。
- **没有 `config` 输入时**（`generation` 模式的正常用法：纯文本 + 框，根本不需要参考图）：
  - **画了框** → 追加一条**只带元素表的 config**（`bbox_table_only=True`，没有 `image`）。编码节点会跳过它的图像处理、但照常读取它的行，所以**布局依然会进 prompt**。此时 `latent` 是纯文本兜底（flux3 为 `1×128×128×128`）；要指定尺寸可另接空 latent 节点。
  - **没画框** → 完全透传 `configs`，`bbox_json` 为 `""`。
- 只接 `config`、不接 `configs` → 直接把装饰后的 config 追加成一条。
- `bbox_json`：整张元素表的 **JSON 数组字符串**（如 `[{"id": "wolf_1", "from": null, "src_bbox": null, "tgt_bbox": [817, 120, 950, 167], "desc": "a cyberpunk dog"}]`），可直接接文本预览 / SaveText。**没有行时返回空字符串 `""`**。串联多个 bbox 节点时输出整条链的行（与 prompt 中的表格一致）。

**使用场景：** 用框告诉 FLUX.3 每个元素放在哪里（`generation`），或在编辑时逐元素 保留/移动/新建/删除（`editing`）。元素表会由编码节点自动以 JSON 数组追加到 prompt 末尾。

### ClearRefLatents_EditUtils
从 conditioning 中剥离参考图潜变量的工具节点，输出不含任何参考潜变量的干净 conditioning。

**输入：**
- `conditioning`：带参考图潜变量的 conditioning

**输出：**
- `conditioning`：已清除参考图潜变量的同一个 conditioning

**使用场景：** 当你只想使用文本编码结果、不需要图像参考时，从 conditioning 中移除参考图潜变量。

### SaveCondition_EditUtils
将 conditioning 张量保存为 `models/conditions` 目录下的 `.ckpt` 文件。

**输入：**
- `condition`：要保存的 conditioning
- `filename`：输出文件名（默认："condition_tensor"）

**使用场景：** 持久化保存 conditioning 张量，之后无需重新编码即可复用。

### LoadCondition_EditUtils
从 `models/conditions` 目录下的 `.ckpt` 文件加载 conditioning 张量。

**输入：**
- `filename`：从 conditions 目录中可用的 `.ckpt` 文件里选择

**输出：**
- `conditioning`：加载到的 conditioning 张量

**使用场景：** 在新的工作流中复用之前保存的 conditioning 张量。

### LoadConditionFromLoras_EditUtils
列出 loras 目录中的文件，并尝试从 conditions 目录加载同名的 `.ckpt` 文件。

**输入：**
- `filename`：从可用的 lora 文件中选择

**输出：**
- `conditioning`：加载到的 conditioning 张量

### DiffMask_EditUtils
生成用于突出显示两张图像差异的蒙版的工具节点。适用于需要定位变化区域的编辑任务。

**输入：**
- `image1`：第一张图
- `image2`：第二张图
- `threshold`：忽略细微差异的阈值（0.0-1.0，默认：0.05）

**输出：**
- `mask`：突出显示两张图差异的蒙版

**使用场景：** 对比原图与编辑后的图像，为选择性编辑或重绘生成蒙版。

### AlphaNoiseFilter_EditUtils
清理 Qwen-Image 2.1 生成图的 alpha 通道。Qwen-Image 2.1 可以生成带透明背景的图，但原始输出通常在透明区域散布着数值很低、肉眼几乎不可见的 alpha 噪点——单独看不明显，却会破坏抠图、合成与基于蒙版的编辑。本节点去除这类噪点，输出干净的 alpha 图与对应蒙版。

**输入：**
- `image`：输入图（RGBA，如 Qwen-Image 2.1 VAE 的 `VAEDecode` 输出）
- `mask`：（可选）输入图没有 alpha 通道（RGB）时作为 alpha 来源
- `alpha_threshold`：低于此值的 alpha 视为低值噪点并裁为 0（0.0-1.0，默认：0.10）
- `min_area`：删除像素数小于此值的独立 alpha 孤岛（0-65536，默认：100）
- `feather`：向内羽化渐变带的宽度（像素）——在 matte 边界处完全透明，向内平滑升到完全不透明（0-32，默认：2.0）
- `invert_mask`：蒙版输出改为 `1 - alpha`（白=透明，ComfyUI 重绘惯例，默认：False）

**输出：**
- `image`：清理后的 RGBA 图
- `mask`：alpha 对应的蒙版（默认白=不透明）

**处理流程：** 低值裁剪 → 小孤岛删除 → 向内羽化。**只修改 alpha 通道**——输出的 RGB 与输入完全一致。

**羽化只向内工作，绝不外扩。** 渐变带整条都落在生成器已经标记为主体的区域内，因此羽化不可能让原本透明的像素变可见。实测 `alpha noise.png`：feather 取 0–16 之间任意值时，轮廓外 alpha>0 的像素数恒为 **0**，且任何位置 alpha 都不会超过原始 matte。渐变从边界处完全透明向内单调升到完全不透明：

| feather | 0 px | 1 px | 2 px | 3 px | 4 px | 5 px | 6 px | 8 px |
|---|---|---|---|---|---|---|---|---|
| 0（原始 matte） | 0.449 | 0.877 | 0.957 | 0.977 | 0.986 | 0.992 | 0.995 | 0.998 |
| 2 | 0.020 | 0.493 | 0.951 | — | — | — | — | — |
| 4 | 0.005 | 0.160 | 0.475 | 0.823 | 0.986 | — | — | — |
| 8 | 0.001 | 0.044 | 0.149 | 0.314 | 0.510 | 0.686 | 0.842 | 0.997 |

**为什么不能用对称模糊：** 对称高斯会把渐变带推到 matte 之外，而那些像素的 RGB 仍然是生成器留下的背景色（Qwen-Image 2.1 的 VAE 是一片紫色 (152,50,196)）。实测 feather=4 时对称模糊会点亮 **13758** 个原本 alpha=0 的像素，它们的 RGB 均值是 (92,47,102)，在深色背景上就是一圈紫晕；改成向内羽化后紫边量恒为 **0.000**。

**代价：** 渐变带内的 alpha 被压低，轮廓面积因此收缩（feather=2 → −2.79%，feather=4 → −5.27%，feather=8 → −10.13%），细结构（如毛发）在羽化带宽大于自身宽度时会明显变淡——羽化带内侧 2px 范围的平均 alpha 从 0.583（不羽化）降到 0.152（feather=2）。这是「不外扩」的必然代价。

**使用场景：** 把 Qwen-Image 2.1 的 alpha 生成图变成干净的抠图/蒙版：`VAEDecode → AlphaNoiseFilter`，然后保存图片，或用蒙版驱动重绘/编辑节点。

### BreastSizeControl_EditUtils
在节点内的画布上直接手绘**胸部尺寸控制图（`_bd` 条件图）**：黑底 + 1~2 个实心椭圆标记，默认填充色 `#FADBAE`，标记的**面积**是尺寸旋钮、**位置**是乳房位置。输出与 `E:\images\breasts_size_control` 里现有 `_bd` 条件图**同一契约**——实测 60 张：每张只有 **2 种颜色**（硬边、无抗锯齿），标记是 `box_px` 的**内切椭圆**（IoU 中位 0.988），60/60 的输出尺寸与 `target_size_px` 完全相同。

画布是 **WYSIWYG** 的：你看到的黑底和色块**就是**渲染出的像素（浏览器 `ctx.ellipse` 与节点光栅化用的是同一个内点判定）。几何量以归一化坐标存放在隐藏的 `markers` 输入里，所以改分辨率不会丢画。

**输入：**
- `width` / `height`：未接 `image` 时的画布尺寸（默认 832x1216）
- `markers`：画布持有的隐藏 JSON 载荷——用画布编辑，不要手改
- `antialias`：`False`（默认）与现有 `_bd` 图的 2 色硬边完全一致；`True` 为 4×4 超采样抗锯齿（面积不变）
- `image`（可选）：接入目标图后，输出被强制为该图的**精确像素尺寸**（batch 取第一帧）——`_bd` 契约要求条件图与目标同尺寸

**输出：**
- `image`：条件图，`(1, H, W, 3)`

**画布操作：** 空白处拖拽新建标记（最多 2 个）；拖本体移动；拖**上/下/左/右边中点手柄**单独改 `ry`/`rx`（这是把圆拖成椭圆的工具，真实 `_bd` 的 bbox 长宽比中位约 0.91）；拖**角手柄**或在标记上**滚轮**等比缩放（保持正圆）；**Shift** + 拖边手柄等比；拖椭圆上方的**旋转手柄**即可**旋转标记**（`↺ 15°` / `↻ 15°` / 角度输入框 / `0°` 用于精确角度）；**右键**删除；**预设按钮**套用实测档位面积；**取色器**改填充色（默认 `#FADBAE`）；**guide** 勾选框把接入图以 45% 透明度垫在黑底上对位，**绝不进入输出像素**。

**旋转：** 角度单位为度，**正数 = 屏幕上顺时针**（与 `krea2_geom_lab` 约定一致），归一化到 `(-180, 180]`。旋转只改朝向、**不改面积**，所以不影响尺寸档与预设；边/角手柄与命中判定都跟着旋转，拖拽始终沿椭圆自身的轴生效。`angle` 非法时只回退成 0，不会丢掉该标记。

**尺寸档预设**（取数据集分析的实测中位面积，按**总面积**计）：

| flat | small | medium | large | huge | gigantic |
|---|---|---|---|---|---|
| 1.49% | 2.94% | 3.22% | 7.18% | 13.95% | 16.50% |

预设作用于**整张图的总面积**并按两标记当前比例分配——若按单标记给面积，两个标记时总面积会翻倍到档位的两倍，与训练分布不符。左右想不一样大，就用边手柄分别拖。

**使用场景：** 制作「改变胸部大小」编辑对的参考图一侧。接入目标图保证尺寸严格一致，一个按钮选好档位，输出直接接进编辑管线的 `_bd` 参考输入。完整可运行的例子（Krea2 + 训练好的控制 LoRA）见 [k2 breasts size control.json](workflows/k2%20breasts%20size%20control.json) —— 参见 [Krea2 胸部尺寸控制](#krea2-胸部尺寸控制)。

**实测：** 六档面积渲染误差 ≤ 0.12%；两标记分离时总面积恰为单标记的 2 倍；重叠时取并集而非相加；旋转在 7 个角度下面积不变，90° 旋转与交换 `rx`/`ry` 完全等价（IoU 1.0000）；用 25 张现有条件图各自的 sidecar 反算几何重渲染，与真实 `_bd.png` 像素的 IoU 中位 **0.9866**、最小 **0.9757**——已达「内切椭圆」模型自身的上限。

## 安装

1. 克隆或下载本仓库到你的 ComfyUI 的 `custom_nodes` 目录。
2. 重启 ComfyUI。
3. 节点将出现在 "advanced/conditioning" 分类下。

## 更新记录

### ComfyUI-EditUtils 与 ComfyUI-QwenEditUtils 的对比
ComfyUI-EditUtils 是 ComfyUI-QwenEditUtils 的后续版本，包含以下改进：
- 多模型支持（Qwen、Qwen-Image 2.1、Flux2Klein、Flux3Klein、Boogu 与 Krea2）
- 采用配置节点的统一节点架构
- 更高的灵活性与模块化程度
- 更好的代码组织与可维护性

## 联系方式
- **Twitter**: [@Lrzjason](https://twitter.com/Lrzjason)  
- **Email**: lrzjason@gmail.com  
- **QQ 群**: 866612947  
- **微信号**: fkdeai
- **Civitai**: [xiaozhijason](https://civitai.com/user/xiaozhijason)

## 赞助我开发更多开源项目：
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
