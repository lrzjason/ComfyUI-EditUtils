# ComfyUI-EditUtils

[English](README.md) | [简体中文](README_zh.md)

一套用于 ComfyUI 高级图像编辑的工具节点集合，支持 Qwen、Qwen-Image 2.1、Flux2Klein、Krea2 等多种 AI 模型。

## 更新日志
20260920 新增 Qwen-Image 2.1 支持（QwenImage21ModelConfig / QwenImage21ConfigPreparer / QwenImage21EditTextEncode / QwenImage21EditApply）。64 通道 16x VAE、Qwen3-VL 文本编码器、vision slot 潜变量拼接以及按参考图独立的 ROPE 偏移。示例工作流：[edit utils qwen image 2.1 example.json](workflows/edit%20utils%20qwen%20image%202.1%20example.json)。需要 ComfyUI 已支持上游的 Qwen-Image 2.1。
20260504 新增 Longest Edge Image Process、Clear Ref Latents、Save/Load Condition 节点，修复 Output Extractors 中 no_refs_cond 输出问题。
20260407 修复额外高度单元填充导致的偏色问题（Extra Height Unit Pad Which Introduce Color Shift）

## 项目简介

ComfyUI-EditUtils 是 ComfyUI-QwenEditUtils 的后续版本，为图像编辑工作流提供更强的能力并支持多种 AI 模型。本项目提供了一套完整的图像编辑工具，具备灵活的配置选项和针对各模型的专门优化。

## 示例

RunningHub 单图工作流：
https://www.runninghub.ai/post/2045207600739913729/?inviteCode=rh-v1279

RunningHub Simple Krea2 Depth 工作流（工作流：Simple Krea Depth）：
https://www.runninghub.ai/post/2082077636234313729/?inviteCode=rh-v1279

## 使用建议

为了在局部编辑时获得更好的一致性，建议配合 Consistency Edit LoRA 使用本工作流：

- Civitai 下载：[Consistency Edit LoRA](https://civitai.com/models/1939453?modelVersionId=2634354)
- Huggingface 下载：[Consistency Edit LoRA](https://huggingface.co/lrzjason/Consistance_Edit_Lora)

## Qwen-Image 2.1

EditUtils 支持 **Qwen-Image 2.1** 编辑：简易路径最多 3 张参考图，通过配置链可支持无限张。Qwen-Image 2.1 与 Qwen-Image 1.0 的工作方式不同，qwen21 系列节点已经为你处理好这些差异：

- **Qwen3-VL 文本编码器**：提示词与参考图一起编码；每张参考图的潜变量会在编码器的 vision slot（`image_slots`）处拼接进文本序列 —— 不再需要 `Picture n:` 这类提示词前缀。
- **全新 VAE**：64 通道潜变量，空间下采样 16x（支持 RGBA）。参考图对齐到 **32 像素的倍数**，因此每个 vision slot 恰好对应 2x2 个潜变量 token。
- **统一的参考图缩放**：视觉塔（vision tower）与 VAE 使用**同一张**缩放后的图像（编码器用白底合成后的 RGB，VAE 用完整 RGBA），因此不再有单独的 `vl_target_size` 流程。

![Qwen-Image 2.1 编辑示例](qwen%20image%2021%20example.png)

*使用 [edit utils qwen image 2.1 example.json](workflows/edit%20utils%20qwen%20image%202.1%20example.json) 生成的结果。*

**节点说明：**

| 节点 | 用途 |
|---|---|
| `QwenImage21ModelConfig_EditUtils` | 模型配置（`qwen_image21` 路径，`vae_unit=32`）。instruction 留空 = 使用内置 T2I 模板；填写自定义 instruction = 使用自定义系统提示词。 |
| `QwenImage21ConfigPreparer_EditUtils` | 单张图片的配置：`to_ref`、`ref_main_image`、`ref_longest_edge`（按 32 对齐）、`ref_crop`、`mask`、`rope_x_offset / rope_y_offset`。多张参考图时串联多个该节点。 |
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
- `rope_x_offset / rope_y_offset` 只有在图中有 `QwenImage21EditApply_EditUtils` 时才生效；偏移全为 0 时模型走原生路径（不影响 prefix KV cache）。
- 请保持 Config Preparer 上的 `to_vl` 启用 —— 关闭后参考图会被拼接到文本序列之后，这是未经训练的分支。
- 建议的起步参数（与示例工作流一致）：Euler / Simple，25 步，CFG 1.0。

## 工作流

示例工作流位于 [workflows](workflows/) 目录：

- **[edit utils qwen image 2.1 example.json](workflows/edit%20utils%20qwen%20image%202.1%20example.json)** - Qwen-Image 2.1 编辑工作流（单张参考图）。使用 `UNETLoader` + `CLIPLoader`（type 选 `qwen_image`）+ `VAELoader` 加载模型，然后接线 `LoadImage → QwenImage21ConfigPreparer_EditUtils → EditTextEncode_EditUtils ← QwenImage21ModelConfig_EditUtils → KSampler → VAEDecode → CropWithPadInfo_EditUtils`，用于去掉主图的填充。
  - 参考图潜变量在 vision slot（`image_slots`）处拼接进文本序列；2.1 VAE 为 64 通道、16x 空间下采样，因此参考图按 32 像素倍数对齐。
  - Config Preparer 上的 `rope_x_offset / rope_y_offset` 在该模型被 `QwenImage21EditApply_EditUtils` 打补丁后，可平移参考图在画布上的位置（区域编辑）—— 示例工作流未包含该节点。
  - 示例中的采样设置：`euler / simple`，25 步，CFG 1.0。需要 ComfyUI 已支持上游的 Qwen-Image 2.1。
- **[Simple Krea2 Depth.json](workflows/Simple%20Krea2%20Depth.json)** - 配合 depth LoRA 的简单 Krea2 编辑工作流。接线 `LoadImage → Krea2ModelConfig_EditUtils → EditTextEncode_EditUtils → Krea2EditApply_EditUtils → KSampler`，通过 `UNETLoader + LoraLoaderModelOnly` 加载模型 —— 参考图潜变量会自动沿 conditioning 链路传递。
  - RunningHub 在线版本：https://www.runninghub.ai/post/2082077636234313729/?inviteCode=rh-v1279
  - Depth LoRA 下载：[Krea2 Depth LoRA (Civitai)](https://civitai.com/models/2815790/krea2-depth-lrzjason-20260729)

> ⚠️ **注意：** Krea2 编辑支持仍在开发中 —— 节点接口与行为可能发生变化。

## 能力特性

- EditUtils 支持无需像素位移的直接高分辨率编辑，最高可达 2xxx ~ 3xxx 分辨率
- 支持多图输入：
  - 简易工作流：最多 3 张图
  - 单图与多图工作流：图片数量不限（串联多个配置节点）

## 节点分类

文档：

- [节点文档](nodes_doc.md)

## 主要特性

- **多模型支持**：可用于 Qwen、Qwen-Image 2.1、Flux2Klein、Boogu 与 Krea2 模型，实现多样化的图像编辑
- **灵活配置**：针对每张图片单独配置参考图与 VL 处理选项
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

**输出：**
- `processed_image`：缩放/填充后的图像
- `pad_info`：填充信息字典
- `scale_by`：缩放系数

**使用场景：** 在传给其他节点之前先做长边缩放预处理，或在编码流程之外复用同一套图像处理管线。

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

## 安装

1. 克隆或下载本仓库到你的 ComfyUI 的 `custom_nodes` 目录。
2. 重启 ComfyUI。
3. 节点将出现在 "advanced/conditioning" 分类下。

## 更新记录

### ComfyUI-EditUtils 与 ComfyUI-QwenEditUtils 的对比
ComfyUI-EditUtils 是 ComfyUI-QwenEditUtils 的后续版本，包含以下改进：
- 多模型支持（Qwen、Qwen-Image 2.1、Flux2Klein、Boogu 与 Krea2）
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
