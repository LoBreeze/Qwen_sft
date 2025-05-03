# 基于Qwen模型的金融情感分析

本仓库提供了一个完整的流程，用于微调和优化Qwen模型进行金融情感分析，采用了多种强化学习技术，包括直接偏好优化（DPO）。
此外，利用广义奖励偏好优化（GRPO）对Qwen模型在弱智吧数据集上的表现能力进行了优化。

## 概述

本项目使用Qwen 2.5 (1.5B)模型对金融文本进行情感分析，重点优化模型的准确性和输出简洁性。完整流程包括：

1. **监督微调 (SFT)**：在FLARE-FIQASA数据集上进行初始微调
2. **直接偏好优化 (DPO)**：进一步优化以减少输出冗长度同时保持准确率
3. **全面评估**：详细比较基础模型、SFT模型和DPO模型的性能指标

## 环境要求

- Python 3.8+
- PyTorch
- Unsloth
- Transformers
- TRL (Transformer强化学习库)
- Rich (用于美化控制台输出)
- Datasets
- Wandb (用于实验跟踪)
- scikit-learn (用于评估指标)

## 项目结构

```
.
├── flare_fiqasa/               # 金融情感数据集
│   ├── data/                   # 数据集文件
│   │   ├── train-*.parquet
│   │   ├── valid-*.parquet
│   │   └── test-*.parquet
│   └── README.md
├── Qwen/                       # 基础Qwen模型
│   └── Qwen2.5-1.5B-Instruct/
├── deepseek-ai/                # DeepSeek模型(用于GRPO)
│   └── DeepSeek-R1-Distill-Qwen-1.5B/
├── ruozhiba_R1/                # 额外数据集
│   └── alpaca_output.jsonl
├── qwen_flare_sft.py           # 监督微调脚本
├── qwen_flare_sft_test.py      # SFT模型评估脚本
├── qwen_flare_dpo.py           # 直接偏好优化脚本
├── qwen_flare_dpo_test.py      # DPO模型评估脚本
└── qwen_r1_grpo.py             # GRPO实现
```

## 数据集

本项目使用FLARE-FIQASA数据集，该数据集包含带有情感标签（积极、消极或中性）的金融文本，专为金融情感分析任务设计。

## 训练流程

### 1. 监督微调 (SFT)

第一步是在FLARE-FIQASA数据集上微调基础Qwen 2.5 (1.5B)模型。

```bash
python qwen_flare_sft.py
```

该脚本会：
- 加载FLARE-FIQASA数据集
- 使用金融情感分析提示模板格式化数据
- 应用LoRA（低秩适应）进行高效微调
- 训练3个轮次
- 保存LoRA适配器和合并后的模型

### 2. 评估SFT模型

微调后，在测试集上评估模型：

```bash
python qwen_flare_sft_test.py
```

提供详细的指标，包括：
- 总体准确率
- 各类别的精确率、召回率和F1分数
- 混淆矩阵

### 3. 直接偏好优化 (DPO)

DPO阶段优化模型，使其产生更简洁的输出同时保持准确性：

```bash
python qwen_flare_dpo.py
```

该脚本会：
- 使用SFT模型生成可能冗长的回答
- 创建偏好（简洁）和拒绝（冗长）回答对
- 应用DPO教导模型偏好简洁输出
- 保存优化后的模型

### 4. 评估DPO模型

比较DPO模型与SFT模型：

```bash
python qwen_flare_dpo_test.py
```

提供详细比较：
- 准确率和F1分数的变化
- 输出长度减少情况
- 各类别性能变化
- 具体改进和退步的示例

### 5. 广义奖励偏好优化 (GRPO)

对于弱智吧实验，仓库包含GRPO实现：

```bash
python qwen_r1_grpo.py
```

实现多种奖励函数来指导模型优化：
- XML格式奖励
- 正确性奖励
- 响应结构奖励

## 模型提示模板

项目对所有模型使用一致的提示模板：

```
Below is an instruction that describes a task, paired with an input that provides further context. 
Write a response that appropriately completes the request.

### Instruction:
You are a financial sentiment analysis expert. 
What is the sentiment of the following financial post: Positive, Negative, or Neutral?

### Input:
{text}

### Response:
```

## 结果

DPO优化通常能够实现：
- 与SFT模型相比保持或提高准确率
- 显著减少输出冗长度
- 更好地专注于情感分类任务
