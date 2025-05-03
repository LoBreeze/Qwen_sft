from unsloth import FastLanguageModel
from datasets import load_dataset
from trl import SFTTrainer
from transformers import TrainingArguments
from unsloth import is_bfloat16_supported
import wandb
import json
import os
from rich import print
from rich.style import Style
from rich.syntax import Syntax
from rich.highlighter import RegexHighlighter
from rich.theme import Theme

# 自定义高亮器，为变量名添加颜色
class VariableHighlighter(RegexHighlighter):
    base_style = "example."
    highlights = [
        r"(?P<variable>\b[a-zA-Z_][a-zA-Z0-9_]*\b)(?=\s*=)",
    ]

# 自定义主题
custom_theme = Theme({
    "info": "bold cyan",
    "success": "bold green",
    "warning": "bold yellow",
    "error": "bold red",
    "highlight": "bold magenta",
    "data": "blue",
    "example.variable": "bold bright_cyan",  # 变量名颜色
})

# 应用主题
from rich.console import Console
console = Console(theme=custom_theme, highlighter=VariableHighlighter())

dir_path = os.path.dirname(os.path.realpath(__file__))
max_seq_length = 2048  # 最大序列长度
dtype = None  # 使用默认数据类型
load_in_4bit = True  # 是否使用4bit量化

# 定义情感分析的提示模板
# 这个模板将用于格式化训练数据
train_prompt_style = """Below is an instruction that describes a task, paired with an input that provides further context. 
Write a response that appropriately completes the request.

### Instruction:
You are a financial sentiment analysis expert. 
What is the sentiment of the following financial post: Positive, Negative, or Neutral?

### Input:
{}

### Response:
{}"""

console.print("[info]正在加载FIQASA数据集...[/info]")
dataset = load_dataset(os.path.join(dir_path, "flare_fiqasa", "data"))
console.print(f"[success]数据集加载完成，训练集大小: {len(dataset['train'])}[/success]")
console.print(f"[data]training set: {dataset['train']}[/data]")

# 从预训练模型加载模型和分词器
console.print("[info]正在加载模型 Qwen/Qwen2.5-1.5B-Instruct...[/info]")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "/root/autodl-tmp/Qwen/Qwen2___5-1___5B-Instruct",
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
)
# 获取EOS token，用于在文本结尾添加
EOS_TOKEN = tokenizer.eos_token

def formatting_prompts_func(examples):
    texts = []
    for i in range(len(examples["query"])):
        # 从数据集中提取文本内容
        text_content = examples["text"][i]
        # 获取正确答案
        answer = examples["answer"][i]
        # 使用模板格式化
        text = train_prompt_style.format(text_content, answer) + EOS_TOKEN
        texts.append(text)
    return {
        "text": texts,
    }
    

# 将数据集映射为训练格式
train_dataset = dataset["train"].map(formatting_prompts_func, batched=True)
val_dataset = dataset["validation"].map(formatting_prompts_func, batched=True)
# 打印第一个样本，检查格式是否正确
console.print("[highlight]第一个训练样本示例:[/highlight]")
console.print(f"[data]{train_dataset['text'][0]}[/data]")
# 打印数据集大小
console.print(f"[success]训练集大小: {len(train_dataset)}[/success]")
console.print(f"[success]验证集大小: {len(val_dataset)}[/success]")





# 配置PEFT（参数高效微调）
console.print("[info]配置PEFT模型...[/info]")


model = FastLanguageModel.get_peft_model(
    model,
    r=16,  # LoRA秩
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],  # 目标模块
    lora_alpha=16,  # LoRA alpha参数
    lora_dropout=0,  # LoRA dropout
    bias="none",  # 是否包含偏置项
    use_gradient_checkpointing="unsloth",  # 使用梯度检查点以节省内存
    random_state=42,  # 随机种子
    use_rslora=False,  # 是否使用RSLoRA
    loftq_config=None,  # LoftQ配置
)


# 初始化wandb进行实验跟踪
wandb.init(
    project="qwen-financial-sentiment",  # 项目名称
    name="qwen2.5-1.5b-fiqasa-sft",  # 实验名称
    config={
        "model": "Qwen/Qwen2.5-1.5B-Instruct",
        "dataset": "flare_fiqasa",
        "task": "sentiment_analysis",
        "epochs": 3,
        "batch_size": 2,
        "learning_rate": 2e-4,
    }
)

# 配置SFT训练器
console.print("[info]配置SFT训练器...[/info]")
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,  # 添加验证集
    dataset_text_field="text",
    max_seq_length=max_seq_length,
    dataset_num_proc=2,
    args=TrainingArguments(
        per_device_train_batch_size=2,  # 每个设备的训练批次大小
        per_device_eval_batch_size=4,   # 每个设备的评估批次大小
        gradient_accumulation_steps=4,  # 梯度累积步数
        num_train_epochs=3,  # 训练轮数
        warmup_steps=5,  # 预热步数
        learning_rate=2e-4,  # 学习率
        fp16=not is_bfloat16_supported(),  # 是否使用fp16
        bf16=is_bfloat16_supported(),  # 是否使用bf16
        logging_steps=10,  # 日志记录步数
        # evaluation_strategy="steps",  # 评估策略
        # eval_steps=50,  # 评估步数
        # save_strategy="steps",  # 保存策略
        # save_steps=50,  # 保存步数
        optim="adamw_8bit",  # 优化器
        weight_decay=0.01,  # 权重衰减
        lr_scheduler_type="linear",  # 学习率调度器类型
        seed=42,  # 随机种子
        output_dir="outputs-qwen-financial",  # 输出目录
        report_to="wandb",  # 报告工具
    ),
)

# 开始训练
console.print("[info]开始训练...[/info]")
trainer_stats = trainer.train()

# 保存模型
console.print("[success]训练完成，保存模型...[/success]")
peft_path = "./output/Qwen2.5-1.5B-Financial-Sentiment-LoRA"
model.save_pretrained(peft_path)  # 保存PEFT模型
tokenizer.save_pretrained(peft_path)  # 保存分词器

# 保存合并后的模型（可选）
new_model_local = "./output/Qwen2.5-1.5B-Financial-Sentiment"
console.print("[info]保存合并后的模型...[/info]")
model.save_pretrained_merged(new_model_local, tokenizer, save_method="merged_16bit")

# 关闭wandb
wandb.finish()

console.print(f"[success]模型已保存到: {new_model_local}[/success]")