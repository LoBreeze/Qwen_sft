from unsloth import FastLanguageModel
from datasets import load_dataset
from trl import DPOTrainer
from transformers import TrainingArguments
from unsloth import is_bfloat16_supported
import wandb
import json
import os
from rich import print
from rich.console import Console
from rich.theme import Theme
import re
from rich.progress import Progress, TextColumn, BarColumn, TaskProgressColumn, TimeElapsedColumn, TimeRemainingColumn, SpinnerColumn

# 自定义主题
custom_theme = Theme({
    "info": "bold cyan",
    "success": "bold green",
    "warning": "bold yellow",
    "error": "bold red",
    "highlight": "bold magenta",
    "data": "blue",
})

# 应用主题
console = Console(theme=custom_theme)

# 配置参数
dir_path = os.path.dirname(os.path.realpath(__file__))
max_seq_length = 2048  # 最大序列长度
dtype = None  # 使用默认数据类型
load_in_4bit = True  # 是否使用4bit量化
dir_path = os.path.dirname(os.path.realpath(__file__))
# 定义SFT训练后的模型路径（这是之前SFT训练产生的模型）
sft_model_path = os.path.join(dir_path, "output", "Qwen2.5-1.5B-Financial-Sentiment")

# 加载之前SFT训练后的模型和分词器
console.print("[info]正在加载SFT训练后的模型...[/info]")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=sft_model_path,
    max_seq_length=max_seq_length,
    dtype=dtype,
    load_in_4bit=load_in_4bit,
)
# 获取EOS token，用于在文本结尾添加
EOS_TOKEN = tokenizer.eos_token

# DPO训练需要的提示模板
prompt_template = """Below is an instruction that describes a task, paired with an input that provides further context. 
Write a response that appropriately completes the request.

### Instruction:
You are a financial sentiment analysis expert. 
What is the sentiment of the following financial post: Positive, Negative, or Neutral?

### Input:
{text}

### Response:
"""

# 函数用于检测输出是否过长（相对于答案）
def is_overlength(output, answer, threshold=2.0):
    """检查输出是否过长（是答案长度的threshold倍以上）"""
    output_len = len(output.split())
    answer_len = len(answer.split())
    return output_len > answer_len * threshold

# 准备DPO训练数据集
# 修改：直接加载FiQASA数据集，无需指定本地路径
console.print("[info]正在加载FIQASA数据集用于DPO训练...[/info]")
dataset = load_dataset(os.path.join(dir_path, "flare_fiqasa", "data"))

# 打印数据集结构信息
console.print(f"[data]数据集结构: {dataset}[/data]")
console.print(f"[data]训练集样本数: {len(dataset['train'])}[/data]")
console.print(f"[data]验证集样本数: {len(dataset['validation'])}[/data]")
console.print(f"[data]测试集样本数: {len(dataset['test'])}[/data]")

# 打印一个示例，了解数据结构
console.print("[highlight]数据集示例:[/highlight]")
console.print(f"[data]{dataset['train'][0]}[/data]")

# 创建SFT模型的推理函数，用于生成回答（可能过长的回答）
def generate_responses(examples, batch_size=4):
    """使用SFT模型生成回答，并显示Rich进度条"""
    responses = []
    if os.path.exists("generated_responses.json"):
        with open("generated_responses.json", "r") as f:
            responses = json.load(f)
        return responses
    
    total_batches = (len(examples["text"]) + batch_size - 1) // batch_size  # 计算总批次数
    total_examples = len(examples["text"])
    
    # 创建一个自定义的Rich进度条
    with Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TextColumn("•"),
        TimeElapsedColumn(),
        TextColumn("•"),
        TimeRemainingColumn(),
    ) as progress:
        # 创建主任务
        main_task = progress.add_task("[info]生成回答中...[/info]", total=total_batches)
        
        for i in range(0, total_examples, batch_size):
            batch_texts = examples["text"][i:i+batch_size]
            batch_prompts = [prompt_template.format(text=text) for text in batch_texts]
            
            # 更新任务描述以显示当前批次和总进度
            current_batch = i // batch_size + 1
            progress.update(
                main_task, 
                description=f"[info]批次 {current_batch}/{total_batches} • 完成 {min(i+batch_size, total_examples)}/{total_examples}[/info]"
            )
            
            # 使用模型生成回答
            outputs = model.generate(
                tokenizer(batch_prompts, return_tensors="pt", padding=True).input_ids.cuda(),
                max_new_tokens=100,  # 允许生成足够长的回答
                temperature=1.0,     # 使用较高温度以产生多样性输出
                top_p=0.9,
                repetition_penalty=1.0,  # 降低重复惩罚以允许生成重复内容
            )
            
            # 解码生成的token
            decoded_outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            
            # 提取生成的回答部分
            for j, output in enumerate(decoded_outputs):
                # 可选：仅显示部分样本信息以减少输出量
                if j == 0 and current_batch % 5 == 0:  # 每5个批次显示一次样本
                    console.print(f"[data]样本回答 (批次 {current_batch}): {output.split('### Response:')[-1].strip()[:50]}...[/data]")
                
                response_part = output.split("### Response:")[-1].strip()
                responses.append(response_part)
            
            # 更新进度
            progress.advance(main_task)
    
    # 在进度完成后显示总结信息
    console.print(f"[success]✓ 成功生成 {len(responses)} 个回答[/success]")
    with open("generated_responses.json", "w") as f:
        json.dump(responses, f, indent=4)
    console.print("[info]生成的回答已保存到 generated_responses.json[/info]")
    return responses

# 准备DPO数据集
def prepare_dpo_dataset(dataset_split):
    """准备DPO训练所需的数据集，包括偏好和拒绝的回答对"""
    # 修改：确保从正确的字段读取数据
    texts = dataset_split["text"]  # 金融文本
    queries = [prompt_template.format(text=text) for text in texts]
    
    # 修改：使用正确的答案字段，确保是小写形式以便于比较
    answers = [answer.lower() for answer in dataset_split["answer"]]
    
    # 使用模型生成可能过长的回答
    generated_responses = generate_responses(dataset_split)
    
    dpo_data = []
    for i in range(len(texts)):
        # 正确答案（简洁回答）作为偏好回答
        chosen = answers[i]
        
        # 修改：确保chosen是格式化好的完整句子，而不仅仅是标签
        # 将简单的标签转换为完整句子
        if chosen in ["positive", "negative", "neutral"]:
            chosen_formatted = f"The sentiment is {chosen}."
        else:
            chosen_formatted = chosen
        
        # 生成的回答 - 可能过长或重复
        generated = generated_responses[i]
        
        # 修改：进行更严格的筛选，确保数据质量
        # 1. 生成的回答与预期答案不同
        # 2. 生成的回答可能过长或重复
        # 3. 生成的回答不是空字符串
        if (generated.lower() != chosen.lower() and 
            is_overlength(generated, chosen_formatted) and 
            len(generated.strip()) > 0):
            
            # 修改：创建DPO训练样本，保存更多元数据以便跟踪
            dpo_data.append({
                "prompt": queries[i],
                "chosen": chosen_formatted,  # 简洁明了的回答
                "rejected": generated,       # 可能冗长的回答
                "original_text": texts[i],   # 原始金融文本
                "gold_label": chosen         # 金标签
            })
    
    console.print(f"[success]创建了{len(dpo_data)}个DPO训练样本[/success]")
    
    # 修改：如果数据集为空，给出警告
    if len(dpo_data) == 0:
        console.print("[warning]警告：DPO数据集为空！请检查数据处理逻辑或生成策略[/warning]")
    
    return dpo_data

# 准备DPO训练数据
console.print("[info]正在准备DPO训练数据...[/info]")
# 修改：仅使用训练集用于DPO训练
train_dpo_data = prepare_dpo_dataset(dataset["train"])

# 将准备好的数据转换为数据集格式
from datasets import Dataset
dpo_dataset = Dataset.from_list(train_dpo_data)
console.print(f"[success]DPO数据集准备完成，大小: {len(dpo_dataset)}[/success]")

# 打印示例检查格式
if len(dpo_dataset) > 0:
    console.print("[highlight]DPO训练样本示例:[/highlight]")
    example_idx = 0  # 第一个样本
    console.print(f"[data]提示: {dpo_dataset[example_idx]['prompt']}[/data]")
    console.print(f"[data]偏好回答: {dpo_dataset[example_idx]['chosen']}[/data]")
    console.print(f"[data]拒绝回答: {dpo_dataset[example_idx]['rejected']}[/data]")
    console.print(f"[data]原始文本: {dpo_dataset[example_idx]['original_text']}[/data]")
    console.print(f"[data]金标签: {dpo_dataset[example_idx]['gold_label']}[/data]")

    # 修改：添加数据统计信息
    chosen_lengths = [len(item['chosen'].split()) for item in train_dpo_data]
    rejected_lengths = [len(item['rejected'].split()) for item in train_dpo_data]
    
    avg_chosen_len = sum(chosen_lengths) / len(chosen_lengths) if chosen_lengths else 0
    avg_rejected_len = sum(rejected_lengths) / len(rejected_lengths) if rejected_lengths else 0
    
    console.print(f"[info]偏好回答平均长度: {avg_chosen_len:.2f} 词[/info]")
    console.print(f"[info]拒绝回答平均长度: {avg_rejected_len:.2f} 词[/info]")
    console.print(f"[info]拒绝/偏好长度比: {(avg_rejected_len/avg_chosen_len):.2f}[/info]")

# 配置PEFT模型用于DPO训练
console.print("[info]配置PEFT模型用于DPO训练...[/info]")
model = FastLanguageModel.get_peft_model(
    model,
    r=16,  # LoRA秩，DPO训练通常使用较小的秩
    target_modules=[
        "q_proj",
        "k_proj", 
        "v_proj",
        "o_proj",
        "gate_proj", 
        "up_proj", 
        "down_proj",
    ],
    lora_alpha=32,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=42,
    use_rslora=False,
    loftq_config=None,
)

# 初始化wandb进行实验跟踪
wandb.init(
    project="qwen-financial-sentiment-dpo",
    name="qwen2.5-1.5b-fiqasa-dpo",
    config={
        "model": "Qwen/Qwen2.5-1.5B-Instruct-DPO",
        "dataset": "flare_fiqasa",
        "task": "sentiment_analysis_dpo",
        "epochs": 2,
        "batch_size": 2,
        "learning_rate": 5e-5,
        "beta": 0.1,  # DPO的温度参数，控制优化强度
        # 修改：添加更多实验配置
        "dataset_size": len(dpo_dataset),
        "chosen_avg_length": avg_chosen_len if len(dpo_dataset) > 0 else 0,
        "rejected_avg_length": avg_rejected_len if len(dpo_dataset) > 0 else 0,
    }
)

# 配置DPO训练器
console.print("[info]配置DPO训练器...[/info]")

# 确保数据集不为空
if len(dpo_dataset) > 0:
    # 修改：提取必要的字段用于DPO训练
    dpo_dataset_for_training = dpo_dataset.select_columns(['prompt', 'chosen', 'rejected'])
    
    trainer = DPOTrainer(
        model=model,
        ref_model=None,  # 使用当前模型作为参考模型
        tokenizer=tokenizer,
        train_dataset=dpo_dataset_for_training,  # 修改：使用处理后的数据集
        args=TrainingArguments(
            per_device_train_batch_size=2,
            gradient_accumulation_steps=4,
            num_train_epochs=2,
            learning_rate=5e-5,
            fp16=not is_bfloat16_supported(),
            bf16=is_bfloat16_supported(),
            logging_steps=10,
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="linear",
            seed=42,
            output_dir="outputs-qwen-financial-dpo",
            report_to="wandb",
            # 修改：添加评估步骤
            # eval_strategy="steps",
            # eval_steps=50,
            save_strategy="steps",
            save_steps=50,
            save_total_limit=2,
        ),
        beta=0.1,  # DPO的温度参数
        max_length=max_seq_length,
        # max_prompt_length=512,
        # max_target_length=256,
    )

    # 开始DPO训练
    console.print("[info]开始DPO训练...[/info]")
    trainer.train()

    # 保存模型
    console.print("[success]DPO训练完成，保存模型...[/success]")
    dpo_model_path = "./output/Qwen2.5-1.5B-Financial-Sentiment-DPO"
    model.save_pretrained(dpo_model_path)
    tokenizer.save_pretrained(dpo_model_path)

    # 修改：添加简单评估逻辑
    console.print("[info]评估DPO训练后的模型...[/info]")
    
    # 从测试集选择几个样本进行评估
    num_eval_samples = min(5, len(dataset["test"]))
    eval_texts = dataset["test"]["text"][:num_eval_samples]
    eval_answers = dataset["test"]["answer"][:num_eval_samples]
    
    console.print("[highlight]DPO训练后模型评估:[/highlight]")
    
    for i, (text, gold_answer) in enumerate(zip(eval_texts, eval_answers)):
        # 构建提示
        prompt = prompt_template.format(text=text)
        
        # 使用DPO训练后的模型生成回答
        inputs = tokenizer(prompt, return_tensors="pt").input_ids.cuda()
        outputs = model.generate(
            inputs,
            max_new_tokens=50,
            temperature=0.1,  # 使用较低温度以获得确定性输出
        )
        
        # 解码生成的token
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # 提取生成的回答部分
        response = generated_text.split("### Response:")[-1].strip()
        
        # 输出结果比较
        console.print(f"[data]样本 {i+1}[/data]")
        console.print(f"[data]文本: {text}[/data]")
        console.print(f"[data]正确答案: {gold_answer}[/data]")
        console.print(f"[data]模型回答: {response}[/data]")
        console.print(f"[data]回答长度: {len(response.split())} 词[/data]")
        console.print("---")
    
    # 关闭wandb
    wandb.finish()

    console.print(f"[success]DPO模型已保存到: {dpo_model_path}[/success]")
else:
    console.print("[error]没有足够的DPO训练数据，请检查数据准备过程[/error]")
    # 修改：添加数据处理排查建议
    console.print("[info]建议排查以下问题:[/info]")
    console.print("[info]1. 确认dataset['train']['text']字段是否存在并包含内容[/info]")
    console.print("[info]2. 检查模型生成回答是否与标准答案过于相似导致筛选失败[/info]")
    console.print("[info]3. 尝试降低is_overlength函数中的threshold阈值[/info]")
    console.print("[info]4. 检查dataset['train']['answer']字段格式是否与预期一致[/info]")
    wandb.finish()