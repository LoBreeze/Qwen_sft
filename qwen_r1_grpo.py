# 导入必要的库
from unsloth import FastLanguageModel, PatchFastRL
PatchFastRL("GRPO", FastLanguageModel)  # 应用 GRPO 补丁到 FastLanguageModel

from unsloth import is_bfloat16_supported  # 检查是否支持 bfloat16
import torch

# 模型参数配置
max_seq_length = 512  # 输入序列的最大长度，控制模型处理的输入和输出文本的最大长度
lora_rank = 32  # LoRA 适配层的秩，较大的秩可以提升模型的表达能力

# 加载预训练模型
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="/home/zhz/DeepSeek-R1-Distill-Qwen-1.5B",  # 模型路径
    max_seq_length=max_seq_length,  # 最大序列长度
    load_in_4bit=True,  # 使用 4bit 量化加载模型，节省显存
    fast_inference=True,  # 启用 vLLM 的高性能推理模式
    max_lora_rank=lora_rank,  # LoRA 最大秩
    gpu_memory_utilization=0.6,  # GPU 内存使用比例
)

# 为模型添加 LoRA 适配层
model = FastLanguageModel.get_peft_model(
    model,
    r=lora_rank,  # LoRA 的秩
    target_modules=[  # 应用 LoRA 的模块
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    lora_alpha=lora_rank,  # LoRA 的缩放因子
    use_gradient_checkpointing="unsloth",  # 启用梯度检查点机制
    random_state=3407,  # 随机种子
)

import re
from datasets import load_dataset, Dataset

# 系统提示模板
SYSTEM_PROMPT = """
Respond in the following format:
<reasoning>
...
</reasoning>
<answer>
...
</answer>
"""

# XML 思维链格式模板
XML_COT_FORMAT = """\
<reasoning>
{reasoning}
</reasoning>
<answer>
{answer}
</answer>
"""

def extract_xml_answer(text: str) -> str:
    """从 XML 格式的文本中提取 <answer> 标签内的内容"""
    answer = text.split("<answer>")[-1]
    answer = answer.split("</answer>")[0]
    return answer.strip()

def extract_think_answer(text: str) -> str:
    """从包含 <think> 标签的文本中提取实际回答内容"""
    return text.split("</think>")[-1].strip()

def get_a_questions(split="train", local_path="/home/zhz/datasets/ruozhiba_R1/alpaca_output.jsonl") -> Dataset:
    """
    加载本地 alpaca_output.jsonl 数据集
    
    Args:
        split (str): 数据集划分，默认为 "train"
        local_path (str): 本地数据集路径
    Returns:
        Dataset: 处理后的数据集
    """
    # 从本地路径加载数据集
    data = load_dataset('json', data_files=local_path, split=split)  # type: ignore
    
    # 根据 Alpaca 格式进行字段映射
    data = data.map(lambda x: {  # type: ignore
        'prompt': [
            {'role': 'system', 'content': SYSTEM_PROMPT},
            {'role': 'user', 'content': x['instruction']}
        ],
        'answer': extract_think_answer(x['output'])  # 提取 <think> 标签后的回答
    })  # type: ignore
    
    return data 

# 加载数据集
dataset = get_a_questions()

# 奖励函数定义
def correctness_reward_func(prompts, completions, answer, **kwargs) -> list[float]:
    """正确性奖励函数：检查回答是否正确"""
    responses = [completion[0]['content'] for completion in completions]
    q = prompts[0][-1]['content']
    extracted_responses = [extract_xml_answer(r) for r in responses]
    print('-'*20, f"Question:\n{q}", f"\nAnswer:\n{answer[0]}", f"\nResponse:\n{responses[0]}", f"\nExtracted:\n{extracted_responses[0]}")
    return [2.0 if r == a else 0.0 for r, a in zip(extracted_responses, answer)]

def int_reward_func(completions, **kwargs) -> list[float]:
    """整数奖励函数：检查回答是否为整数"""
    responses = [completion[0]['content'] for completion in completions]
    extracted_responses = [extract_xml_answer(r) for r in responses]
    return [0.5 if r.isdigit() else 0.0 for r in extracted_responses]

def strict_format_reward_func(completions, **kwargs) -> list[float]:
    """严格格式奖励函数：检查是否符合严格的 XML 格式"""
    pattern = r"^<reasoning>\n.*?\n</reasoning>\n<answer>\n.*?\n</answer>\n$"
    responses = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, r) for r in responses]
    return [0.5 if match else 0.0 for match in matches]

def soft_format_reward_func(completions, **kwargs) -> list[float]:
    """宽松格式奖励函数：检查是否符合宽松的 XML 格式"""
    pattern = r"<reasoning>.*?</reasoning>\s*<answer>.*?</answer>"
    responses = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, r) for r in responses]
    return [0.5 if match else 0.0 for match in matches]

def count_xml(text) -> float:
    """计算 XML 标签数量并评分"""
    count = 0.0
    if text.count("<reasoning>\n") == 1:
        count += 0.125
    if text.count("\n</reasoning>\n") == 1:
        count += 0.125
    if text.count("\n<answer>\n") == 1:
        count += 0.125
        count -= len(text.split("\n</answer>\n")[-1])*0.001
    if text.count("\n</answer>") == 1:
        count += 0.125
        count -= (len(text.split("\n</answer>")[-1] - 1))*0.001
    return count

def xmlcount_reward_func(completions, **kwargs) -> list[float]:
    """XML 标签计数奖励函数"""
    contents = [completion[0]["content"] for completion in completions]
    return [count_xml(c) for c in contents]

from trl import GRPOConfig, GRPOTrainer

# 配置 GRPO 训练参数
training_args = GRPOConfig(
    use_vllm=True,  # 使用 vLLM 进行快速推理
    learning_rate=5e-6,  # 学习率
    adam_beta1=0.9,  # Adam 优化器的 beta1 参数
    adam_beta2=0.99,  # Adam 优化器的 beta2 参数
    weight_decay=0.1,  # 权重衰减
    warmup_ratio=0.1,  # 学习率预热比例
    lr_scheduler_type="cosine",  # 学习率调度器类型
    optim="paged_adamw_8bit",  # 优化器类型
    logging_steps=1,  # 日志记录步数
    bf16=is_bfloat16_supported(),  # 是否支持 bfloat16
    fp16=not is_bfloat16_supported(),  # 是否支持 fp16
    per_device_train_batch_size=1,  # 每个设备的训练批量大小
    gradient_accumulation_steps=1,  # 梯度累积步数
    num_generations=6,  # 生成数量
    max_prompt_length=256,  # 最大提示长度
    max_completion_length=200,  # 最大完成长度
    max_steps=300,  # 最大训练步数
    save_steps=300,  # 保存步数
    max_grad_norm=0.1,  # 最大梯度范数
    report_to="none",  # 日志报告目标
    output_dir="outputs",  # 输出目录
)

# 创建 GRPO 训练器
trainer = GRPOTrainer(
    model=model,  # 模型
    processing_class=tokenizer,  # 分词器
    reward_funcs=[  # 奖励函数列表
        xmlcount_reward_func,  # XML结构奖励
        soft_format_reward_func,  # 宽松格式奖励
        strict_format_reward_func,  # 严格格式奖励
        int_reward_func,  # 整数奖励
        correctness_reward_func,  # 正确性奖励
    ],
    args=training_args,  # 训练参数
    train_dataset=dataset,  # 训练数据集
)

# 开始训练
trainer.train()

# 生成示例
text = tokenizer.apply_chat_template([
    {"role": "user", "content": "你是谁，开始你的表演"},  # 用户输入
], tokenize=False, add_generation_prompt=True)  # 将用户请求封装成对话模板

from vllm import SamplingParams

# 定义采样参数
sampling_params = SamplingParams(
    temperature=0.8,  # 控制输出的多样性
    top_p=0.95,  # 核采样参数
    max_tokens=1024,  # 生成文本的最大长度
)

# 调用模型生成文本
output = model.fast_generate(
    [text],  # 输入文本内容
    sampling_params=sampling_params,  # 采样参数
    lora_request=None,  # LoRA 请求
)[0].outputs[0].text  # 获取生成结果中的第一个样本的文本内容

print(output)  # 输出生成的文本

# 使用保存的 LoRA 权重生成回答
text = tokenizer.apply_chat_template([
    {"role": "system", "content": SYSTEM_PROMPT},  # 系统提示
    {"role": "user", "content": "你是谁，开始你的表演"},  # 用户输入
], tokenize=False, add_generation_prompt=True)  # 将用户请求封装成对话模板

# 定义采样参数
sampling_params = SamplingParams(
    temperature=0.8,  # 控制输出的多样性
    top_p=0.95,  # 核采样参数
    max_tokens=1024,  # 生成文本的最大长度
)

# 调用模型生成文本
output = model.fast_generate(
    text,  # 输入文本内容
    sampling_params=sampling_params,  # 采样参数
    lora_request=model.load_lora("grpo_saved_lora"),  # 加载 LoRA 权重
)[0].outputs[0].text  # 获取生成结果中的第一个样本的文本内容

print(output)  # 输出生成的文本

# 保存合并后的模型
model.save_pretrained_merged("model", tokenizer, save_method="merged_16bit")