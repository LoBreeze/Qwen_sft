from unsloth import FastLanguageModel
from datasets import load_dataset
import torch
import os
import json
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from rich.console import Console
from rich.table import Table
from rich.theme import Theme
from rich.syntax import Syntax
from rich.highlighter import RegexHighlighter
from tqdm import tqdm

# 自定义主题
custom_theme = Theme({
    "info": "bold cyan",
    "success": "bold green",
    "warning": "bold yellow",
    "error": "bold red",
    "highlight": "bold magenta",
    "data": "blue",
    "result": "bold white on blue",
})

# 应用主题
console = Console(theme=custom_theme)

# 设置路径和参数
dir_path = os.path.dirname(os.path.realpath(__file__))
max_seq_length = 2048  # 最大序列长度
dtype = None  # 使用默认数据类型
load_in_4bit = True  # 是否使用4bit量化

# 定义情感分析的测试提示模板
test_prompt_style = """Below is an instruction that describes a task, paired with an input that provides further context. 
Write a response that appropriately completes the request.

### Instruction:
You are a financial sentiment analysis expert. 
What is the sentiment of the following financial post: Positive, Negative, or Neutral?

### Input:
{}

### Response:
"""

# 加载模型路径（使用训练好的模型路径）
MODEL_PATH = "./output/Qwen2.5-1.5B-Financial-Sentiment"
if not os.path.exists(MODEL_PATH):
    console.print(f"[error]模型路径不存在: {MODEL_PATH}[/error]")
    console.print("[info]请修改MODEL_PATH为正确的模型路径[/info]")
    exit(1)

# 加载数据集
console.print("[info]正在加载FIQASA测试数据集...[/info]")
try:
    dataset = load_dataset(os.path.join(dir_path, "flare_fiqasa", "data"))
    test_dataset = dataset["test"] if "test" in dataset else dataset["validation"]
    console.print(f"[success]测试集加载完成，大小: {len(test_dataset)}[/success]")
except Exception as e:
    console.print(f"[error]加载数据集失败: {str(e)}[/error]")
    exit(1)

# 加载微调后的模型和分词器
try:
    console.print(f"[info]正在加载微调模型: {MODEL_PATH}...[/info]")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_PATH,
        max_seq_length=max_seq_length,
        dtype=dtype,
        load_in_4bit=load_in_4bit,
    )
    console.print("[success]模型加载成功[/success]")
except Exception as e:
    console.print(f"[error]加载模型失败: {str(e)}[/error]")
    exit(1)

# 确保GPU可用
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
console.print(f"[info]使用设备: {device}[/info]")

# 定义生成函数
def generate_response(text_input, max_new_tokens=50):
    prompt = test_prompt_style.format(text_input)
    
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.eos_token_id,
            do_sample=False  # 使用贪婪解码，不进行采样
        )
    
    # 解码生成的文本
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # 提取回答部分
    response = generated_text.split("### Response:")[-1].strip()
    
    return response

# 处理预测结果
def process_prediction(prediction):
    prediction = prediction.lower().strip()
    
    # 匹配可能的预测结果变种
    if "positive" in prediction:
        return "positive"
    elif "negative" in prediction:
        return "negative"
    elif "neutral" in prediction:
        return "neutral"
    else:
        # 如果无法确定，默认为中性
        return "neutral"

# 评估模型
console.print("[info]开始评估模型...[/info]")

# 设置小一点的样本用于测试（可选，完整测试时移除此限制）
MAX_TEST_SAMPLES = len(test_dataset)  # 使用全部测试集
# MAX_TEST_SAMPLES = min(100, len(test_dataset))  # 使用部分测试集进行快速测试

predictions = []
true_labels = []

# 使用tqdm显示进度条
for i in tqdm(range(MAX_TEST_SAMPLES), desc="测试进度"):
    sample = test_dataset[i]
    text_content = sample["text"]
    true_label = sample["answer"]
    
    try:
        # 生成预测
        prediction_text = generate_response(text_content)
        prediction = process_prediction(prediction_text)
        
        predictions.append(prediction)
        true_labels.append(true_label)
        
        # 每10个样本显示一次示例（可选）
        if i % 10 == 0:
            console.print(f"[data]样本 #{i}:[/data]")
            console.print(f"文本: {text_content[:100]}...")
            console.print(f"真实标签: {true_label}")
            console.print(f"预测标签: {prediction}")
            console.print(f"原始预测: {prediction_text}")
            console.print("---")
    
    except Exception as e:
        console.print(f"[error]处理样本 #{i} 失败: {str(e)}[/error]")

# 计算评估指标
accuracy = accuracy_score(true_labels, predictions)

# 获取数据集中实际存在的标签
unique_labels = sorted(list(set(true_labels + predictions)))
console.print(f"[info]数据集中的标签: {unique_labels}[/info]")

precision, recall, f1, _ = precision_recall_fscore_support(
    true_labels, predictions, average=None, labels=unique_labels
)
conf_matrix = confusion_matrix(true_labels, predictions, labels=unique_labels)

# 显示结果
console.print("\n[result]模型评估结果[/result]")

# 显示总体准确率
console.print(f"总体准确率: {accuracy:.4f}")

# 创建表格显示各类别的指标
results_table = Table(title="各情感类别评估指标")
results_table.add_column("情感类别", style="cyan")
results_table.add_column("准确率 (Precision)", style="green")
results_table.add_column("召回率 (Recall)", style="yellow")
results_table.add_column("F1分数", style="magenta")

for i, category in enumerate(unique_labels):
    results_table.add_row(
        category,
        f"{precision[i]:.4f}",
        f"{recall[i]:.4f}",
        f"{f1[i]:.4f}"
    )

console.print(results_table)

# 显示混淆矩阵
console.print("\n[result]混淆矩阵[/result]")
cm_table = Table(title="混淆矩阵 (行:真实标签, 列:预测标签)")
cm_table.add_column("", style="cyan")
for label in unique_labels:
    cm_table.add_column(label)

for i, true_label in enumerate(unique_labels):
    cm_table.add_row(
        true_label,
        *[str(conf_matrix[i][j]) for j in range(len(unique_labels))]
    )

console.print(cm_table)

# 保存评估结果到文件
results = {
    "accuracy": float(accuracy),
    "precision": {label: float(p) for label, p in zip(unique_labels, precision)},
    "recall": {label: float(r) for label, r in zip(unique_labels, recall)},
    "f1": {label: float(f) for label, f in zip(unique_labels, f1)},
    "confusion_matrix": conf_matrix.tolist(),
    "labels": unique_labels
}

with open("sentiment_model_evaluation_results.json", "w") as f:
    json.dump(results, f, indent=4)

console.print("[success]评估完成，结果已保存到 sentiment_model_evaluation_results.json[/success]")