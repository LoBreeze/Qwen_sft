import torch
from unsloth import FastLanguageModel
from transformers import pipeline
from rich import print
from rich.console import Console
from rich.theme import Theme
import os
import pandas as pd
import numpy as np
from sklearn.metrics import f1_score, accuracy_score
from datasets import load_dataset

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
max_seq_length = 2048
dtype = None  # 使用默认数据类型
load_in_4bit = True

# 定义模型路径
sft_model_path = os.path.join(dir_path, "output", "Qwen2.5-1.5B-Financial-Sentiment")
dpo_model_path = os.path.join(dir_path, "output", "Qwen2.5-1.5B-Financial-Sentiment-DPO")

# 加载flare_fiqasa数据集
def load_test_dataset():
    """加载flare_fiqasa测试数据集"""
    console.print("[info]正在加载flare_fiqasa数据集...[/info]")
    try:
        dataset = load_dataset("flare-fiqasa")
        console.print(f"[success]成功加载数据集，测试集包含{len(dataset['test'])}个样本[/success]")
        
        # 抽取测试集数据
        test_data = []
        for item in dataset['test']:
            test_sample = {
                "id": item["id"],
                "text": item["text"],
                "expected": item["answer"],
                "choices": item["choices"],
                "gold_index": item["gold"]
            }
            test_data.append(test_sample)
        
        return test_data
    except Exception as e:
        console.print(f"[error]加载数据集失败: {e}[/error]")
        # 如果加载失败，使用少量样本作为备选
        console.print("[warning]使用少量样本作为备选测试集[/warning]")
        return [
            {
                "id": "sample1",
                "text": "Whats up with $LULU? Numbers looked good, not great, but good. I think conference call will instill confidence.",
                "expected": "neutral",
                "choices": ["negative", "positive", "neutral"],
                "gold_index": 2
            },
            {
                "id": "sample2",
                "text": "Apple's latest quarterly earnings report shows a 15% increase in revenue, beating analyst expectations.",
                "expected": "positive",
                "choices": ["negative", "positive", "neutral"],
                "gold_index": 1
            },
            {
                "id": "sample3",
                "text": "Tesla's stock fell 7% after concerns about production delays and increasing competition in the EV market.",
                "expected": "negative",
                "choices": ["negative", "positive", "neutral"],
                "gold_index": 0
            }
        ]

# 提示模板 - 与训练时使用的一致
prompt_template = """Below is an instruction that describes a task, paired with an input that provides further context. 
Write a response that appropriately completes the request.

### Instruction:
You are a financial sentiment analysis expert. 
What is the sentiment of the following financial post: Positive, Negative, or Neutral?

### Input:
{text}

### Response:
"""

def load_model(model_path):
    """加载模型和分词器"""
    console.print(f"[info]正在加载模型: {model_path}...[/info]")
    try:
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_path,
            max_seq_length=max_seq_length,
            dtype=dtype,
            load_in_4bit=load_in_4bit,
        )
        return model, tokenizer
    except Exception as e:
        console.print(f"[error]加载模型失败: {e}[/error]")
        return None, None

def evaluate_model(model, tokenizer, test_data, sample_limit=None):
    """评估模型在测试数据上的表现"""
    results = []
    predictions = []
    ground_truth = []
    output_lengths = []
    
    # 限制测试样本数量，用于快速测试
    if sample_limit and sample_limit > 0:
        test_data = test_data[:sample_limit]
    
    console.print(f"[info]开始模型评估...使用{len(test_data)}个测试样本[/info]")
    
    for i, sample in enumerate(test_data):
        # 构建提示
        prompt = prompt_template.format(text=sample["text"])
        
        # 生成回答
        inputs = tokenizer(prompt, return_tensors="pt").input_ids.cuda()
        outputs = model.generate(
            inputs,
            max_new_tokens=100,
            temperature=0.1,  # 使用较低温度以获得确定性输出
            top_p=0.9,
            do_sample=False,  # 确定性生成
        )
        
        # 解码生成的token
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # 提取生成的回答部分
        response = generated_text.split("### Response:")[-1].strip()
        
        # 提取预测标签（通过检测选项中的单词）
        prediction = "unknown"
        response_lower = response.lower()
        
        # 检查所有可能的选项
        for choice in sample["choices"]:
            if choice.lower() in response_lower:
                prediction = choice
                break
                
        # 如果无法匹配任何选项，尝试模糊匹配
        if prediction == "unknown":
            if any(pos_word in response_lower for pos_word in ["positive", "good", "bullish", "up"]):
                prediction = "positive"
            elif any(neg_word in response_lower for neg_word in ["negative", "bad", "bearish", "down"]):
                prediction = "negative"
            elif any(neu_word in response_lower for neu_word in ["neutral", "mixed", "unclear"]):
                prediction = "neutral"
        
        # 计算输出长度
        output_length = len(response.split())
        output_lengths.append(output_length)
        
        # 保存预测和真实标签
        predictions.append(prediction)
        ground_truth.append(sample["expected"])
        
        # 添加结果
        results.append({
            "sample_id": sample["id"],
            "text": sample["text"],
            "expected": sample["expected"],
            "prediction": prediction,
            "correct": prediction == sample["expected"],
            "full_response": response,
            "output_length": output_length
        })
        
        # 显示进度（每10个样本）
        if (i + 1) % 10 == 0 or i == 0 or i == len(test_data) - 1:
            console.print(f"[info]已处理 {i + 1}/{len(test_data)} 样本...[/info]")
        
        # 只显示前几个和错误样本的详细信息，避免输出过多
        if i < 5 or prediction != sample["expected"]:
            console.print(f"[data]样本 {sample['id']}[/data]")
            console.print(f"[data]文本: {sample['text']}[/data]")
            console.print(f"[data]预期: {sample['expected']}[/data]")
            console.print(f"[data]预测: {prediction}[/data]")
            console.print(f"[data]回答: {response}[/data]")
            console.print(f"[data]长度: {output_length} 词[/data]")
            console.print("---")
    
    # 计算指标
    accuracy = accuracy_score(ground_truth, predictions)
    f1 = f1_score(ground_truth, predictions, average='weighted')
    avg_length = np.mean(output_lengths)
    
    console.print(f"[success]评估完成[/success]")
    console.print(f"[info]准确率: {accuracy:.4f}[/info]")
    console.print(f"[info]F1分数: {f1:.4f}[/info]")
    console.print(f"[info]平均输出长度: {avg_length:.2f} 词[/info]")
    
    # 按类别计算指标
    class_metrics = {}
    for choice in set(ground_truth):
        indices = [i for i, label in enumerate(ground_truth) if label == choice]
        if indices:
            class_predictions = [predictions[i] for i in indices]
            class_ground_truth = [ground_truth[i] for i in indices]
            class_accuracy = accuracy_score(class_ground_truth, class_predictions)
            class_metrics[choice] = {
                "count": len(indices),
                "accuracy": class_accuracy
            }
    
    console.print("[highlight]按类别的准确率:[/highlight]")
    for choice, metrics in class_metrics.items():
        console.print(f"[info]{choice}: {metrics['accuracy']:.4f} (样本数: {metrics['count']})[/info]")
    
    return {
        "results": results,
        "metrics": {
            "accuracy": accuracy,
            "f1": f1,
            "avg_length": avg_length,
            "class_metrics": class_metrics
        }
    }

def check_overlength_ratio(results, threshold=2.0):
    """检查产生过长输出的样本比例"""
    # 在实际场景中，你需要一个基准长度作为参考
    # 这里我们假设理想回答长度是5个词
    reference_length = 5
    
    overlength_count = 0
    for result in results:
        if result["output_length"] > reference_length * threshold:
            overlength_count += 1
    
    overlength_ratio = overlength_count / len(results) if results else 0
    console.print(f"[info]过长输出比例: {overlength_ratio:.2%}[/info]")
    return overlength_ratio

def compare_models(sft_results, dpo_results):
    """比较SFT模型和DPO模型的表现"""
    console.print("[highlight]模型比较结果:[/highlight]")
    
    # 准确率对比
    acc_change = dpo_results["metrics"]["accuracy"] - sft_results["metrics"]["accuracy"]
    console.print(f"[info]准确率变化: {acc_change:.4f} ({'+' if acc_change >= 0 else ''}{acc_change*100:.2f}%)[/info]")
    
    # F1对比
    f1_change = dpo_results["metrics"]["f1"] - sft_results["metrics"]["f1"]
    console.print(f"[info]F1分数变化: {f1_change:.4f} ({'+' if f1_change >= 0 else ''}{f1_change*100:.2f}%)[/info]")
    
    # 输出长度对比
    length_change = dpo_results["metrics"]["avg_length"] - sft_results["metrics"]["avg_length"]
    length_change_pct = length_change / sft_results["metrics"]["avg_length"] * 100
    console.print(f"[info]平均输出长度变化: {length_change:.2f} 词 ({'+' if length_change >= 0 else ''}{length_change_pct:.2f}%)[/info]")
    
    # 样本级对比
    correct_sft = sum(1 for r in sft_results["results"] if r["correct"])
    correct_dpo = sum(1 for r in dpo_results["results"] if r["correct"])
    
    # 按类别比较
    console.print("[highlight]按类别的准确率比较:[/highlight]")
    all_classes = set(list(sft_results["metrics"]["class_metrics"].keys()) + 
                    list(dpo_results["metrics"]["class_metrics"].keys()))
    
    for choice in all_classes:
        sft_acc = sft_results["metrics"]["class_metrics"].get(choice, {}).get("accuracy", 0)
        dpo_acc = dpo_results["metrics"]["class_metrics"].get(choice, {}).get("accuracy", 0)
        acc_diff = dpo_acc - sft_acc
        console.print(f"[info]{choice}: SFT={sft_acc:.4f}, DPO={dpo_acc:.4f}, 变化={acc_diff:+.4f}[/info]")
    
    # 详细的样本对比
    console.print("[highlight]样本级别对比:[/highlight]")
    
    # 找出预测发生变化的样本
    changed_predictions = []
    for i in range(len(sft_results["results"])):
        sft_result = sft_results["results"][i]
        dpo_result = dpo_results["results"][i]
        
        if sft_result["prediction"] != dpo_result["prediction"]:
            changed_predictions.append({
                "id": sft_result["sample_id"],
                "text": sft_result["text"],
                "expected": sft_result["expected"],
                "sft_prediction": sft_result["prediction"],
                "dpo_prediction": dpo_result["prediction"],
                "sft_length": sft_result["output_length"],
                "dpo_length": dpo_result["output_length"],
                "improved": dpo_result["prediction"] == dpo_result["expected"] and sft_result["prediction"] != sft_result["expected"]
            })
    
    # 统计变化
    improvements = [p for p in changed_predictions if p["improved"]]
    regressions = [p for p in changed_predictions if not p["improved"] and p["sft_prediction"] == p["expected"]]
    
    console.print(f"[info]预测发生变化的样本: {len(changed_predictions)}[/info]")
    console.print(f"[info]DPO改进的样本数: {len(improvements)}[/info]")
    console.print(f"[info]DPO退步的样本数: {len(regressions)}[/info]")
    
    # 显示几个例子
    if improvements:
        console.print("[success]DPO改进的例子:[/success]")
        for i, example in enumerate(improvements[:3]):  # 最多显示3个
            console.print(f"[data]样本 {example['id']}[/data]")
            console.print(f"[data]文本: {example['text']}[/data]")
            console.print(f"[data]预期: {example['expected']}[/data]")
            console.print(f"[data]SFT预测: {example['sft_prediction']} (长度: {example['sft_length']})[/data]")
            console.print(f"[data]DPO预测: {example['dpo_prediction']} (长度: {example['dpo_length']})[/data]")
            console.print("---")
    
    if regressions:
        console.print("[warning]DPO退步的例子:[/warning]")
        for i, example in enumerate(regressions[:3]):  # 最多显示3个
            console.print(f"[data]样本 {example['id']}[/data]")
            console.print(f"[data]文本: {example['text']}[/data]")
            console.print(f"[data]预期: {example['expected']}[/data]")
            console.print(f"[data]SFT预测: {example['sft_prediction']} (长度: {example['sft_length']})[/data]")
            console.print(f"[data]DPO预测: {example['dpo_prediction']} (长度: {example['dpo_length']})[/data]")
            console.print("---")
    
    # 返回比较结果摘要
    return {
        "accuracy_change": acc_change,
        "f1_change": f1_change,
        "length_change": length_change,
        "length_change_pct": length_change_pct,
        "correct_sft": correct_sft,
        "correct_dpo": correct_dpo,
        "improvements": len(improvements),
        "regressions": len(regressions),
    }

def save_results_to_csv(sft_results, dpo_results, output_path="model_comparison_results.csv"):
    """将评估结果保存到CSV文件"""
    # 合并结果
    combined_results = []
    for i in range(len(sft_results["results"])):
        sft_result = sft_results["results"][i]
        dpo_result = dpo_results["results"][i]
        
        row = {
            "sample_id": sft_result["sample_id"],
            "text": sft_result["text"],
            "expected": sft_result["expected"],
            "sft_prediction": sft_result["prediction"],
            "dpo_prediction": dpo_result["prediction"],
            "sft_correct": sft_result["correct"],
            "dpo_correct": dpo_result["correct"],
            "sft_length": sft_result["output_length"],
            "dpo_length": dpo_result["output_length"],
            "prediction_changed": sft_result["prediction"] != dpo_result["prediction"],
            "improved": dpo_result["correct"] and not sft_result["correct"],
            "regression": sft_result["correct"] and not dpo_result["correct"],
        }
        combined_results.append(row)
    
    # 创建DataFrame并保存
    df = pd.DataFrame(combined_results)
    df.to_csv(output_path, index=False)
    console.print(f"[success]结果已保存到 {output_path}[/success]")

def main():
    """主函数"""
    console.print("[highlight]金融情感分析模型DPO效果评估[/highlight]")
    
    # 加载测试数据集
    test_data = load_test_dataset()
    if not test_data:
        console.print("[error]没有可用的测试数据，评估终止[/error]")
        return
    
    # 可以设置一个较小的样本量进行快速测试
    sample_limit = None  # 设置为None使用全部数据，或者设置一个整数进行快速测试
    
    # 加载SFT模型
    sft_model, sft_tokenizer = load_model(sft_model_path)
    if sft_model is None:
        console.print("[error]SFT模型加载失败，评估终止[/error]")
        return
    
    # 评估SFT模型
    console.print("[highlight]评估SFT模型[/highlight]")
    sft_eval = evaluate_model(sft_model, sft_tokenizer, test_data, sample_limit)
    sft_overlength = check_overlength_ratio(sft_eval["results"])
    
    # 加载DPO模型
    dpo_model, dpo_tokenizer = load_model(dpo_model_path)
    if dpo_model is None:
        console.print("[error]DPO模型加载失败，评估终止[/error]")
        return
    
    # 评估DPO模型
    console.print("[highlight]评估DPO模型[/highlight]")
    dpo_eval = evaluate_model(dpo_model, dpo_tokenizer, test_data, sample_limit)
    dpo_overlength = check_overlength_ratio(dpo_eval["results"])
    
    # 比较两个模型
    comparison = compare_models(sft_eval, dpo_eval)
    
    # 保存结果到CSV
    save_results_to_csv(sft_eval, dpo_eval)
    
    # 输出DPO改进的关键指标
    console.print("[highlight]DPO改进总结:[/highlight]")
    console.print(f"[info]过长输出比例: SFT={sft_overlength:.2%}, DPO={dpo_overlength:.2%}, 减少={(sft_overlength-dpo_overlength)*100:.2f}个百分点[/info]")
    console.print(f"[info]准确率变化: {comparison['accuracy_change']*100:+.2f}%[/info]")
    console.print(f"[info]输出长度减少: {-comparison['length_change_pct']:+.2f}%[/info]")
    console.print(f"[info]改进的样本数: {comparison['improvements']}[/info]")
    console.print(f"[info]退步的样本数: {comparison['regressions']}[/info]")
    
    if dpo_overlength < sft_overlength and comparison['accuracy_change'] >= 0:
        console.print("[success]DPO优化成功: 减少了冗长输出同时保持或提高了准确率[/success]")
    elif dpo_overlength < sft_overlength:
        console.print("[warning]DPO部分优化: 减少了冗长输出但准确率有所下降[/warning]")
    elif comparison['accuracy_change'] > 0:
        console.print("[warning]DPO部分优化: 提高了准确率但未减少冗长输出[/warning]")
    else:
        console.print("[error]DPO优化不明显: 建议调整DPO训练参数[/error]")

if __name__ == "__main__":
    main()