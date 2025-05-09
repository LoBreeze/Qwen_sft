root@autodl-container-f2c94a94e7-0be5bdc8:~/autodl-tmp# /root/miniconda3/bin/python /root/autodl-tmp/qwen_flare_dpo_test.py
🦥 Unsloth: Will patch your computer to enable 2x faster free finetuning.
Unsloth: Failed to patch Gemma3ForConditionalGeneration.
🦥 Unsloth Zoo will now patch everything to make training faster!
金融情感分析模型DPO效果评估
正在加载flare_fiqasa数据集...
加载数据集失败: Value.__init__() missing 1 required positional argument: 'dtype'
使用少量样本作为备选测试集
正在加载模型: /root/autodl-tmp/output/Qwen2.5-1.5B-Financial-Sentiment...
==((====))==  Unsloth 2025.3.19: Fast Qwen2 patching. Transformers: 4.51.2.
   \\   /|    NVIDIA vGPU-32GB. Num GPUs = 1. Max memory: 31.503 GB. Platform: Linux.
O^O/ \_/ \    Torch: 2.6.0+cu124. CUDA: 8.9. CUDA Toolkit: 12.4. Triton: 3.2.0
\        /    Bfloat16 = TRUE. FA [Xformers = 0.0.29.post3. FA2 = False]
 "-____-"     Free license: http://github.com/unslothai/unsloth
Unsloth: Fast downloading is enabled - ignore downloading bars which are red colored!
Sliding Window Attention is enabled but not implemented for `eager`; unexpected results may be encountered.
评估SFT模型
开始模型评估...使用3个测试样本
The attention mask is not set and cannot be inferred from input because pad token is same as eos token. As a consequence, you may observe unexpected behavior. Please pass your input's `attention_mask` to obtain reliable results.
已处理 1/3 样本...
样本 sample1
文本: Whats up with $LULU? Numbers looked good, not great, but good. I think conference call will instill confidence.
预期: neutral
预测: negative
回答: Based on the given information, it seems like the sentiment of this financial post is **Positive**. The post mentions "good" numbers and 
suggests that the conference call may instill confidence, which both positive indicators. However, there's also a hint of uncertainty in the 
phrase "not great," so we can't be completely certain about the overall sentiment. Overall, though, the tone appears to lean towards positivity. 

Please note that while the post contains some mixed signals (positive and negative), the
长度: 79 词
---
样本 sample2
文本: Apple's latest quarterly earnings report shows a 15% increase in revenue, beating analyst expectations.
预期: positive
预测: positive
回答: Based on the information provided, the sentiment of this financial post is **Positive**. The statement indicates that Apple's earnings have 
increased by 15%, which is considered a significant positive development for the company and its shareholders. Analysts' expectations were met or 
exceeded, suggesting investor confidence and satisfaction with the company's performance. Therefore, the overall tone and content of the post 
convey a positive sentiment towards Apple's financial results.
长度: 68 词
---
已处理 3/3 样本...
样本 sample3
文本: Tesla's stock fell 7% after concerns about production delays and increasing competition in the EV market.
预期: negative
预测: negative
回答: Based on the information provided, the sentiment of this financial post appears to be **Negative**. The statement mentions that Tesla's 
stock fell by 7%, which indicates a decline in value. Additionally, it highlights concerns about production delays and increased competition in 
the electric vehicle (EV) market, both of which could negatively impact the company's performance. Therefore, the overall tone and implications 
suggest a negative sentiment towards Tesla's stock.
长度: 68 词
---
评估完成
准确率: 0.6667
F1分数: 0.5556
平均输出长度: 71.67 词
按类别的准确率:
neutral: 0.0000 (样本数: 1)
negative: 1.0000 (样本数: 1)
positive: 1.0000 (样本数: 1)
过长输出比例: 100.00%
正在加载模型: /root/autodl-tmp/output/Qwen2.5-1.5B-Financial-Sentiment-DPO...
==((====))==  Unsloth 2025.3.19: Fast Qwen2 patching. Transformers: 4.51.2.
   \\   /|    NVIDIA vGPU-32GB. Num GPUs = 1. Max memory: 31.503 GB. Platform: Linux.
O^O/ \_/ \    Torch: 2.6.0+cu124. CUDA: 8.9. CUDA Toolkit: 12.4. Triton: 3.2.0
\        /    Bfloat16 = TRUE. FA [Xformers = 0.0.29.post3. FA2 = False]
 "-____-"     Free license: http://github.com/unslothai/unsloth
Unsloth: Fast downloading is enabled - ignore downloading bars which are red colored!
Unsloth 2025.3.19 patched 28 layers with 28 QKV layers, 28 O layers and 28 MLP layers.
评估DPO模型
开始模型评估...使用3个测试样本
已处理 1/3 样本...
样本 sample1
文本: Whats up with $LULU? Numbers looked good, not great, but good. I think conference call will instill confidence.
预期: neutral
预测: positive
回答: The sentiment is positive.
长度: 4 词
---
样本 sample2
文本: Apple's latest quarterly earnings report shows a 15% increase in revenue, beating analyst expectations.
预期: positive
预测: positive
回答: The sentiment is positive.
长度: 4 词
---
已处理 3/3 样本...
样本 sample3
文本: Tesla's stock fell 7% after concerns about production delays and increasing competition in the EV market.
预期: negative
预测: negative
回答: The sentiment is negative.
长度: 4 词
---
评估完成
准确率: 0.6667
F1分数: 0.5556
平均输出长度: 4.00 词
按类别的准确率:
neutral: 0.0000 (样本数: 1)
negative: 1.0000 (样本数: 1)
positive: 1.0000 (样本数: 1)
过长输出比例: 0.00%
模型比较结果:
准确率变化: 0.0000 (+0.00%)
F1分数变化: 0.0000 (+0.00%)
平均输出长度变化: -67.67 词 (-94.42%)
按类别的准确率比较:
neutral: SFT=0.0000, DPO=0.0000, 变化=+0.0000
negative: SFT=1.0000, DPO=1.0000, 变化=+0.0000
positive: SFT=1.0000, DPO=1.0000, 变化=+0.0000
样本级别对比:
预测发生变化的样本: 1
DPO改进的样本数: 0
DPO退步的样本数: 0
结果已保存到 model_comparison_results.csv
DPO改进总结:
过长输出比例: SFT=100.00%, DPO=0.00%, 减少=100.00个百分点
准确率变化: +0.00%
输出长度减少: +94.42%
改进的样本数: 0
退步的样本数: 0
DPO优化成功: 减少了冗长输出同时保持或提高了准确率