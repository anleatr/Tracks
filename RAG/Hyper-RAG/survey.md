## 依赖数据集标注

### Evaluating Object Hallucination in Large Vision-Language Models（https://arxiv.org/abs/2305.10355）
- 评价方法：  幻觉与细节一致性探测法POPE (Polling-based Object Probing Evaluation)
- 对于每张图像中存在若干个GT标注物体，生成一系列二值问题并用`Is there a [object] in the image？`提问llm，模型回答yes or no， 最后统计accuracy, precison, recall, f1指标来进行比较

可能原始图片没有真值，是不是可以用VLM去做一些GT，比如我们准备一份问题列表（“这张图片里有没有人？”...），让LLaVA去生成一份GT，再用这份GT和其他llm输出去计算指标，但是如何确保GT的真实性？如果图片数量较少可以用人工校正，但是图片太多就不现实了。

###  Uncertainty-o: One Model-agnostic Framework for Unveiling Uncertainty in Large Multimodal Models (https://arxiv.org/abs/2506.07575)

- 评价方法：对原始的多模态输入（如图像+指令）进行微小的扰动（如改变图片缩放、调整指令措辞），生成多个变体。对这些多次生成的文本解析结果进行聚类。如果解析结果在不同扰动下高度一致（聚类集中），则熵值低，表示模型“确定”；如果解析结果五花八门（聚类分散），则熵值高，表示模型存在高度不确定性，解析结果极大概率不可信。

### Spectral entropies as information-theoretic tools for complex network comparison（https://arxiv.org/abs/1607.08822）

- 评价方法： SPICE (Scene Graph)，将模型生成的候选描述（Candidate）与人类编写的参考描述（Reference）进行语义图比对,将描述转化为场景图（物体、属性、关系）。输出 F-score。衡量解析出的语义三元组与参考事实的覆盖率。需要人工描述进行比对
- 好像不太适合我们这个任务

### Visual Instruction Tuning （https://arxiv.org/abs/2304.08485）
- 评价方法： LLaVA-Bench (In-the-wild) 
- 提供一组真实的野外场景图片和问题，由 GPT-4 根据图像的“金标准描述”对模型的回答进行打分（0-10分），通过对比模型输出与 GPT-4 生成的参考答案在准确性、细节丰富度、推理逻辑上的差异来给出评分。 
- 这算是使用了大模型评价？

## 不依赖数据集标注

###  GenCeption: Evaluate Vision LLMs with Unlabeled Unimodal Data （https://arxiv.org/abs/2402.14973）
- 评价方法： GenCeption: 构建一个“图像 -> 文本 -> 图像”的闭环。首先让待测 LMM 生成图像描述，再利用预训练的文生图模型（论文里用的DALL-E 3）还原图像，把新图像在喂给LLM比对和原图差异，多轮迭代，最后通过CLIP或ViT提取原图与还原图的特征，计算原图与重构图的偏差。偏差越小，说明 LMM 的解析越完善。
- 论文里面也做了视觉密集型样本和文本密集型样本的实验，在文本翻译，OCR领域也有不错效果

### CLIPScore: A Reference-free Evaluation Metric for Image Captioning (https://arxiv.org/abs/2104.08718)
- 评价方法：计算描述文本向量和图像向量的余弦相似度， 称为CLIPScore
- 可以直接使用作为评价指标，但是CLIP文本编码器上限为77个token，不能用来评测太长句子，并且主要用来评价整体图文一致性而不是完善性
- 或许修改一些是符合要求的简单方案？
### An Online Reference-Free Evaluation Framework for Flowchart Image-to-Code Generation（https://arxiv.org/abs/2602.13376）
主要用来评估流程图描述质量
- 评价方法：分为召回评估和幻觉检测两部分
- 召回评估：用一个高质量的OCR模型（如Gemini 1.5 Pro）提取原图中的所有文本，将其作为“代理参考标准”。然后检查LLM生成Mermaid代码中包含了多少OCR提取出的文本。
- 幻觉检测：将LLM生成的代码拆解成一个个节点和边，然后对照原图向模型提问，计算准确率