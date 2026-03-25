请你调研一下，有没有哪些用于比较不同的llm对于同一张图片解析准确性的评价方法，用来看哪个输出信息更完善，请你将其分为依赖数据集标注和不依赖两类，并且按照论文的arxiv地址（地址使用两个分割的代码块，确保域名为arxiv.org）-评价方法-相关文献的格式给出

这篇文章是如何比较不同llm对于同一张图片解析准确率的，依赖数据集标注吗，步骤是什么，使用的哪些数据集
## 依赖数据集标注

### Evaluating Object Hallucination in Large Vision-Language Models（https://arxiv.org/abs/2305.10355）
- 评价方法：  幻觉与细节一致性探测法POPE (Polling-based Object Probing Evaluation)
- 数据集： MSCOCO
- 对于每张图像中存在若干个GT标注物体，生成一系列二值问题并用`Is there a [object] in the image？`提问llm，模型回答yes or no， 最后统计accuracy, precison, recall, f1指标来进行比较

可能原始图片没有真值，是不是可以用VLM去做一些GT，比如我们准备一份问题列表（“这张图片里有没有人？”...），让LLaVA去生成一份GT，再用这份GT和其他llm输出去计算指标，但是如何确保GT的真实性？如果图片数量较少可以用人工校正，但是图片太多就不现实了。

### Spectral entropies as information-theoretic tools for complex network comparison（https://arxiv.org/abs/1607.08822）
- 评价方法： SPICE (Scene Graph)，将模型生成的候选描述（Candidate）与人类编写的参考描述（Reference）进行语义图比对
- 数据集： Microsoft COCO 2014 & 2015 挑战赛数据， Flickr 8K
- 将描述转化为场景图（物体、属性、关系）。输出 F-score。衡量解析出的语义三元组与参考事实的覆盖率。需要人工描述进行比对

### MM-Vet: Evaluating Large Multimodal Models for Integrated Capabilities （https://arxiv.org/abs/2308.02490）
- 评价方法：MM-Vet 
- 
由于多模态模型在实际应用中通常输出开放式回答（如解释表情包、撰写长段落、进行数学推导），传统的单一正确选项或关键词匹配无法准确衡量回答质量。因此，作者为测试集中的每一个问题都配备了标准答案，主要由人工标注完

###  Uncertainty-o: One Model-agnostic Framework for Unveiling Uncertainty in Large Multimodal Models (https://arxiv.org/abs/2506.07575)
- 评价方法: 
第一步：获取初始回答与真实标签 (Initial Answer Obtaining) 模型首先接收原始的图片和文本提示（Prompt），生成一个初始回答。随后，将这个初始回答与数据集的 Ground Truth 进行比对，标记该回答是“正确的”还是“存在幻觉的（错误的）”。这个标记将作为后续评估的绝对标准。
第二步：多模态提示扰动 (Multimodal Prompt Perturbation) 为了测试模型“对自己回答的确定程度”，研究人员会对原始输入进行多维度的轻微扰动。对于图片，会应用空间变换（如旋转）和属性扭曲（如模糊或亮度调整）；对于文本，会进行同义改写或词汇替换。然后将这些不同扰动程度的输入多次喂给模型，收集它产生的多个回答。
第三步：提取语义与计算不确定性 (Multimodal Semantic Uncertainty) 由于模型每次生成的文本可能字面上不同但意思一样，文章使用了一个大语言模型（如 Qwen2.5-7B）对收集到的所有回答进行“语义聚类”。意思相同的回答被归为一类，然后计算这些语义类别的分布熵（Entropy）。如果模型给出各种相互矛盾的回答，说明分布熵高，模型非常“不确定”；如果回答高度一致，说明分布熵低，模型很“自信”。
第四步：通过检测指标进行比较 (Hallucination Detection Metrics) 最后，研究人员会观察模型计算出的“不确定性分数”与第一步得出的“真实幻觉标签”是否对齐。文章使用了三个主要指标来比较不同模型：
AUROC / AURAC（越高越好）：衡量模型分数区分对错的排序能力。
ECE（越低越好）：预期校准误差，衡量模型给出的不确定性与实际错误率的拟合程度。

## 不依赖数据集标注

###  GenCeption: Evaluate Vision LLMs with Unlabeled Unimodal Data （https://arxiv.org/abs/2402.14973）
- 评价方法： GenCeption: 构建一个“图像 -> 文本 -> 图像”的闭环。首先让待测 LMM 生成图像描述，再利用预训练的文生图模型（论文里用的DALL-E 3）还原图像，把新图像在喂给LLM比对和原图差异，多轮迭代，最后通过CLIP或ViT提取原图与还原图的特征，计算原图与重构图的偏差。偏差越小，说明 LMM 的解析越完善。
- 论文里面也做了视觉密集型样本和文本密集型样本的实验，在文本翻译，OCR领域也有不错效果



### CLIPScore: A Reference-free Evaluation Metric for Image Captioning (https://arxiv.org/abs/2104.08718)
- 评价方法：计算描述文本向量和图像向量的余弦相似度， 称为CLIPScore
- 可以直接使用作为评价指标，但是CLIP文本编码器上限为77个token，不能用来评测太长句子，并且主要用来评价整体图文一致性而不是完善性

### Visual Instruction Tuning （https://arxiv.org/abs/2304.08485）
- 评价方法： LLaVA-Bench (In-the-wild) 
- 提供一组真实的野外场景图片和问题，由 GPT-4 根据图像的“金标准描述”对模型的回答进行打分（0-10分），通过对比模型输出与 GPT-4 生成的参考答案在准确性、细节丰富度、推理逻辑上的差异来给出评分。 这算是使用了大模型评价？

