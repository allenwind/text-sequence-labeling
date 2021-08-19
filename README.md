# text-sequence-labeling

NLP中序列标注任务，包括分词（WS）、实体识别（NER）、词性标注（POS）。中文分词见[chinese-cut-word](https://github.com/allenwind/chinese-cut-word)，另外CRF提供一个简单的实现和例子，见[tensorflow-crf](https://github.com/allenwind/tensorflow-crf)。这里主要是NER相关的模型、tricks，当然需要强调，解决NER问题不一定需要序列标注方法，像中文分词类似的词典匹配方法也能，而像邮箱、网址这类规则性强的实体则直接使用正则表达式。



持续更新中~


## 优化角度



从**特征**角度：

- 字词（word-level）特征
- 字形特征
- 词性特征
- 句法特征
- 知识图谱



从**编码器**角度：

- BiLSTM
- CNN
- BERT
- BERT-BiLSTM



从**解码**角度：

- CRF
- MLP + softmax（不考虑标签约束的逐位置分类）
- 指针网络（Pointer Network）


合并连续的同类型实体

数据增强：

- 实体替换
- 引入词汇信息



多任务：

- 联合分词任务



其他：

- 对抗训练
- 各种Loss（focal loss）


## 参考

[1] [A Survey on Deep Learning for Named Entity Recognition](https://arxiv.org/abs/1812.09449)
