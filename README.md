# Event-Extraction（事件抽取资料综述总结）更新中...
近年来事件抽取方法总结，包括中文事件抽取、开放域事件抽取、事件数据生成、跨语言事件抽取、小样本事件抽取、零样本事件抽取等类型，DMCNN、FramNet、DLRNN、DBRNN、GCN、DAG-GRU、JMEE、PLMEE等方法


# Table of Contents 目录

- [Surveys（综述论文）](#Surveys)
- [Shallow Learning Models（浅层学习模型）](#Shallow-Learning-Models)
- [Deep Learning Models（深度学习模型）](#Deep-Learning-Models)
- [Datasets（数据集）](#Datasets)
- [Evaluation Metrics（评价指标）](#Evaluation-Metrics)
- [Future Research Challenges（未来研究挑战）](#Future-Research-Challenges)
- [Tools and Repos（工具与资源）](#tools-and-repos)
</p></blockquote></details>

---


## Surveys(综述论文)
[:arrow_up:](#table-of-contents)

<details/>
<summary/>
<a href="https://arxiv.org/pdf/2008.00364.pdf">元事件抽取研究综述，2019</a> by<i>GAO Li-zheng, ZHOU Gang, LUO Jun-yong, LAN Ming-jing
</a></summary><blockquote><p align="justify">
事件抽取是信息抽取领域的一个重要研究方向,在情报收集、知识提取、文档摘要、知识问答等领域有着广泛应用。对当前事件抽取领域研究得较多的元事件抽取进行了综述。首先,简要介绍了元事件和元事件抽取的基本概念,以及元事件抽取的主要实现方法。然后,重点阐述了元事件抽取的主要任务,详细介绍了元事件检测过程,并对其他相关任务进行了概述。最后,总结了元事件抽取面临的问题,在此基础上展望了元事件抽取的发展趋势。
</p></blockquote></details>


details/>
<summary/>
<a href="https://arxiv.org/pdf/2008.00364.pdf">An Overview of Event Extraction from Text，2011</a> by<i>Frederik Hogenboom, Flavius Frasincar, Uzay Kaymak, Franciska de Jong:
</a></summary><blockquote><p align="justify">
One common application of text mining is event extraction,which encompasses deducing specific knowledge concerning incidents re-ferred to in texts. Event extraction can be applied to various types ofwritten text, e.g., (online) news messages, blogs, and manuscripts. Thisliterature survey reviews text mining techniques that are employed forvarious event extraction purposes. It provides general guidelines on howto choose a particular event extraction technique depending on the user,the available content, and the scenario of use.
</p></blockquote></details>


## Deep Learning Models（深度学习模型）
[:arrow_up:](#table-of-contents)

#### 2020
 <details/>
<summary/>
  <a href="https://transacl.org/ojs/index.php/tacl/article/view/1853">Spanbert: Improving pre-training by representing and predicting spans</a>  --- SpanBERT主要贡献：Span Mask机制，不再对随机的单个token添加mask，随机对邻接分词添加mask；Span Boundary Objective(SBO)训练，使用分词边界表示预测被添加mask分词的内容；一个句子的训练效果更好 --- (<a href="https://github.com/facebookresearch/SpanBERT">Github</a>)</summary><blockquote><p align="justify">
We present SpanBERT, a pre-training method that is designed to better represent and predict spans of text. Our approach extends BERT by (1) masking contiguous random spans, rather than random tokens, and (2) training the span boundary representations to predict the entire content of the masked span, without relying on the individual token representations within it. SpanBERT consistently outperforms BERT and our better-tuned baselines, with substantial gains on span selection tasks such as question answering and coreference resolution. In particular, with the same training data and model size as BERT-Large, our single model obtains 94.6% and 88.7% F1 on SQuAD 1.1 and 2.0 respectively. We also achieve a new state of the art on the OntoNotes coreference resolution task (79.6% F1), strong performance on the TACRED relation extraction benchmark, and even gains on GLUE.
  提出了一种名为SpanBERT的预训练方法，旨在更好地表示和预测文本范围。我们的方法通过(1)屏蔽连续的随机跨度而不是随机标记来扩展BERT，以及(2)训练跨度边界表示来预测屏蔽跨度的整个内容，而不依赖于其中的单个标记表示。斯潘伯特的表现始终优于伯特和我们优化后的基线，在跨度选择任务(如问题回答和共参照解决)上取得了实质性的进展。特别是，在训练数据和模型尺寸与伯特- large相同的情况下，我们的单模型在1.1和2.0阵容上分别得到F1的94.6%和88.7%。我们还实现了OntoNotes共参考分辨率任务(79.6% F1)的新水平，在TACRED关系提取基准测试上的强劲性能，甚至在GLUE上也取得了进展。
 
![image](https://github.com/xiaoqian19940510/text-classification-surveys/blob/master/picture3.png)

</p></blockquote></details>



 <details/>
<summary/>
  <a href="https://openreview.net/forum?id=H1eA7AEtvS">ALBERT: A lite BERT for self-supervised learning of language representations</a> --- ALBERT论文主要贡献：瘦身成功版BERT，全新的参数共享机制。对embedding因式分解，隐层embedding带有上线文信息；跨层参数共享，全连接和attention层都进行参数共享，效果下降，参数减少，训练时间缩短；句间连贯 --- (<a href="https://github.com/google-research/ALBERT">Github</a>)</summary><blockquote><p align="justify">
Increasing model size when pretraining natural language representations often results in improved performance on downstream tasks. However, at some point further model increases become harder due to GPU/TPU memory limitations and longer training times. To address these problems,  we present two parameter-reduction techniques to lower memory consumption and increase the training speed of BERT~\citep{devlin2018bert}. Comprehensive empirical evidence shows that our proposed methods lead to models that scale much better compared to the original BERT. We also use a self-supervised loss that focuses on modeling inter-sentence coherence, and show it consistently helps downstream tasks with multi-sentence inputs. As a result, our best model establishes new state-of-the-art results on the GLUE, RACE, and \squad benchmarks while having fewer parameters compared to BERT-large. The code and the pretrained models are available at https://github.com/google-research/ALBERT.
  在对自然语言表示进行预训练时增加模型大小通常会提高下游任务的性能。然而，在某种程度上，由于GPU/TPU内存的限制和更长的训练时间，进一步的模型增加会变得更加困难。为了解决这些问题，提出了两种参数减少技术来降低内存消耗和提高BERT的训练速度~\citep{devlin2018bert}。全面的经验证据表明，提出的方法导致的模型，规模比原来的BERT更好。还使用了一种关注于句子间连贯性建模的自我监督丢失，并表明它始终有助于多句子输入的下游任务。因此，最好的模型建立了新的最先进的结果在胶水，比赛，\队基准，而拥有更少的参数，比伯特-大。代码和预先训练的模型可以在https://github.com/google-research/ALBERT下载。
</p></blockquote></details>

#### 2019
 <details/>
<summary/>
  <a href="https://arxiv.org/abs/1907.11692">Roberta: A robustly optimized BERT pretraining approach</a> --- Roberta主要贡献：更多训练数据、更大batch size、训练时间更长；去掉NSP；训练序列更长；动态调整Masking机制，数据copy十份，每句话会有十种不同的mask方式 --- (<a href="https://github.com/pytorch/fairseq">Github</a>)</summary><blockquote><p align="justify">
Language model pretraining has led to significant performance gains but careful comparison between different approaches is challenging. Training is computationally expensive, often done on private datasets of different sizes, and, as we will show, hyperparameter choices have significant impact on the final results. We present a replication study of BERT pretraining (Devlin et al., 2019) that carefully measures the impact of many key hyperparameters and training data size. We find that BERT was significantly undertrained, and can match or exceed the performance of every model published after it. Our best model achieves state-of-the-art results on GLUE, RACE and SQuAD. These results highlight the importance of previously overlooked design choices, and raise questions about the source of recently reported improvements. We release our models and code.
  语言模型预训练已经导致了显著的性能提高，但仔细比较不同的方法是具有挑战性的。训练的计算开销很大，通常是在不同大小的私有数据集上进行的，而且，正如我们将展示的，超参数的选择对最终结果有很大的影响。我们提出了一项BERT预训练的复制研究(Devlin et al.， 2019)，该研究仔细测量了许多关键超参数和训练数据大小的影响。我们发现BERT明显训练不足，可以匹配或超过其后发布的每个模型的性能。我们最好的模型在GLUE, RACE和SQuAD上达到了最先进的效果。这些结果突出了以前被忽略的设计选择的重要性，并对最近报告的改进的来源提出了疑问。我们发布我们的模型和代码。
</p></blockquote></details>

 <details/>
<summary/>
  <a href="http://papers.nips.cc/paper/8812-xlnet-generalized-autoregressive-pretraining-for-language-understanding">Xlnet: Generalized autoregressive pretraining for language understanding</a> --- Xlnet主要贡献：采用自回归（AR）模型替代自编码（AE）模型，解决mask带来的负面影响；双流自注意力机制；引入transformer-xl，解决超长序列的依赖问题；采用相对位置编码 --- (<a href="https://github.com/zihangdai/xlnet">Github</a>)</summary><blockquote><p align="justify">
With the capability of modeling bidirectional contexts, denoising autoencoding based pretraining like BERT achieves better performance than pretraining approaches based on autoregressive language modeling. However, relying on corrupting the input with masks, BERT neglects dependency between the masked positions and suffers from a pretrain-finetune discrepancy. In light of these pros and cons, we propose XLNet, a generalized autoregressive pretraining method that (1) enables learning bidirectional contexts by maximizing the expected likelihood over all permutations of the factorization order and (2) overcomes the limitations of BERT thanks to its autoregressive formulation. Furthermore, XLNet integrates ideas from Transformer-XL, the state-of-the-art autoregressive model, into pretraining. Empirically, under comparable experiment setting, XLNet outperforms BERT on 20 tasks, often by a large margin, including question answering, natural language inference, sentiment analysis, and document ranking.
  由于具有双向上下文建模的能力，像BERT这样的基于自编码的去噪预训练方法的性能优于基于自回归语言建模的预训练方法。然而，BERT依靠用掩模破坏输入，忽略了掩模位置之间的依赖关系，并遭受了预训练-微调误差。鉴于这些优点和缺点，提出了XLNet，这是一种广义的自回归预训练方法，它(1)通过最大化因数分解顺序的所有排列的期望似然，使学习双向上下文成为可能;(2)由于它的自回归公式，克服了BERT的局限性。此外，XLNet将Transformer-XL(最先进的自回归模型)的思想集成到预训练中。从经验上看，在可比的实验设置下，XLNet在20项任务上的表现优于BERT，通常都是遥遥领先的，包括问题回答、自然语言推理、情感分析和文档排序。
  
  ![image](https://github.com/xiaoqian19940510/text-classification-surveys/blob/master/picture4.png)  
  
</p></blockquote></details>




 <details/>
<summary/>
  <a href="https://www.aclweb.org/anthology/P19-1441/">Multi-task deep neural networks for natural language understanding</a> --- MT-DNN主要贡献：多任务学习机制训练模型，提高模型的泛化性能 --- (<a href="https://github.com/namisan/mt-dnn">Github</a>)</summary><blockquote><p align="justify">
In this paper, we present a Multi-Task Deep Neural Network (MT-DNN) for learning representations across multiple natural language understanding (NLU) tasks. MT-DNN not only leverages large amounts of cross-task data, but also benefits from a regularization effect that leads to more general representations to help adapt to new tasks and domains. MT-DNN extends the model proposed in Liu et al. (2015) by incorporating a pre-trained bidirectional transformer language model, known as BERT (Devlin et al., 2018). MT-DNN obtains new state-of-the-art results on ten NLU tasks, including SNLI, SciTail, and eight out of nine GLUE tasks, pushing the GLUE benchmark to 82.7% (2.2% absolute improvement) as of February 25, 2019 on the latest GLUE test set. We also demonstrate using the SNLI and SciTail datasets that the representations learned by MT-DNN allow domain adaptation with substantially fewer in-domain labels than the pre-trained BERT representations. Our code and pre-trained models will be made publicly available.
  在本文中，提出了一个多任务深度神经网络(MT-DNN)，用于跨多个自然语言理解(NLU)任务学习表示。MT-DNN不仅利用了大量的跨任务数据，而且还受益于正则化效应，从而产生更通用的表示，以帮助适应新的任务和领域。MT-DNN扩展了Liu等人(2015)提出的模型，加入了一个预训练的双向transformer语言模型，称为BERT (Devlin et al.， 2018)。MT-DNN获得新的先进的结果十NLU任务,包括SNLI SciTail,和九胶水的任务,把胶水基准82.7%(2.2%绝对改进)2月25日,2019年最新胶水测试集。我们还演示使用SNLI和SciTail数据集,表示学习通过MT-DNN允许域适应在域标签明显少于pre-trained伯特表示。代码和预训练的模型将公开提供。
  
![image](https://github.com/xiaoqian19940510/text-classification-surveys/blob/master/picture5.png)  
  
</p></blockquote></details>

  


 <details/>
<summary/>
  <a href="https://doi.org/10.18653/v1/n19-1423">BERT: pre-training of deep bidirectional transformers for language understanding</a> --- BERT主要贡献：BERT是双向的Transformer block连接，增加词向量模型泛化能力，充分描述字符级、词级、句子级关系特征。真正的双向encoding，Masked LM类似完形填空；transformer做encoder实现上下文相关，而不是bi-LSTM，模型更深，并行性更好；学习句子关系表示，句子级负采样 --- (<a href="https://github.com/google-research/bert">Github</a>)</summary><blockquote><p align="justify">
We introduce a new language representation model called BERT, which stands for Bidirectional Encoder Representations from Transformers. Unlike recent language representation models (Peters et al., 2018a; Radford et al., 2018), BERT is designed to pre-train deep bidirectional representations from unlabeled text by jointly conditioning on both left and right context in all layers. As a result, the pre-trained BERT model can be fine-tuned with just one additional output layer to create state-of-the-art models for a wide range of tasks, such as question answering and language inference, without substantial task-specific architecture modifications. BERT is conceptually simple and empirically powerful. It obtains new state-of-the-art results on eleven natural language processing tasks, including pushing the GLUE score to 80.5 (7.7 point absolute improvement), MultiNLI accuracy to 86.7% (4.6% absolute improvement), SQuAD v1.1 question answering Test F1 to 93.2 (1.5 point absolute improvement) and SQuAD v2.0 Test F1 to 83.1 (5.1 point absolute improvement).
  引入了一种新的语言表示模型BERT，它代表来自转换器的双向编码器表示。不同于最近的语言表示模型(Peters et al.， 2018a;(Radford et al.， 2018)， BERT被设计用于预训练未标记文本的深层双向表示，方法是联合作用于所有层中的左右上下文。因此，只需一个额外的输出层就可以对预先训练好的BERT模型进行微调，从而为广泛的任务创建最先进的模型，比如问题回答和语言推理，而无需对特定于任务的架构进行实质性的修改。伯特在概念上是简单的，在经验上是强大的。它获得新的先进的结果十一自然语言处理任务,包括推动胶分数80.5(7.7点绝对改进),MultiNLI精度86.7%绝对改善(4.6%),球队v1.1问答测试F1 93.2(1.5点绝对改进)和阵容v2.0测试F1到83.1(5.1点绝对改善)。
  
![image](https://github.com/xiaoqian19940510/text-classification-surveys/blob/master/picture6.png)  
  
</p></blockquote></details>

  

 <details/>
<summary/>
  <a href="https://wvvw.aaai.org/ojs/index.php/AAAI/article/view/4725">Graph convolutional networks for text classification</a> --- TextGCN主要贡献：构建基于文本和词的异构图，在GCN上进行半监督文本分类，包含文本节点和词节点，document-word边的权重是TF-IDF，word-word边的权重是PMI，即词的共现频率 --- (<a href="https://github.com/yao8839836/text_gcn">Github</a>)</summary><blockquote><p align="justify">
Text classification is an important and classical problem in natural language processing. There have been a number of studies that applied convolutional neural networks (convolution on regular grid, e.g., sequence) to classification. However, only a limited number of studies have explored the more flexible graph convolutional neural networks (convolution on non-grid, e.g., arbitrary graph) for the task. In this work, we propose to use graph convolutional networks for text classification. We build a single text graph for a corpus based on word co-occurrence and document word relations, then learn a Text Graph Convolutional Network (Text GCN) for the corpus. Our Text GCN is initialized with one-hot representation for word and document, it then jointly learns the embeddings for both words and documents, as supervised by the known class labels for documents. Our experimental results on multiple benchmark datasets demonstrate that a vanilla Text GCN without any external word embeddings or knowledge outperforms state-of-the-art methods for text classification. On the other hand, Text GCN also learns predictive word and document embeddings. In addition, experimental results show that the improvement of Text GCN over state-of-the-art comparison methods become more prominent as we lower the percentage of training data, suggesting the robustness of Text GCN to less training data in text classification.
  文本分类是自然语言处理中的一个重要而经典的问题。已经有很多研究将卷积神经网络(规则网格上的卷积，例如序列)应用于分类。然而，只有有限的研究探索了更灵活的图形卷积神经网络(在非网格上卷积，如任意图)的任务。在这项工作中，我们提出使用图卷积网络来进行文本分类。基于词的共现关系和文档词的关系，为语料库构建单个文本图，然后学习用于语料库的文本图卷积网络(text GCN)。我们的文本GCN对word和document使用单热表示进行初始化，然后在已知文档类标签的监督下联合学习单词和文档的嵌入。我们在多个基准数据集上的实验结果表明，一个没有任何外部词嵌入或知识的普通文本GCN优于最先进的文本分类方法。另一方面，Text GCN也学习预测词和文档嵌入。此外，实验结果表明，当我们降低训练数据的百分比时，文本GCN相对于现有比较方法的改进更加显著，说明在文本分类中，文本GCN对较少的训练数据具有鲁棒性。
  
![image](https://github.com/xiaoqian19940510/text-classification-surveys/blob/master/figure7.png)  

</p></blockquote></details>



#### 2018

 <details/>
<summary/>
  <a href="https://doi.org/10.18653/v1/d18-1380">Multi-grained attention network for aspect-level sentiment classification</a> --- MGAN主要贡献：多粒度注意力网络，结合粗粒度和细粒度注意力来捕捉aspect和上下文在词级别上的交互；aspect对齐损失来描述拥有共同上下文的aspect之间的aspect级别上的相互影响。 --- (<a href="https://github.com/songyouwei/ABSA-PyTorch">Github</a>)</summary><blockquote><p align="justify">
We propose a novel multi-grained attention network (MGAN) model for aspect level sentiment classification. Existing approaches mostly adopt coarse-grained attention mechanism, which may bring information loss if the aspect has multiple words or larger context. We propose a fine-grained attention mechanism, which can capture the word-level interaction between aspect and context. And then we leverage the fine-grained and coarse-grained attention mechanisms to compose the MGAN framework. Moreover, unlike previous works which train each aspect with its context separately, we design an aspect alignment loss to depict the aspect-level interactions among the aspects that have the same context. We evaluate the proposed approach on three datasets: laptop and restaurant are from SemEval 2014, and the last one is a twitter dataset. Experimental results show that the multi-grained attention network consistently outperforms the state-of-the-art methods on all three datasets. We also conduct experiments to evaluate the effectiveness of aspect alignment loss, which indicates the aspect-level interactions can bring extra useful information and further improve the performance.
  提出了一种新的面向级情绪分类的多粒度关注网络模型。现有的方法多采用粗粒度注意机制，如果方面有多个词或较大的上下文，可能会造成信息丢失。我们提出了一种精细的注意机制，可以捕捉到方面和上下文之间的字级交互。然后我们利用细粒度和粗粒度的注意机制来组成MGAN框架。此外，与之前用上下文分别训练每个方面的工作不同，我们设计了一个方面对齐损失来描述具有相同上下文的方面之间的方面级交互。我们在三个数据集上评估提出的方法:笔记本和餐厅来自2014年SemEval，最后一个数据集是twitter数据集。实验结果表明，在这三个数据集上，多粒度注意力网络的性能始终优于现有的方法。我们还进行了实验来评估方面对齐丢失的有效性，表明方面级交互可以带来额外的有用信息，并进一步提高性能。
</p></blockquote></details>

 <details/>
<summary/>
  <a href="https://doi.org/10.18653/v1/d18-1350">Investigating capsule networks with dynamic routing for text classification</a> --- TextCapsule主要贡献： --- (<a href="https://github.com/andyweizhao/capsule_text_classification">Github</a>)</summary><blockquote><p align="justify">
In this study, we explore capsule networks with dynamic routing for text classification. We propose three strategies to stabilize the dynamic routing process to alleviate the disturbance of some noise capsules which may contain “background” information or have not been successfully trained. A series of experiments are conducted with capsule networks on six text classification benchmarks. Capsule networks achieve state of the art on 4 out of 6 datasets, which shows the effectiveness of capsule networks for text classification. We additionally show that capsule networks exhibit significant improvement when transfer single-label to multi-label text classification over strong baseline methods. To the best of our knowledge, this is the first work that capsule networks have been empirically investigated for text modeling.
  在本研究中，我们探索带有动态路由的胶囊网络用于文本分类。我们提出了三种稳定动态路由过程的策略，以减轻一些可能包含“背景”信息或未成功训练的噪声胶囊的干扰。在六个文本分类基准上用胶囊网络进行了一系列的实验。胶囊网络在6个数据集中的4个数据集上达到了最新的分类水平，显示了胶囊网络在文本分类中的有效性。此外，我们还发现，与强基线方法相比，胶囊网络在将单标签转换为多标签文本分类时表现出了显著的改进。就我们所知，这是第一个工作胶囊网络已被实证研究文本建模。
</p></blockquote></details>

 <details/>
<summary/>
  <a href="https://doi.org/10.24963/ijcai.2018/584">Constructing narrative event evolutionary graph for script event prediction</a> SGNN (<a href="https://github.com/eecrazy/ConstructingNEEG_IJCAI_2018">Github</a>)</summary><blockquote><p align="justify">
Script event prediction requires a model to predict the subsequent event given an existing event context. Previous models based on event pairs or event chains cannot make full use of dense event connections, which may limit their capability of event prediction. To remedy this, we propose constructing an event graph to better utilize the event network information for script event prediction. In particular, we first extract narrative event chains from large quantities of news corpus, and then construct a narrative event evolutionary graph (NEEG) based on the extracted chains. NEEG can be seen as a knowledge base that describes event evolutionary principles and patterns. To solve the inference problem on NEEG, we present a scaled graph neural network (SGNN) to model event interactions and learn better event representations. Instead of computing the representations on the whole graph, SGNN processes only the concerned nodes each time, which makes our model feasible to large-scale graphs. By comparing the similarity between input context event representations and candidate event representations, we can choose the most reasonable subsequent event. Experimental results on widely used New York Times corpus demonstrate that our model significantly outperforms state-of-the-art baseline methods, by using standard multiple choice narrative cloze evaluation.
  脚本事件预测需要一个模型来预测给定现有事件上下文的后续事件。以往基于事件对或事件链的模型不能充分利用密集的事件连接，从而限制了它们对事件的预测能力。为了弥补这一点，我们提出构造一个事件图，以便更好地利用事件网络信息进行脚本事件预测。具体来说，我们首先从大量的新闻语料库中提取叙事事件链，然后基于提取的叙事事件链构造一个叙事事件演化图。NEEG可以看作是一个描述事件进化原理和模式的知识库。为了解决NEEG上的推理问题，我们提出了一种尺度图神经网络(SGNN)来建模事件交互并学习更好的事件表示。SGNN每次只处理相关节点，而不需要计算整个图的表示，这使得我们的模型对大规模图可行。通过比较输入上下文事件表示与候选事件表示的相似性，可以选择最合理的后续事件。在广泛使用的纽约时报语料库上的实验结果表明，通过使用标准的多项选择叙事完形填空评价，我们的模型显著优于目前最先进的基线方法。
</p></blockquote></details>

 <details/>
<summary/>
  <a href="https://www.aclweb.org/anthology/C18-1330/">SGM: sequence generation model for multi-label classification</a> SGM (<a href="https://github.com/lancopku/SGM">Github</a>)</summary><blockquote><p align="justify">
Multi-label classification is an important yet challenging task in natural language processing. It is more complex than single-label classification in that the labels tend to be correlated. Existing methods tend to ignore the correlations between labels. Besides, different parts of the text can contribute differently for predicting different labels, which is not considered by existing models. In this paper, we propose to view the multi-label classification task as a sequence generation problem, and apply a sequence generation model with a novel decoder structure to solve it. Extensive experimental results show that our proposed methods outperform previous work by a substantial margin. Further analysis of experimental results demonstrates that the proposed methods not only capture the correlations between labels, but also select the most informative words automatically when predicting different labels.
  多标签分类是自然语言处理中的一项重要而富有挑战性的任务。它比单标签分类更加复杂，因为标签往往是相关的。现有的方法往往忽略标签之间的相关性。此外，文本的不同部分对不同标签的预测有不同的贡献，现有的模型没有考虑到这一点。本文提出将多标号分类任务看作是一个序列生成问题，并应用序列生成模型和一种新的解码器结构来解决该问题。大量的实验结果表明，我们提出的方法比以前的工作有很大的优势。实验结果表明，该方法不仅能够捕获标签之间的相关性，而且能够在预测不同标签时自动选择信息最丰富的词语。
</p></blockquote></details>


 <details/>
<summary/>
  <a href="https://www.aclweb.org/anthology/P18-1216/">Joint embedding of words and labelsfor text classification</a> LEAM (<a href="https://github.com/guoyinwang/LEAM">Github</a>)</summary><blockquote><p align="justify">
Word embeddings are effective intermediate representations for capturing semantic regularities between words, when learning the representations of text sequences. We propose to view text classification as a label-word joint embedding problem: each label is embedded in the same space with the word vectors. We introduce an attention framework that measures the compatibility of embeddings between text sequences and labels. The attention is learned on a training set of labeled samples to ensure that, given a text sequence, the relevant words are weighted higher than the irrelevant ones. Our method maintains the interpretability of word embeddings, and enjoys a built-in ability to leverage alternative sources of information, in addition to input text sequences. Extensive results on the several large text datasets show that the proposed framework outperforms the state-of-the-art methods by a large margin, in terms of both accuracy and speed.
  在学习文本序列表示时，单词嵌入是捕获单词之间语义规律的有效中间表示。我们提出将文本分类看作是一个标签-单词联合嵌入问题:每个标签被嵌入到与单词向量相同的空间中。我们介绍了一个注意力框架，用来衡量文本序列和标签之间的嵌入的兼容性。注意力是在一组标记样本的训练集上学习的，以确保在给定的文本序列中，相关词的权重高于不相关词。我们的方法维护了单词嵌入的可解释性，并享有利用除了输入文本序列之外的其他信息源的内置能力。在几个大型文本数据集上的广泛结果表明，提出的框架在精度和速度方面都比最先进的方法有很大的优势。
</p></blockquote></details>

 <details/>
<summary/>
  <a href="https://www.aclweb.org/anthology/P18-1031/">Universal language model fine-tuning for text classification</a> ULMFiT (<a href="http://nlp.fast.ai/category/classification.html">Github</a>)</summary><blockquote><p align="justify">
Inductive transfer learning has greatly impacted computer vision, but existing approaches in NLP still require task-specific modifications and training from scratch. We propose Universal Language Model Fine-tuning (ULMFiT), an effective transfer learning method that can be applied to any task in NLP, and introduce techniques that are key for fine-tuning a language model. Our method significantly outperforms the state-of-the-art on six text classification tasks, reducing the error by 18-24% on the majority of datasets. Furthermore, with only 100 labeled examples, it matches the performance of training from scratch on 100 times more data. We open-source our pretrained models and code.
  归纳迁移学习对计算机视觉产生了很大的影响，但现有的神经语言处理方法仍需要对任务进行针对性的修改和从零开始的训练。我们提出了通用语言模型微调(ULMFiT)，一种有效的迁移学习方法，可以应用于NLP中的任何任务，并介绍了微调语言模型的关键技术。我们的方法在六个文本分类任务上显著优于最新的技术，在大多数数据集上减少了18-24%的错误。此外，由于只有100个带标签的例子，它可以在100倍以上的数据上匹配从零开始训练的性能。我们开放了预先训练好的模型和代码。
</p></blockquote></details>

 <details/>
<summary/>
  <a href="https://dl.acm.org/doi/10.1145/3178876.3186005">Large-scale hierarchical text classification withrecursively regularized deep graph-cnn</a> DGCNN (<a href="https://github.com/HKUST-KnowComp/DeepGraphCNNforTexts">Github</a>)</summary><blockquote><p align="justify">
Text classification to a hierarchical taxonomy of topics is a common and practical problem. Traditional approaches simply use bag-of-words and have achieved good results. However, when there are a lot of labels with different topical granularities, bag-of-words representation may not be enough. Deep learning models have been proven to be effective to automatically learn different levels of representations for image data. It is interesting to study what is the best way to represent texts. In this paper, we propose a graph-CNN based deep learning model to first convert texts to graph-of-words, and then use graph convolution operations to convolve the word graph. Graph-of-words representation of texts has the advantage of capturing non-consecutive and long-distance semantics. CNN models have the advantage of learning different level of semantics. To further leverage the hierarchy of labels, we regularize the deep architecture with the dependency among labels. Our results on both RCV1 and NYTimes datasets show that we can significantly improve large-scale hierarchical text classification over traditional hierarchical text classification and existing deep models.
  文本分类对主题进行分层分类是一个常见而实用的问题。传统的方法只是简单地使用词汇袋，取得了良好的效果。然而，当有许多带有不同主题粒度的标签时，词汇袋表示可能不够。深度学习模型已经被证明能够有效地自动学习图像数据的不同层次的表示。研究什么是表现文本的最佳方式是很有趣的。在本文中，我们提出了一个基于graph- cnn的深度学习模型，首先将文本转换为图形的单词，然后使用图形卷积操作对单词图形进行卷积。文本的文字图表示具有捕获非连续和长距离语义的优点。CNN模型具有学习不同层次语义的优势。为了进一步利用标签的层次结构，我们使用标签之间的依赖关系来规范深层架构。我们在RCV1和NYTimes数据集上的结果表明，与传统的分层文本分类和现有的深度模型相比，我们可以显著改进大规模分层文本分类。
</p></blockquote></details>


 <details/>
<summary/>
  <a href="https://doi.org/10.18653/v1/n18-1202">Deep contextualized word rep-resentations</a> ELMo (<a href="https://github.com/flairNLP/flair">Github</a>)</summary><blockquote><p align="justify">
We introduce a new type of deep contextualized word representation that models both (1) complex characteristics of word use (e.g., syntax and semantics), and (2) how these uses vary across linguistic contexts (i.e., to model polysemy). Our word vectors are learned functions of the internal states of a deep bidirectional language model (biLM), which is pre-trained on a large text corpus. We show that these representations can be easily added to existing models and significantly improve the state of the art across six challenging NLP problems, including question answering, textual entailment and sentiment analysis. We also present an analysis showing that exposing the deep internals of the pre-trained network is crucial, allowing downstream models to mix different types of semi-supervision signals.
  我们引入了一种新型的深层语境化词汇表示，它既模拟了(1)复杂的词汇使用特征(例如，句法和语义)，也模拟了(2)这些用法如何在不同的语言语境中变化(例如，模拟多义词)。我们的词向量是深度双向语言模型(biLM)内部状态的学习函数，该模型是在大型文本语料库上预先训练的。我们表明，这些表示可以很容易地添加到现有的模型中，并在六个具有挑战性的NLP问题(包括问题回答、文本蕴涵和情绪分析)中显著提高技术水平。我们还提供了一项分析，显示出预先训练过的网络的深层内在是至关重要的，允许下游模型混合不同类型的半监督信号。
</p></blockquote></details>


#### 2017
 <details/>
<summary/>
  <a href="https://www.aclweb.org/anthology/D17-1047/">Recurrent Attention Network on Memory for Aspect Sentiment Analysis</a> RAM (<a href="https://github.com/songyouwei/ABSA-PyTorch">Github</a>)</summary><blockquote><p align="justify">
We propose a novel framework based on neural networks to identify the sentiment of opinion targets in a comment/review. Our framework adopts multiple-attention mechanism to capture sentiment features separated by a long distance, so that it is more robust against irrelevant information. The results of multiple attentions are non-linearly combined with a recurrent neural network, which strengthens the expressive power of our model for handling more complications. The weighted-memory mechanism not only helps us avoid the labor-intensive feature engineering work, but also provides a tailor-made memory for different opinion targets of a sentence. We examine the merit of our model on four datasets: two are from SemEval2014, i.e. reviews of restaurants and laptops; a twitter dataset, for testing its performance on social media data; and a Chinese news comment dataset, for testing its language sensitivity. The experimental results show that our model consistently outperforms the state-of-the-art methods on different types of data.
  我们提出了一个基于神经网络的新框架来识别评论/评论中意见目标的情绪。我们的框架采用了多注意机制来捕获远距离分离的情绪特征，从而对无关信息具有更强的鲁棒性。多重关注的结果与递归神经网络进行非线性结合，这增强了我们的模型处理更多并发症的表达能力。加权记忆机制不仅帮助我们避免了劳动密集型的特征工程工作，而且为句子的不同观点目标提供了量身定制的记忆。我们在四个数据集上检验了我们的模型的优点:两个数据集来自2014年上半年，即对餐馆和笔记本电脑的评论;一个twitter数据集，用于测试其在社交媒体数据上的表现;以及一个中文新闻评论数据集，用于测试其语言敏感性。实验结果表明，我们的模型在不同类型的数据上始终优于最新的方法。
</p></blockquote></details>

 <details/>
<summary/>
  <a href="https://doi.org/10.18653/v1/d17-1169">Using millions of emoji occurrences to learn any-domain representations for detecting sentiment, emotion and sarcasm</a> DeepMoji (<a href="https://github.com/bfelbo/DeepMoji">Github</a>)</summary><blockquote><p align="justify">
NLP tasks are often limited by scarcity of manually annotated data. In social media sentiment analysis and related tasks, researchers have therefore used binarized emoticons and specific hashtags as forms of distant supervision. Our paper shows that by extending the distant supervision to a more diverse set of noisy labels, the models can learn richer representations. Through emoji prediction on a dataset of 1246 million tweets containing one of 64 common emojis we obtain state-of-the-art performance on 8 benchmark datasets within emotion, sentiment and sarcasm detection using a single pretrained model. Our analyses confirm that the diversity of our emotional labels yield a performance improvement over previous distant supervision approaches.
  NLP任务常常受到手工注释数据稀缺的限制。因此，在社交媒体情绪分析和相关任务中，研究人员使用了二值化的表情符号和特定的标签作为远程监督的形式。我们的论文表明，通过将远程监控扩展到更多样化的噪声标签集合，模型可以学习更丰富的表示。通过对包含64个常见表情符号之一的1.46亿条推文数据集的表情符号预测，我们通过使用单一的预训练模型在8个基准数据集上获得了最先进的情绪、情绪和讽刺检测性能。我们的分析证实，情感标签的多样性比以前的远程监督方法产生了表现改善。
</p></blockquote></details>

 <details/>
<summary/>
  <a href="https://www.ijcai.org/Proceedings/2017/568">Interactive attention networks for aspect-level sentiment classification</a> IAN (<a href="https://github.com/songyouwei/ABSA-PyTorch">Github</a>)</summary><blockquote><p align="justify">
Aspect-level sentiment classification aims at identifying the sentiment polarity of specific target in its context. Previous approaches have realized the importance of targets in sentiment classification and developed various methods with the goal of precisely modeling thier contexts via generating target-specific representations. However, these studies always ignore the separate modeling of targets. In this paper, we argue that both targets and contexts deserve special treatment and need to be learned their own representations via interactive learning. Then, we propose the interactive attention networks (IAN) to interactively learn attentions in the contexts and targets, and generate the representations for targets and contexts separately. With this design, the IAN model can well represent a target and its collocative context, which is helpful to sentiment classification. Experimental results on SemEval 2014 Datasets demonstrate the effectiveness of our model.
  方面级情绪分类旨在识别特定对象在其语境中的情绪极性。以往的方法已经认识到目标在情感分类中的重要性，并发展了各种方法，目的是通过生成特定于目标的表示来精确地建模它们的上下文。然而，这些研究往往忽略了目标的单独建模。在本文中，我们认为目标和语境都需要特殊对待，需要通过交互学习来学习它们各自的表征。在此基础上，我们提出了交互式注意网络(IAN)来交互地学习上下文和目标中的注意，并分别生成目标和上下文的表示。通过这样的设计，IAN模型可以很好地表示目标及其搭配语境，有利于情感分类。在SemEval 2014数据集上的实验结果证明了我们的模型的有效性。
</p></blockquote></details>

 <details/>
<summary/>
  <a href="https://doi.org/10.18653/v1/P17-1052">Deep pyramid convolutional neural networks for text categorization</a> DPCNN (<a href="https://github.com/Cheneng/DPCNN">Github</a>)</summary><blockquote><p align="justify">
This paper proposes a low-complexity word-level deep convolutional neural network (CNN) architecture for text categorization that can efficiently represent long-range associations in text. In the literature, several deep and complex neural networks have been proposed for this task, assuming availability of relatively large amounts of training data. However, the associated computational complexity increases as the networks go deeper, which poses serious challenges in practical applications. Moreover, it was shown recently that shallow word-level CNNs are more accurate and much faster than the state-of-the-art very deep nets such as character-level CNNs even in the setting of large training data. Motivated by these findings, we carefully studied deepening of word-level CNNs to capture global representations of text, and found a simple network architecture with which the best accuracy can be obtained by increasing the network depth without increasing computational cost by much. We call it deep pyramid CNN. The proposed model with 15 weight layers outperforms the previous best models on six benchmark datasets for sentiment classification and topic categorization.
  本文提出了一种低复杂度的词级深度卷积神经网络(CNN)用于文本分类的架构，该架构可以有效地表示文本中的远程关联。在文献中，已经提出了几种深度和复杂的神经网络来完成这个任务，假设有相对大量的训练数据可用。然而，随着网络的深入，相关的计算复杂度增加，这在实际应用中提出了严重的挑战。此外，最近的研究表明，即使在设置大的训练数据时，浅词级的cnn网络也比目前最先进的深度网络(如字符级的cnn网络)更准确、更快。基于这些发现，我们仔细研究了单词级CNNs的加深以捕获文本的全局表示，并发现了一种简单的网络架构，通过增加网络深度，在不增加太多计算成本的情况下，可以获得最佳的精确度。我们称之为深度金字塔CNN。在情感分类和主题分类的六个基准数据集上，该模型在15个权重层上的表现优于以往的最佳模型。
</p></blockquote></details>


 <details/>
<summary/>
  <a href="https://openreview.net/forum?id=rJbbOLcex">Topicrnn: A recurrent neural network with long-range semantic dependency</a> TopicRNN (<a href="https://github.com/dangitstam/topic-rnn">Github</a>)</summary><blockquote><p align="justify">
In this paper, we propose TopicRNN, a recurrent neural network (RNN)-based language model designed to directly capture the global semantic meaning relating words in a document via latent topics. Because of their sequential nature, RNNs are good at capturing the local structure of a word sequence – both semantic and syntactic – but might face difficulty remembering long-range dependencies. Intuitively, these long-range dependencies are of semantic nature. In contrast, latent topic models are able to capture the global underlying semantic structure of a document but do not account for word ordering. The proposed TopicRNN model integrates the merits of RNNs and latent topic models: it captures local (syntactic) dependencies using an RNN and global (semantic) dependencies using latent topics. Unlike previous work on contextual RNN language modeling, our model is learned end-to-end. Empirical results on word prediction show that TopicRNN outperforms existing contextual RNN baselines. In addition, TopicRNN can be used as an unsupervised feature extractor for documents. We do this for sentiment analysis on the IMDB movie review dataset and report an error rate of 6.28%. This is comparable to the state-of-the-art 5.91% resulting from a semi-supervised approach. Finally, TopicRNN also yields sensible topics, making it a useful alternative to document models such as latent Dirichlet allocation.
  本文提出一种基于递归神经网络(RNN)的语言模型TopicRNN，旨在通过潜在主题直接捕获文档中相关词的全局语义意义。由于它们的顺序性质，rnn善于捕捉单词序列的局部结构——包括语义和句法——但可能难以记住长期依赖关系。直观地说，这些长期依赖具有语义性质。相反，潜在主题模型能够捕获文档的全局底层语义结构，但不考虑单词排序。该模型集成了网络神经网络和潜在主题模型的优点:利用网络神经网络捕获局部(句法)依赖，利用潜在主题捕获全局(语义)依赖。与之前的上下文RNN语言建模不同，我们的模型是端到端学习的。在词汇预测方面的经验结果表明，TopicRNN的性能优于已有的上下文RNN基线。此外，TopicRNN还可以用作文档的无监督特性提取器。我们这样做是为了对IMDB电影评论数据集进行情感分析，并报告错误率为6.28%。这可与半监督方法所产生的5.91%的最先进水平相媲美。最后，TopicRNN还生成合理的主题，这使得它成为文档模型(如潜在的Dirichlet分配)的有用替代品。
</p></blockquote></details>


 <details/>
<summary/>
  <a href="https://openreview.net/forum?id=r1X3g2_xl">Adversarial training methods for semi-supervised text classification</a> Miyato et al. (<a href="https://github.com/tensorflow/models/tree/master/adversarial_text">Github</a>)</summary><blockquote><p align="justify">
Adversarial training provides a means of regularizing supervised learning algorithms while virtual adversarial training is able to extend supervised learning algorithms to the semi-supervised setting. However, both methods require making small perturbations to numerous entries of the input vector, which is inappropriate for sparse high-dimensional inputs such as one-hot word representations. We extend adversarial and virtual adversarial training to the text domain by applying perturbations to the word embeddings in a recurrent neural network rather than to the original input itself. The proposed method achieves state of the art results on multiple benchmark semi-supervised and purely supervised tasks. We provide visualizations and analysis showing that the learned word embeddings have improved in quality and that while training, the model is less prone to overfitting.
  对抗式训练提供了一种规范监督学习算法的方法，而虚拟对抗式训练则能够将监督学习算法扩展到半监督环境中。然而，这两种方法都需要对输入向量的多个条目进行小的扰动，这对于稀疏高维输入(如单热字表示)是不合适的。我们将对敌训练和虚拟对敌训练扩展到文本领域，方法是对递归神经网络中的嵌入词施加扰动，而不是对原始输入本身施加扰动。该方法在多基准测试、半监督和纯监督任务上都取得了较好的效果。我们提供的可视化和分析显示，学习到的单词嵌入在质量上有了提高，而且在训练时，模型不太容易出现过拟合。
</p></blockquote></details>


 <details/>
<summary/>
  <a href="https://doi.org/10.18653/v1/e17-2068">Bag of tricks for efficient text classification</a> FastText (<a href="https://github.com/SeanLee97/short-text-classification">Github</a>)</summary><blockquote><p align="justify">
This paper explores a simple and efficient baseline for text classification. Our experiments show that our fast text classifier fastText is often on par with deep learning classifiers in terms of accuracy, and many orders of magnitude faster for training and evaluation. We can train fastText on more than one billion words in less than ten minutes using a standard multicore CPU, and classify half a million sentences among 312K classes in less than a minute.
  本文探讨了一种简单有效的文本分类基准。我们的实验表明，我们的快速文本分类器在准确率方面与深度学习分类器相当，在训练和评估方面比深度学习分类器快很多个数量级。使用标准的多核CPU，我们可以在不到10分钟的时间内对fastText进行10亿个单词的训练，并在不到一分钟的时间内对312K个类中的50万个句子进行分类。
</p></blockquote></details>



#### 2016

 <details/>
<summary/>
  <a href="https://doi.org/10.18653/v1/d16-1053">Long short-term memory-networks for machine reading</a> LSTMN (<a href="https://github.com/JRC1995/Abstractive-Summarization">Github</a>)</summary><blockquote><p align="justify">
In this paper we address the question of how to render sequence-level networks better at handling structured input. We propose a machine reading simulator which processes text incrementally from left to right and performs shallow reasoning with memory and attention. The reader extends the Long Short-Term Memory architecture with a memory network in place of a single memory cell. This enables adaptive memory usage during recurrence with neural attention, offering a way to weakly induce relations among tokens. The system is initially designed to process a single sequence but we also demonstrate how to integrate it with an encoder-decoder architecture. Experiments on language modeling, sentiment analysis, and natural language inference show that our model matches or outperforms the state of the art.
  在本文中，我们讨论了如何渲染序列级网络来更好地处理结构化输入的问题。我们提出一种机器阅读模拟器，它可以从左向右逐步处理文本，并通过记忆和注意力进行浅层推理。读者扩展长短期记忆架构与记忆网络在一个单一的记忆细胞。这使得神经关注在递归期间能够自适应记忆使用，提供了一种方法来弱地诱导标记之间的关系。该系统最初设计用于处理单个序列，但我们也演示了如何将其与编码器-解码器体系结构集成。在语言建模、情感分析和自然语言推理方面的实验表明，我们的模型符合或优于当前的技术水平。
</p></blockquote></details>

 <details/>
<summary/>
  <a href="https://www.ijcai.org/Abstract/16/408">Recurrent neural network for text classification with multi-task learning</a> Multi-Task (<a href="https://github.com/baixl/text_classification">Github</a>)</summary><blockquote><p align="justify">
Neural network based methods have obtained great progress on a variety of natural language processing tasks. However, in most previous works, the models are learned based on single-task supervised objectives, which often suffer from insufficient training data. In this paper, we use the multi-task learning framework to jointly learn across multiple related tasks. Based on recurrent neural network, we propose three different mechanisms of sharing information to model text with task-specific and shared layers. The entire network is trained jointly on all these tasks. Experiments on four benchmark text classification tasks show that our proposed models can improve the performance of a task with the help of other related tasks.
  基于神经网络的方法在各种自然语言处理任务中取得了很大进展。但是，在以往的工作中，大多数模型都是基于单任务监督目标学习的，这往往存在训练数据不足的问题。在本文中，我们使用多任务学习框架来共同学习多个相关的任务。基于递归神经网络，我们提出了三种不同的信息共享机制来对具有任务特定层和共享层的文本进行建模。整个网络共同接受所有这些任务的培训。在四个基准文本分类任务上的实验表明，我们所提出的模型可以在其他相关任务的帮助下提高任务的性能。
</p></blockquote></details>

 <details/>
<summary/>
  <a href="https://doi.org/10.18653/v1/n16-1174">Hierarchical attention networks for document classification</a> HAN(<a href="https://github.com/richliao/textClassifier">Github</a>)</summary><blockquote><p align="justify">
We propose a hierarchical attention networkfor document classification.  Our model hastwo distinctive characteristics: (i) it has a hier-archical structure that mirrors the hierarchicalstructure of documents; (ii) it has two levelsof attention mechanisms applied at the word-and sentence-level, enabling it to attend dif-ferentially to more and less important con-tent when constructing the document repre-sentation. Experiments conducted on six largescale text classification tasks demonstrate thatthe proposed architecture outperform previousmethods by a substantial margin. Visualiza-tion of the attention layers illustrates that themodel selects qualitatively informative wordsand sentences.
  提出了一种用于文档分类的分层关注网络。我们的模型有两个显著的特点:(1)它有一个更高级的结构，反映了文件的层次结构;(二)在词语和句子层面上有两种注意机制，使其在构建文件表达时，对重要的内容和次要的内容有不同程度的注意。在六个大规模文本分类任务上的实验表明，所提出的架构大大优于以前的方法。注意层的视觉化说明了该模式选择有定性信息的词和句子。
</p></blockquote></details>


#### 2015

 <details/>
<summary/>
  <a href="http://papers.nips.cc/paper/5782-character-level-convolutional-networks-for-text-classification">Character-level convolutional networks for text classification</a> CharCNN (<a href="https://github.com/mhjabreel/CharCNN">Github</a>)</summary><blockquote><p align="justify">
This article offers an empirical exploration on the use of character-level convolutional networks (ConvNets) for text classification. We constructed several large-scale datasets to show that character-level convolutional networks could achieve state-of-the-art or competitive results. Comparisons are offered against traditional models such as bag of words, n-grams and their TFIDF variants, and deep learning models such as word-based ConvNets and recurrent neural networks.
  本文对使用字符级卷积网络(ConvNets)进行文本分类提供了一个经验探索。我们构建了几个大规模数据集来表明字符级卷积网络可以获得最先进的或有竞争力的结果。与传统的模型(如单词包、n-grams及其TFIDF变体)和深度学习模型(如基于单词的ConvNets和递归神经网络)进行比较。
</p></blockquote></details>

 <details/>
<summary/>
  <a href="https://doi.org/10.3115/v1/p15-1150">Improved semantic representations from tree-structured long short-term memory networks</a> Tree-LSTM (<a href="https://github.com/stanfordnlp/treelstm">Github</a>)</summary><blockquote><p align="justify">
Because  of  their  superior  ability  to  pre-serve   sequence   information   over   time,Long  Short-Term  Memory  (LSTM)  net-works,   a  type  of  recurrent  neural  net-work with a more complex computationalunit, have obtained strong results on a va-riety  of  sequence  modeling  tasks.Theonly underlying LSTM structure that hasbeen  explored  so  far  is  a  linear  chain.However,  natural  language  exhibits  syn-tactic properties that would naturally com-bine words to phrases.  We introduce theTree-LSTM, a generalization of LSTMs totree-structured network topologies.  Tree-LSTMs  outperform  all  existing  systemsand strong LSTM baselines on two tasks:predicting the semantic relatedness of twosentences  (SemEval  2014,  Task  1)  andsentiment  classification  (Stanford  Senti-ment Treebank).
  长短期记忆(LSTM)网络是一种具有更复杂计算单元的循环神经网络，由于它们具有随时间预服务序列信息的卓越能力，因此在一系列序列建模任务中获得了强大的结果。到目前为止，我们研究的唯一底层LSTM结构是一个线性链。然而，自然语言表现出的句法策略特性会自然地将单词组合成短语。我们介绍了树- lstm，它是LSTMs树结构网络拓扑的一般化。在两项任务上，Tree-LSTMs的表现胜过所有现有系统，而且在两项任务上都有较强的LSTM基线:预测两个句子的语义相关性(SemEval 2014, Task 1)和情绪分类(Stanford Senti-ment Treebank)。
</p></blockquote></details>


 <details/>
<summary/>
  <a href="https://doi.org/10.3115/v1/p15-1162">Deep unordered composition rivals syntactic methods for text classification</a> DAN(<a href="https://github.com/miyyer/dan">Github</a>)</summary><blockquote><p align="justify">
Many  existing  deep  learning  models  fornatural language processing tasks focus onlearning thecompositionalityof their in-puts, which requires many expensive com-putations. We present a simple deep neuralnetwork that competes with and, in somecases,  outperforms  such  models  on  sen-timent  analysis  and  factoid  question  an-swering tasks while taking only a fractionof the training time.  While our model issyntactically-ignorant, we show significantimprovements over previous bag-of-wordsmodels by deepening our network and ap-plying a novel variant of dropout.  More-over, our model performs better than syn-tactic models on datasets with high syn-tactic variance.  We show that our modelmakes similar errors to syntactically-awaremodels, indicating that for the tasks we con-sider, nonlinearly transforming the input ismore important than tailoring a network toincorporate word order and syntax.
  许多现有的用于自然语言处理任务的深度学习模型都关注于学习输入的组合性，这需要许多昂贵的组合。我们提出了一种简单的深层神经网络，它在感知分析和模拟问题转换任务上与这些模型竞争，并且在某些方面优于这些模型，而只需要训练时间的一小部分。虽然我们的模型在句法上是无知的，但我们通过深化我们的网络和使用dropout的新变体，显示了对以前的词汇包模型的重大改进。此外，我们的模型在具有高同步策略方差的数据集上的表现优于同步策略模型。我们发现我们的模型与句法认知模型犯了类似的错误，这表明对于我们所考虑的任务来说，非线性地转换输入比裁剪一个网络以包含词序和语法更重要。
</p></blockquote></details>


 <details/>
<summary/>
  <a href="http://www.aaai.org/ocs/index.php/AAAI/AAAI15/paper/view/9745">Recurrent convolutional neural networks for text classification</a> TextRCNN(<a href="https://github.com/roomylee/rcnn-text-classification">Github</a>)</summary><blockquote><p align="justify">
Text classification is a foundational task in many NLP applications. Traditional text classifiers often rely on many human-designed features, such as dictionaries, knowledge bases and special tree kernels. In contrast to traditional methods, we introduce a recurrent convolutional neural network for text classification without human-designed features. In our model, we apply a recurrent structure to capture contextual information as far as possible when learning word representations, which may introduce considerably less noise compared to traditional window-based neural networks. We also employ a max-pooling layer that automatically judges which words play key roles in text classification to capture the key components in texts. We conduct experiments on four commonly used datasets. The experimental results show that the proposed method outperforms the state-of-the-art methods on several datasets, particularly on document-level datasets.
  文本分类是许多自然语言处理应用中的一项基本任务。传统的文本分类器通常依赖于许多人为设计的特性，如字典、知识库和特殊的树核。与传统方法相比，我们引入了一种循环卷积神经网络来进行文本分类，而不需要人为设计特征。在我们的模型中，在学习单词表示时，我们使用递归结构来尽可能捕获上下文信息，与传统的基于窗口的神经网络相比，这可以引入相当少的噪音。我们还使用了一个最大池层，自动判断哪些词在文本分类中扮演关键角色，以捕获文本中的关键组件。我们在四个常用数据集上进行实验。实验结果表明，该方法在多个数据集，特别是文档级数据集上的性能优于现有的方法。
</p></blockquote></details>



#### 2014
 <details/>
<summary/>
  <a href="http://proceedings.mlr.press/v32/le14.html">Distributed representations of sentences and documents</a> Paragraph-Vec (<a href="https://github.com/inejc/paragraph-vectors">Github</a>)</summary><blockquote><p align="justify">
Many machine learning algorithms require the input to be represented as a fixed length feature vector. When it comes to texts, one of the most common representations is bag-of-words. Despite their popularity, bag-of-words models have two major weaknesses: they lose the ordering of the words and they also ignore semantics of the words. For example, "powerful," "strong" and "Paris" are equally distant. In this paper, we propose an unsupervised algorithm that learns vector representations of sentences and text documents. This algorithm represents each document by a dense vector which is trained to predict words in the document. Its construction gives our algorithm the potential to overcome the weaknesses of bag-of-words models. Empirical results show that our technique outperforms bag-of-words models as well as other techniques for text representations. Finally, we achieve new state-of-the-art results on several text classification and sentiment analysis tasks.
  许多机器学习算法要求输入以固定长度的特征向量表示。说到文本，最常见的一种表现形式是词汇袋。尽管词包模型很流行，但它们有两个主要的弱点:它们失去了单词的顺序，而且它们还忽略了单词的语义。例如，“powerful”、“strong”和“Paris”的距离一样远。在本文中，我们提出了一种学习句子和文本文档的向量表示的无监督算法。该算法通过密集向量来表示每个文档，密集向量被训练用来预测文档中的单词。它的构造使我们的算法有可能克服单词包模型的缺点。实验结果表明，我们的技术在文本表示方面优于词汇袋模型和其他技术。最后，我们在几个文本分类和情绪分析任务上取得了最新的结果。
</p></blockquote></details>

 <details/>
<summary/>
  <a href="https://doi.org/10.3115/v1/p14-1062">A convolutional neural network for modelling sentences</a> DCNN (<a href="https://github.com/kinimod23/ATS_Project">Github</a>)</summary><blockquote><p align="justify">
The ability to accurately represent sentences is central to language understanding. We describe a convolutional architecture dubbed the Dynamic Convolutional Neural Network (DCNN) that we adopt for the semantic modelling of sentences. The network uses Dynamic k-Max Pooling, a global pooling operation over linear sequences. The network handles input sentences of varying length and induces a feature graph over the sentence that is capable of explicitly capturing short and long-range relations. The network does not rely on a parse tree and is easily applicable to any language. We test the DCNN in four experiments: small scale binary and multi-class sentiment prediction, six-way question classification and Twitter sentiment prediction by distant supervision. The network achieves excellent performance in the first three tasks and a greater than 25% error reduction in the last task with respect to the strongest baseline.
  准确表达句子的能力是语言理解的核心。我们描述了一种称为动态卷积神经网络(DCNN)的卷积架构，我们将其用于句子的语义建模。该网络使用动态k-Max池，一种线性序列上的全局池操作。该网络处理不同长度的输入句子，并在句子上归纳出一个特征图，能够明确地捕捉短关系和长关系。该网络不依赖于解析树，并且很容易适用于任何语言。我们通过四组实验对DCNN进行了测试:小尺度二值多类情绪预测、六向问题分类和推特远程监控情绪预测。该网络在前三个任务中都取得了优异的性能，在最后一个任务中相对于最强基线的误差减少了25%以上。
</p></blockquote></details>

 <details/>
<summary/>
  <a href="https://www.aclweb.org/anthology/D14-1181.pdf">Convolutional Neural Networks for Sentence Classification</a> TextCNN (<a href="https://github.com/alexander-rakhlin/CNN-for-Sentence-Classification-in-Keras">Github</a>)</summary><blockquote><p align="justify">
We report on a series of experiments with convolutional neural networks (CNN) trained on top of pre-trained word vectors for sentence-level classification tasks. We show that a simple CNN with little hyperparameter tuning and static vectors achieves excellent results on multiple benchmarks. Learning task-specific vectors through fine-tuning offers further gains in performance. We additionally propose a simple modification to the architecture to allow for the use of both task-specific and static vectors. The CNN models discussed herein improve upon the state of the art on 4 out of 7 tasks, which include sentiment analysis and question classification.
  本文报告了一系列卷积神经网络(CNN)的实验，这些网络是在预先训练好的单词向量上训练的，用于句子级别的分类任务。我们证明了一个简单的CNN具有小的超参数调优和静态向量在多个基准测试中取得了很好的结果。通过微调学习特定任务向量可以进一步提高性能。此外，我们还建议对架构进行简单修改，以允许同时使用特定于任务的和静态向量。本文讨论的CNN模型在7项任务中的4项上改进了现有的技术，其中包括情绪分析和问题分类。
</p></blockquote></details>

#### 2013
 <details/>
<summary/>
  <a href="https://www.aclweb.org/anthology/D13-1170/">Recursive deep models for semantic compositionality over a sentiment treebank</a> RNTN (<a href=" https://github.com/pondruska/DeepSentiment">Github</a>)</summary><blockquote><p align="justify">
Semantic word spaces have been very useful but cannot express the meaning of longer phrases in a principled way. Further progress towards understanding compositionality in tasks such as sentiment detection requires richer supervised training and evaluation resources and more powerful models of composition. To remedy this, we introduce a Sentiment Treebank. It includes fine grained sentiment labels for 215,154 phrases in the parse trees of 11,855 sentences and presents new challenges for sentiment composition-ality. To address them, we introduce the Recursive Neural Tensor Network. When trained on the new treebank, this model outperforms all previous methods on several metrics. It pushes the state of the art in single sentence positive/negative classification from 80% up to 85.4%. The accuracy of predicting fine-grained sentiment labels for all phrases reaches 80.7%, an improvement of 9.7% over bag of features baselines. Lastly, it is the only model that can accurately capture the effects of negation and its scope at various tree levels for both positive and negative phrases.
  语义词空间已经非常有用，但不能以一种原则的方式表达较长的短语的意义。在情感检测等任务中进一步理解作文性，需要更丰富的监督训练和评估资源以及更强大的作文模型。为了解决这个问题，我们引入了一个情绪树银行。它为11,855个句子的解析树中的215,154个短语包含了细粒度的情绪标签，并对情绪的组合提出了新的挑战。为了解决这个问题，我们引入递归神经张量网络。当在新的treebank上进行训练时，这个模型在几个指标上都优于之前所有的方法。它将单句肯定/否定分类的技术水平从80%提升到85.4%。预测所有短语的精细情绪标签的准确率达到80.7%，比包特征基线提高了9.7%。最后，该模型是唯一能够准确地捕捉否定语句在不同树级上的影响及其范围的模型。
</p></blockquote></details>


#### 2012
 <details/>
<summary/>
  <a href="https://www.aclweb.org/anthology/D12-1110/">Semantic compositionality through recursive matrix-vector spaces</a> MV-RNN (<a href="https://github.com/github-pengge/MV_RNN">Github</a>)</summary><blockquote><p align="justify">
Single-word vector space models have been very successful at learning lexical information. However, they cannot capture the compositional meaning of longer phrases, preventing them from a deeper understanding of language. We introduce a recursive neural network (RNN) model that learns compositional vector representations for phrases and sentences of arbitrary syntactic type and length. Our model assigns a vector and a matrix to every node in a parse tree: the vector captures the inherent meaning of the constituent, while the matrix captures how it changes the meaning of neighboring words or phrases. This matrix-vector RNN can learn the meaning of operators in propositional logic and natural language. The model obtains state of the art performance on three different experiments: predicting fine-grained sentiment distributions of adverb-adjective pairs; classifying sentiment labels of movie reviews and classifying semantic relationships such as cause-effect or topic-message between nouns using the syntactic path between them.
  单词向量空间模型在学习词汇信息方面非常成功。然而，他们不能捕捉长短语的组成意义，阻碍了他们对语言的更深的理解。我们介绍了一个递归神经网络(RNN)模型，学习组合向量表示的短语和句子的任意句法类型和长度。我们的模型为解析树中的每个节点分配一个向量和一个矩阵:向量捕获组成部分的内在含义，而矩阵捕获它如何改变邻近单词或短语的含义。该矩阵向量神经网络可以学习算子在命题逻辑和自然语言中的意义。该模型通过三个不同的实验得到艺术表现的状态:预测副词-形容词对的精细情绪分布;对电影评论的情感标签进行分类，利用名词之间的句法路径对名词之间的因果关系、主题信息等语义关系进行分类。
</p></blockquote></details>

#### 2011
 <details/>
<summary/>
  <a href="https://www.aclweb.org/anthology/D14-1181.pdf">Semi-supervised recursive autoencoders forpredicting sentiment distributions</a> RAE (<a href="https://github.com/alexander-rakhlin/CNN-for-Sentence-Classification-in-Keras">Github</a>)</summary><blockquote><p align="justify">
We introduce a novel machine learning frame-work based on recursive autoencoders for sentence-level prediction of sentiment labeldistributions. Our method learns vector spacerepresentations for multi-word phrases. In sentiment prediction tasks these represen-tations outperform other state-of-the-art ap-proaches on commonly used datasets, such asmovie reviews, without using any pre-definedsentiment lexica or polarity shifting rules. Wealso  evaluate  the  model’s  ability to predict sentiment distributions on a new dataset basedon confessions from the experience project. The dataset consists of personal user storiesannotated with multiple labels which, whenaggregated, form a multinomial distributionthat captures emotional reactions. Our algorithm can more accurately predict distri-butions over such labels compared to severalcompetitive baselines.
  提出了一种基于递归自动编码器的句子级情绪标签分布预测机器学习框架。我们的方法学习多词短语的向量空间扫描。在情绪预测任务中，这些代表在不使用任何预定义的情绪词汇或极性转换规则的情况下，在常用数据集(如电影评论)上胜过其他先进的应用程序。我们还评估了该模型预测情绪分布的能力，基于经验项目的告白。数据集由带有多个标签的个人用户故事组成，当聚合时，这些故事就形成了一个多项分布，可以捕捉人们的情绪反应。与几个有竞争力的基线相比，我们的算法可以更准确地预测这些标签的销量。
</p></blockquote></details>




## Shallow Learning Models(浅层学习模型)
[:arrow_up:](#table-of-contents)

#### 2017
 <details/>
<summary/>
  <a href="http://papers.nips.cc/paper/6907-lightgbm-a-highly-efficient-gradient-boosting-decision-tree">Lightgbm: A highly efficient gradient boosting decision tree</a> LightGBM (<a href="https://github.com/creatist/text_classify">Github</a>)</summary><blockquote><p align="justify">
Gradient Boosting Decision Tree (GBDT) is a popular machine learning algorithm, and has quite a few effective implementations such as XGBoost and pGBRT. Although many engineering optimizations have been adopted in these implementations, the efficiency and scalability are still unsatisfactory when the feature dimension is high and data size is large. A major reason is that for each feature, they need to scan all the data instances to estimate the information gain of all possible split points, which is very time consuming. To tackle this problem, we propose two novel techniques: \emph{Gradient-based One-Side Sampling} (GOSS) and \emph{Exclusive Feature Bundling} (EFB). With GOSS, we exclude a significant proportion of data instances with small gradients, and only use the rest to estimate the information gain. We prove that, since the data instances with larger gradients play a more important role in the computation of information gain, GOSS can obtain quite accurate estimation of the information gain with a much smaller data size. With EFB, we bundle mutually exclusive features (i.e., they rarely take nonzero values simultaneously), to reduce the number of features. We prove that finding the optimal bundling of exclusive features is NP-hard, but a greedy algorithm can achieve quite good approximation ratio (and thus can effectively reduce the number of features without hurting the accuracy of split point determination by much). We call our new GBDT implementation with GOSS and EFB \emph{LightGBM}. Our experiments on multiple public datasets show that, LightGBM speeds up the training process of conventional GBDT by up to over 20 times while achieving almost the same accuracy.
  梯度增强决策树(Gradient Boosting Decision Tree, GBDT)是一种流行的机器学习算法，它有很多有效的实现，如XGBoost和pGBRT。虽然在这些实现中采用了很多工程优化，但在特征维数高、数据量大的情况下，效率和可伸缩性仍然不能令人满意。一个主要原因是对于每个特征，他们需要扫描所有的数据实例来估计所有可能的分割点的信息增益，这是非常耗时的。为了解决这一问题，我们提出了两种新技术:基于梯度的单边采样}(GOSS)和专属特性绑定}(EFB)。使用GOSS，我们排除了具有小梯度的大量数据实例，只使用其余部分来估计信息增益。我们证明了，由于梯度较大的数据实例在信息增益的计算中起着更重要的作用，高斯函数可以在数据量小得多的情况下获得相当准确的信息增益估计。使用EFB，我们捆绑互斥的特性(即，它们很少同时取非零值)，以减少特性的数量。我们证明了寻找唯一特征的最优捆绑是np困难的，但贪婪算法可以达到很好的逼近率(从而在不影响分割点确定精度的前提下有效地减少特征的数量)。我们使用GOSS和EFB \emph{LightGBM}调用我们的新GBDT实现。我们在多个公共数据集上的实验表明，LightGBM将传统GBDT的训练过程提高了20多倍，同时达到几乎相同的准确率。
</p></blockquote></details>

#### 2016
 <details/>
<summary/>
  <a href="https://dl.acm.org/doi/10.1145/2939672.2939785">Xgboost: A scalable tree boosting system</a> XGBoost(<a href="https://xgboost.readthedocs.io/en/latest">Github</a>)</summary><blockquote><p align="justify">
Tree boosting is a highly effective and widely used machine learning method. In this paper, we describe a scalable end-to-end tree boosting system called XGBoost, which is used widely by data scientists to achieve state-of-the-art results on many machine learning challenges. We propose a novel sparsity-aware algorithm for sparse data and weighted quantile sketch for approximate tree learning. More importantly, we provide insights on cache access patterns, data compression and sharding to build a scalable tree boosting system. By combining these insights, XGBoost scales beyond billions of examples using far fewer resources than existing systems.
  树boosting是一种高效、应用广泛的机器学习方法。在本文中，我们描述了一个名为XGBoost的可扩展的端到端树形增强系统，该系统被数据科学家广泛使用，在许多机器学习挑战中取得了最新的成果。我们提出了一种新的稀疏数据稀疏感知算法和加权分位数草图的近似树学习。更重要的是，我们提供了关于缓存访问模式、数据压缩和分片的见解来构建一个可伸缩的树增强系统。通过结合这些见解，XGBoost可以使用比现有系统少得多的资源扩展数十亿个示例。
</p></blockquote></details>


#### 2001
<details/>
<summary/>
 <a href="https://link.springer.com/article/10.1023%2FA%3A1010933404324">Random Forests (RF)</a> (<a href="https://github.com/hexiaolang/RandomForest-In-text-classification">{Github}</a>) </summary><blockquote><p align="justify">
  Random forests are a combination of tree predictors such that each tree depends on the values of a random vector sampled independently and with the same distribution for all trees in the forest. The generalization error for forests converges a.s. to a limit as the number of trees in the forest becomes large. The generalization error of a forest of tree classifiers depends on the strength of the individual trees in the forest and the correlation between them. Using a random selection of features to split each node yields error rates that compare favorably to Adaboost (Y. Freund & R. Schapire, Machine Learning: Proceedings of the Thirteenth International conference, ***, 148–156), but are more robust with respect to noise. Internal estimates monitor error, strength, and correlation and these are used to show the response to increasing the number of features used in the splitting. Internal estimates are also used to measure variable importance. These ideas are also applicable to regression.
  随机森林是树预测器的组合，因此每棵树都依赖于独立采样的随机向量的值，并且对森林中的所有树具有相同的分布。森林的泛化误差随着森林中树木数量的增加而收敛到一个极限。树分类器的森林泛化误差取决于森林中单个树的强度和它们之间的相关性。使用随机选择的特征来分割每个节点产生的错误率与Adaboost (Y. Freund & R. Schapire，机器学习:第十三届国际会议论文集，***，148-156)不相上下，但在噪声方面更健壮。内部估计监视错误、强度和相关性，这些用于显示对分割中使用的特性数量增加的响应。内部估计也被用来衡量变量的重要性。这些思想也适用于回归。
</p></blockquote></details>

#### 1998
<details/>
<summary/>
<a href="https://xueshu.baidu.com/usercenter/paper/show?paperid=58aa6cfa340e6ae6809c5deadd07d88e&site=xueshu_se">Text categorization with Support Vector Machines: Learning with many relevant features (SVM)</a> (<a href="https://github.com/Gunjitbedi/Text-Classification">{Github}</a>) </summary><blockquote><p align="justify">
  This paper explores the use of Support Vector Machines (SVMs) for learning text classifiers from examples. It analyzes the particular properties of learning with text data and identifies why SVMs are appropriate for this task. Empirical results support the theoretical findings. SVMs achieve substantial improvements over the currently best performing methods and behave robustly over a variety of different learning tasks. Furthermore they are fully automatic, eliminating the need for manual parameter tuning.
  本文通过实例探讨了支持向量机在文本分类器学习中的应用。它分析了文本数据学习的特殊属性，并确定了支持向量机为什么适合这项任务。实证结果支持理论发现。支持向量机在当前性能最好的方法中取得了实质性的改进，并且在各种不同的学习任务中表现得很稳健。此外，它们是全自动的，不需要手动调整参数。
</p></blockquote></details>

#### 1993
<details/>
<summary/>
<a href="https://link.springer.com/article/10.1007/BF00993309">C4.5: Programs for Machine Learning (C4.5)</a> (<a href="https://github.com/Cater5009/Text-Classify">{Github}</a>) </summary><blockquote><p align="justify">
  C4.5算法是由Ross Quinlan开发的用于产生决策树的算法，该算法是对Ross Quinlan之前开发的ID3算法的一个扩展。C4.5算法主要应用于统计分类中，主要是通过分析数据的信息熵建立和修剪决策树。
</p></blockquote></details>

#### 1984
<details/>
<summary/>
<a href="https://dblp.org/img/paper.dark.empty.16x16.png">Classification and Regression Trees (CART)</a> (<a href="https://github.com/sayantann11/all-classification-templetes-for-ML">{Github}</a>) </summary><blockquote><p align="justify">
  分类与回归树CART是由Loe Breiman等人在1984年提出的，自提出后被广泛的应用。CART既能用于分类也能用于回归，和决策树相比较，CART把选择最优特征的方法从信息增益（率）换成了基尼指数。
</p></blockquote></details>

#### 1967
<details/>
<summary/>
<a href="https://dl.acm.org/doi/10.1145/321075.321084">Nearest neighbor pattern classification (k-nearest neighbor classification,KNN)</a> (<a href="https://github.com/raimonbosch/knn.classifier">{Github}</a>) </summary><blockquote><p align="justify">
  The nearest neighbor decision rule assigns to an unclassified sample point the classification of the nearest of a set of previously classified points. This rule is independent of the underlying joint distribution on the sample points and their classifications, and hence the probability of errorRof such a rule must be at least as great as the Bayes probability of errorR^{\ast}--the minimum probability of error over all decision rules taking underlying probability structure into account. However, in a large sample analysis, we will show in theM-category case thatR^{\ast} \leq R \leq R^{\ast}(2 --MR^{\ast}/(M-1)), where these bounds are the tightest possible, for all suitably smooth underlying distributions. Thus for any number of categories, the probability of error of the nearest neighbor rule is bounded above by twice the Bayes probability of error. In this sense, it may be said that half the classification information in an infinite sample set is contained in the nearest neighbor.
  最近邻决策规则将一组已分类点中最接近的分类分配给一个未分类的样本点。这个规则独立于底层的联合分布的采样点及其分类的概率,因此error R概率这样的规则必须至少一样伟大的贝叶斯概率error R ^ {\ ast},可以使误差概率最小决策规则考虑潜在的概率结构。然而，在一个大型样本分析中，我们将在它们-类别的情况下显示R^{\ast} \leq R \leq R^{\ast}/(2—MR^{\ast}/(M-1))，其中对于所有适当平滑的底层分布，这些边界是可能最紧的。因此，对于任意数量的类别，最近邻规则的错误概率是贝叶斯错误概率的两倍以上。从这个意义上说，可以说一个无限样本集中一半的分类信息都包含在最近邻中。
</p></blockquote></details>


#### 1961 

<details/>
<summary/>
<a href="https://dl.acm.org/doi/10.1145/321075.321084">Automatic indexing: An experimental inquiry 采用贝叶斯公式进行文本分类</a> (<a href="https://github.com/Gunjitbedi/Text-Classification">{Github}</a>) </summary><blockquote><p align="justify">
  This inquiry examines a technique for automatically classifying (indexing) documents according to their subject content. The task, in essence, is to have a computing machine read a document and on the basis of the occurrence of selected clue words decide to which of many subject categories the document in question belongs. This paper describes the design, execution and evaluation of a modest experimental study aimed at testing empirically one statistical technique for automatic indexing.
  这个调查研究了一种根据主题内容自动分类(索引)文档的技术。本质上，任务是让计算机读取文档，并根据所选线索词的出现情况来决定所涉文档属于哪个主题类别。本文描述了一个适度的实验研究的设计、执行和评估，旨在实证地测试一种统计技术的自动索引。
</p></blockquote></details>





## Data（数据）
[:arrow_up:](#table-of-contents)

#### Sentiment Analysis (SA) 情感分析
SA is the process of analyzing and reasoning the subjective text withinemotional color. It is crucial to get information on whether it supports a particular point of view fromthe text that is distinct from the traditional text classification that analyzes the objective content ofthe text. SA can be binary or multi-class. Binary SA is to divide the text into two categories, includingpositive and negative. Multi-class SA classifies text to multi-level or fine-grained labels. （情感分析是对带有情感色彩的主观文本进行分析和推理的过程。从文本中获取是否支持特定观点的信息是至关重要的，而传统的文本分类是分析文本的客观内容。情感分析可以是二进制的或多类的。二元情感分类是将文本分为正反两类。多类情感分类将文本分类为多层或细粒度的标签。）

<details/>
<summary/> <a href="http://www.cs.cornell.edu/people/pabo/movie-review-data/">Movie Review (MR) 电影评论数据集</a></summary><blockquote><p align="justify">
The MR is a movie review dataset, each of which correspondsto a sentence. The corpus has 5,331 positive data and 5,331 negative data. 10-fold cross-validationby random splitting is commonly used to test MR. MR是一个影评数据集，每个影评是一个句子。该语料库有正样本和负样本各5331个。十折交叉验证常用来测试MR。
</p></blockquote></details>

<details/>
<summary/> <a href="http://www.cs.uic.edu/∼liub/FBS/sentiment-analysis.html">Stanford Sentiment Treebank (SST) 斯坦福情感库</a></summary><blockquote><p align="justify">
The SST [175] is an extension of MR. It has two cate-gories. SST-1 with fine-grained labels with five classes. It has 8,544 training texts and 2,210 testtexts, respectively. Furthermore, SST-2 has 9,613 texts with binary labels being partitioned into6,920 training texts, 872 development texts, and 1,821 testing texts.
 SST是MR的扩展版本，一共有两种类型。SST-1有五个类别标签，有8,544个训练文本和2,210个测试文本。SST-2有两个类别，共9,613个文本，被划分为6,920个训练文本、872个开发文本和1,821个测试文本。
</p></blockquote></details>

<details/>
<summary/> <a href="http://www.cs.pitt.edu/mpqa/">The Multi-Perspective Question Answering (MPQA)多视角问答数据集</a></summary><blockquote><p align="justify">
The MPQA is an opinion dataset. It has two class labels and also an MPQA dataset of opinion polarity detection sub-tasks.MPQA includes 10,606 sentences extracted from news articles from various news sources. It shouldbe noted that it contains 3,311 positive texts and 7,293 negative texts without labels of each text.
 MPQA是一个观点数据集。它有两个类标签和一个MPQA数据集的观点极性检测子任务。MPQA包含了10,606个句子，这些句子是从各种新闻来源的新闻文章中提取的。应当指出的是，其中有3311个正样本和7293个负样本，每个文本没有标签。
</p></blockquote></details>

<details/>
<summary/> <a href="https://dblp.org/rec/bib/conf/kdd/DiaoQWSJW14">IMDB reviews IMDB评论</a></summary><blockquote><p align="justify">
The IMDB review is developed for binary sentiment classification of filmreviews with the same amount in each class. It can be separated into training and test groups onaverage, by 25,000 comments per group.
 IMDB影评是对每类影评进行相同数量的二元情感分类。它平均可以被分成训练组和测试组，每组有25000条评论。
</p></blockquote></details>

<details/>
<summary/> <a href="https://dblp.org/rec/bib/conf/emnlp/TangQL15">Yelp reviews Yelp评论</a></summary><blockquote><p align="justify">
The Yelp review is summarized from the Yelp Dataset Challenges in 2013,2014, and 2015. This dataset has two categories. Yelp-2 of these were used for negative and positiveemotion classification tasks, including 560,000 training texts and 38,000 test texts. Yelp-5 is used todetect fine-grained affective labels with 650,000 training and 50,000 test texts in all classes.
 Yelp评论数据来自于2013年、2014年和2015年的Yelp数据集挑战。这个数据集有两个类别。其中的Yelp-2被用于消极和积极情绪分类任务，包括560,000篇训练文本和38,000篇测试文本。使用Yelp-5对65万篇训练文本和5万篇测试文本进行精细的情感标签检测。
</p></blockquote></details>

<details/>
<summary/> <a href="https://www.kaggle.com/datafiniti/consumer-reviews-of-amazon-products">Amazon Reviews (AM) 亚马逊评论数据集</a></summary><blockquote><p align="justify">
The AM is a popular corpus formed by collecting Amazon websiteproduct reviews [190]. This dataset has two categories. The Amazon-2 with two classes includes 3,600,000 training sets and 400,000 testing sets. Amazon-5, with five classes, includes 3,000,000 and650,000 comments for training and testing.
 AM是一个收集亚马逊网站产品评论的流行语料库。这个数据集有两个类别。带有两个类的Amazon-2包括360万个训练集和40万个测试集。Amazon-5有五个类，包含300万条和65万条培训和测试评论。
</p></blockquote></details>


#### News Classification (NC) 新闻分类数据集
News content is one of the most crucial information sources which hasa critical influence on people. The NC system facilitates users to get vital knowledge in real-time.News classification applications mainly encompass: recognizing news topics and recommendingrelated news according to user interest. The news classification datasets include 20NG, AG, R8, R52,Sogou, and so on. Here we detail several of the primary datasets.（新闻内容是影响人们生活的最重要的信息来源之一。数控系统便于用户实时获取重要知识。新闻分类应用主要包括:识别新闻话题，根据用户兴趣推荐相关新闻。新闻分类数据集包括20NG、AG、R8、R52、Sogou等。这里我们详细介绍了几个主要数据集。）

<details/>
<summary/>
<a href="http://ana.cachopo.org/datasets-for-single-label-text-categorization">20 Newsgroups (20NG)</a></summary><blockquote><p align="justify">
 The 20NG is a newsgroup text dataset. It has 20 categories withthe same number of each category and includes 18,846 texts.
 20NG是一个新闻组文本数据集。它有20个类别，每个类别的数目相同，包括18,846个文本。
</p></blockquote></details>

<details/>
<summary/> <a href="http://www.di.unipi.it/~gulli/AG_corpus_of_news_articles.html">AG News (AG)</a></summary><blockquote><p align="justify">
The AG News is a search engine for news from academia, choosingthe four largest classes. It uses the title and description fields of each news. AG contains 120,000texts for training and 7,600 texts for testing.
 AG新闻是一个搜索来自学术界的新闻的引擎，选择了四个最大的类别。它使用每个新闻的标题和描述字段。AG包含120,000篇训练文本和7,600篇测试文本。
</p></blockquote></details>

<details/>
<summary/> <a href="https://www.cs.umb.edu/~smimarog/textmining/datasets/">R8 and R52</a></summary><blockquote><p align="justify">
R8 and R52 are two subsets which are the subset of Reuters. R8 has 8categories, divided into 2,189 test files and 5,485 training courses. R52 has 52 categories, split into6,532 training files and 2,568 test files.
 R8和R52是两个子集，它们是Reuters的子集。R8有8个类别，分为2189个测试文件和5485个训练文件。R52有52个类别，分为6,532个训练文件和2,568个测试文件。
</p></blockquote></details>

<details/>
<summary/> <a href="https://dblp.org/rec/conf/cncl/SunQXH19.bib">Sogou News (Sogou) 搜狗新闻</a></summary><blockquote><p align="justify">
The Sogou News combines two datasets, including SogouCA andSogouCS news sets. The label of each text is the domain names in the URL.
 搜狗新闻结合了两个数据集，包括SogouCA和sogoucs新闻集。每个文本的标签是URL中的域名。
</p></blockquote></details>

#### Topic Labeling (TL) 话题标签
The topic analysis attempts to get the meaning of the text by defining thesophisticated text theme. The topic labeling is one of the essential components of the topic analysistechnique, intending to assign one or more subjects for each document to simplify the topic analysis.（主题分析试图通过对文本主题的界定来获得文本的意义。话题标签是主题分析技术的重要组成部分之一，用于为每个文档分配一个或多个主题，以简化主题分析。）

<details/>
<summary/> <a href="https://dblp.org/rec/journals/semweb/LehmannIJJKMHMK15.bib">DBpedia</a></summary><blockquote><p align="justify">
The DBpedia is a large-scale multi-lingual knowledge base generated usingWikipedia’s most ordinarily used infoboxes. It publishes DBpedia each month, adding or deletingclasses and properties in every version. DBpedia’s most prevalent version has 14 classes and isdivided into 560,000 training data and 70,000 test data. 
 DBpedia是使用wikipedia最常用的信息框生成的大型多语言知识库。它每月发布DBpedia，在每个版本中添加或删除类和属性。DBpedia最流行的版本有14个类，分为560,000个训练数据和70,000个测试数据。
</p></blockquote></details>

<details/>
<summary/> <a href="http://davis.wpi.edu/xmdv/datasets/ohsumed.html">Ohsumed</a></summary><blockquote><p align="justify">
The Ohsumed belongs to the MEDLINE database. It includes 7,400 texts andhas 23 cardiovascular disease categories. All texts are medical abstracts and are labeled into one ormore classes.
 Ohsumed属于MEDLINE数据库。它包括7400个文本，有23个心血管疾病类别。所有文本都是医学摘要，并被标记为一个或多个类。
</p></blockquote></details>

<details/>
<summary/> <a href="https://dblp.org/rec/bib/conf/nips/ZhangZL15">Yahoo answers (YahooA) 雅虎问答</a></summary><blockquote><p align="justify">
The YahooA is a topic labeling task with 10 classes. It includes140,000 training data and 5,000 test data. All text contains three elements, being question titles,question contexts, and best answers, respectively.
 YahooA是一个包含10个类的主题标记任务。它包括140,000个训练数据和5,000个测试数据。所有文本包含三个元素，分别是问题标题、问题上下文和最佳答案。
</p></blockquote></details>


#### Question Answering (QA) 问答
The QA task can be divided into two types: the extractive QA and thegenerative QA. The extractive QA gives multiple candidate answers for each question to choosewhich one is the right answer. Thus, the text classification models can be used for the extractiveQA task. The QA discussed in this paper is all extractive QA. The QA system can apply the textclassification model to recognize the correct answer and set others as candidates. The questionanswering datasets include SQuAD, MS MARCO, TREC-QA, WikiQA, and Quora [209]. Here wedetail several of the primary datasets.（QA任务可以分为两种类型:抽取型QA和生成型QA。抽取式QA为每个问题提供多个候选答案，以选择哪一个是正确答案。因此，文本分类模型可以用于抽取式QA任务。本文所讨论的质量保证都是抽取式质量保证。QA系统可以应用文本分类模型来识别正确答案，并将其他答案设置为考生。问答的数据集包括SQuAD、MS MARCO、treci - qa、WikiQA和Quora。这里我们详细介绍了几个主要的数据集。）


<details/>
<summary/> <a href="https://dblp.org/rec/bib/conf/nips/ZhangZL15">Stanford Question Answering Dataset (SQuAD） 斯坦福问答数据集</a></summary><blockquote><p align="justify">
The SQuAD is a set of question and answer pairs obtained from Wikipedia articles. The SQuAD has two categories. SQuAD1.1 contains 536 pairs of 107,785 Q&A items. SQuAD2.0 combines 100,000 questions in SQuAD1.1 with morethan 50,000 unanswerable questions that crowd workers face in a form similar to answerable questions.
这个数据集是一组从维基百科文章中获得的问题和答案。球队分为两类。SQuAD1.1包含536对107,785个问答题。SQuAD2.0将SQuAD1.1中的100,000个问题与5万多个无法回答的问题组合在一起，这些问题以类似于可回答问题的形式出现。
</p></blockquote></details>

<details/>
<summary/> <a href="https://dblp.org/rec/bib/conf/nips/ZhangZL15">MS MARCO</a></summary><blockquote><p align="justify">
The MS MARCO contains questions and answers. The questions and part ofthe answers are sampled from actual web texts by the Bing search engine. Others are generative. Itis used for developing generative QA systems released by Microsoft.
MS MARCO包含了问题和答案。这些问题和部分答案都是由Bing搜索引擎从实际的网络文本中抽取的，其他的则是生成式的。它用于开发由微软发布的生成式QA系统。
</p></blockquote></details>

<details/>
<summary/> <a href="https://dblp.org/rec/bib/conf/nips/ZhangZL15">TREC-QA</a></summary><blockquote><p align="justify">
The TREC-QA includes 5,452 training texts and 500 testing texts. It has two versions. TREC-6 contains 6 categories, and TREC-50 has 50 categories.
TREC-QA包括5,452个训练文本和500个测试文本。它有两个版本。TREC-6包含6个类别，TREC-50包含50个类别。
</p></blockquote></details>

<details/>
<summary/> <a href="https://dblp.org/rec/bib/conf/nips/ZhangZL15">WikiQA</a></summary><blockquote><p align="justify">
The WikiQA dataset includes questions with no correct answer, which needs toevaluate the answer.
WikiQA数据集包含没有正确答案的问题，需要对答案进行评估。
</p></blockquote></details>



#### Natural Language Inference (NLI) 自然语言推理
NLI is used to predict whether the meaning of one text canbe deduced from another. Paraphrasing is a generalized form of NLI. It uses the task of measuringthe semantic similarity of sentence pairs to decide whether one sentence is the interpretation ofanother. The NLI datasets include SNLI, MNLI, SICK, STS, RTE, SciTail, MSRP, etc. Here we detailseveral of the primary datasets.（NLI是用来预测一个文本的意思是否可以从另一个文本中推断出来。释义是NLI的一种概括形式。它的任务是测量句子对的语义相似度，以决定一个句子是否是另一个句子的解释。NLI数据集包括SNLI, MNLI, SICK, STS, RTE, SciTail, MSRP等。这里我们详细介绍了几个主要数据集。）

<details/>
<summary/> <a href="https://dblp.org/rec/bib/conf/nips/ZhangZL15">The Stanford Natural Language Inference (SNLI)</a></summary><blockquote><p align="justify">
The SNLI is generally applied toNLI tasks. It contains 570,152 human-annotated sentence pairs, including training, development,and test sets, which are annotated with three categories: neutral, entailment, and contradiction.
  SNLI一般应用于各种任务。它包含570,152对人注释句子，包括训练集、开发集和测试集，并以中性句、隐含句和矛盾句三大类注释.
</p></blockquote></details>

<details/>
<summary/> <a href="https://dblp.org/rec/bib/conf/nips/ZhangZL15">Multi-Genre Natural Language Inference (MNLI)</a></summary><blockquote><p align="justify">
The Multi-NLI is an expansion of SNLI, embracing a broader scope of written and spoken text genres. It includes 433,000 sentencepairs annotated by textual entailment labels.
  Multi-NLI是SNLI的扩展，包括更广泛的书面和口语文本类型。它包括433,000个句子，并附有文本蕴涵标签。
</p></blockquote></details>

<details/>
<summary/> <a href="https://dblp.org/rec/bib/conf/nips/ZhangZL15">Sentences Involving Compositional Knowledge (SICK)</a></summary><blockquote><p align="justify">
The SICK contains almost10,000 English sentence pairs. It consists of neutral, entailment and contradictory labels.
The SICK包含近10,000对英语句子。它由中性、含蓄和矛盾的标签构成。
</p></blockquote></details>

<details/>
<summary/> <a href="https://dblp.org/rec/bib/conf/nips/ZhangZL15">Microsoft Research Paraphrase (MSRP)</a></summary><blockquote><p align="justify">
The MSRP consists of sentence pairs, usuallyfor the text-similarity task. Each pair is annotated by a binary label to discriminate whether theyare paraphrases. It respectively includes 1,725 training and 4,076 test sets.
MSRP由句子对组成，通常用于文本相似任务。每一对都由一个二进制标签注释，以区分它们是否是意译。分别包含1725个训练集和4076个测试集。
</p></blockquote></details>



#### Dialog Act Classification (DAC) 对话行为分类
A dialog act describes an utterance in a dialog based on semantic,pragmatic, and syntactic criteria. DAC labels a piece of a dialog according to its category of meaningand helps learn the speaker’s intentions. It is to give a label according to dialog. Here we detailseveral of the primary datasets, including DSTC 4, MRDA, and SwDA.（对话行为描述基于语义、语用和句法标准的对话中的话语。DAC根据对话的意义类别给对话片贴上标签，并帮助了解讲话者的意图。根据对话给一个标签。这里我们详细介绍了几个主要数据集，包括DSTC 4、MRDA和SwDA。）

<details/>
<summary/> <a href="https://dblp.org/rec/bib/conf/nips/ZhangZL15">Dialog State Tracking Challenge 4 (DSTC 4)</a></summary><blockquote><p align="justify">
The DSTC 4 is used for dialog act classi-fication. It has 89 training classes, 24,000 training texts, and 6,000 testing texts.
DSTC 4用于对话行为分类。它有89个培训班，24000个培训文本，6000个测试文本。
</p></blockquote></details>

<details/>
<summary/> <a href="https://dblp.org/rec/bib/conf/nips/ZhangZL15">ICSI Meeting Recorder Dialog Act (MRDA)</a></summary><blockquote><p align="justify">
The MRDA is used for dialog act classifi-cation. It has 5 training classes, 51,000 training texts, 11,000 testing texts, and 11,000 validation texts.
MRDA用于对话行为分类。它有5个培训类、51,000个培训文本、11,000个测试文本和11,000个验证文本。
</p></blockquote></details>

<details/>
<summary/> <a href="https://dblp.org/rec/bib/conf/nips/ZhangZL15">Switchboard Dialog Act (SwDA)</a></summary><blockquote><p align="justify">
The SwDA is used for dialog act classification. It has43 training classes, 1,003,000 training texts, 19,000 testing texts and 112,000 validation texts.
SwDA用于对话行为分类。它有43个培训课程，1,003,000个培训文本，19,000个测试文本和112,000个验证文本。
</p></blockquote></details>


#### Multi-label datasets 多标签数据集
In multi-label classification, an instance has multiple labels, and each la-bel can only take one of the multiple classes. There are many datasets based on multi-label textclassification. It includes Reuters, Education, Patent, RCV1, RCV1-2K, AmazonCat-13K, BlurbGen-reCollection, WOS-11967, AAPD, etc. Here we detail several of the main datasets.（在多标签分类中，一个实例有多个标签，每个la-bel只能获取多个类中的一个。有许多基于多标签文本分类的数据集。包括Reuters、Education、Patent、RCV1、RCV1-2k、AmazonCat-13K、BlurbGen-reCollection, WOS-11967、AAPD等。这里我们详细介绍了几个主要的数据集。）

<details/>
<summary/> <a href="https://dblp.org/rec/bib/conf/nips/ZhangZL15">Reuters news</a></summary><blockquote><p align="justify">
The Reuters is a popularly used dataset for text classification fromReuters financial news services. It has 90 training classes, 7,769 training texts, and 3,019 testingtexts, containing multiple labels and single labels. There are also some Reuters sub-sets of data,such as R8, BR52, RCV1, and RCV1-v2.
Reuters是一个广泛使用的数据集，用于Reuters财经新闻服务的文本分类。它有90个训练类别，7,769个训练文本和3,019个测试文本，包含多个标签和单个标签。还有一些Reuters数据子集，如R8、BR52、RCV1和RCV1-v2。
</p></blockquote></details>


<details/>
<summary/> <a href="https://dblp.org/rec/bib/conf/nips/ZhangZL15">Patent Dataset</a></summary><blockquote><p align="justify">
The Patent Dataset is obtained from USPTO1, which is a patent system gratingU.S. patents containing textual details such title and abstract. It contains 100,000 US patents awardedin the real-world with multiple hierarchical categories.
该数据集来自美国专利系统USPTO1。专利包含文本细节，如标题和摘要。它包含了现实世界中获得的10万项美国专利，具有多重等级类别。
</p></blockquote></details>


<details/>
<summary/> <a href="https://dblp.org/rec/bib/conf/nips/ZhangZL15">Reuters Corpus Volume I (RCV1) and RCV1-2K</a></summary><blockquote><p align="justify">
The RCV1 is collected from Reuters News articles from 1996-1997, which is human-labeled with 103 categories. It consists of 23,149 training and 784,446 testing texts, respectively. The RCV1-2K dataset has the same features as the RCV1. However, the label set of RCV1-2K has been expanded with some new labels. It contains2456 labels.
RCV1收集自1996-1997年Reuters News的新闻文章，被人为标记为103个类别。它分别由23,149个培训文本和784,446个测试文本组成。RCV1- 2k数据集具有与RCV1相同的特征。然而，RCV1-2K的标签集已经扩展了一些新的标签。它包含2456标签。
</p></blockquote></details>


<details/>
<summary/> <a href="https://dblp.org/rec/bib/conf/nips/ZhangZL15">Web of Science (WOS-11967)</a></summary><blockquote><p align="justify">
The WOS-11967 is crawled from the Web of Science,consisting of abstracts of published papers with two labels for each example. It is shallower, butsignificantly broader, with fewer classes in total.
WOS-11967抓取自Web of Science，其中包含已发表论文的摘要，每个例子有两个标签。它较浅，但明显较宽，类总数较少。
</p></blockquote></details>

<details/>
<summary/> <a href="https://dblp.org/rec/bib/conf/nips/ZhangZL15">Arxiv Academic Paper Dataset (AAPD)</a></summary><blockquote><p align="justify">
The AAPD is a large dataset in the computer science field for the multi-label text classification from website2. It has 55,840 papers, including the abstract and the corresponding subjects with 54 labels in total. The aim is to predict the corresponding subjects of each paper according to the abstract.
AAPD是计算机科学领域中用于website2多标签文本分类的大型数据集。论文55,840篇，包括摘要及相应科目，共计54个标签。目的是根据摘要预测每篇论文对应的主题。
</p></blockquote></details>


#### Others 其他
There are some datasets for other applications, such as Geonames toponyms, Twitter posts,and so on.（还有一些用于其他应用程序的数据集，比如Geonames toponyms、Twitter帖子等等。）

## Evaluation Metrics（评价指标）
[:arrow_up:](#table-of-contents)

In terms of evaluating text classification models, accuracy and F1 score are the most used to assessthe text classification methods. Later, with the increasing difficulty of classification tasks or theexistence of some particular tasks, the evaluation metrics are improved. For example, evaluationmetrics such as P@K and Micro-F1 are used to evaluate multi-label text classification performance,and MRR is usually used to estimate the performance of QA tasks.（在评价文本分类模型方面，评价文本分类方法最常用的是accuracy和F1分。随后，随着分类任务难度的增加或某些特定任务的存在，改进了评价指标。例如，评价指标如P@K和Micro-F1用于评价多标签文本分类性能，MRR通常用于评价QA任务的性能。）

#### Single-label metrics 单标签评价指标
Single-label text classification divides the text into one of the most likelycategories applied in NLP tasks such as QA, SA, and dialogue systems [9]. For single-label textclassification, one text belongs to just one catalog, making it possible not to consider the relationsamong labels. Here we introduce some evaluation metrics used for single-label text classificationtasks.（单标签文本分类将文本分成一个最可能应用于NLP任务的类别，如QA、SA和对话系统。对于单标签文本分类，一个文本只属于一个目录，使得不考虑标签之间的关系成为可能。在这里，我们介绍一些评价指标用于单标签文本分类任务。）


<details/>
<summary/> <a href="https://dblp.org/rec/bib/conf/nips/ZhangZL15">Accuracy and Error Rate</a></summary><blockquote><p align="justify">
Accuracy and Error Rate are the fundamental metrics for a text classification model. The Accuracy and Error Rate are respectively defined as
</p></blockquote></details>

<details/>
<summary/> <a href="https://dblp.org/rec/bib/conf/nips/ZhangZL15">Precision, Recall and F1</a></summary><blockquote><p align="justify">
These are vital metrics utilized for unbalanced test sets regardless ofthe standard type and error rate. For example, most of the test samples have a class label. F1 is theharmonic average of Precision and Recall. Accuracy, Recall, and F1 as defined
  
  The desired results will be obtained when the accuracy, F1 and recall value reach 1. On the contrary,when the values become 0, the worst result is obtained. For the multi-class classification problem,the precision and recall value of each class can be calculated separately, and then the performanceof the individual and whole can be analyzed.
</p></blockquote></details>

<details/>
<summary/> <a href="https://dblp.org/rec/bib/conf/nips/ZhangZL15">Exact Match (EM)</a></summary><blockquote><p align="justify">
The EM is a metric for QA tasks measuring the prediction that matches all theground-truth answers precisely. It is the primary metric utilized on the SQuAD dataset.

</p></blockquote></details>


<details/>
<summary/> <a href="https://dblp.org/rec/bib/conf/nips/ZhangZL15">Mean Reciprocal Rank (MRR)</a></summary><blockquote><p align="justify">
The EM is a metric for QA tasks measuring the prediction that matches all theground-truth answers precisely. It is the primary metric utilized on the SQuAD dataset.
</p></blockquote></details>

<details/>
<summary/> <a href="https://dblp.org/rec/bib/conf/nips/ZhangZL15">Hamming-loss (HL)</a></summary><blockquote><p align="justify">
The HL assesses the score of misclassified instance-label pairs wherea related label is omitted or an unrelated is predicted.
</p></blockquote></details>


#### Multi-label metrics 多标签评价指标
Compared with single-label text classification, multi-label text classifica-tion divides the text into multiple category labels, and the number of category labels is variable. These metrics are designed for single label text classification, which are not suitable for multi-label tasks. Thus, there are some metrics designed for multi-label text classification.（与单标签文本分类相比，多标签文本分类将文本分为多个类别标签，类别标签的数量是可变的。这些度量标准是针对单标签文本分类而设计的，不适用于多标签任务。因此，有一些度量标准为多标签文本分类设计。）

<details/>
<summary/> <a href="https://dblp.org/rec/bib/conf/nips/ZhangZL15">Micro−F1</a></summary><blockquote><p align="justify">
The Micro−F1 is a measure that considers the overall accuracy and recall of alllabels. The Micro−F1is defined as
</p></blockquote></details>


<details/>
<summary/> <a href="https://dblp.org/rec/bib/conf/nips/ZhangZL15">Macro−F1</a></summary><blockquote><p align="justify">
The Macro−F1 calculates the average F1 of all labels. Unlike Micro−F1, which setseven weight to every example, Macro−F1 sets the same weight to all labels in the average process. Formally, Macro−F1is defined as
</p></blockquote></details>

In addition to the above evaluation metrics, there are some rank-based evaluation metrics forextreme multi-label classification tasks, including P@K and NDCG@K.

<details/>
<summary/> <a href="https://dblp.org/rec/bib/conf/nips/ZhangZL15">Precision at Top K (P@K)</a></summary><blockquote><p align="justify">
The P@K is the precision at the top k. ForP@K, each text has a set of L ground truth labels Lt={l0,l1,l2...,lL−1}, in order of decreasing probability Pt=p0,p1,p2...,pQ−1.The precision at k is
</p></blockquote></details>

<details/>
<summary/> <a href="https://dblp.org/rec/bib/conf/nips/ZhangZL15">Normalized Discounted Cummulated Gains (NDCG@K)</a></summary><blockquote><p align="justify">
The NDCG at k is
</p></blockquote></details>


## Future Research Challenges（未来研究挑战）
[:arrow_up:](#table-of-contents)

文本分类-作为有效的信息检索和挖掘技术-在管理文本数据中起着至关重要的作用。它使用NLP，数据挖掘，机器学习和其他技术来自动分类和发现不同的文本类型。文本分类将多种类型的文本作为输入，并且文本由预训练模型表示为矢量。然后将向量馈送到DNN中进行训练，直到达到终止条件为止，最后，下游任务验证了训练模型的性能。现有的模型已经显示出它们在文本分类中的有用性，但是仍有许多可能的改进需要探索。尽管一些新的文本分类模型反复擦写了大多数分类任务的准确性指标，但它无法指示模型是否像人类一样从语义层面“理解”文本。此外，随着噪声样本的出现，小的样本噪声可能导致决策置信度发生实质性变化，甚至导致决策逆转。因此，需要在实践中证明该模型的语义表示能力和鲁棒性。此外，由词向量表示的预训练语义表示模型通常可以提高下游NLP任务的性能。关于上下文无关单词向量的传输策略的现有研究仍是相对初步的。因此，我们从数据，模型和性能的角度得出结论，文本分类主要面临以下挑战：





#### 数据层面

对于文本分类任务，无论是浅层学习还是深度学习方法，数据对于模型性能都是必不可少的。研究的文本数据主要包括多章，短文本，跨语言，多标签，少样本文本。对于这些数据的特征，现有的技术挑战如下：


<details/>
<summary/>
<a href="https://link.springer.com/article/10.1007/BF00993309">Zero-shot/Few-shot learning</a> (<a href="https://github.com/Cater5009/Text-Classify">{Github}</a>) </summary><blockquote><p align="justify">
  当前的深度学习模型过于依赖大量标记数据。这些模型的性能在零镜头或少镜头学习中受到显着影响。
</p></blockquote></details>

<details/>
<summary/>
<a href="https://link.springer.com/article/10.1007/BF00993309">外部知识</a> (<a href="https://github.com/Cater5009/Text-Classify">{Github}</a>) </summary><blockquote><p align="justify">
  我们都知道，输入的有益信息越多，DNN的性能就越好。因此，认为添加外部知识(知识库或知识图)是提高模型性能的有效途径。然而，如何添加以及添加什么仍然是一个挑战。
</p></blockquote></details>

<details/>
<summary/>
<a href="https://link.springer.com/article/10.1007/BF00993309">多标签文本分类任务</a> (<a href="https://github.com/Cater5009/Text-Classify">{Github}</a>) </summary><blockquote><p align="justify">
  多标签文本分类需要充分考虑标签之间的语义关系，并且模型的嵌入和编码是有损压缩的过程。因此，如何减少训练过程中层次语义的丢失以及如何保留丰富而复杂的文档语义信息仍然是一个亟待解决的问题。
</p></blockquote></details>

<details/>
<summary/>
<a href="https://link.springer.com/article/10.1007/BF00993309">具有许多术语词汇的特殊领域</a> (<a href="https://github.com/Cater5009/Text-Classify">{Github}</a>) </summary><blockquote><p align="justify">
  特定领域的文本（例如金融和医学文本）包含许多特定的单词或领域专家，可理解的语，缩写等，这使现有的预训练单词向量难以使用。
</p></blockquote></details>


#### 模型层面

现有的浅层和深度学习模型的大部分结构都被尝试用于文本分类，包括集成方法。BERT学习了一种语言表示法，可以用来对许多NLP任务进行微调。主要的方法是增加数据，提高计算能力和设计训练程序，以获得更好的结果如何在数据和计算资源和预测性能之间权衡是值得研究的。

#### 性能评估层面

浅层模型和深层模型可以在大多数文本分类任务中取得良好的性能，但是需要提高其结果的抗干扰能力。如何实现对深度模型的解释也是一个技术挑战。

<details/>
<summary/>
<a href="https://link.springer.com/article/10.1007/BF00993309">模型的语义鲁棒性</a> (<a href="https://github.com/Cater5009/Text-Classify">{Github}</a>) </summary><blockquote><p align="justify">
  近年来，研究人员设计了许多模型来增强文本分类模型的准确性。但是，如果数据集中有一些对抗性样本，则模型的性能会大大降低。因此，如何提高模型的鲁棒性是当前研究的热点和挑战。
</p></blockquote></details>

<details/>
<summary/>
<a href="https://link.springer.com/article/10.1007/BF00993309">模型的可解释性</a> (<a href="https://github.com/Cater5009/Text-Classify">{Github}</a>) </summary><blockquote><p align="justify">
  DNN在特征提取和语义挖掘方面具有独特的优势，并且已经完成了出色的文本分类任务。但是，深度学习是一个黑盒模型，训练过程难以重现，隐式语义和输出可解释性很差。它对模型进行了改进和优化，丢失了明确的准则。此外，我们无法准确解释为什么该模型可以提高性能。
</p></blockquote></details>


## Tools and Repos（工具与资源）
[:arrow_up:](#table-of-contents)


<details>
<summary><a href="https://github.com/Tencent/NeuralNLP-NeuralClassifier">NeuralClassifier</a></summary><blockquote><p align="justify">
腾讯的开源NLP项目
</p></blockquote></details>


<details>
<summary><a href="https://github.com/nocater/baidu_nlp_project2">baidu_nlp_project2</a></summary><blockquote><p align="justify">
百度NLP项目
</p></blockquote></details>



<details>
<summary><a href="https://github.com/TianWuYuJiangHenShou/textClassifier">Multi-label</a></summary><blockquote><p align="justify">
多标签文本分类项目
</p></blockquote></details>
