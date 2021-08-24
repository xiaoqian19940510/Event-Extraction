# Event-Extraction（事件抽取资料综述总结）更新中...
近年来事件抽取方法总结，包括中文事件抽取、开放域事件抽取、事件数据生成、跨语言事件抽取、小样本事件抽取、零样本事件抽取等类型，DMCNN、FramNet、DLRNN、DBRNN、GCN、DAG-GRU、JMEE、PLMEE等方法。


# Table of Contents

- [Define](#define)
- [Surveys](#surveys)
- [Models](#models)
- [Datasets](#datasets)
- [Future Research Challenges](#future-research-challenges)
</p></blockquote></details>



# Define
[:arrow_up:](#table-of-contents)

### Closed-domain

Closed-domain事件抽取使用预定义的事件模式从文本中发现和提取所需的特定类型的事件。事件模式包含多个事件类型及其相应的事件结构。D.Ahn首先提出将ACE事件抽取任务分成四个子任务:触发词检测、事件/触发词类型识别、事件论元检测和参数角色识别。我们使用ACE术语来介绍如下事件结构:

<details/>
<summary/>
<a >事件提及</summary><blockquote><p align="justify">
描述事件的短语或句子，包括触发词和几个参数。
</p></blockquote></details>


<details/>
<summary/>
<a >事件触发词</summary><blockquote><p align="justify">
最清楚地表达事件发生的主要词，一般指动词或名词。
</p></blockquote></details>



<details/>
<summary/>
<a >事件论元</summary><blockquote><p align="justify">
 一个实体,时间表达式，作为参与者的值和在事件中具有特定角色的属性。
</p></blockquote></details>


<details/>
<summary/>
<a >论元角色</summary><blockquote><p align="justify">
论元与它所参与的事件之间的关系。
</p></blockquote></details>




### Open domain

在没有预定义的事件模式的情况下，开放域事件抽取的目的是从文本中检测事件，在大多数情况下，还可以通过提取的事件关键词聚类相似的事件。事件关键词指的是那些主要描述事件的词/短语，有时关键词还进一步分为触发器和参数。

<details/>
<summary/>
<a >故事分割</summary><blockquote><p align="justify">
 从新闻中检测故事的边界。
</p></blockquote></details>


<details/>
<summary/>
<a >第一个故事检测</summary><blockquote><p align="justify">
 检测新闻流中讨论新话题的故事。
</p></blockquote></details>


<details/>
<summary/>
<a >话题检测</summary><blockquote><p align="justify">
 根据讨论的主题将故事分组。
</p></blockquote></details>

<details/>
<summary/>
<a >话题追踪</summary><blockquote><p align="justify">
 检测讨论先前已知话题的故事。
</p></blockquote></details>


<details/>
<summary/>
<a >故事链检测</summary><blockquote><p align="justify">
决定两个故事是否讨论同一个主题。
</p></blockquote></details>


前两个任务主要关注事件检测;其余三个任务用于事件集群。虽然这五项任务之间的关系很明显，但每一项任务都需要一个不同的评价过程，并鼓励采用不同的方法来解决特定问题。


# Surveys
[:arrow_up:](#table-of-contents)

### 事件抽取综述

<details/>
<summary/>
<a href="https://arxiv.org/abs/2107.02126">	A Comprehensive Survey on Schema-based Event Extraction with Deep Learning, arxiv 2021</a> by <i>Qian Li, Hao Peng, Jianxin Li, Yiming Hei, Rui Sun, Jiawei Sheng, Shu Guo, Lihong Wang, Philip S. Yu
</a></summary><blockquote><p align="justify">
基于模式的事件提取是及时理解事件本质内容的关键技术。随着深度学习技术的快速发展，基于深度学习的事件提取技术成为研究热点。文献中提出了大量的方法、数据集和评价指标，因此需要进行全面和更新的调查。本文通过回顾最新的方法填补了这一空白，重点关注基于深度学习的模型。我们总结了基于模式的事件提取的任务定义、范式和模型，然后详细讨论每一个。我们引入了支持预测和评估指标测试的基准数据集。本调查还提供了不同技术之间的综合比较。最后，总结了今后的研究方向。
</p></blockquote></details>



<details/>
<summary/>
<a href="https://doi.org/10.11896/j.issn.1002-137X.2019.08.002">元事件抽取研究综述, 2019</a> by <i>GAO Li-zheng, ZHOU Gang, LUO Jun-yong, LAN Ming-jing
</a></summary><blockquote><p align="justify">
事件抽取是信息抽取领域的一个重要研究方向,在情报收集、知识提取、文档摘要、知识问答等领域有着广泛应用。写了一篇对当前事件抽取领域研究得较多的元事件抽取任务的综述。首先,简要介绍了元事件和元事件抽取的基本概念,以及元事件抽取的主要实现方法。然后,重点阐述了元事件抽取的主要任务,详细介绍了元事件检测过程,并对其他相关任务进行了概述。最后,总结了元事件抽取面临的问题,在此基础上展望了元事件抽取的发展趋势。
</p></blockquote></details>


<details/>
<summary/>
<a href="http://ceur-ws.org/Vol-779/derive2011_submission_1.pdf">An Overview of Event Extraction from Text, 2019</a> by <i>Frederik Hogenboom, Flavius Frasincar, Uzay Kaymak, Franciska de Jong:
</a></summary><blockquote><p align="justify">
文本挖掘的一个常见应用是事件抽取，它包括推导出与事件相关的特定知识，这些知识重新映射到文本中。事件抽取可处理各种类型的文本，如(在线)新闻消息、博客和手稿。本文献回顾了用于各种事件抽取目的的文本挖掘技术。它提供了关于如何根据用户、可用内容和使用场景选择特定事件抽取技术的一般指南。
</p></blockquote></details>


<details/>
<summary/>
<a href="https://doi.org/10.1109/ACCESS.2019.2956831">A Survey of Event Extraction from Text, 2019</a> by <i>Wei Xiang, Bang Wang </a></summary><blockquote><p align="justify">
事件抽取的任务定义、数据源和性能评估，还为其解决方案方法提供了分类。在每个解决方案组中，提供了最具代表性的方法的详细分析，特别是它们的起源、基础、优势和弱点。最后，对未来的研究方向进行了展望。
</p></blockquote></details>



<details/>
<summary/>
<a href="http://ceur-ws.org/Vol-1988/LPKM2017_paper_15.pdf">A Survey of Textual Event Extraction from Social Networks, 2017</a> by <i>Mohamed Mejri, Jalel Akaichi </a></summary><blockquote><p align="justify">
过去的十年中，在社交网络上挖掘文本内容以抽取相关数据和有用的知识已成为无所不在的任务。文本挖掘的一种常见应用是事件抽取，它被认为是一个复杂的任务，分为不同难度的多个子任务。在本文中，我们对现有的主要文本挖掘技术进行了概述，这些技术可用于许多不同的事件抽取目标。首先，我们介绍基于统计模型将数据转换为知识的主要数据驱动方法。其次，我们介绍了基于专家知识的知识驱动方法，通常通过基于模式的方法来抽取知识。然后，我们介绍结合了数据驱动和知识驱动方法的主要现有混合方法。最后，我们比较社交网络事件抽取研究，概括了每种提出的方法的主要特征。
</p></blockquote></details>



<details/>
<summary/>
<a href="https://doi.org/10.1016/j.dss.2016.02.006">A Survey of event extraction methods from text for decision support systems, 2016</a> by <i>Frederik Hogenboom, Flavius Frasincar, Uzay Kaymak, Franciska de Jong, Emiel Caron </a></summary><blockquote><p align="justify">
事件抽取是一种可以追溯到20世纪80年代的专门的信息抽取流程，由于大数据的出现以及文本挖掘和自然语言处理等相关领域的发展，事件抽取技术得到了极大的普及。
然而，到目前为止，对这一特殊领域的概述仍然是难以捉摸的。
因此，我们总结了文本数据的事件抽取技术，划分成数据驱动、知识驱动和混合方法三类，并对这些方法进行了定性评价。
此外，还讨论了从文本语料库中抽取事件的常见决策支持应用。
最后，对事件抽取系统的评价进行了阐述，并指出了当前的研究问题。
</p></blockquote></details>





# Models
[:arrow_up:](#table-of-contents)


### 事件抽取


#### 2021

 <details/>
<summary/>
  <a href="https://www.aclweb.org/anthology/2021.naacl-main.6/">Event Time Extraction and Propagation via Graph Attention Networks, NAACL-HLT 2021 </a> by <i> Haoyang Wen, Yanru Qu, Heng Ji, Qiang Ning, Jiawei Han, Avi Sil, Hanghang Tong and Dan Roth(<a>Github</a>)</summary><blockquote><p align="justify">
  
主要思想：
将事件固定在一个精确的时间轴上对自然语言理解很重要，但在最近的研究中受到的关注有限。
由于语言固有的模糊性和信息在相互关联的事件上传播的要求，这个问题具有挑战性。
本文首先提出了一种用于实体槽填充的四元组时态表示方法，使我们能够更方便地表示模糊时间跨度。
然后，我们提出了一种基于图注意网络的方法，在由共享实体参数和时间关系构造的文档级事件图上传播时间信息。
为了更好地评估我们的方法，我们在ACE2005语料库上提出了一个具有挑战性的新基准，其中78%以上的事件在其本地上下文中没有明确提到时间跨度。
该方法比上下文化的嵌入方法的匹配率提高了7.0%，比句子级人工事件时间参数标注方法的匹配率提高了16.3%。


数据集：ACE
</p></blockquote></details>


 <details/>
<summary/>
  <a href="https://www.aclweb.org/anthology/2021.eacl-main.52/">GRIT: Generative Role-filler Transformers for Document-level Event Entity Extraction, EACL 2021 </a> by <i> Xinya Du, Alexander M. Rush and Claire Cardie(<a>Github</a>)</summary><blockquote><p align="justify">
  
主要思想：
事件参数提取是事件提取中的一项基本任务，在资源匮乏的情况下尤其具有挑战性。
我们从两个方面来解决现有研究在资源匮乏情况下存在的问题。
从模型的角度来看，现有的方法往往存在参数共享不足的问题，没有考虑角色的语义，不利于处理稀疏数据。
而从数据的角度来看，现有的方法大多侧重于数据生成和数据增强。
然而，这些方法严重依赖外部资源，创建外部资源比获取未标记的数据更加费力。
在本文中,我们提出DualQA,小说的框架,这事件模型参数提取的任务问题回答缓解数据稀疏的问题,利用事件参数识别的二元性问“什么扮演的角色”,以及事件的角色识别这是问“角色”,
相互完善。
在两个数据集上的实验结果证明了我们的方法的有效性，特别是在资源极低的情况下。


数据集：MUC-4
</p></blockquote></details>


 <details/>
<summary/>
  <a href="https://ojs.aaai.org/index.php/AAAI/article/view/17720">What the Role is vs. What Plays the Role: Semi-Supervised Event Argument Extraction via Dual Question Answering, AAAI 2021 </a> by <i> Yang Zhou, Yubo Chen, Jun Zhao, Yin Wu, Jiexin Xu and JinLong Li(<a>Github</a>)</summary><blockquote><p align="justify">
  
主要思想：
事件参数提取是事件提取中的一项基本任务，在资源匮乏的情况下尤其具有挑战性。
我们从两个方面来解决现有研究在资源匮乏情况下存在的问题。
从模型的角度来看，现有的方法往往存在参数共享不足的问题，没有考虑角色的语义，不利于处理稀疏数据。
而从数据的角度来看，现有的方法大多侧重于数据生成和数据增强。
然而，这些方法严重依赖外部资源，创建外部资源比获取未标记的数据更加费力。
在本文中,我们提出DualQA,小说的框架,这事件模型参数提取的任务问题回答缓解数据稀疏的问题,利用事件参数识别的二元性问“什么扮演的角色”,以及事件的角色识别这是问“角色”,
相互完善。
在两个数据集上的实验结果证明了我们的方法的有效性，特别是在资源极低的情况下。


数据集：ACE, FewFC
</p></blockquote></details>



 <details/>
<summary/>
  <a href="https://ojs.aaai.org/index.php/AAAI/article/view/17478">GATE: Graph Attention Transformer Encoder for Cross-lingual Relation and Event Extraction, AAAI 2021 </a> by <i> Wasi Uddin Ahmad, Nanyun Peng and KaiWei Chang(<a>Github</a>)</summary><blockquote><p align="justify">
  
主要思想：
跨语言关系和事件提取的最新进展是使用具有通用依赖解析的图卷积网络(GCNs)来学习与语言无关的句子表示，这样在一种语言上训练的模型可以应用于其他语言。
然而，GCNs很难对具有长期依赖关系的词进行建模，或者不能在依赖树中直接连接。
为了解决这些问题，我们建议利用自我注意机制，明确融合结构信息来学习不同句法距离单词之间的依赖关系。
我们引入了GATE，一种图注意转换器编码器，并在关系和事件提取任务中测试了它的跨语言可移植性。
我们在ACE05数据集上执行实验，该数据集包括三种类型不同的语言:英语、汉语和阿拉伯语。
评价结果表明，GATE算法的性能优于最近提出的三种方法。
我们的详细分析显示，由于对语法依赖的依赖，GATE产生了有助于跨语言迁移的健壮表示。


数据集：ACE
</p></blockquote></details>




 <details/>
<summary/>
  <a href="https://arxiv.org/abs/2108.10038">Event Extraction by Associating Event Types and Argument Roles, arXiv 2021 </a> by <i> Qian Li, Shu Guo, Jia Wu, Jianxin Li, Jiawei Sheng, Lihong Wang, Xiaohan Dong, and Hao Peng </summary><blockquote><p align="justify">
  
从文本中获取结构化事件知识的事件抽取可分为事件类型分类和元素抽取(即识别不同角色模式下的触发器和参数)两个子任务。由于不同的事件类型总是有不同的抽取模式(即角色模式)，以往关于情感表达的研究通常遵循一个孤立的学习范式，对不同的事件类型独立地进行元素抽取。它忽略了事件类型和参数角色之间有意义的关联，导致较少频繁的类型/角色的性能相对较差。本文提出了一种新的情感表达任务神经关联框架。给定一个文档，首先通过构建文档级图来关联不同类型的句子节点进行类型分类，然后采用图注意网络来学习句子嵌入。然后，通过构建参数角色的通用模式来实现元素提取，使用参数继承机制来增强提取元素的角色优先级。因此，我们的模型考虑了情感表达过程中的类型和角色关联，实现了它们之间的隐式信息共享。实验结果表明，我们的方法在两个子任务中都优于最先进的情感表达方法。特别是对于训练数据较少的类型/角色，其性能优于现有方法。

  ![image](https://github.com/xiaoqian19940510/Event-Extraction/blob/master/figures/Framework.png)

数据集：ACE
</p></blockquote></details>




 <details/>
<summary/>
  <a href="https://arxiv.org/abs/2106.12384">Reinforcement Learning-based Dialogue Guided Event Extraction to Exploit Argument Relations, arXiv 2021 </a> by <i> Qian Li, Hao Peng, Jianxin Li, Yuanxing Ning, Lihong Wang, Philip S. Yu, and Zheng Wang(<a href="https://github.com/xiaoqian19940510/TASLP-EAREE">Github</a>)</summary><blockquote><p align="justify">
  
主要思想：事件提取是自然语言处理的一项基本任务。
找到事件参数(如事件参与者)的角色是提取事件的关键。
然而，在真实的事件描述中这样做是具有挑战性的，因为一个论点的作用在不同的语境中往往是不同的。
虽然多个参数之间的关系和交互对于解决参数角色是有用的，但是这些信息很大程度上被现有的方法忽略了。
本文通过显式地利用事件参数的关系，提出了一种更好的事件提取方法。
我们通过一个精心设计的面向任务的对话系统来实现这一点。
为了对参数关系进行建模，我们采用了强化学习和增量学习的方法，通过一个多轮、迭代的过程提取多个参数。
我们的方法利用对同一句子中已经提取的论据的知识来确定那些很难单独决定的论据的作用。
然后，它使用新获得的信息来改进以前提取的参数的决策。
这种双向反馈的过程允许我们利用论证关系，有效地解决论证角色，导致更好的句子理解和事件提取。
实验结果表明，在事件分类、参数角色和参数识别方面，该方法始终优于目前最先进的7种事件提取方法。

  ![image](https://github.com/xiaoqian19940510/Event-Extraction/blob/master/figures/Framework.png)

数据集：ACE
</p></blockquote></details>



 <details/>
<summary/>
  <a href="https://arxiv.org/abs/2107.01583">CasEE: A Joint Learning Framework with Cascade Decoding for Overlapping Event Extraction, Findings of ACL 2021 </a> by <i> Jiawei Sheng, Shu Guo, Bowen Yu, Qian Li, Yiming Hei, Lihong Wang, Tingwen Liu, Hongbo Xu(<a href="https://github.com/JiaweiSheng">Github</a>)</summary><blockquote><p align="justify">
  
事件提取(EE)是一项重要的信息提取任务，旨在提取文本中的事件信息。现有的方法大多假设事件出现在没有重叠的句子中，不适用于复杂的重叠事件抽取。本研究系统地研究了现实事件重叠问题，即一个词可以作为具有多种类型或不同角色的触发器。为了解决上述问题，我们提出了一种新的基于级联解码的重叠事件提取联合学习框架，称为CasEE。特别是，CasEE依次执行类型检测、触发器提取和参数提取，其中重叠的目标根据特定的前一个预测分别提取。所有子任务在一个框架中共同学习，以捕获子任务之间的依赖关系。对公共事件提取基准FewFC的评估表明，与以前的竞争方法相比，CasEE在重叠事件提取方面取得了显著改进。



数据集： FewFC
</p></blockquote></details>



#### 2020
 <details/>
<summary/>
  <a href="https://arxiv.org/abs/1912.01586">Reading the Manual: Event Extraction as Definition Comprehension, EMNLP 2020</a> by <i> Yunmo Chen, Tongfei Chen, Seth Ebner, Benjamin Van Durme.
</summary><blockquote><p align="justify">
动机：提出一种新颖的事件抽取方法，为模型提供带有漂白语句（实体用通用的方式指代）的模型。漂白语句是指基于注释准则、描述事件发生的通常情况的机器可读的自然语言句子。实验结果表明，模型能够提取封闭本体下的事件，并且只需阅读新的漂白语句即可将其推广到未知的事件类型。
 
  ![image](https://github.com/xiaoqian19940510/Event-Extraction/blob/master/figures/1.png)
  
主要思想：提出了一种新的事件抽取方法，该方法考虑了通过将文本中的实体用指代的方式表示，如人用someone表示，以这种方式构造语料库；提出了一个多跨度的选择模型，该模型演示了事件抽取方法的可行性以及零样本或少样本设置的可行性。

数据集：ACE 2005

</p></blockquote></details>



 <details/>
<summary/>
  <a href="http://arxiv.org/abs/1912.11334">Open-domain Event Extraction and Embedding for Natural Gas Market Prediction, arxiv 2020 </a> by <i> Chau, Minh Triet and Esteves, Diego and Lehmann, Jens
(<a href="https://github.com/minhtriet/gas_market">Github</a>)</summary><blockquote><p align="justify">
动机：以前的方法大多数都将价格视为可推断的时间序列，那些分析价格和新闻之间的关系的方法是根据公共新闻数据集相应地修正其价格数据、手动注释标题或使用现成的工具。与现成的工具相比，我们的事件抽取方法不仅可以检测现象的发生，还可以由公共来源检测变化的归因和特征。
 
  ![image](https://github.com/xiaoqian19940510/Event-Extraction/blob/master/figures/2-1.png)

  ![image](https://github.com/xiaoqian19940510/Event-Extraction/blob/master/figures/2-2.png)
  
主要思想：依靠公共新闻API的标题，我们提出一种方法来过滤不相关的标题并初步进行事件抽取。价格和文本均被反馈到3D卷积神经网络，以学习事件与市场动向之间的相关性。

数据集：NYTf、FT、TG
</p></blockquote></details>




 <details/>
<summary/>
  <a href="https://arxiv.org/abs/2004.13625">Event Extraction by Answering (Almost) Natural Questions, EMNLP 2020 </a> by <i> Xinya Du and Claire Cardie(<a href="https://github.com/xinyadu/eeqa">Github</a>)</summary><blockquote><p align="justify">
  
主要思想：事件提取问题需要检测事件触发并提取其相应的参数。
事件参数提取中的现有工作通常严重依赖于作为预处理/并发步骤的实体识别，这导致了众所周知的错误传播问题。
为了避免这个问题，我们引入了一种新的事件抽取范式，将其形式化为问答(QA)任务，该任务以端到端的方式提取事件论元。
实证结果表明，我们的框架优于现有的方法;
此外，它还能够提取训练时未见角色的事件论元。

  ![image](https://github.com/xiaoqian19940510/Event-Extraction/blob/master/figures/0.png)

数据集：ACE
</p></blockquote></details>





#### 2019


 <details/>
<summary/>
  <a href="https://doi.org/10.18653/v1/p19-1522" >Exploring Pre-trained Language Models for Event Extraction and Generation, ACL 2019</a> by <i> Yang, Sen and Feng, Dawei and Qiao, Linbo and Kan, Zhigang and Li, Dongsheng
</summary><blockquote><p align="justify">
 
动机：
ACE事件抽取任务的传统方法通常依赖被手工标注过的数据，但是手工标注数据非常耗费精力并且也限制了数据集的规模。我们提出了一个方法来克服这个问题。

 ![image](https://github.com/xiaoqian19940510/Event-Extraction/blob/master/figures/16-1.png)
  ![image](https://github.com/xiaoqian19940510/Event-Extraction/blob/master/figures/16.png)
  
 
主要思想：
本文提出了一个基于预训练语言模型的框架，该框架包含一个作为基础的事件抽取模型以及一种生成被标注事件的方法。我们提出的事件抽取模型由触发词抽取器和论元抽取器组成，论元抽取器用前者的结果进行推理。此外，我们根据角色的重要性对损失函数重新进行加权，从而提高了论元抽取器的性能。

数据集：ACE2005

Keywords: Context-aware word representation, LSTM, Tensor layer

</p></blockquote></details>


 <details/>
<summary/>
  <a href="https://doi.org/10.18653/v1/p19-1276" >Open Domain Event Extraction Using Neural Latent Variable Models, ACL2019</a> by <i> Xiao Liu and Heyan Huang and Yue Zhang
(<a href="https://github.com/lx865712528/ACL2019-ODEE">Github</a>)</summary><blockquote><p align="justify">
 
动机：
我们考虑开放域的事件抽取，即从新闻集群中抽取无约束的事件类型的任务。结果表明，与最新的事件模式归纳方法相比，这种无监督模型具有更好的性能。

 ![image](https://github.com/xiaoqian19940510/Event-Extraction/blob/master/figures/15-1.png)
 
  ![image](https://github.com/xiaoqian19940510/Event-Extraction/blob/master/figures/15.png)

主要思想：
以前关于生成模式归纳的研究非常依赖人工生成的指标特征，而我们引入了由神经网络产生的潜在变量来获得更好的表示能力。我们设计了一种新颖的图形模型，该模型具有潜在的事件类型矢量以及实体的文本冗余特征，而这些潜在的事件类型矢量来自全局参数化正态分布的新闻聚类。

数据集：GNBusiness

</p></blockquote></details>


 <details/>
<summary/>
  <a href="https://doi.org/10.18653/v1/p19-3006" >Rapid Customization for Event Extraction, ACL 2019</a> by <i> Yee Seng Chan, Joshua Fasching, Haoling Qiu, Bonan Min
(<a href="https://github.com/BBN-E/Rapid-customization-events-acl19">Github</a>)</summary><blockquote><p align="justify">
 
动机：
从文本中获取事件发生的时间、地点、人物以及具体做了什么是很多应用程序（例如网页搜索和问题解答）的核心信息抽取任务之一。本文定义了一种快速自定义事件抽取功能的系统，用于查找新的事件类型以及他们的论元。

 ![image](https://github.com/xiaoqian19940510/Event-Extraction/blob/master/figures/14.png)

主要思想：
为了能够抽取新类型的事件，我们提出了一种新颖的方法：让用户通过探索无标注的语料库来查找，扩展和过滤事件触发词。然后，系统将自动生成相应级别的事件标注，并训练神经网络模型以查找相应事件。

数据集：ACE2005

</p></blockquote></details>


 <details/>
<summary/>
  <a href="https://www.aclweb.org/anthology/D19-1030/" >Cross-lingual Structure Transfer for Relation and Event Extraction, EMNLP 2019</a> by <i> Ananya Subburathinam, Di Lu, Heng Ji, Jonathan May, Shih-Fu Chang, Avirup Sil, Clare Voss
</summary><blockquote><p align="justify">
 
动机：
从资源不足以及标注不足的语料库中进行复杂语义结构的识别（例如事件和实体关系）是很困难的，这已经变成了一个很有挑战性的信息抽取任务。

 ![image](https://github.com/xiaoqian19940510/Event-Extraction/blob/master/figures/13.png)

主要思想：
通过使用卷积神经网络，将所有实体信息片段、事件触发词、事件背景放入一个复杂的、结构化的多语言公共空间，然后我们可以从源语言注释中训练一个事件抽取器，并将它应用于目标语言。

数据集：ACE2005

</p></blockquote></details>



 <details/>
<summary/>
   <a href="https://www.aclweb.org/anthology/D19-1032/" >Doc2EDAG: An End-to-End Document-level Framework for Chinese Financial Event Extraction, EMNLP2019</a> by <i> Shun Zheng, Wei Cao, Wei Xu, Jiang Bian
</summary><blockquote><p align="justify">
 
任务:与其他研究不同，该任务被定义为：事件框架填充：也就是论元检测+识别
 
不同点有：不需要触发词检测;文档级的抽取;论元有重叠

动机:解码论元需要一定顺序，先后有关

 ![image](https://github.com/xiaoqian19940510/Event-Extraction/blob/master/figures/12.png)
 
主要思想:发布数据集，具有特性：arguments-scattering and multi-event,先对事件是否触发进行预测；然后，按照一定顺序先后来分别解码论元

数据集:ten years (2008-2018) Chinese financial announcements：ChFinAnn;Crawling from http://www.cninfo.com.cn/new/index
</p></blockquote></details>


 <details/>
<summary/>
  <a href="https://www.aclweb.org/anthology/D19-1585" >Entity, Relation, and Event Extraction with Contextualized Span Representations, CCL 2016</a> by <i> David Wadden, Ulme Wennberg, Yi Luan, Hannaneh Hajishirzi
(<a href="https://github.com/dwadden/dygiepp">Github</a>)</summary><blockquote><p align="justify">
 
动机：
许多信息提取任务（例如命名实体识别，关系抽取，事件抽取和共指消解）都可以从跨句子的全局上下文或无局部依赖性的短语中获益。

 ![image](https://github.com/xiaoqian19940510/Event-Extraction/blob/master/figures/11.png)
 
主要思想：
（1）将事件抽取作为附加任务执行，并在事件触发词与其论元的关系图形中进行跨度更新。
（2）在多句子BERT编码的基础上构建跨度表示形式。

数据集：ACE2005

</p></blockquote></details>



 <details/>
<summary/>
  <a href="https://www.aclweb.org/anthology/D19-1584/">HMEAE: Hierarchical Modular Event Argument Extraction, EMNLP 2019 short(<a href="https://github.com/thunlp/HMEAE">Github</a>)</summary><blockquote><p align="justify">
任务:事件角色分类

 ![image](https://github.com/xiaoqian19940510/Event-Extraction/blob/master/figures/10-1.png)
 
  ![image](https://github.com/xiaoqian19940510/Event-Extraction/blob/master/figures/10.png)

动机:论元的类型（如PERSON）会给论元之间的关联带来影响

数据集:ACE 2005
</p></blockquote></details>



 <details/>
<summary/>
  <a href="https://www.aclweb.org/anthology/D19-1041/" >Joint Event and Temporal Relation Extraction with Shared Representations and Structured Prediction, EMNLP 2019</a> by <i> Rujun Han, Qiang Ning, Nanyun Peng
</summary><blockquote><p align="justify">
 
动机：
事件之间的时序关系的提取是一项重要的自然语言理解（NLU）任务，可以使许多下游任务受益。我们提出了一种事件和事件时序关系的联合抽取模型，该模型可以进行共享表示学习和结构化预测。

 ![image](https://github.com/xiaoqian19940510/Event-Extraction/blob/master/figures/9.png)
 
  ![image](https://github.com/xiaoqian19940510/Event-Extraction/blob/master/figures/9-2.png)
 
主要思想：
（1）提出了一个同时进行事件和事件时序关系抽取的联合模型。这样做的好处是：如果我们使用非事件之间的NONE关系训练关系分类器，则它可能具有修正事件抽取错误的能力。
（2）通过在事件抽取和时序关系抽取模块之间首次共享相同的上下文嵌入和神经表示学习器来改进事件的表示。

数据集：TB-Dense and MATRES datasets

</p></blockquote></details>


 <details/>
<summary/>
  <a href="https://www.aclweb.org/anthology/D19-1027/" >Open Event Extraction from Online Text using a Generative Adversarial Network, EMNLP 2019</a> by <i> Rui Wang, Deyu Zhou, Yulan He
</summary><blockquote><p align="justify">
 
动机：
提取开放域事件的结构化表示的方法通常假定文档中的所有单词都是从单个事件中生成的，因此他们通常不适用于诸如新闻文章之类的长文本。为了解决这些局限性，我们提出了一种基于生成对抗网络的事件抽取模型，称为对抗神经事件模型（AEM）。

 ![image](https://github.com/xiaoqian19940510/Event-Extraction/blob/master/figures/8.png)


主要思想：
AEM使用Dirichlet先验对事件建模，并使用生成器网络来捕获潜在事件的模式，鉴别器用于区分原始文档和从潜在事件中重建的文档，鉴别器网络生成的特征允许事件抽取的可视化。

数据集：Twitter, and Google datasets

</p></blockquote></details>


 <details/>
<summary/>
  <a href="https://arxiv.org/pdf/1909.02126.pdf" >Reporting the unreported: Event Extraction for Analyzing the Local Representation of Hate Crimes, EMNLP 2019</a> by <i> Aida Mostafazadeh Davani etal.
(<a href="https://github.com/aiida-/HateCrime">Github</a>)</summary><blockquote><p align="justify">
 
动机：
将事件抽取和多实例学习应用于本地新闻文章的语料库，可以用来预测仇恨犯罪的发生。

 ![image](https://github.com/xiaoqian19940510/Event-Extraction/blob/master/figures/7.png)

主要思想：
根据是否为仇恨罪标记每篇文章的任务被定义为多实例学习（MIL）问题。我们通过使用文章所有句子中嵌入的信息来确定文章是否报道了仇恨犯罪。在一组带标注的文章上测试了模型之后，我们将被训练过的模型应用于联邦调查局没有报道过的城市，并对这些城市中仇恨犯罪的发生频率进行了下界估计。

</p></blockquote></details>


 <details/>
<summary/>
  <a href="https://www.ijcai.org/proceedings/2019/753" >Extracting entities and events as a single task using a transition-based neural model, IJCAI 2019</a> by <i> Zhang, Junchi and Qin, Yanxia and Zhang, Yue and Liu, Mengchi and Ji, Donghong
(<a href="https://github.com/zjcerwin/TransitionEvent">Github</a>)</summary><blockquote><p align="justify">
 
动机：
事件抽取任务包括许多子任务：实体抽取，事件触发词抽取，论元角色抽取。传统的方法是使用pipeline的方式解决这些任务，没有利用到任务间相互关联的信息。已有一些联合学习的模型对这些任务进行处理，然而由于技术上的挑战，还没有模型将其看作一个单一的任务，预测联合的输出结构。本文提出了一个transition-based的神经网络框架，以state-transition的过程，递进地预测复杂的联合结构。

 ![image](https://github.com/xiaoqian19940510/Event-Extraction/blob/master/figures/6.png)

主要思想：
使用transition-based的框架，通过使用递增的output-building行为的state-transition过程，构建一个复杂的输出结构。在本文中我们设计了一个transition系统以解决事件抽取问题，从左至右递增地构建出结构，不使用可分的子任务结构。本文还是第一个使transition-based模型，并将之用于实体和事件的联合抽取任务的研究。模型实现了对3个子任务完全的联合解码，实现了更好的信息组合。

数据集：ACE2005

Keywords: Context-aware word representation, LSTM, Tensor layer

</p></blockquote></details>


 <details/>
<summary/>
  <a href="https://www.aclweb.org/anthology/N19-1145/" >Biomedical Event Extraction based on Knowledge-driven Tree-LSTM, CCL 2016</a> by <i> Diya Li, Lifu Huang, Heng Ji, Jiawei Han
</summary><blockquote><p align="justify">
 
动机：
生物医学领域的事件抽取比一般新闻领域的事件抽取更具挑战性，因为它需要更广泛地获取领域特定的知识并加深对复杂情境的理解。为了更好地对上下文信息和外部背景知识进行编码，我们提出了一种新颖的知识库（KB）驱动的树结构长短期记忆网络（Tree-LSTM）框架。

 ![image](https://github.com/xiaoqian19940510/Event-Extraction/blob/master/figures/5-1.png)
 
  ![image](https://github.com/xiaoqian19940510/Event-Extraction/blob/master/figures/5.png)

主要思想：
该框架合并了两种功能：（1）抓取上下文背景的依赖结构（2）通过实体链接从外部本体获得实体属性（类型和类别描述）。

数据集：Genia dataset

Keywords: Knowledge-driven Tree-LSTM

</p></blockquote></details>


 <details/>
<summary/>
 <a href="https://ieeexplore.ieee.org/document/8643786" >Joint Event Extraction Based on Hierarchical Event Schemas From FrameNet, EMNLP 2019 short</a> by <i> Wei Li , Dezhi Cheng, Lei He, Yuanzhuo Wang, Xiaolong Jin
</summary><blockquote><p align="justify">
 
动机：事件抽取对于许多实际应用非常有用，例如新闻摘要和信息检索。但是目前很流行的ACE事件抽取仅定义了非常有限且粗糙的事件模式，这可能不适合实际应用。 FrameNet是一种语言语料库，它定义了完整的语义框架和框架间的关系。由于FrameNet中的框架与ACE中的事件架构共享高度相似的结构，并且许多框架实际上表达了事件，因此，我们建议基于FrameNet重新定义事件架构。

 ![image](https://github.com/xiaoqian19940510/Event-Extraction/blob/master/figures/4-1.png)
 
 ![image](https://github.com/xiaoqian19940510/Event-Extraction/blob/master/figures/4.png)
  
主要思想：（1）提取FrameNet中表示事件的所有框架，并利用框架与框架之间的关系建立事件模式的层次结构。（2）适当利用全局信息（例如事件间关系）和事件抽取必不可少的局部特征（例如词性标签和依赖项标签）。基于一种利用事件抽取结果的多文档摘要无监督抽取方法，我们使用了一种图排序方法。

数据集：ACE 2005，FrameNet 1.7 corpus
</p></blockquote></details>


 <details/>
<summary/>
  <a >One for All: Neural Joint Modeling of Entities and Events, AAAI 2019</a> by <i> Trung Minh Nguyen∗ Alt Inc.
</summary><blockquote><p align="justify">
 
事件抽取之前的工作主要关注于对事件触发器和论元角色的预测，将实体提及视为由人工标注提供的。
这是不现实的，因为实体提及通常是由一些现有工具包预测的，它们的错误可能会传播到事件触发器和论元角色识别。
最近很少有研究通过联合预测实体提及、事件触发器和论元来解决这个问题。
然而，这种工作仅限于使用离散的工程特征来表示单个任务及其交互的上下文信息。
在这项工作中，我们提出了一个基于共享的隐层表示的新的模型来联合执行实体提及，事件触发和论元的预测。
实验证明了该方法的优点，实现了最先进性能的事件抽取。

 ![image](https://github.com/xiaoqian19940510/Event-Extraction/blob/master/figures/3.png)


数据集：ACE 2005 

</p></blockquote></details>




#### 2018

 <details/>
<summary/>
  <a href="https://arxiv.org/pdf/1712.03665.pdf" >Scale up event extraction learning via automatic training data generation, AAAI 2018</a> by <i> Zeng, Ying and Feng, Yansong and Ma, Rong and Wang, Zheng and Yan, Rui and Shi, Chongde and Zhao, Dongyan
</summary><blockquote><p align="justify">
 
动机：现有的训练数据必须通过专业领域知识以及大量的参与者来手动生成，这样生成的数据规模很小，严重影响训练出来的模型的质量。因此我们开发了一种自动生成事件抽取训练数据的方法。

 ![image](https://github.com/xiaoqian19940510/Event-Extraction/blob/master/figures/25.png)

主要思想：我们提出了一种基于神经网络和线性规划的事件抽取框架，该模型不依赖显式触发器，而是使用一组关键论元来表征事件类型。这样就不需要明确识别事件的触发因素，进而降低了人力参与的需求。

数据集：Wikipedia article

Keywords: Data Generation, Distant Supervision

</p></blockquote></details>


<details/>
<summary/>
  <a href="https://www.aclweb.org/anthology/P18-4009" >DCFFE: A Document-level Chinese Financial Event Extraction System based on Automatically Labelled Training Data, ACL 2018 </a> by <i> Yang, Hang and Chen, Yubo and Liu, Kang and Xiao, Yang and Zhao, Jun
</summary><blockquote><p align="justify">
 
动机：在某些特定领域，例如金融，医疗和司法领域，由于数据标注过程的成本很高，因此标注数据不足。此外，当前大多数方法都关注于从一个句子中提取事件，但通常在一个文档中，一个事件由多个句子表示。我们提出一种方法来解决这个问题。

 
主要思想：我们提供了一个名为DCFEE的框架，该框架可以从被自动标注过的训练数据中抽取文档级事件。我们使用一个序列标注模型来自动抽取句子级事件，并且提出了一个关键事件检测模型和一个论元填充策略，进而从文档中提取整个事件。

数据集：Chinese financial event dataset

Keywords: Automatically Labelled, Chinese Financial EE

</p></blockquote></details>


<details/>
<summary/>
  <a href="https://shalei120.github.io/docs/sha2018Joint.pdf" >Jointly Extraction Event Triggers and Arguments by Dependency-Bridge RNN and Tensor-Based Argument Interaction, AAAI 2018 </a> by <i> Sha, Lei and Qian, Feng and Chang, Baobao and Sui, Zhifang
</summary><blockquote><p align="justify">
 
动机：传统的事件抽取很大程度上依赖词汇和句法特征，需要大量的人工工程，并且模型通用性不强。另一方面，深度神经网络可以自动学习底层特征，但是现有的网络却没有充分利用句法关系。因此本文在对每个单词建模时，使用依赖桥来增强它的信息表示。说明在RNN模型中同时应用树结构和序列结构比只使用顺序RNN具有更好的性能。另外，利用张量层来同时捕获论元之间的关系以及其在事件中的角色。实验表明，模型取得了很好地效果。

 ![image](https://github.com/xiaoqian19940510/Event-Extraction/blob/master/figures/23-1.png)
 
  ![image](https://github.com/xiaoqian19940510/Event-Extraction/blob/master/figures/23.png)

主要思想：
（1）实现了事件触发词以及论元的联合抽取，避开了Pipeline方法中错误的触发词识别结果会在网络中传播的问题；同时联合抽取的过程中，有可能通过元素抽取的步骤反过来纠正事件检测的结果。
（2）将元素的互信息作为影响元素抽取结果的因素。
（3）在构建模型的过程中使用了句法信息。

数据集：ACE2005

Keywords: dbRNN

</p></blockquote></details>

 <details/>
<summary/>
  <a href="https://www.aclweb.org/anthology/P18-1201" >Zero-Shot Transfer Learning for Event Extraction, ACL2018</a> by <i> Lifu Huang, Heng Ji, Kyunghyun Cho, Ido Dagan, Sebastian Riedel, Clare R. Voss
(<a href="https://github.com/wilburOne/ZeroShotEvent">Github</a>)</summary><blockquote><p align="justify">
动机：以前大多数受监督的事件抽取方法都依赖手工标注派生的特征，因此，如果没有额外的标注工作，这些方法便无法应对于新的事件类型。我们设计了一个新的框架来解决这个问题。
 
 ![image](https://github.com/xiaoqian19940510/Event-Extraction/blob/master/figures/22.png)
 
主要思想：每个事件都有由候选触发词和论元组成的结构，同时这个结构具有和事件类型及论元相一致的预定义的名字和标签。 我们增加了事件类型以及事件信息片段的语义代表( semantic representations)，并根据目标本体中定义的事件类型和事件信息片段的语义相似性来决定事件的类型

数据集：ACE2005

Keywords: Zero-Shot Transfer

</p></blockquote></details>


 <details/>
<summary/>
  <a href="https://www.aclweb.org/anthology/P18-4009" >DCFFE: A Document-level Chinese Financial Event Extraction System based on Automatically Labelled Training Data, ACL 2018 </a> by <i> Yang, Hang and Chen, Yubo and Liu, Kang and Xiao, Yang and Zhao, Jun
</summary><blockquote><p align="justify">
 
动机：在某些特定领域，例如金融，医疗和司法领域，由于数据标记过程的成本很高，因此标记数据不足。此外，当前大多数方法都关注于从一个句子中提取事件，但通常在一个文档中，一个事件由多个句子表示。我们提出一种方法来解决这个问题。

 ![image](https://github.com/xiaoqian19940510/Event-Extraction/blob/master/figures/21-1.png)
 
 ![image](https://github.com/xiaoqian19940510/Event-Extraction/blob/master/figures/21-2.png)
  
 ![image](https://github.com/xiaoqian19940510/Event-Extraction/blob/master/figures/21-3.png)

主要思想：我们提供了一个名为DCFEE的框架，该框架可以从被自动标记过的训练数据中抽取文档级事件。我们使用一个序列标记模型来自动抽取句子级事件，并且提出了一个关键事件检测模型和一个论元填充策略，进而从文档中提取整个事件。

数据集：Chinese financial event dataset

Keywords: Automatically Labelled, Chinese Financial EE

</p></blockquote></details>


 <details/>
<summary/>
  <a href="https://blender.cs.illinois.edu/paper/imitation2019.pdf" >Joint Entity and Event Extraction with Generative Adversarial Imitation Learning, CCL 2016 </a> by <i> Tongtao Zhang and Heng Ji and Avirup Sil
</summary><blockquote><p align="justify">
 
动机:我们提出了一种基于生成对抗的模仿学习的实体与事件抽取框架，这种学习是一种使用生成对抗网络（GAN）的逆强化学习方法。该框架的实际表现优于目前最先进的方法。

 ![image](https://github.com/xiaoqian19940510/Event-Extraction/blob/master/figures/20-1.png)
 
 ![image](https://github.com/xiaoqian19940510/Event-Extraction/blob/master/figures/20-2.png)
 
主要思想：在本文中，我们提出了一种动态机制——逆强化学习，直接评估实体和事件抽取中实例的正确和错误标签。 我们为案例分配明确的分数，或者根据强化学习（RL）给予奖励，并采用来自生成对抗网络（GAN）的鉴别器来估计奖励价值。

</p></blockquote></details>


 <details/>
<summary/>
  <a href="https://www.aclweb.org/anthology/D18-1156" >Joint Multiple Event Extraction via Attention-based Graph Information Aggregration, EMNLP 2018 </a> by <i> Liu, Xiao and Luo, Zhunchen and Huang, Heyan
(<a href="https://github.com/lx865712528/EMNLP2018-JMEE/">Github</a>)</summary><blockquote><p align="justify">
 
动机：比抽取单个事件更困难。在以往的工作中，由于捕获远距离的依赖关系效率很低，因此通过顺序建模的方法在对事件之间的联系进行建模很难成功。本文提出了一种新的框架来解决这个问题。

 ![image](https://github.com/xiaoqian19940510/Event-Extraction/blob/master/figures/19-1.png)
 
  ![image](https://github.com/xiaoqian19940510/Event-Extraction/blob/master/figures/19.png)
 
主要思想：本文提出JMEE模型（Jointly Multiple Events Extraction），面向的应用是从一个句子中抽取出多个事件触发器和参数（arguments）。JMEE模型引入了syntactic shortcut arcs来增强信息流并且使用基于attention的GCN建模图数据。实验结果表明本文的方法和目前最顶级的方法相比，有着可以媲美的效果。

数据集：ACE2005

Keywords: JMEE

</p></blockquote></details>


 <details/>
<summary/>
  <a href="https://www.aclweb.org/anthology/N18-2058/" >Semi-supervised event extraction with paraphrase clusters, NAACL 2018</a> by <i> Ferguson, James and Lockard, Colin and Weld, Daniel and Hajishirzi, Hannaneh
</summary><blockquote><p align="justify">
 
动机：
受监督的事件抽取系统由于缺乏可用的训练数据而其准确性受到限制。我们提出了一种通过对额外的训练数据进行重复抽样来使事件抽取系统自我训练的方法。这种方法避免了训练数据缺乏导致的问题。

 
主要思想：
我们通过详细的事件描述自动生成被标记过的训练数据，然后用这些数据进行事件触发词识别。具体来说，首先，将提及该事件的片段聚集在一起，形成一个聚类。然后用每个聚类中的简单示例来给整个聚类贴一个标签。最后，我们将新示例与原始训练集结合在一起，重新训练事件抽取器。


数据集：ACE2005, TAC-KBP 2015

Keywords: Semi-supervised

</p></blockquote></details>



 <details/>
<summary/>
  <a href="https://www.cips-cl.org/static/anthology/CCL-2016/CCL-16-081.pdf" >Jointly multiple events extraction via attention-based graph information aggregation, EMNLP 2018 </a> by <i> Xiao Liu, Zhunchen Luo‡ and Heyan Huang
</summary><blockquote><p align="justify">
 
任务:
触发词分类；论元分类

动机:
论元的语法依存关系

 ![image](https://github.com/xiaoqian19940510/Event-Extraction/blob/master/figures/17-1.png)
 
 ![image](https://github.com/xiaoqian19940510/Event-Extraction/blob/master/figures/17.png)

主要思想：
用GCN增强论元之间的依存关系
用自注意力来增强触发词分类的效果

数据集：ACE2005

</p></blockquote></details>



#### 2017
 <details/>
<summary/>
  <a href="https://www.aclweb.org/anthology/P17-1038" >Automatically Labeled Data Generation for Large Scale Event Extraction, ACL 2017 </a> by <i> Chen, Yubo and Liu, Shulin and Zhang, Xiang and Liu, Kang and Zhao, Jun
(<a href="https://github.com/acl2017submission/event-data">Github</a>)</summary><blockquote><p align="justify">
 
动机：手动标记的训练数据成本太高，事件类型覆盖率低且规模有限，这种监督的方法很难从知识库中抽取大量事件。

 ![image](https://github.com/xiaoqian19940510/Event-Extraction/blob/master/figures/26.png)

主要思想：1）提出了一种按重要性排列论元并且为每种事件类型选取关键论元或代表论元方法。2）仅仅使用关键论元来标记事件，并找出关键词。3）用外部语言知识库FrameNet来过滤噪声触发词并且扩展触发词库。

数据集：ACE2005

Keywords: Data Generation, Distant Supervision

</p></blockquote></details>




#### 2016
 <details/>
<summary/>
  <a href="https://www.aclweb.org/anthology/P16-1116" >RBPB Regularization Based Pattern Balancing Method for Event Extraction,ACL2016 </a> by <i> Sha, Lei and Liu, Jing and Lin, Chin-Yew and Li, Sujian and Chang, Baobao and Sui, Zhifang
</summary><blockquote><p align="justify">
动机：在最近的工作中，当确定事件类型（触发器分类）时，大多数方法要么是仅基于模式（pattern），要么是仅基于特征。此外，以往的工作在识别和文类论元的时候，忽略了论元之间的关系，只是孤立的考虑每个候选论元。
 
 ![image](https://github.com/xiaoqian19940510/Event-Extraction/blob/master/figures/32.png)
 
主要思想：在本文中，我们同时使用‘模式’和‘特征’来识别和分类‘事件触发器’。 此外，我们使用正则化方法对候选自变量之间的关系进行建模，以提高自变量识别的性能。 我们的方法称为基于正则化的模式平衡方法。

数据集：ACE2005

Keywords: Embedding & Pattern features, Regularization method

</p></blockquote></details>


 <details/>
<summary/>
  <a href="https://www.aclweb.org/anthology/C16-1114" >Leveraging Multilingual Training for Limited Resource Event Extraction, COLING 2016 </a> by <i> Hsi, Andrew and Yang, Yiming and Carbonell, Jaime and Xu, Ruochen
</summary><blockquote><p align="justify">
 
动机：迄今为止，利用跨语言培训来提高性能的工作非常有限。因此我们提出了一种新的事件抽取方法。

 ![image](https://github.com/xiaoqian19940510/Event-Extraction/blob/master/figures/31.png)

主要思想：在本文中，我们提出了一种新颖的跨语言事件抽取方法，该方法可在多种语言上进行训练，并利用依赖于语言的特征和不依赖于语言的特征来提高性能。使用这种系统，我们旨在同时利用可用的多语言资源（带注释的数据和引入的特征）来克服目标语言中的注释稀缺性问题。 从经验上我们认为，我们的方法可以极大地提高单语系统对中文事件论元提取任务的性能。 与现有工作相比，我们的方法是新颖的，我们不依赖于使用高质量的机器翻译的或手动对齐的文档，这因为这种需求对于给定的目标语言可能是无法满足的。

数据集：ACE2005

Keywords: Training on multiple languages using a combination of both language-dependent and language-independent features

</p></blockquote></details>



 <details/>
<summary/>
  <a href="https://www.cips-cl.org/static/anthology/CCL-2016/CCL-16-081.pdf" >Event Extraction via Bidirectional Long Short-Term Memory Tensor Neural Network, CCL 2016 </a> by <i> Chen, Yubo and Liu, Shulin and He, Shizhu and Liu, Kang and Zhao, Jun
</summary><blockquote><p align="justify">
 
动机：


主要思想：

数据集：ACE2005

Keywords: Context-aware word representation, LSTM, Tensor layer

</p></blockquote></details>


 <details/>
<summary/>
  <a href="https://www.aclweb.org/anthology/P18-1145.pdf" >A convolution bilstm neural network model for chinese event extraction, NLPCC 2016 </a> by <i> Lin, Hongyu and Lu, Yaojie and Han, Xianpei and Sun, Le
(<a href="https://github.com/sanmusunrise/NPNs">Github</a>)</summary><blockquote><p align="justify">
 
动机：在中文的事件抽取中，以前的方法非常依赖复杂的特征工程以及复杂的自然语言处理工具。本文提出了一种卷积双向LSTM神经网络，该神经网络将LSTM和CNN结合起来，可以捕获句子级和词汇信息，而无需任何人为提供的特征。

 ![image](https://github.com/xiaoqian19940510/Event-Extraction/blob/master/figures/30-1.png)

 ![image](https://github.com/xiaoqian19940510/Event-Extraction/blob/master/figures/30.png)

主要思想：首先使用双向LSTM将整个句子中的单词的语义编码为句子级特征，不做任何句法分析。然后，我们利用卷积神经网络来捕获突出的局部词法特征来消除触发器的歧义，整个过程无需来自POS标签或NER的任何帮助。

数据集：ACE2005， KBP2017 Corpus

</p></blockquote></details>



 <details/>
<summary/>
  <a href="https://www.aclweb.org/anthology/P16-1025/" >Liberal Event Extraction and Event Schema Induction, AACL 2016 </a> by <i> Lifu Huang, Taylor Cassidy, Xiaocheng Feng, Heng Ji, Clare R. Voss, Jiawei Han, Avirup Sil
</summary><blockquote><p align="justify">
 
动机：结合了象征式的（例如抽象含义表示）和分布式的语义来检测和表示事件结构，并采用同一个类型框架来同时提取事件类型和论元角色并发现事件模式。这种模式的提取性能可以与被预定义事件类型标记过的大量数据训练的监督模型相媲美。 

 ![image](https://github.com/xiaoqian19940510/Event-Extraction/blob/master/figures/29-1.png)
 
 ![image](https://github.com/xiaoqian19940510/Event-Extraction/blob/master/figures/29-2.png)

主要思想：我们试图将事件触发器和事件论元聚类，每个聚类代表一个事件类型。 我们将分布的相似性用于聚类的距离度量。分布假设指出，经常出现在相似语境中的单词往往具有相似的含义。

两个基本假设：1）出现在相似的背景中并且有相同作用的事件触发词往往具有相似的类型。2）除了特定事件触发器的词汇语义外，事件类型还取决于其论元和论元的作用，以及上下文中与触发器关联的其他单词。


数据集：ERE (Entity Relation Event)

</p></blockquote></details>




 <details/>
<summary/>
  <a href="https://www.aclweb.org/anthology/N16-1049/" >Joint Learning Templates and Slots for Event Schema Induction, NAACL 2016 </a> by <i> Lei Sha, Sujian Li, Baobao Chang, Zhifang Sui
(<a href="https://github.com/shenglih/normalized_cut/tree/master">Github</a>)</summary><blockquote><p align="justify">
 
动机：我们提出了一个联合实体驱动模型，这种模型可以根据同一句子中模板和各种信息槽（例如attribute slot和participate slot）的限制，同时学习模板和信息槽。这样的模型会得到比以前的方法更好的结果。

主要思想：为了更好地建立实体之间的内在联系的模型，我们借用图像分割中的标准化切割作为聚类标准。同时我们用模板之间的约束以及一个句子中的信息槽之间的约束来改善AESI结果。


数据集：MUC-4

</p></blockquote></details>


 <details/>
<summary/>
  <a href="https://www.aclweb.org/anthology/N16-1034" >Joint Event Extraction via Recurrent Neural Networks, NAACL 2016 </a> by <i> Chen, Yubo and Liu, Shulin and He, Shizhu and Liu, Kang and Zhao, Jun
(<a href="https://github.com/anoperson/jointEE-NN">Github</a>)</summary><blockquote><p align="justify">
 
任务:给定实体标签；通过序列标注识别触发词和论元

动机:论元之间有着相关关系，某些论元已经识别出来可能会导致一些论元共现,RNN减少错误传播

 ![image](https://github.com/xiaoqian19940510/Event-Extraction/blob/master/figures/27.png)

主要思想:使用RNN来标注要素，通过记忆矩阵来增强要素之间的关联。

数据集：ACE2005

Keywords: RNN, Joint Event Extraction

</p></blockquote></details>




#### 2015
 <details/>
<summary/>
  <a href="https://www.aclweb.org/anthology/P15-1017" >Event Extraction via Dynamic Multi-Pooling Convolutional Neural Networks, ACL2015 </a> by <i> Yubo Chen, Liheng Xu, Kang Liu, Daojian Zeng and Jun Zhao 
</summary><blockquote><p align="justify">
任务：给定候选实体的位置；完成触发词识别，触发词分类，论元识别，论元分类
 
动机：在于一个句子中可能会有多个事件，如果只用一个池化将导致多个事件的句子级特征没有区别。因此引入动态多池化
 
 ![image](https://github.com/xiaoqian19940510/Event-Extraction/blob/master/figures/34.png)
 
主要思想：采用动态多池化的方式，以trigger和candidate作为分隔符[-trigger-candidate-]，将句子池化成三段；动机在于一个句子中可能会有多个事件，如果只用一个池化将导致多个事件的句子级特征没有区别。将任务目标转换成句子分类任务，从而完成任务。
 

数据集：ACE2005

keywords: DMCNN, CNN, Dynamic Multi-Pooling

</p></blockquote></details>



 <details/>
<summary/>
  <a href="https://www.aclweb.org/anthology/P15-1019/" >Generative Event Schema Induction with Entity Disambiguation, AACL2015 </a> by <i> Kiem-Hieu Nguyen, Xavier Tannier, Olivier Ferret, Romaric Besançon
</summary><blockquote><p align="justify">
动机：以往文献中的方法仅仅使用中心词来代表实体，然而除了中心词，别的元素也包含了很多重要的信息。这篇论文提出了一种事件模式归纳的生成模型来解决这个问题。
 
 ![image](https://github.com/xiaoqian19940510/Event-Extraction/blob/master/figures/33-1.png)
  
 ![image](https://github.com/xiaoqian19940510/Event-Extraction/blob/master/figures/33.png)
 
主要思想：模式归纳是指从没有被标记的文本中无监督的学习模板（一个模板定义了一个与实体的语义角色有关的特定事件的类型）。想法是：基于事件模板中相同角色对应的这些实体的相似性，将他们分组在一起。例如，在有关恐怖袭击的语料库中，可以将要被杀死，要被攻击的对象的实体组合在一起，并以名为VICTIM的角色为它们的特征。

数据集：ACE2005

keywords: DMCNN, CNN, Dynamic Multi-Pooling

</p></blockquote></details>






### Few-shot or zero-shot

#### 2020
 <details/>
<summary/>
  <a href="https://dl.acm.org/doi/10.1145/3336191.3371796">Meta-Learning with Dynamic-Memory-Based Prototypical Network for Few-Shot Event Detection, WSDM 2020</a> by <i> Shumin Deng, Ningyu Zhang, Jiaojian Kang, Yichi Zhang, Wei Zhang, Huajun Chen
</summary><blockquote><p align="justify">
 
事件检测(ED)是事件抽取的一个子任务，涉及到识别触发器和对提到的事件进行分类。
现有的方法主要依赖于监督学习，并需要大规模的标记事件数据集，不幸的是，这些数据集在许多实际应用中并不容易获得。
在这篇论文中，我们考虑并重新制定了一个有限标记数据的教育任务作为一个少样本的学习问题。
我们提出了一个基于动态记忆的原型网络(DMB-PN)，它利用动态记忆网络(DMN)不仅可以更好地学习事件类型的原型，还可以为提到事件生成更健壮的句子编码。
传统的原型网络只使用一次事件提及次数，通过平均计算事件原型，与之不同的是，由于DMNs的多跳机制，我们的模型更加健壮，能够从多次提及的事件中提取上下文信息。
实验表明，与一系列基线模型相比，DMB-PN不仅能更好地解决样本稀缺问题，而且在事件类型变化较大、实例数量极小的情况下性能更强。
 
</p></blockquote></details>

 <details/>
<summary/>
   <a href="https://arxiv.org/abs/2002.05295">Exploiting the Matching Information in the Support Set for Few Shot Event Classification, PAKDD 2020</a> by <i> 	Viet Dac Lai, Franck Dernoncourt, Thien Huu Nguyen
</summary><blockquote><p align="justify">
 现有的事件分类(EC)的工作主要集中在传统的监督学习设置，其中模型无法提取的事件提到新的/看不见的事件类型。
尽管EC模型能够将其操作扩展到未观察到的事件类型，但在这一领域还没有研究过少样本习。
为了填补这一空白，在本研究中，我们调查了在少样本学习设置下的事件分类。
针对这一问题，我们提出了一种新的训练方法，即在训练过程中扩展利用支持集。
特别地，除了将查询示例与用于训练的支持集中的示例进行匹配之外，我们还试图进一步匹配支持集中本身的示例。
该方法为模型提供了更多的训练信息，可应用于各种基于度量学习的少样本学习方法。
我们在两个EC基准数据集上的广泛实验表明，该方法可以提高事件分类准确率达10%
 
</p></blockquote></details>

 <details/>
<summary/>
   <a href="https://www.aclweb.org/anthology/2020.lrec-1.216/">Towards Few-Shot Event Mention Retrieval : An Evaluation Framework and A Siamese Network Approach, LREC 2020</a> by <i> Bonan Min, Yee Seng Chan, Lingjun Zhao
</summary><blockquote><p align="justify">

在大量的文本中自动分析事件对于情境意识和决策是至关重要的。
以前的方法将事件抽取视为“一刀切”，并预先定义了本体。
所建立的提取模型用于提取本体中的类型。
这些方法不能很容易地适应新的事件类型或感兴趣的新领域。
为了满足以事件为中心的个性化信息需求，本文引入了少样本事件提及检索(EMR)任务:给定一个由少量事件提及组成的用户提供的查询，返回在语料库中找到的相关事件提及。
这个公式支持“按例查询”，这大大降低了指定以事件为中心的信息需求的门槛。
检索设置还支持模糊搜索。
我们提供了一个利用现有事件数据集(如ACE)的评估框架。

</p></blockquote></details>



#### 2018
 <details/>
<summary/>
  <a href="https://www.aclweb.org/anthology/P18-1201/">Zero-Shot Transfer Learning for Event Extraction, ACL 2018</a> by <i> Lifu Huang, Heng Ji, Kyunghyun Cho, Ido Dagan, Sebastian Riedel, Clare R. Voss
</summary><blockquote><p align="justify">
 
 以前的大多数事件抽取研究都严重依赖于从标注的事件提及中衍生出来的特性，因此如果不进行注释就不能应用于新的事件类型。
在这项工作中，我们重新审视事件抽取，并将其建模为一个接地问题。
我们设计一个Transfer的神经结构,映射事件提及和类型共同到一个共享语义空间使用神经网络结构和组成,每个事件提及的类型可以由所有候选人的最亲密的类型。
通过利用一组现有事件类型可用的手工标注和现有事件本体，我们的框架应用于新的事件类型而不需要额外的标注。
在现有事件类型(如ACE、ERE)和新事件类型(如FrameNet)上的实验证明了我们的方法的有效性。
对于23种新的事件类型，我们的zero-shot框架实现了可以与最先进的监督模型相比较的性能，该模型是从500个事件提及的标注数据中训练出来的。
 
</p></blockquote></details>




### 中文事件抽取


#### 2019
 <details/>
<summary/>
   <a href="https://www.aclweb.org/anthology/D19-1032/" >Doc2EDAG: An End-to-End Document-level Framework for Chinese Financial Event Extraction, EMNLP2019 </a> by <i> Shun Zheng, Wei Cao, Wei Xu, Jiang Bian
</summary><blockquote><p align="justify">
 
任务:与其他研究不同，该任务被定义为：事件框架填充：也就是论元检测+识别
 
不同点有：不需要触发词检测;文档级的抽取;论元有重叠

动机:解码论元需要一定顺序，先后有关

主要思想:发布数据集，具有特性：arguments-scattering and multi-event,先对事件是否触发进行预测；然后，按照一定顺序先后来分别解码论元

数据集:ten years (2008-2018) Chinese financial announcements：ChFinAnn;Crawling from http://www.cninfo.com.cn/new/index
</p></blockquote></details>


 <details/>
<summary/>
  <a href="https://www.aclweb.org/anthology/D19-1030/" >Cross-lingual Structure Transfer for Relation and Event Extraction, EMNLP 2019) </a> by <i> Ananya Subburathinam, Di Lu, Heng Ji, Jonathan May, Shih-Fu Chang, Avirup Sil, Clare Voss
</summary><blockquote><p align="justify">
 
动机：
从资源不足以及注释不足的语料库中进行复杂语义结构的识别（例如事件和实体关系）是很困难的，这已经变成了一个很有挑战性的信息抽取任务。

 ![image](https://github.com/xiaoqian19940510/Event-Extraction/blob/master/figures/13.png)

主要思想：
通过使用卷积神经网络，将所有实体信息片段、事件触发词、事件背景放入一个复杂的、结构化的多语言公共空间，然后我们可以从源语言注释中训练一个事件抽取器，并将它应用于目标语言。

数据集：ACE2005

</p></blockquote></details>



#### 2018
<details/>
<summary/>
  <a href="https://www.aclweb.org/anthology/P18-4009" >DCFFE: A Document-level Chinese Financial Event Extraction System based on Automatically Labelled Training Data, ACL 2018 </a> by <i> Yang, Hang and Chen, Yubo and Liu, Kang and Xiao, Yang and Zhao, Jun
</summary><blockquote><p align="justify">
 
动机：在某些特定领域，例如金融，医疗和司法领域，由于数据标记过程的成本很高，因此标记数据不足。此外，当前大多数方法都关注于从一个句子中提取事件，但通常在一个文档中，一个事件由多个句子表示。我们提出一种方法来解决这个问题。

 ![image](https://github.com/xiaoqian19940510/Event-Extraction/blob/master/figures/21-1.png)
 
 ![image](https://github.com/xiaoqian19940510/Event-Extraction/blob/master/figures/21-2.png)
 
 ![image](https://github.com/xiaoqian19940510/Event-Extraction/blob/master/figures/21-3.png)
 
主要思想：我们提供了一个名为DCFEE的框架，该框架可以从被自动标记过的训练数据中抽取文档级事件。我们使用一个序列标记模型来自动抽取句子级事件，并且提出了一个关键事件检测模型和一个论元填充策略，进而从文档中提取整个事件。

数据集：Chinese financial event dataset

Keywords: Automatically Labelled, Chinese Financial EE

</p></blockquote></details>




#### 2016
 <details/>
<summary/>
  <a href="https://www.aclweb.org/anthology/P18-1145.pdf" >A convolution bilstm neural network model for chinese event extraction, NLPCC 2016 </a> by <i> Lin, Hongyu and Lu, Yaojie and Han, Xianpei and Sun, Le
(<a href="https://github.com/sanmusunrise/NPNs">Github</a>)</summary><blockquote><p align="justify">
 
动机：在中文的事件抽取中，以前的方法非常依赖复杂的特征工程以及复杂的自然语言处理工具。本文提出了一种卷积双向LSTM神经网络，该神经网络将LSTM和CNN结合起来，可以捕获句子级和词汇信息，而无需任何人为提供的特征。

 ![image](https://github.com/xiaoqian19940510/Event-Extraction/blob/master/figures/30-1.png)
 
 ![image](https://github.com/xiaoqian19940510/Event-Extraction/blob/master/figures/30.png)

主要思想：首先使用双向LSTM将整个句子中的单词的语义编码为句子级特征，不做任何句法分析。然后，我们利用卷积神经网络来捕获突出的局部词法特征来消除触发器的歧义，整个过程无需来自POS标签或NER的任何帮助。

数据集：ACE2005， KBP2017 Corpus

</p></blockquote></details>






### 半监督\远程监督事件抽取

#### 2018
 <details/>
<summary/>
  <a href="https://www.aclweb.org/anthology/N18-2058/" >Semi-supervised event extraction with paraphrase clusters, NAACL 2018</a> by <i> Ferguson, James and Lockard, Colin and Weld, Daniel and Hajishirzi, Hannaneh
</summary><blockquote><p align="justify">
 
动机：
受监督的事件抽取系统由于缺乏可用的训练数据而其准确性受到限制。我们提出了一种通过对额外的训练数据进行重复抽样来使事件抽取系统自我训练的方法。这种方法避免了训练数据缺乏导致的问题。

主要思想：
我们通过详细的事件描述自动生成被标记过的训练数据，然后用这些数据进行事件触发词识别。具体来说，首先，将提及该事件的片段聚集在一起，形成一个聚类。然后用每个聚类中的简单示例来给整个聚类贴一个标签。最后，我们将新示例与原始训练集结合在一起，重新训练事件抽取器。


数据集：ACE2005, TAC-KBP 2015

Keywords: Semi-supervised

</p></blockquote></details>





### 开放域事件抽取

#### 2020
 <details/>
<summary/>
  <a>Open-domain Event Extraction and Embedding for Natural Gas Market Prediction, arxiv 2020 (<a href="https://github.com/">Github</a>)</summary><blockquote><p align="justify">
动机：以前的方法大多数都将价格视为可推断的时间序列，那些分析价格和新闻之间的关系的方法是根据公共新闻数据集相应地修正其价格数据、手动注释标题或使用现成的工具。与现成的工具相比，我们的事件抽取方法不仅可以检测现象的发生，还可以由公共来源检测变化的归因和特征。
 
  ![image](https://github.com/xiaoqian19940510/Event-Extraction/blob/master/figures/2-1.png)
  
  ![image](https://github.com/xiaoqian19940510/Event-Extraction/blob/master/figures/2-2.png)
  
主要思想：依靠公共新闻API的标题，我们提出一种方法来过滤不相关的标题并初步进行事件抽取。价格和文本均被反馈到3D卷积神经网络，以学习事件与市场动向之间的相关性。

数据集：NYTf、FT、TG
</p></blockquote></details>


#### 2019
 <details/>
<summary/>
  <a href="https://www.aclweb.org/anthology/P19-1276/" >Open Domain Event Extraction Using Neural Latent Variable Models, ACL2019 </a> by <i> Xiao Liu and Heyan Huang and Yue Zhang
(<a href="https://github.com/lx865712528/ACL2019-ODEE">Github</a>)</summary><blockquote><p align="justify">
 
动机：
我们考虑开放领域的事件抽取，即从新闻集群中抽取无约束的事件类型的任务。结果表明，与最新的事件模式归纳方法相比，这种无监督模型具有更好的性能。

主要思想：
以前关于生成模式归纳的研究非常依赖人工生成的指标特征，而我们引入了由神经网络产生的潜在变量来获得更好的表示能力。我们设计了一种新颖的图模型，该模型具有潜在的事件类型矢量以及实体的文本冗余特征，而这些潜在的事件类型矢量来自全局参数化正态分布的新闻聚类。

数据集：GNBusiness

</p></blockquote></details>


 <details/>
<summary/>
  <a href="https://arxiv.org/pdf/1909.02126.pdf" >Reporting the unreported: Event Extraction for Analyzing the Local Representation of Hate Crimes, EMNLP 2019</a> by <i> Aida Mostafazadeh Davani etal.
(<a href="https://github.com/aiida-/HateCrime">Github</a>)</summary><blockquote><p align="justify">
 
动机：
将事件抽取和多实例学习应用于本地新闻文章的语料库，可以用来预测仇恨犯罪的发生。

 ![image](https://github.com/xiaoqian19940510/Event-Extraction/blob/master/figures/7.png)

主要思想：
根据是否为仇恨罪标记每篇文章的任务被定义为多实例学习（MIL）问题。我们通过使用文章所有句子中嵌入的信息来确定文章是否报道了仇恨犯罪。在一组带注释的文章上测试了模型之后，我们将被训练过的模型应用于联邦调查局没有报道过的城市，并对这些城市中仇恨犯罪的发生频率进行了下界估计。

</p></blockquote></details>



### 多语言事件抽取

#### 2019
 <details/>
<summary/>
  <a href="https://www.aclweb.org/anthology/D19-1030/" >Cross-lingual Structure Transfer for Relation and Event Extraction, EMNLP 2019) </a> by <i> Ananya Subburathinam, Di Lu, Heng Ji, Jonathan May, Shih-Fu Chang, Avirup Sil, Clare Voss
</summary><blockquote><p align="justify">
 
动机：
从资源不足以及标注不足的语料库中进行复杂语义结构的识别（例如事件和实体关系）是很困难的，这已经变成了一个很有挑战性的信息抽取任务。

 ![image](https://github.com/xiaoqian19940510/Event-Extraction/blob/master/figures/13.png)

主要思想：
通过使用卷积神经网络，将所有实体信息片段、事件触发词、事件背景放入一个复杂的、结构化的多语言公共空间，然后我们可以从源语言标注数据中训练一个事件抽取器，并将它应用于目标语言。

数据集：ACE2005

</p></blockquote></details>



#### 2016
 <details/>
<summary/>
  <a href="https://www.aclweb.org/anthology/C16-1114" >Leveraging Multilingual Training for Limited Resource Event Extraction, COLING 2016 </a> by <i> Hsi, Andrew and Yang, Yiming and Carbonell, Jaime and Xu, Ruochen
</summary><blockquote><p align="justify">
 
动机：迄今为止，利用跨语言培训来提高性能的工作非常有限。因此我们提出了一种新的事件抽取方法。

 ![image](https://github.com/xiaoqian19940510/Event-Extraction/blob/master/figures/31.png)

主要思想：在本文中，我们提出了一种新颖的跨语言事件抽取方法，该方法可在多种语言上进行训练，并利用依赖于语言的特征和不依赖于语言的特征来提高性能。使用这种系统，我们旨在同时利用可用的多语言资源（带标注的数据和引入的特征）来克服目标语言中的标注数据稀缺性问题。 从经验上我们认为，我们的方法可以极大地提高单语系统对中文事件论元提取任务的性能。 与现有工作相比，我们的方法是新颖的，我们不依赖于使用高质量的机器翻译的或手动对齐的文档，这因为这种需求对于给定的目标语言可能是无法满足的。

数据集：ACE2005

Keywords: Training on multiple languages using a combination of both language-dependent and language-independent features

</p></blockquote></details>





### 数据生成


#### 2019
 <details/>
<summary/>
  <a href="https://www.aclweb.org/anthology/P19-1522" >Exploring Pre-trained Language Models for Event Extraction and Geenration, ACL 2019</a> by <i> Yang, Sen and Feng, Dawei and Qiao, Linbo and Kan, Zhigang and Li, Dongsheng
</summary><blockquote><p align="justify">
 
动机：
ACE事件抽取任务的传统方法通常依赖被手动标注过的数据，但是手动标注数据非常耗费精力并且也限制了数据集的规模。我们提出了一个方法来克服这个问题。

 ![image](https://github.com/xiaoqian19940510/Event-Extraction/blob/master/figures/16-1.png)
 
  ![image](https://github.com/xiaoqian19940510/Event-Extraction/blob/master/figures/16.png)
 
主要思想：
本文提出了一个基于预训练语言模型的框架，该框架包含一个作为基础的事件抽取模型以及一种生成被标注事件的方法。我们提出的事件抽取模型由触发词抽取器和论元抽取器组成，论元抽取器用前者的结果进行推理。此外，我们根据角色的重要性对损失函数重新进行加权，从而提高了论元抽取器的性能。

数据集：ACE2005

Keywords: Context-aware word representation, LSTM, Tensor layer

</p></blockquote></details>

 <details/>
<summary/>
  <a href="https://www.aclweb.org/anthology/D19-1027/" >Open Event Extraction from Online Text using a Generative Adversarial Network, EMNLP 2019 </a> by <i> Rui Wang, Deyu Zhou, Yulan He
</summary><blockquote><p align="justify">
 
动机：
提取开放域事件的结构化表示的方法通常假定文档中的所有单词都是从单个事件中生成的，因此他们通常不适用于诸如新闻文章之类的长文本。为了解决这些局限性，我们提出了一种基于生成对抗网络的事件抽取模型，称为对抗神经事件模型（AEM）。

 ![image](https://github.com/xiaoqian19940510/Event-Extraction/blob/master/figures/8.png)


主要思想：
AEM使用Dirichlet先验对事件建模，并使用生成器网络来捕获潜在事件的模式。鉴别符用于区分原始文档和从潜在事件中重建的文档。鉴别器的副产品是鉴别器网络生成的特征允许事件抽取的可视化。

数据集：Twitter, and Google datasets

</p></blockquote></details>

 <details/>
<summary/>
  <a href="https://arxiv.org/pdf/1909.02126.pdf" >Reporting the unreported: Event Extraction for Analyzing the Local Representation of Hate Crimes, EMNLP 2019</a> by <i> Aida Mostafazadeh Davani etal.
(<a href="https://github.com/aiida-/HateCrime">Github</a>)</summary><blockquote><p align="justify">
 
动机：
将事件抽取和多实例学习应用于本地新闻文章的语料库，可以用来预测仇恨犯罪的发生。

 ![image](https://github.com/xiaoqian19940510/Event-Extraction/blob/master/figures/7.png)

主要思想：
根据是否为仇恨罪标记每篇文章的任务被定义为多实例学习（MIL）问题。我们通过使用文章所有句子中嵌入的信息来确定文章是否报道了仇恨犯罪。在一组带标注数据的文章上测试了模型之后，我们将被训练过的模型应用于联邦调查局没有报道过的城市，并对这些城市中仇恨犯罪的发生频率进行了下界估计。

</p></blockquote></details>


#### 2017
 <details/>
<summary/>
  <a href="https://www.aclweb.org/anthology/P17-1038" >Automatically Labeled Data Generation for Large Scale Event Extraction, ACL 2017 </a> by <i> Chen, Yubo and Liu, Shulin and Zhang, Xiang and Liu, Kang and Zhao, Jun
(<a href="https://github.com/acl2017submission/event-data">Github</a>)</summary><blockquote><p align="justify">
 
动机：手动标记的训练数据成本太高，事件类型覆盖率低且规模有限，这种监督的方法很难从知识库中抽取大量事件。

 ![image](https://github.com/xiaoqian19940510/Event-Extraction/blob/master/figures/26.png)

主要思想：1）提出了一种按重要性排列论元并且为每种事件类型选取关键论元或代表论元方法。2）仅仅使用关键论元来标注事件，并找出关键词。3）用外部语言知识库FrameNet来过滤噪声触发词并且扩展触发词库。

数据集：ACE2005

Keywords: Data Generation, Distant Supervision

</p></blockquote></details>



### 阅读理解式事件抽取

#### 2021
 <details/>
<summary/>
  <a href="https://arxiv.org/abs/2106.12384">Reinforcement Learning-based Dialogue Guided Event Extraction to Exploit Argument Relations, arXiv 2021 </a> by <i> Qian Li, Hao Peng, Jianxin Li, Yuanxing Ning, Lihong Wang, Philip S. Yu, and Zheng Wang(<a href="https://github.com/xiaoqian19940510/TASLP-EAREE">Github</a>)</summary><blockquote><p align="justify">
  
主要思想：事件提取是自然语言处理的一项基本任务。
找到事件参数(如事件参与者)的角色是提取事件的关键。
然而，在真实的事件描述中这样做是具有挑战性的，因为一个论点的作用在不同的语境中往往是不同的。
虽然多个参数之间的关系和交互对于解决参数角色是有用的，但是这些信息很大程度上被现有的方法忽略了。
本文通过显式地利用事件参数的关系，提出了一种更好的事件提取方法。
我们通过一个精心设计的面向任务的对话系统来实现这一点。
为了对参数关系进行建模，我们采用了强化学习和增量学习的方法，通过一个多轮、迭代的过程提取多个参数。
我们的方法利用对同一句子中已经提取的论据的知识来确定那些很难单独决定的论据的作用。
然后，它使用新获得的信息来改进以前提取的参数的决策。
这种双向反馈的过程允许我们利用论证关系，有效地解决论证角色，导致更好的句子理解和事件提取。
实验结果表明，在事件分类、参数角色和参数识别方面，该方法始终优于目前最先进的7种事件提取方法。

  ![image](https://github.com/xiaoqian19940510/Event-Extraction/blob/master/figures/Framework.png)

数据集：ACE
</p></blockquote></details>




#### 2020
 <details/>
<summary/>
  <a href="https://arxiv.org/abs/2004.13625">Event Extraction by Answering (Almost) Natural Questions, EMNLP 2020 </a> by <i> Xinya Du and Claire Cardie(<a href="https://github.com/xinyadu/eeqa">Github</a>)</summary><blockquote><p align="justify">
  
主要思想：事件抽取问题需要检测事件触发并提取其相应的参数。
事件参数抽取中的现有工作通常严重依赖于作为预处理/并发步骤的实体识别，这导致了众所周知的错误传播问题。
为了避免这个问题，我们引入了一种新的事件抽取范式，将其形式化为问答(QA)任务，该任务以端到端的方式抽取事件论元。
实证结果表明，我们的框架优于现有的方法;
此外，它还能够抽取训练时未见角色的事件论元。

  ![image](https://github.com/xiaoqian19940510/Event-Extraction/blob/master/figures/0.png)

数据集：ACE
</p></blockquote></details>


#### 2019
 <details/>
<summary/>
  <a href="https://www.aclweb.org/anthology/D19-1068/" >Neural Cross-Lingual Event Detection with Minimal Parallel Resources, EMNLP2019</a> by <i> Jian Liu, Yubo Chen, Kang Liu, Jun Zhao
</summary><blockquote><p align="justify">
 
标注数据的缺乏给事件检测带来了巨大的挑战。
跨语言教育旨在解决这一挑战，通过在不同语言之间传递知识，提高性能。
但是，以前用于ED的跨语言方法对并行资源有严重依赖，这可能限制了它们的适用性。
在本文中，我们提出了一种跨语言的ED的新方法，证明了并行资源的最小依赖性。
具体来说，为了构建不同语言之间的词汇映射，我们设计了一种上下文依赖的翻译方法;
为了解决语序差异问题，我们提出了一种用于多语言联合训练的共享句法顺序事件检测器。
通过在两个标准数据集上的大量实验，研究了该方法的有效性。
实证结果表明，我们的方法在执行不同方向的跨语言迁移和解决注解不足的情况下是有效的。

</p></blockquote></details>


 <details/>
<summary/>
  <a href="https://arxiv.org/abs/1905.05529" >Entity-relation extraction as multi-turn question answering, ACL2019</a> by <i> Xiaoya Li, Fan Yin, Zijun Sun, Xiayu Li, Arianna Yuan, Duo Chai, Mingxin Zhou, Jiwei Li
</summary><blockquote><p align="justify">
提出了一种新的实体-关系抽取的范式。
我们把任务作为多向问答的问题,也就是说,实体和关系的抽取转化为确定答案的任务。
这种多轮QA形式化有几个关键的优点:首先，问题查询为我们想要识别的实体/关系类编码重要的信息;
其次，QA为实体与关系的联合建模提供了一种自然的方式;
第三，它允许我们开发良好的机器阅读理解(MRC)模型。
在ACE和CoNLL04语料库上的实验表明，提出的范式显著优于之前的最佳模型。
我们能够在所有的ACE04、ACE05和CoNLL04数据集上获得最先进的结果，将这三个数据集上的SOTA结果分别提高到49.4(+1.0)、60.2(+0.6)和68.9(+2.1)。
此外，我们构建了一个新开发的中文数据集恢复，它需要多步推理来构建实体依赖关系，而不是以往数据集的三元提取的单步依赖关系抽取。
提出的多轮质量保证模型在简历数据集上也取得了最好的效果。

</p></blockquote></details>



#### 2019
 <details/>
<summary/>
  <a href="https://arxiv.org/abs/1706.04115" >Zero-shot relation extraction via reading comprehension, CoNLL 2017</a> by <i> Jian Liu, Yubo Chen, Kang Liu, Jun Zhao
</summary><blockquote><p align="justify">
 
通过将一个或多个自然语言问题与每个关系槽相关联，可以将关系提取简化为回答简单的阅读理解问题。
减少有几个好处:我们可以(1)学习relation-extraction模型通过扩展最近神经阅读理解技术,(2)为这些模型相结合构建大训练集关系专用众包与远方监督问题,甚至(3)zero-shot学习通过提取新关系类型,只有指定的测试时间,我们没有标签的训练例子。
在Wikipedia填槽任务上的实验表明，该方法可以高精度地将已知关系类型的新问题概括为新问题，并且在较低的精度水平下，Zero-shot地概括为不可见的关系类型是可能的，这为该任务的未来工作设置了标准。

</p></blockquote></details>



# Datasets
[:arrow_up:](#table-of-contents)

#### English Corpus

<details/>
<summary/> <a href="https://catalog.ldc.upenn.edu/LDC2006T06">ACE2005 English Corpus</a></summary><blockquote><p align="justify">
ACE 2005多语种训练语料库包含了用于2005年自动内容抽取(ACE)技术评价的完整的英语、阿拉伯语和汉语训练数据集。
语料库由语言数据联盟(LDC)为实体、关系和事件注释的各种类型的数据组成，该联盟得到了ACE计划的支持和LDC的额外帮助。
 
</p></blockquote></details>

<details/>
<summary/> <a href="https://www.aclweb.org/old_anthology/W/W15/W15-0812.pdf">Rich ERE</a></summary><blockquote><p align="justify">
Rich ERE扩展了实体、关系和事件本体，并扩展了什么是taggable的概念。
Rich ERE还引入了事件跳跃的概念，以解决普遍存在的事件共引用的挑战，特别是关于在文档内和文档之间的事件提及和事件参数粒度变化，从而为创建(分层的或嵌套的)跨文档的事件表示铺平了道路。
 
</p></blockquote></details>



<details/>
<summary/> <a href="https://tac.nist.gov//2015/KBP/Event/index.html">TAC2015</a></summary><blockquote><p align="justify">
TAC KBP事件跟踪的目标是提取关于事件的信息，以便这些信息适合作为知识库的输入。轨迹包括用于检测和链接事件的事件块任务，以及用于提取属于同一事件的事件参数和链接参数的事件参数(EA)任务。2015年TAC KBP赛事轨迹分为5个子任务
</p></blockquote></details>


<details/>
<summary/> <a href="https://tac.nist.gov/2017/KBP/">KBP2017</a></summary><blockquote><p align="justify">
TAC知识库填充(KBP)的目标是开发和评估从非结构化文本中填充知识库的技术。
KBP包括为KBP开发特定组件和功能的组件跟踪，以及称为“冷启动”的端到端KB构建任务，该任务通过在技术成熟时集成选定的组件从头开始构建KB。
与在冷启动KB任务中执行的功能相比，组件跟踪中所需的功能可以“更多”，也可以“更少”。
组件轨道比冷启动“更多”，因为每个轨道可能探索未立即集成到冷启动任务中的试点任务;
他们是“少”,将组件集成到一个KB需要额外协调与和解各个组件之间的不匹配,这样KB符合知识库模式(例如,知识库不能断言一个实体是一个事件的“地方”如果它还断言,实体是一个“人”)。
</p></blockquote></details>


<details/>
<summary/> <a>others</a></summary><blockquote><p align="justify">
Genia2011 dataset, Spainish ERE Corpus, Wikipedia article, BioNLP Cancer Genetics (CG) Shared Task 2013
</p></blockquote></details>



#### Chinese Corpus

<details/>
<summary/> <a href="https://catalog.ldc.upenn.edu/LDC2006T06">ACE2005 Chinese Corpus</a></summary><blockquote><p align="justify">
ACE 2005多语种训练语料库包含了用于2005年自动内容抽取(ACE)技术评价的完整的英语、阿拉伯语和汉语训练数据集。
语料库由语言数据联盟(LDC)为实体、关系和事件注释的各种类型的数据组成，该联盟得到了ACE计划的支持和LDC的额外帮助。
</p></blockquote></details>




# Future Research Challenges
[:arrow_up:](#table-of-contents)



#### 数据层面
领域数据难构造，标注成本大

生成标注数据 or 无标注式事件抽取论元


#### 模型层面

pipeline方式存在错误信息的传递，如何减小错误信息传递

论元之间的关联关系的有效利用


#### 性能评估层面

无标注数据的评价指标设计


