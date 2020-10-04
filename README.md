# Event-Extraction（事件抽取资料综述总结）更新中...
近年来事件抽取方法总结，包括中文事件抽取、开放域事件抽取、事件数据生成、跨语言事件抽取、小样本事件抽取、零样本事件抽取等类型，DMCNN、FramNet、DLRNN、DBRNN、GCN、DAG-GRU、JMEE、PLMEE等方法


# Table of Contents 目录

- [Define（综述论文）](#Define)
- [Surveys（综述论文）](#Surveys)
- [Shallow Learning Models（浅层学习模型）](#Shallow-Learning-Models)
- [Deep Learning Models（深度学习模型）](#Deep-Learning-Models)
- [Datasets（数据集）](#Datasets)
- [Evaluation Metrics（评价指标）](#Evaluation-Metrics)
- [Future Research Challenges（未来研究挑战）](#Future-Research-Challenges)
- [Tools and Repos（工具与资源）](#tools-and-repos)
</p></blockquote></details>

---


## Define(事件抽取的定义)
[:arrow_up:](#Define)

### Closed-domain

Closed-domain事件抽取使用预定义的事件模式从文本中发现和提取所需的特定类型的事件。事件模式包含多个事件类型及其相应的事件结构。我们使用ACE术语来介绍如下事件结构:


<details/>
<summary/>
<a >Event mention</summary><blockquote><p align="justify">
描述事件的短语或句子，包括触发词和几个参数。
</p></blockquote></details>


<details/>
<summary/>
<a >Event trigger</summary><blockquote><p align="justify">
最清楚地表达事件发生的主要词，尤指动词或名词。
</p></blockquote></details>



<details/>
<summary/>
<a >Event argument</summary><blockquote><p align="justify">
 一个实体,时间表达式，作为参与者的值和在事件中具有特定角色的属性。
</p></blockquote></details>


<details/>
<summary/>
<a >Argument role</summary><blockquote><p align="justify">
 论元与它所参与的事件之间的关系。
</p></blockquote></details>


D.Ahn首先提出将ACE事件提取任务分成四个子任务:触发词检测、事件/触发词类型识别、事件参数检测和参数角色识别。

<details/>
<summary/>
<a >Event mention</summary><blockquote><p align="justify">
 描述事件的短语或句子，包括触发词和参数。
</p></blockquote></details>


<details/>
<summary/>
<a >Event trigger</summary><blockquote><p align="justify">
 最清楚地表示事件发生的主要词(ACE事件触发词通常是动词或名词)。
</p></blockquote></details>

<details/>
<summary/>
<a >Event argument</summary><blockquote><p align="justify">
an entity mention, temporal expression or value (e.g. Job-Title) that is involved in an event (viz., participants).
 在事件(即参与者)中涉及的实体提及、时间表达或值(例如工作头衔)。
</p></blockquote></details>

<details/>
<summary/>
<a >Argument role</summary><blockquote><p align="justify">
the relationship between an argument to the event in which it participates.
 论元和事件同参与者之间的关系。
</p></blockquote></details>



### Open domain

在没有预定义的事件模式的情况下，开放域事件提取的目的是从文本中检测事件，在大多数情况下，还可以通过提取的事件关键字聚类相似的事件。事件关键字指的是那些主要描述事件的词/短语，有时关键字还进一步分为触发器和参数。

<details/>
<summary/>
<a >Story segmentation</summary><blockquote><p align="justify">
detecting the boundaries of a story from news articles.
 从新闻中检测故事的边界。
</p></blockquote></details>


<details/>
<summary/>
<a >First story detection</summary><blockquote><p align="justify">
 检测新闻流中讨论新话题的故事。
</p></blockquote></details>


<details/>
<summary/>
<a >Topic detection</summary><blockquote><p align="justify">
 根据讨论的主题将故事分组。
</p></blockquote></details>

<details/>
<summary/>
<a >Topic tracking</summary><blockquote><p align="justify">
 检测讨论先前已知话题的故事。
</p></blockquote></details>


<details/>
<summary/>
<a >Story link detection</summary><blockquote><p align="justify">
决定两个故事是否讨论同一个主题。
</p></blockquote></details>


前两个任务主要关注事件检测;其余三个任务用于事件集群。虽然这五项任务之间的关系很明显，但每一项任务都需要一个不同的评价过程，并鼓励采用不同的方法来解决特定问题。


## Surveys(综述论文)
[:arrow_up:](#Surveys)

### 事件抽取综述

<details/>
<summary/>
<a href="https://arxiv.org/pdf/2008.00364.pdf">元事件抽取研究综述，2019</a> by<i>GAO Li-zheng, ZHOU Gang, LUO Jun-yong, LAN Ming-jing
</a></summary><blockquote><p align="justify">
事件抽取是信息抽取领域的一个重要研究方向,在情报收集、知识提取、文档摘要、知识问答等领域有着广泛应用。对当前事件抽取领域研究得较多的元事件抽取进行了综述。首先,简要介绍了元事件和元事件抽取的基本概念,以及元事件抽取的主要实现方法。然后,重点阐述了元事件抽取的主要任务,详细介绍了元事件检测过程,并对其他相关任务进行了概述。最后,总结了元事件抽取面临的问题,在此基础上展望了元事件抽取的发展趋势。
</p></blockquote></details>


<details/>
<summary/>
<a href="https://arxiv.org/pdf/2008.00364.pdf">An Overview of Event Extraction from Text，2019</a> by<i>Frederik Hogenboom, Flavius Frasincar, Uzay Kaymak, Franciska de Jong:
</a></summary><blockquote><p align="justify">
文本挖掘的一个常见应用是事件提取，它包括推导出与事件相关的特定知识，这些知识重新映射到文本中。事件提取可应用于各种类型的书面文本，如(在线)新闻消息、博客和手稿。本文献调查回顾了用于各种事件提取目的的文本挖掘技术。它提供了关于如何根据用户、可用内容和使用场景选择特定事件提取技术的一般指南。
</p></blockquote></details>


<details/>
<summary/>
<a href="https://arxiv.org/pdf/2008.00364.pdf">A Survey of Event Extraction from Text，2019</a> by<i>Wei Xiang, Bang Wang </a></summary><blockquote><p align="justify">
事件提取的任务定义、数据源和性能评估，还为其解决方案方法提供了分类。在每个解决方案组中，提供了最具代表性的方法的详细分析，特别是它们的起源、基础、优势和弱点。最后，对未来的研究方向进行了展望。
</p></blockquote></details>



<details/>
<summary/>
<a href="https://arxiv.org/pdf/2008.00364.pdf">A Survey of Textual Event Extraction from Social Networks，2017</a> by<i>Mohamed Mejri, Jalel Akaichi </a></summary><blockquote><p align="justify">
过去的十年中，在社交网络上挖掘文本内容以抽取相关数据和有用的知识已成为无所不在的任务。文本挖掘的一种常见应用是事件抽取，它被认为是一个复杂的任务，分为不同难度的多个子任务。在本文中，我们对现有的主要文本挖掘技术进行了概述，这些技术可用于许多不同的事件抽取目标。首先，我们介绍基于统计模型将数据转换为知识的主要数据驱动方法。其次，我们介绍了基于专家知识的知识驱动方法，通常通过基于模式的方法来抽取知识。然后，我们介绍结合了数据驱动和知识驱动方法的主要现有混合方法。我们以比较研究结束本文，该研究概括了每种提出的方法的主要特征。
</p></blockquote></details>



<details/>
<summary/>
<a href="https://arxiv.org/pdf/2008.00364.pdf">A Survey of event extraction methods from text for decision support systems，2016</a> by<i>Frederik Hogenboom, Flavius Frasincar, Uzay Kaymak, Franciska de Jong, Emiel Caron </a></summary><blockquote><p align="justify">
事件抽取是一种可以追溯到20世纪80年代的专门的信息抽取流程，由于大数据的出现以及文本挖掘和自然语言处理等相关领域的发展，它得到了极大的普及。
然而，到目前为止，对这一特殊领域的概述仍然是难以捉摸的。
因此，我们总结了文本数据的事件提取技术，区分了数据驱动、知识驱动和混合方法，并对这些方法进行了定性评价。
此外，还讨论了从文本语料库中抽取事件的常见决策支持应用。
最后，对事件抽取系统的评价进行了阐述，并指出了当前的研究问题。
</p></blockquote></details>





## Deep Learning Models（深度学习模型）
[:arrow_up:](#table-of-contents)


### 事件抽取


#### 2020
 <details/>
<summary/>
  <a >Event Extraction as Definitation Comprehension, arxiv 2020(<a href="https://link.zhihu.com/?target=https%3A//github.com/231sm/Low_Resource_KBP">Github</a>)</summary><blockquote><p align="justify">
动机：提出一种新颖的事件提取方法，为模型提供带有漂白语句的模型。漂白语句是指基于注释准则、描述事件发生的通常情况的机器可读的自然语言句子。实验结果表明，模型能够提取封闭本体下的事件，并且只需阅读新的漂白语句即可将其推广到未知的事件类型。
 
  ![image](https://github.com/xiaoqian19940510/Event-Extraction/blob/master/figure1.png)
  
主要思想：提出了一种新的事件抽取方法，该方法考虑了通过漂白语句的注释准则；提出了一个多跨度的选择模型，该模型演示了事件抽取方法的可行性以及零样本或少样本设置的可行性。

数据集：ACE 2005

</p></blockquote></details>



 <details/>
<summary/>
  <a>Open-domain Event Extraction and Embedding for Natural Gas Market Prediction, arxiv 2020 (<a href="https://github.com/">Github</a>)</summary><blockquote><p align="justify">
动机：以前的方法大多数都将价格视为可推断的时间序列，那些分析价格和新闻之间的关系的方法是根据公共新闻数据集相应地修正其价格数据、手动注释标题或使用现成的工具。与现成的工具相比，我们的事件提取方法不仅可以检测现象的发生，还可以由公共来源检测变化的归因和特征。
 
  ![image](https://github.com/xiaoqian19940510/Event-Extraction/blob/master/figure1.png)
  
主要思想：依靠公共新闻API的标题，我们提出一种方法来过滤不相关的标题并初步进行事件抽取。价格和文本均被反馈到3D卷积神经网络，以学习事件与市场动向之间的相关性。

数据集：NYTf、FT、TG
</p></blockquote></details>




#### 2019


 <details/>
<summary/>
  <a href="https://link.zhihu.com/?target=https%3A//www.aclweb.org/anthology/P19-1522" >Exploring Pre-trained Language Models for Event Extraction and Geenration, ACL 2019</a> by<i> Yang, Sen and Feng, Dawei and Qiao, Linbo and Kan, Zhigang and Li, Dongsheng
(<a>Github</a>)</summary><blockquote><p align="justify">
 
动机：
ACE事件抽取任务的传统方法通常依赖被手动注释过的数据，但是手动注释数据非常耗费精力并且也限制了数据集的规模。我们提出了一个方法来克服这个问题。

 ![image](https://github.com/xiaoqian19940510/Event-Extraction/blob/master/figure1.png)
 
主要思想：
本文提出了一个基于预训练语言模型的框架，该框架包含一个作为基础的事件抽取模型以及一种生成被标记事件的方法。我们提出的事件抽取模型由触发词抽取器和论元抽取器组成，论元抽取器用前者的结果进行推理。此外，我们根据角色的重要性对损失函数重新进行加权，从而提高了论元抽取器的性能。

数据集：ACE2005

Keywords: Context-aware word representation, LSTM, Tensor layer

</p></blockquote></details>


 <details/>
<summary/>
  <a href="https://link.zhihu.com/?target=https%3A//www.aclweb.org/anthology/P19-1276/" >Open Domain Event Extraction Using Neural Latent Variable Models, ACL2019 </a> by<i> Xiao Liu and Heyan Huang and Yue Zhang
(<a href="https://link.zhihu.com/?target=https%3A//github.com/lx865712528/ACL2019-ODEE">Github</a>)</summary><blockquote><p align="justify">
 
动机：
我们考虑开放领域的事件提取，即从新闻集群中抽取无约束的事件类型的任务。结果表明，与最新的事件模式归纳方法相比，这种无监督模型具有更好的性能。

 ![image](https://github.com/xiaoqian19940510/Event-Extraction/blob/master/figure1.png)

主要思想：
以前关于生成模式归纳的研究非常依赖人工生成的指标特征，而我们引入了由神经网络产生的潜在变量来获得更好的表示能力。我们设计了一种新颖的图形模型，该模型具有潜在的事件类型矢量以及实体的文本冗余特征，而这些潜在的事件类型矢量来自全局参数化正态分布的新闻聚类。

数据集：GNBusiness

</p></blockquote></details>


 <details/>
<summary/>
  <a href="https://link.zhihu.com/?target=https%3A//www.aclweb.org/anthology/P19-3006/" >Rapid Customization for Event Extraction, ACL 2019 </a> by<i> Yee Seng Chan, Joshua Fasching, Haoling Qiu, Bonan Min
(<a href="https://link.zhihu.com/?target=https%3A//github.com/BBN-E/Rapid-customization-events-acl19">Github</a>)</summary><blockquote><p align="justify">
 
动机：
从文本中获取事件发生的时间、地点、人物以及具体做了什么是很多应用程序（例如网页搜索和问题解答）的核心信息抽取任务之一。本文定义了一种快速自定义事件抽取功能的系统，用于查找新的事件类型以及他们的论元。

 ![image](https://github.com/xiaoqian19940510/Event-Extraction/blob/master/figure1.png)

主要思想：
为了能够提抽新类型的事件，我们提出了一种新颖的方法：让用户通过探索无注释的语料库来查找，扩展和过滤事件触发词。然后，系统将自动生成相应级别的事件注释，并训练神经网络模型以查找相应事件。

数据集：ACE2005

</p></blockquote></details>


 <details/>
<summary/>
  <a href="https://link.zhihu.com/?target=https%3A//www.aclweb.org/anthology/D19-1030/" >Cross-lingual Structure Transfer for Relation and Event Extraction, EMNLP 2019) </a> by<i> Ananya Subburathinam, Di Lu, Heng Ji, Jonathan May, Shih-Fu Chang, Avirup Sil, Clare Voss
(<a>Github</a>)</summary><blockquote><p align="justify">
 
动机：
从资源不足以及注释不足的语料库中进行复杂语义结构的识别（例如事件和实体关系）是很困难的，这已经变成了一个很有挑战性的信息抽取任务。

 ![image](https://github.com/xiaoqian19940510/Event-Extraction/blob/master/figure1.png)

主要思想：
通过使用卷积神经网络，将所有实体信息片段、事件触发词、事件背景放入一个复杂的、结构化的多语言公共空间，然后我们可以从源语言注释中训练一个事件抽取器，并将它应用于目标语言。

数据集：ACE2005

</p></blockquote></details>



 <details/>
<summary/>
   <a href="https://link.zhihu.com/?target=https%3A//www.aclweb.org/anthology/D19-1032/" >Doc2EDAG: An End-to-End Document-level Framework for Chinese Financial Event Extraction, EMNLP2019 </a> by<i> Shun Zheng, Wei Cao, Wei Xu, Jiang Bian
(<a>Github</a>)</summary><blockquote><p align="justify">
 
任务:与其他研究不同，该任务被定义为：事件框架填充：也就是论元检测+识别
 
不同点有：不需要触发词检测;文档级的抽取;论元有重叠

动机:解码论元需要一定顺序，先后有关

主要思想:发布数据集，具有特性：arguments-scattering and multi-event,先对事件是否触发进行预测；然后，按照一定顺序先后来分别解码论元

数据集:ten years (2008-2018) Chinese financial announcements：ChFinAnn;Crawling from http://www.cninfo.com.cn/new/index
</p></blockquote></details>


 <details/>
<summary/>
  <a href="https://link.zhihu.com/?target=https%3A//www.aclweb.org/anthology/D19-1585" >Entity, Relation, and Event Extraction with Contextualized Span Representations, CCL 2016 </a> by<i> David Wadden, Ulme Wennberg, Yi Luan, Hannaneh Hajishirzi
(<a href="https://link.zhihu.com/?target=https%3A//github.com/dwadden/dygiepp">Github</a>)</summary><blockquote><p align="justify">
 
动机：
许多信息提取任务（例如被命名的实体的识别，关系提取，事件抽取和共指消解）都可以从跨句子的全局上下文或无局部依赖性的短语中获益。

 ![image](https://github.com/xiaoqian19940510/Event-Extraction/blob/master/figure1.png)
 
主要思想：
（1）将事件提取作为附加任务执行，并在事件触发词与其论元的关系图形中进行跨度更新。
（2）在多句子BERT编码的基础上构建跨度表示形式。

数据集：ACE2005

</p></blockquote></details>



 <details/>
<summary/>
  <a href="https://link.zhihu.com/?target=https%3A//www.aclweb.org/anthology/D19-1584/">HMEAE: Hierarchical Modular Event Argument Extraction, EMNLP 2019 short(<a href="https://link.zhihu.com/?target=https%3A//github.com/thunlp/HMEAE">Github</a>)</summary><blockquote><p align="justify">
任务:事件角色分类
 
动机:论元的类型（如PERSON）会给论元之间的关联带来影响

数据集:ACE 2005
</p></blockquote></details>



 <details/>
<summary/>
  <a href="https://link.zhihu.com/?target=https%3A//www.aclweb.org/anthology/D19-1041/" >Joint Event and Temporal Relation Extraction with Shared Representations and Structured Prediction, EMNLP 2019 </a> by<i> Rujun Han, Qiang Ning, Nanyun Peng
(<a>Github</a>)</summary><blockquote><p align="justify">
 
动机：
事件之间的时间关系的提取是一项重要的自然语言理解（NLU）任务，可以使许多下游任务受益。我们提出了一种事件和时间关系的联合抽取模型，该模型可以进行共享表示学习和结构化预测。

 ![image](https://github.com/xiaoqian19940510/Event-Extraction/blob/master/figure1.png)
 
主要思想：
（1）提出了一个同时进行事件和时间关系抽取的联合模型。这样做的好处是：如果我们使用非事件之间的NONE关系训练关系分类器，则它可能具有修正事件抽取错误的能力。
（2）通过在事件抽取和时间关系抽取模块之间首次共享相同的上下文嵌入和神经表示学习器来改进事件的表示。

数据集：TB-Dense and MATRES datasets

</p></blockquote></details>


 <details/>
<summary/>
  <a href="https://link.zhihu.com/?target=https%3A//www.aclweb.org/anthology/D19-1027/" >Open Event Extraction from Online Text using a Generative Adversarial Network, EMNLP 2019 </a> by<i> Rui Wang, Deyu Zhou, Yulan He
(<a>Github</a>)</summary><blockquote><p align="justify">
 
动机：
提取开放域事件的结构化表示的方法通常假定文档中的所有单词都是从单个事件中生成的，因此他们通常不适用于诸如新闻文章之类的长文本。为了解决这些局限性，我们提出了一种基于生成对抗网络的事件抽取模型，称为对抗神经事件模型（AEM）。

 ![image](https://github.com/xiaoqian19940510/Event-Extraction/blob/master/figure1.png)


主要思想：
AEM使用Dirichlet先验对事件建模，并使用生成器网络来捕获潜在事件的模式。鉴别符用于区分原始文档和从潜在事件中重建的文档。鉴别器的副产品是鉴别器网络生成的特征允许事件抽取的可视化。

数据集：Twitter, and Google datasets

</p></blockquote></details>


 <details/>
<summary/>
  <a href="https://link.zhihu.com/?target=https%3A//arxiv.org/pdf/1909.02126.pdf" >Reporting the unreported: Event Extraction for Analyzing the Local Representation of Hate Crimes, EMNLP 2019</a> by<i> Aida Mostafazadeh Davani etal.
(<a href="https://link.zhihu.com/?target=https%3A//github.com/aiida-/HateCrime">Github</a>)</summary><blockquote><p align="justify">
 
动机：
将事件提取和多实例学习应用于本地新闻文章的语料库，可以用来预测仇恨犯罪的发生。

 ![image](https://github.com/xiaoqian19940510/Event-Extraction/blob/master/figure1.png)

主要思想：
根据是否为仇恨罪标记每篇文章的任务被定义为多实例学习（MIL）问题。我们通过使用文章所有句子中嵌入的信息来确定文章是否报道了仇恨犯罪。在一组带注释的文章上测试了模型之后，我们将被训练过的模型应用于联邦调查局没有报道过的城市，并对这些城市中仇恨犯罪的发生频率进行了下界估计。

</p></blockquote></details>


 <details/>
<summary/>
  <a href="https://link.zhihu.com/?target=https%3A//www.ijcai.org/proceedings/2019/753" >Extracting entities and events as a single task using a transition-based neural model, IJCAI 2019 </a> by<i> Zhang, Junchi and Qin, Yanxia and Zhang, Yue and Liu, Mengchi and Ji, Donghong
(<a href="https://link.zhihu.com/?target=https%3A//github.com/zjcerwin/TransitionEvent">Github</a>)</summary><blockquote><p align="justify">
 
动机：
事件抽取任务包括许多子任务：实体抽取，事件触发词抽取，元素角色抽取。传统的方法是使用pipeline的方式解决这些任务，没有利用到任务间相互关联的信息。已有一些联合学习的模型对这些任务进行处理，然而由于技术上的挑战，还没有模型将其看作一个单一的任务，预测联合的输出结构。本文提出了一个transition-based的神经网络框架，以state-transition的过程，递进地预测复杂的联合结构。

 ![image](https://github.com/xiaoqian19940510/Event-Extraction/blob/master/figure1.png)

主要思想：
使用transition-based的框架，通过使用递增的output-building行为的state-transition过程，构建一个复杂的输出结构。在本文中我们设计了一个transition系统以解决EE问题，从左至右递增地构建出结构，不使用可分的子任务结构。本文还是第一个使transition-based模型，并将之用于实体和事件的联合抽取任务的研究。模型实现了对3个子任务完全的联合解码，实现了更好的信息组合。

数据集：ACE2005

Keywords: Context-aware word representation, LSTM, Tensor layer

</p></blockquote></details>


 <details/>
<summary/>
  <a href="https://link.zhihu.com/?target=https%3A//www.aclweb.org/anthology/N19-1145/" >Biomedical Event Extraction based on Knowledge-driven Tree-LSTM, CCL 2016 </a> by<i> Diya Li, Lifu Huang, Heng Ji, Jiawei Han
(<a>Github</a>)</summary><blockquote><p align="justify">
 
动机：
生物医学领域的事件抽取比一般新闻领域的事件抽取更具挑战性，因为它需要更广泛地获取领域特定的知识并加深对复杂情境的理解。为了更好地对上下文信息和外部背景知识进行编码，我们提出了一种新颖的知识库（KB）驱动的树结构长短期记忆网络（Tree-LSTM）框架。

 ![image](https://github.com/xiaoqian19940510/Event-Extraction/blob/master/figure1.png)

主要思想：
该框架合并了两种功能：（1）抓取上下文背景的依赖结构（2）（2）通过实体链接从外部本体获得实体属性（类型和类别描述）。

数据集：Genia dataset

Keywords: Knowledge-driven Tree-LSTM

</p></blockquote></details>


 <details/>
<summary/>
 <a href="https://link.zhihu.com/?target=https%3A//ieeexplore.ieee.org/document/8643786" >Joint Event Extraction Based on Hierarchical Event Schemas From FrameNet, EMNLP 2019 short</a> by<i> Wei Li , Dezhi Cheng, Lei He, Yuanzhuo Wang, Xiaolong Jin
(<a>Github</a>)</summary><blockquote><p align="justify">
 
动机：事件抽取对于许多实际应用非常有用，例如新闻摘要和信息检索。但是目前很流行的自动上下文抽取（ACE）事件抽取程序仅定义了非常有限且粗糙的事件模式，这可能不适合实际应用。 FrameNet是一种语言语料库，它定义了完整的语义框架和框架间的关系。由于FrameNet中的框架与ACE中的事件架构共享高度相似的结构，并且许多框架实际上表达了事件，因此，我们建议基于FrameNet重新定义事件架构。

主要思想：（1）提取FrameNet中表示事件的所有框架，并利用框架与框架之间的关系建立事件模式的层次结构。（2）适当利用全局信息（例如事件间关系）和事件抽取必不可少的局部特征（例如词性标签和依赖项标签）。基于一种利用事件抽取结果的多文档摘要无监督抽取方法，我们实行了一种图排序。

数据集：ACE 2005，FrameNet 1.7 corpus
</p></blockquote></details>


 <details/>
<summary/>
  <a >One for All: Neural Joint Modeling of Entities and Events, AAAI 2019 </a> by<i> Trung Minh Nguyen∗ Alt Inc.
(<a>Github</a>)</summary><blockquote><p align="justify">
 
动机：

 ![image](https://github.com/xiaoqian19940510/Event-Extraction/blob/master/figure1.png)

主要思想：

数据集：

</p></blockquote></details>




#### 2018

 <details/>
<summary/>
  <a href="https://link.zhihu.com/?target=https%3A//arxiv.org/pdf/1712.03665.pdf" >Scale up event extraction learning via automatic training data generation, AAAI 2018</a> by<i> Zeng, Ying and Feng, Yansong and Ma, Rong and Wang, Zheng and Yan, Rui and Shi, Chongde and Zhao, Dongyan
(<a>Github</a>)</summary><blockquote><p align="justify">
 
动机：现有的训练数据必须通过专业领域知识以及大量的参与者来手动生成，这样生成的数据规模很小，严重影响训练出来的模型的质量。因此我们开发了一种自动生成事件抽取训练数据的方法。

 ![image](https://github.com/xiaoqian19940510/Event-Extraction/blob/master/figure1.png)

主要思想：我们提出了一种基于神经网络和线性规划的事件抽取框架，该模型不依赖显式触发器，而是使用一组关键论元来表征事件类型。这样就去不需要明确识别事件的触发因素，进而降低了人力参与的需求。

数据集：Wikipedia article

Keywords: Data Generation, Distant Supervision

</p></blockquote></details>


<details/>
<summary/>
  <a href="https://link.zhihu.com/?target=https%3A//www.aclweb.org/anthology/P18-4009" >DCFFE: A Document-level Chinese Financial Event Extraction System based on Automatically Labelled Training Data, ACL 2018 </a> by<i> Yang, Hang and Chen, Yubo and Liu, Kang and Xiao, Yang and Zhao, Jun
(<a>Github</a>)</summary><blockquote><p align="justify">
 
动机：在某些特定领域，例如金融，医疗和司法领域，由于数据标记过程的成本很高，因此标记数据不足。此外，当前大多数方法都关注于从一个句子中提取事件，但通常在一个文档中，一个事件由多个句子表示。我们提出一种方法来解决这个问题。

 ![image](https://github.com/xiaoqian19940510/Event-Extraction/blob/master/figure1.png)
 
主要思想：我们提供了一个名为DCFEE的框架，该框架可以从被自动标记过的训练数据中抽取文档级事件。我们使用一个序列标记模型来自动抽取句子级事件，并且提出了一个关键事件检测模型和一个论元填充策略，进而从文档中提取整个事件。

数据集：Chinese financial event dataset

Keywords: Automatically Labelled, Chinese Financial EE

</p></blockquote></details>


<details/>
<summary/>
  <a href="https://link.zhihu.com/?target=http%3A//shalei120.github.io/docs/sha2018Joint.pdf" >Jointly Extraction Event Triggers and Arguments by Dependency-Bridge RNN and Tensor-Based Argument Interaction, AAAI 2018 </a> by<i> Sha, Lei and Qian, Feng and Chang, Baobao and Sui, Zhifang
(<a>Github</a>)</summary><blockquote><p align="justify">
 
动机：传统的事件抽取很大程度上依赖词汇和句法特征，需要大量的人工工程，并且模型通用性不强。另一方面，深度神经网络可以自动学习底层特征，但是现有的网络却没有充分利用句法关系。因此本文在对每个单词建模时，使用依赖桥来增强它的信息表示。说明在RNN模型中同时应用树结构和序列结构比只使用顺序RNN具有更好的性能。另外，利用张量层来同时捕获论元之间的关系以及其在事件中的角色。实验表明，模型取得了很好地效果。

 ![image](https://github.com/xiaoqian19940510/Event-Extraction/blob/master/figure1.png)

主要思想：
（1）实现了事件触发词以及Argument的联合抽取，避开了Pipeline方法中错误的触发词识别结果会在网络中传播的问题；同时联合抽取的过程中，有可能通过元素抽取的步骤反过来纠正事件检测的结果。
（2）将元素的互信息作为影响元素抽取结果的因素。
（3）在构建模型的过程中使用了句法信息。

数据集：ACE2005

Keywords: dbRNN

</p></blockquote></details>

 <details/>
<summary/>
  <a href="https://www.aclweb.org/anthology/P18-1201" >Zero-Shot Transfer Learning for Event Extraction, ACL2018 </a> by<i> Lifu Huang, Heng Ji, Kyunghyun Cho, Ido Dagan, Sebastian Riedel, Clare R. Voss
(<a href="https://github.com/wilburOne/ZeroShotEvent">Github</a>)</summary><blockquote><p align="justify">
动机：以前大多数受监督的事件提取方法都依赖手动注释派生的特征，因此，如果没有额外的注释工作，这些方法便无法应对于新的事件类型。我们设计了一个新的框架来解决这个问题。
 
 ![image](https://github.com/xiaoqian19940510/Event-Extraction/blob/master/figure1.png)
 
主要思想：每个事件都有由候选触发词和论元组成的结构，同时这个结构具有和事件类型及论元相一致的预定义的名字和标签。 我们增加了事件类型以及事件信息片段的语义代表( semantic representations)，并根据目标本体中定义的事件类型和事件信息片段的语义相似性来决定事件的类型

数据集：ACE2005

Keywords: Zero-Shot Transfer

</p></blockquote></details>


 <details/>
<summary/>
  <a href="https://link.zhihu.com/?target=https%3A//www.aclweb.org/anthology/P18-4009" >DCFFE: A Document-level Chinese Financial Event Extraction System based on Automatically Labelled Training Data, ACL 2018 </a> by<i> Yang, Hang and Chen, Yubo and Liu, Kang and Xiao, Yang and Zhao, Jun
(<a>Github</a>)</summary><blockquote><p align="justify">
 
动机：在某些特定领域，例如金融，医疗和司法领域，由于数据标记过程的成本很高，因此标记数据不足。此外，当前大多数方法都关注于从一个句子中提取事件，但通常在一个文档中，一个事件由多个句子表示。我们提出一种方法来解决这个问题。

 ![image](https://github.com/xiaoqian19940510/Event-Extraction/blob/master/figure1.png)

主要思想：我们提供了一个名为DCFEE的框架，该框架可以从被自动标记过的训练数据中抽取文档级事件。我们使用一个序列标记模型来自动抽取句子级事件，并且提出了一个关键事件检测模型和一个论元填充策略，进而从文档中提取整个事件。

数据集：Chinese financial event dataset

Keywords: Automatically Labelled, Chinese Financial EE

</p></blockquote></details>


 <details/>
<summary/>
  <a href="https://link.zhihu.com/?target=https%3A//blender.cs.illinois.edu/paper/imitation2019.pdf" >Joint Entity and Event Extraction with Generative Adversarial Imitation Learning, CCL 2016 </a> by<i> Tongtao Zhang and Heng Ji and Avirup Sil
(<a>Github</a>)</summary><blockquote><p align="justify">
 
动机:我们提出了一种基于生成对抗的模仿学习的实体与事件抽取框架，这种学习是一种使用生成对抗网络（GAN）的逆强化学习方法。该框架的实际表现优于目前最先进的方法。

 ![image](https://github.com/xiaoqian19940510/Event-Extraction/blob/master/figure1.png)
 
主要思想：在本文中，我们提出了一种动态机制——逆强化学习，直接评估实体和事件提取中实例的正确和错误标签。 我们为案例分配明确的分数，或者根据强化学习（RL）给予奖励，并采用来自生成对抗网络（GAN）的鉴别器来估计奖励价值。

</p></blockquote></details>


 <details/>
<summary/>
  <a href="https://link.zhihu.com/?target=https%3A//www.aclweb.org/anthology/D18-1156" >Joint Multiple Event Extraction via Attention-based Graph Information Aggregration, EMNLP 2018 </a> by<i> Liu, Xiao and Luo, Zhunchen and Huang, Heyan
(<a href="https://link.zhihu.com/?target=https%3A//github.com/lx865712528/EMNLP2018-JMEE/">Github</a>)</summary><blockquote><p align="justify">
 
动机：比提取单个事件更困难。在以往的工作中，由于捕获远距离的依赖关系效率很低，因此通过顺序建模的方法在对事件之间的联系进行建模很难成功。本文提出了一种新的框架来解决这个问题。

 ![image](https://github.com/xiaoqian19940510/Event-Extraction/blob/master/figure1.png)
 
主要思想：本文提出JMEE模型（Jointly Multiple Events Extraction），面向的应用是从一个句子中抽取出多个事件触发器和参数（arguments）。JMEE模型引入了syntactic shortcut arcs来增强信息流并且使用基于attention的GCN建模图数据。实验结果表明本文的方法和目前最顶级的方法相比，有着可以媲美的效果。

数据集：ACE2005

Keywords: JMEE

</p></blockquote></details>


 <details/>
<summary/>
  <a href="https://link.zhihu.com/?target=https%3A//www.aclweb.org/anthology/N18-2058/" >Semi-supervised event extraction with paraphrase clusters, NAACL 2018</a> by<i> Ferguson, James and Lockard, Colin and Weld, Daniel and Hajishirzi, Hannaneh
(<a>Github</a>)</summary><blockquote><p align="justify">
 
动机：
受监督的事件抽取系统由于缺乏可用的训练数据而其准确性受到限制。我们提出了一种通过对额外的训练数据进行重复抽样来使事件提取系统自我训练的方法。这种方法避免了训练数据缺乏导致的问题。

 ![image](https://github.com/xiaoqian19940510/Event-Extraction/blob/master/figure1.png)
 
主要思想：
我们通过详细的事件描述自动生成被标记过的训练数据，然后用这些数据进行事件触发词识别。具体来说，首先，将提及该事件的片段聚集在一起，形成一个聚类。然后用每个聚类中的简单示例来给整个聚类贴一个标签。最后，我们将新示例与原始训练集结合在一起，重新训练事件抽取器。


数据集：ACE2005, TAC-KBP 2015

Keywords: Semi-supervised

</p></blockquote></details>



 <details/>
<summary/>
  <a href="https://link.zhihu.com/?target=http%3A//www.cips-cl.org/static/anthology/CCL-2016/CCL-16-081.pdf" >Jointly multiple events extraction via attention-based graph information aggregation, EMNLP 2018 </a> by<i> Xiao Liu, Zhunchen Luo‡ and Heyan Huang
(<a>Github</a>)</summary><blockquote><p align="justify">
 
任务:
触发词分类；论元分类

动机:
论元的语法依存关系

 ![image](https://github.com/xiaoqian19940510/Event-Extraction/blob/master/figure1.png)

主要思想：
用GCN增强论元之间的依存关系
用自注意力来增强触发词分类的效果

数据集：ACE2005

</p></blockquote></details>



#### 2017
 <details/>
<summary/>
  <a href="https://link.zhihu.com/?target=https%3A//www.aclweb.org/anthology/P17-1038" >Automatically Labeled Data Generation for Large Scale Event Extraction, ACL 2017 </a> by<i> Chen, Yubo and Liu, Shulin and Zhang, Xiang and Liu, Kang and Zhao, Jun
(<a href="https://link.zhihu.com/?target=https%3A//github.com/acl2017submission/event-data">Github</a>)</summary><blockquote><p align="justify">
 
动机：手动标记的训练数据成本太高，事件类型覆盖率低且规模有限，这种监督的方法很难从知识库中抽取大量事件。

 ![image](https://github.com/xiaoqian19940510/Event-Extraction/blob/master/figure1.png)

主要思想：1）提出了一种按重要性排列论元并且为每种事件类型选取关键论元或代表论元方法。2）仅仅使用关键论元来标记事件，并找出关键词。3）用外部语言知识库FrameNet来过滤噪声触发词并且扩展触发词库。

数据集：ACE2005

Keywords: Data Generation, Distant Supervision

</p></blockquote></details>




#### 2016
 <details/>
<summary/>
  <a href="https://www.aclweb.org/anthology/P16-1116" >RBPB Regularization Based Pattern Balancing Method for Event Extraction,ACL2016 </a> by<i> Sha, Lei and Liu, Jing and Lin, Chin-Yew and Li, Sujian and Chang, Baobao and Sui, Zhifang
(<a>Github</a>)</summary><blockquote><p align="justify">
动机：在最近的工作中，当确定事件类型（触发器分类）时，大多数方法要么是仅基于模式（pattern），要么是仅基于特征。此外，以往的工作在识别和文类论元的时候，忽略了论元之间的关系，只是孤立的考虑每个候选论元。
 
 ![image](https://github.com/xiaoqian19940510/Event-Extraction/blob/master/figure1.png)
 
主要思想：在本文中，我们同时使用‘模式’和‘特征’来识别和分类‘事件触发器’。 此外，我们使用正则化方法对候选自变量之间的关系进行建模，以提高自变量识别的性能。 我们的方法称为基于正则化的模式平衡方法。

数据集：ACE2005

Keywords: Embedding & Pattern features, Regularization method

</p></blockquote></details>


 <details/>
<summary/>
  <a href="https://link.zhihu.com/?target=https%3A//www.aclweb.org/anthology/C16-1114" >Leveraging Multilingual Training for Limited Resource Event Extraction, COLING 2016 </a> by<i> Hsi, Andrew and Yang, Yiming and Carbonell, Jaime and Xu, Ruochen
(<a>Github</a>)</summary><blockquote><p align="justify">
 
动机：迄今为止，利用跨语言培训来提高性能的工作非常有限。因此我们提出了一种新的事件抽取方法。

 ![image](https://github.com/xiaoqian19940510/Event-Extraction/blob/master/figure1.png)

主要思想：在本文中，我们提出了一种新颖的跨语言事件抽取方法，该方法可在多种语言上进行训练，并利用依赖于语言的特征和不依赖于语言的特征来提高性能。使用这种系统，我们旨在同时利用可用的多语言资源（带注释的数据和引入的特征）来克服目标语言中的注释稀缺性问题。 从经验上我们认为，我们的方法可以极大地提高单语系统对中文事件论元提取任务的性能。 与现有工作相比，我们的方法是新颖的，我们不依赖于使用高质量的机器翻译的或手动对齐的文档，这因为这种需求对于给定的目标语言可能是无法满足的。

数据集：ACE2005

Keywords: Training on multiple languages using a combination of both language-dependent and language-independent features

</p></blockquote></details>



 <details/>
<summary/>
  <a href="https://link.zhihu.com/?target=http%3A//www.cips-cl.org/static/anthology/CCL-2016/CCL-16-081.pdf" >Event Extraction via Bidirectional Long Short-Term Memory Tensor Neural Network, CCL 2016 </a> by<i> Chen, Yubo and Liu, Shulin and He, Shizhu and Liu, Kang and Zhao, Jun
(<a>Github</a>)</summary><blockquote><p align="justify">
 
动机：

 ![image](https://github.com/xiaoqian19940510/Event-Extraction/blob/master/figure1.png)

主要思想：

数据集：ACE2005

Keywords: Context-aware word representation, LSTM, Tensor layer

</p></blockquote></details>


 <details/>
<summary/>
  <a href="https://link.zhihu.com/?target=https%3A//www.aclweb.org/anthology/P18-1145.pdf" >A convolution bilstm neural network model for chinese event extraction, NLPCC 2016 </a> by<i> Lin, Hongyu and Lu, Yaojie and Han, Xianpei and Sun, Le
(<a href="https://link.zhihu.com/?target=https%3A//github.com/sanmusunrise/NPNs">Github</a>)</summary><blockquote><p align="justify">
 
动机：在中文的事件抽取中，以前的方法非常依赖复杂的特征工程以及复杂的自然语言处理工具。本文提出了一种卷积双向LSTM神经网络，该神经网络将LSTM和CNN结合起来，可以捕获句子级和词汇信息，而无需任何人为提供的特征。



 ![image](https://github.com/xiaoqian19940510/Event-Extraction/blob/master/figure1.png)

主要思想：首先使用双向LSTM将整个句子中的单词的语义编码为句子级特征，不做任何句法分析。然后，我们利用卷积神经网络来捕获突出的局部词法特征来消除触发器的歧义，整个过程无需来自POS标签或NER的任何帮助。

数据集：ACE2005， KBP2017 Corpus

</p></blockquote></details>



 <details/>
<summary/>
  <a href="https://link.zhihu.com/?target=https%3A//www.aclweb.org/anthology/P16-1025/" >Liberal Event Extraction and Event Schema Induction, AACL 2016 </a> by<i> Lifu Huang, Taylor Cassidy, Xiaocheng Feng, Heng Ji, Clare R. Voss, Jiawei Han, Avirup Sil
(<a>Github</a>)</summary><blockquote><p align="justify">
 
动机：结合了象征式的（例如抽象含义表示）和分布式的语义来检测和表示事件结构，并采用同一个类型框架来同时提取事件类型和论元角色并发现事件模式。这种模式的提取性能可以与被预定义事件类型标记过的大量数据训练的监督模型相媲美。 

 ![image](https://github.com/xiaoqian19940510/Event-Extraction/blob/master/figure1.png)

主要思想：我们试图将事件触发器和事件论元聚类，每个聚类代表一个事件类型。 我们将分布的相似性用于聚类的距离度量。分布假设指出，经常出现在相似语境中的单词往往具有相似的含义。

两个基本假设：1）出现在相似的背景中并且有相同作用的事件触发词往往具有相似的类型。2）除了特定事件触发器的词汇语义外，事件类型还取决于其论元和论元的作用，以及上下文中与触发器关联的其他单词。


数据集：ERE (Entity Relation Event)

</p></blockquote></details>




 <details/>
<summary/>
  <a href="https://link.zhihu.com/?target=https%3A//www.aclweb.org/anthology/N16-1049/" >Joint Learning Templates and Slots for Event Schema Induction, NAACL 2016 </a> by<i> Lei Sha, Sujian Li, Baobao Chang, Zhifang Sui
(<a href="https://link.zhihu.com/?target=https%3A//github.com/shenglih/normalized_cut/tree/master">Github</a>)</summary><blockquote><p align="justify">
 
动机：我们提出了一个联合实体驱动模型，这种模型可以根据同一句子中模板和各种信息槽（例如attribute slot和participate slot）的限制，同时学习模板和信息槽。这样的模型会得到比以前的方法更好的结果。

 ![image](https://github.com/xiaoqian19940510/Event-Extraction/blob/master/figure1.png)

主要思想：为了更好地建立实体之间的内在联系的模型，我们借用图像分割中的标准化切割作为聚类标准。同时我们用模板之间的约束以及一个句子中的信息槽之间的约束来改善AESI结果。


数据集：MUC-4

</p></blockquote></details>


 <details/>
<summary/>
  <a href="https://link.zhihu.com/?target=https%3A//www.aclweb.org/anthology/N16-1034" >Joint Event Extraction via Recurrent Neural Networks, NAACL 2016 </a> by<i> Chen, Yubo and Liu, Shulin and He, Shizhu and Liu, Kang and Zhao, Jun
(<a>Github</a>)</summary><blockquote><p align="justify">
 
任务:给定实体标签；通过序列标注识别触发词和论元

动机:论元之间有着相关关系，某些论元已经识别出来可能会导致一些论元共现,RNN减少错误传播

 ![image](https://github.com/xiaoqian19940510/Event-Extraction/blob/master/figure1.png)

主要思想:使用RNN来标注要素，通过记忆矩阵来增强要素之间的关联。

数据集：ACE2005

Keywords: RNN, Joint Event Extraction

</p></blockquote></details>




#### 2015
 <details/>
<summary/>
  <a href="https://link.zhihu.com/?target=https%3A//www.aclweb.org/anthology/P15-1017" >Event Extraction via Dynamic Multi-Pooling Convolutional Neural Networks, ACL2015 </a> by<i> Yubo Chen, Liheng Xu, Kang Liu, Daojian Zeng and Jun Zhao 
(<a>Github</a>)</summary><blockquote><p align="justify">
任务：给定候选实体的位置；完成触发词识别，触发词分类，论元识别，论元分类
 
动机：在于一个句子中可能会有多个事件，如果只用一个池化将导致多个事件的句子级特征没有区别。因此引入动态多池化
 
 ![image](https://github.com/xiaoqian19940510/Event-Extraction/blob/master/figure1.png)
 
主要思想：采用动态多池化的方式，以trigger和candidate作为分隔符[-trigger-candidate-]，将句子池化成三段；动机在于一个句子中可能会有多个事件，如果只用一个池化将导致多个事件的句子级特征没有区别。将任务目标转换成句子分类任务，从而完成任务。
 

数据集：ACE2005

keywords: DMCNN, CNN, Dynamic Multi-Pooling

</p></blockquote></details>



 <details/>
<summary/>
  <a href="https://link.zhihu.com/?target=https%3A//www.aclweb.org/anthology/P15-1019/" >Generative Event Schema Induction with Entity Disambiguation, AACL2015 </a> by<i> Kiem-Hieu Nguyen, Xavier Tannier, Olivier Ferret, Romaric Besançon
(<a>Github</a>)</summary><blockquote><p align="justify">
动机：以往文献中的方法仅仅使用中心词来代表实体，然而除了中心词，别的元素也包含了很多重要的信息。这篇论文提出了一种事件模式归纳的生成模型来解决这个问题。
 
 ![image](https://github.com/xiaoqian19940510/Event-Extraction/blob/master/figure1.png)
 
主要思想：模式归纳是指从没有被标记的文本中无监督的学习模板（一个模板定义了一个与实体的语义角色有关的特定事件的类型）。想法是：基于事件模板中相同角色对应的这些实体的相似性，将他们分组在一起。例如，在有关恐怖袭击的语料库中，可以将要被杀死，要被攻击的对象的实体组合在一起，并以名为VICTIM的角色为它们的特征。

数据集：ACE2005

keywords: DMCNN, CNN, Dynamic Multi-Pooling

</p></blockquote></details>



### Few-shot or zero-shot

#### 2020
 <details/>
<summary/>
  <a href="https://arxiv.org/abs/1907.11692">Meta-Learning with Dynamic-Memory-Based Prototypical Network for Few-Shot Event Detection, WSDM 2020(<a href="https://github.com/pytorch/fairseq">Github</a>)</summary><blockquote><p align="justify">
Shumin Deng, Ningyu Zhang, Jiaojian Kang, Yichi Zhang, Wei Zhang, Huajun Chen
</p></blockquote></details>

 <details/>
<summary/>
  <a href="https://arxiv.org/abs/1907.11692">Exploiting the Matching Information in the Support Set for Few Shot Event Classification, PAKDD 2020(<a href="https://github.com/pytorch/fairseq">Github</a>)</summary><blockquote><p align="justify">
Viet Lai, Franck Dernoncourt, Thien Huu Nguyen
</p></blockquote></details>

 <details/>
<summary/>
  <a href="https://arxiv.org/abs/1907.11692">Towards Few-Shot Event Mention Retrieval : An Evaluation Framework and A Siamese Network Approach, 2020(<a href="https://github.com/pytorch/fairseq">Github</a>)</summary><blockquote><p align="justify">

</p></blockquote></details>

#### 2018
 <details/>
<summary/>
  <a href="https://arxiv.org/abs/1907.11692">Zero-Shot Transfer Learning for Event Extraction, ACL 2018(<a href="https://github.com/pytorch/fairseq">Github</a>)</summary><blockquote><p align="justify">
Lifu Huang, Heng Ji, Kyunghyun Cho, Ido Dagan, Sebastian Riedel, Clare R. Voss
</p></blockquote></details>




### 中文事件抽取


#### 2019
 <details/>
<summary/>
   <a href="https://link.zhihu.com/?target=https%3A//www.aclweb.org/anthology/D19-1032/" >Doc2EDAG: An End-to-End Document-level Framework for Chinese Financial Event Extraction, EMNLP2019 </a> by<i> Shun Zheng, Wei Cao, Wei Xu, Jiang Bian
(<a>Github</a>)</summary><blockquote><p align="justify">
 
任务:与其他研究不同，该任务被定义为：事件框架填充：也就是论元检测+识别
 
不同点有：不需要触发词检测;文档级的抽取;论元有重叠

动机:解码论元需要一定顺序，先后有关

主要思想:发布数据集，具有特性：arguments-scattering and multi-event,先对事件是否触发进行预测；然后，按照一定顺序先后来分别解码论元

数据集:ten years (2008-2018) Chinese financial announcements：ChFinAnn;Crawling from http://www.cninfo.com.cn/new/index
</p></blockquote></details>


 <details/>
<summary/>
  <a href="https://link.zhihu.com/?target=https%3A//www.aclweb.org/anthology/D19-1030/" >Cross-lingual Structure Transfer for Relation and Event Extraction, EMNLP 2019) </a> by<i> Ananya Subburathinam, Di Lu, Heng Ji, Jonathan May, Shih-Fu Chang, Avirup Sil, Clare Voss
(<a>Github</a>)</summary><blockquote><p align="justify">
 
动机：
从资源不足以及注释不足的语料库中进行复杂语义结构的识别（例如事件和实体关系）是很困难的，这已经变成了一个很有挑战性的信息抽取任务。

 ![image](https://github.com/xiaoqian19940510/Event-Extraction/blob/master/figure1.png)

主要思想：
通过使用卷积神经网络，将所有实体信息片段、事件触发词、事件背景放入一个复杂的、结构化的多语言公共空间，然后我们可以从源语言注释中训练一个事件抽取器，并将它应用于目标语言。

数据集：ACE2005

</p></blockquote></details>



#### 2018
<details/>
<summary/>
  <a href="https://link.zhihu.com/?target=https%3A//www.aclweb.org/anthology/P18-4009" >DCFFE: A Document-level Chinese Financial Event Extraction System based on Automatically Labelled Training Data, ACL 2018 </a> by<i> Yang, Hang and Chen, Yubo and Liu, Kang and Xiao, Yang and Zhao, Jun
(<a>Github</a>)</summary><blockquote><p align="justify">
 
动机：在某些特定领域，例如金融，医疗和司法领域，由于数据标记过程的成本很高，因此标记数据不足。此外，当前大多数方法都关注于从一个句子中提取事件，但通常在一个文档中，一个事件由多个句子表示。我们提出一种方法来解决这个问题。

 ![image](https://github.com/xiaoqian19940510/Event-Extraction/blob/master/figure1.png)
 
主要思想：我们提供了一个名为DCFEE的框架，该框架可以从被自动标记过的训练数据中抽取文档级事件。我们使用一个序列标记模型来自动抽取句子级事件，并且提出了一个关键事件检测模型和一个论元填充策略，进而从文档中提取整个事件。

数据集：Chinese financial event dataset

Keywords: Automatically Labelled, Chinese Financial EE

</p></blockquote></details>




#### 2016
 <details/>
<summary/>
  <a href="https://link.zhihu.com/?target=https%3A//www.aclweb.org/anthology/P18-1145.pdf" >A convolution bilstm neural network model for chinese event extraction, NLPCC 2016 </a> by<i> Lin, Hongyu and Lu, Yaojie and Han, Xianpei and Sun, Le
(<a href="https://link.zhihu.com/?target=https%3A//github.com/sanmusunrise/NPNs">Github</a>)</summary><blockquote><p align="justify">
 
动机：在中文的事件抽取中，以前的方法非常依赖复杂的特征工程以及复杂的自然语言处理工具。本文提出了一种卷积双向LSTM神经网络，该神经网络将LSTM和CNN结合起来，可以捕获句子级和词汇信息，而无需任何人为提供的特征。

 ![image](https://github.com/xiaoqian19940510/Event-Extraction/blob/master/figure1.png)

主要思想：首先使用双向LSTM将整个句子中的单词的语义编码为句子级特征，不做任何句法分析。然后，我们利用卷积神经网络来捕获突出的局部词法特征来消除触发器的歧义，整个过程无需来自POS标签或NER的任何帮助。

数据集：ACE2005， KBP2017 Corpus

</p></blockquote></details>






### 半监督\远程监督事件抽取

#### 2018
 <details/>
<summary/>
  <a href="https://link.zhihu.com/?target=https%3A//www.aclweb.org/anthology/N18-2058/" >Semi-supervised event extraction with paraphrase clusters, NAACL 2018</a> by<i> Ferguson, James and Lockard, Colin and Weld, Daniel and Hajishirzi, Hannaneh
(<a>Github</a>)</summary><blockquote><p align="justify">
 
动机：
受监督的事件抽取系统由于缺乏可用的训练数据而其准确性受到限制。我们提出了一种通过对额外的训练数据进行重复抽样来使事件提取系统自我训练的方法。这种方法避免了训练数据缺乏导致的问题。

 ![image](https://github.com/xiaoqian19940510/Event-Extraction/blob/master/figure1.png)
 
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
动机：以前的方法大多数都将价格视为可推断的时间序列，那些分析价格和新闻之间的关系的方法是根据公共新闻数据集相应地修正其价格数据、手动注释标题或使用现成的工具。与现成的工具相比，我们的事件提取方法不仅可以检测现象的发生，还可以由公共来源检测变化的归因和特征。
 
  ![image](https://github.com/xiaoqian19940510/Event-Extraction/blob/master/figure1.png)
  
主要思想：依靠公共新闻API的标题，我们提出一种方法来过滤不相关的标题并初步进行事件抽取。价格和文本均被反馈到3D卷积神经网络，以学习事件与市场动向之间的相关性。

数据集：NYTf、FT、TG
</p></blockquote></details>


#### 2019
 <details/>
<summary/>
  <a href="https://link.zhihu.com/?target=https%3A//www.aclweb.org/anthology/P19-1276/" >Open Domain Event Extraction Using Neural Latent Variable Models, ACL2019 </a> by<i> Xiao Liu and Heyan Huang and Yue Zhang
(<a href="https://link.zhihu.com/?target=https%3A//github.com/lx865712528/ACL2019-ODEE">Github</a>)</summary><blockquote><p align="justify">
 
动机：
我们考虑开放领域的事件提取，即从新闻集群中抽取无约束的事件类型的任务。结果表明，与最新的事件模式归纳方法相比，这种无监督模型具有更好的性能。

 ![image](https://github.com/xiaoqian19940510/Event-Extraction/blob/master/figure1.png)

主要思想：
以前关于生成模式归纳的研究非常依赖人工生成的指标特征，而我们引入了由神经网络产生的潜在变量来获得更好的表示能力。我们设计了一种新颖的图形模型，该模型具有潜在的事件类型矢量以及实体的文本冗余特征，而这些潜在的事件类型矢量来自全局参数化正态分布的新闻聚类。

数据集：GNBusiness

</p></blockquote></details>


 <details/>
<summary/>
  <a href="https://link.zhihu.com/?target=https%3A//arxiv.org/pdf/1909.02126.pdf" >Reporting the unreported: Event Extraction for Analyzing the Local Representation of Hate Crimes, EMNLP 2019</a> by<i> Aida Mostafazadeh Davani etal.
(<a href="https://link.zhihu.com/?target=https%3A//github.com/aiida-/HateCrime">Github</a>)</summary><blockquote><p align="justify">
 
动机：
将事件提取和多实例学习应用于本地新闻文章的语料库，可以用来预测仇恨犯罪的发生。

 ![image](https://github.com/xiaoqian19940510/Event-Extraction/blob/master/figure1.png)

主要思想：
根据是否为仇恨罪标记每篇文章的任务被定义为多实例学习（MIL）问题。我们通过使用文章所有句子中嵌入的信息来确定文章是否报道了仇恨犯罪。在一组带注释的文章上测试了模型之后，我们将被训练过的模型应用于联邦调查局没有报道过的城市，并对这些城市中仇恨犯罪的发生频率进行了下界估计。

</p></blockquote></details>



### 多语言事件抽取

#### 2019
 <details/>
<summary/>
  <a href="https://link.zhihu.com/?target=https%3A//www.aclweb.org/anthology/D19-1030/" >Cross-lingual Structure Transfer for Relation and Event Extraction, EMNLP 2019) </a> by<i> Ananya Subburathinam, Di Lu, Heng Ji, Jonathan May, Shih-Fu Chang, Avirup Sil, Clare Voss
(<a>Github</a>)</summary><blockquote><p align="justify">
 
动机：
从资源不足以及注释不足的语料库中进行复杂语义结构的识别（例如事件和实体关系）是很困难的，这已经变成了一个很有挑战性的信息抽取任务。

 ![image](https://github.com/xiaoqian19940510/Event-Extraction/blob/master/figure1.png)

主要思想：
通过使用卷积神经网络，将所有实体信息片段、事件触发词、事件背景放入一个复杂的、结构化的多语言公共空间，然后我们可以从源语言注释中训练一个事件抽取器，并将它应用于目标语言。

数据集：ACE2005

</p></blockquote></details>



#### 2016
 <details/>
<summary/>
  <a href="https://link.zhihu.com/?target=https%3A//www.aclweb.org/anthology/C16-1114" >Leveraging Multilingual Training for Limited Resource Event Extraction, COLING 2016 </a> by<i> Hsi, Andrew and Yang, Yiming and Carbonell, Jaime and Xu, Ruochen
(<a>Github</a>)</summary><blockquote><p align="justify">
 
动机：迄今为止，利用跨语言培训来提高性能的工作非常有限。因此我们提出了一种新的事件抽取方法。

 ![image](https://github.com/xiaoqian19940510/Event-Extraction/blob/master/figure1.png)

主要思想：在本文中，我们提出了一种新颖的跨语言事件抽取方法，该方法可在多种语言上进行训练，并利用依赖于语言的特征和不依赖于语言的特征来提高性能。使用这种系统，我们旨在同时利用可用的多语言资源（带注释的数据和引入的特征）来克服目标语言中的注释稀缺性问题。 从经验上我们认为，我们的方法可以极大地提高单语系统对中文事件论元提取任务的性能。 与现有工作相比，我们的方法是新颖的，我们不依赖于使用高质量的机器翻译的或手动对齐的文档，这因为这种需求对于给定的目标语言可能是无法满足的。

数据集：ACE2005

Keywords: Training on multiple languages using a combination of both language-dependent and language-independent features

</p></blockquote></details>





### 数据生成

#### 2017
 <details/>
<summary/>
  <a href="https://link.zhihu.com/?target=https%3A//www.aclweb.org/anthology/P17-1038" >Automatically Labeled Data Generation for Large Scale Event Extraction, ACL 2017 </a> by<i> Chen, Yubo and Liu, Shulin and Zhang, Xiang and Liu, Kang and Zhao, Jun
(<a href="https://link.zhihu.com/?target=https%3A//github.com/acl2017submission/event-data">Github</a>)</summary><blockquote><p align="justify">
 
动机：手动标记的训练数据成本太高，事件类型覆盖率低且规模有限，这种监督的方法很难从知识库中抽取大量事件。

 ![image](https://github.com/xiaoqian19940510/Event-Extraction/blob/master/figure1.png)

主要思想：1）提出了一种按重要性排列论元并且为每种事件类型选取关键论元或代表论元方法。2）仅仅使用关键论元来标记事件，并找出关键词。3）用外部语言知识库FrameNet来过滤噪声触发词并且扩展触发词库。

数据集：ACE2005

Keywords: Data Generation, Distant Supervision

</p></blockquote></details>



#### 2019
 <details/>
<summary/>
  <a href="https://link.zhihu.com/?target=https%3A//www.aclweb.org/anthology/P19-1522" >Exploring Pre-trained Language Models for Event Extraction and Geenration, ACL 2019</a> by<i> Yang, Sen and Feng, Dawei and Qiao, Linbo and Kan, Zhigang and Li, Dongsheng
(<a>Github</a>)</summary><blockquote><p align="justify">
 
动机：
ACE事件抽取任务的传统方法通常依赖被手动注释过的数据，但是手动注释数据非常耗费精力并且也限制了数据集的规模。我们提出了一个方法来克服这个问题。

 ![image](https://github.com/xiaoqian19940510/Event-Extraction/blob/master/figure1.png)
 
主要思想：
本文提出了一个基于预训练语言模型的框架，该框架包含一个作为基础的事件抽取模型以及一种生成被标记事件的方法。我们提出的事件抽取模型由触发词抽取器和论元抽取器组成，论元抽取器用前者的结果进行推理。此外，我们根据角色的重要性对损失函数重新进行加权，从而提高了论元抽取器的性能。

数据集：ACE2005

Keywords: Context-aware word representation, LSTM, Tensor layer

</p></blockquote></details>

 <details/>
<summary/>
  <a href="https://link.zhihu.com/?target=https%3A//www.aclweb.org/anthology/D19-1027/" >Open Event Extraction from Online Text using a Generative Adversarial Network, EMNLP 2019 </a> by<i> Rui Wang, Deyu Zhou, Yulan He
(<a>Github</a>)</summary><blockquote><p align="justify">
 
动机：
提取开放域事件的结构化表示的方法通常假定文档中的所有单词都是从单个事件中生成的，因此他们通常不适用于诸如新闻文章之类的长文本。为了解决这些局限性，我们提出了一种基于生成对抗网络的事件抽取模型，称为对抗神经事件模型（AEM）。

 ![image](https://github.com/xiaoqian19940510/Event-Extraction/blob/master/figure1.png)


主要思想：
AEM使用Dirichlet先验对事件建模，并使用生成器网络来捕获潜在事件的模式。鉴别符用于区分原始文档和从潜在事件中重建的文档。鉴别器的副产品是鉴别器网络生成的特征允许事件抽取的可视化。

数据集：Twitter, and Google datasets

</p></blockquote></details>

 <details/>
<summary/>
  <a href="https://link.zhihu.com/?target=https%3A//arxiv.org/pdf/1909.02126.pdf" >Reporting the unreported: Event Extraction for Analyzing the Local Representation of Hate Crimes, EMNLP 2019</a> by<i> Aida Mostafazadeh Davani etal.
(<a href="https://link.zhihu.com/?target=https%3A//github.com/aiida-/HateCrime">Github</a>)</summary><blockquote><p align="justify">
 
动机：
将事件提取和多实例学习应用于本地新闻文章的语料库，可以用来预测仇恨犯罪的发生。

 ![image](https://github.com/xiaoqian19940510/Event-Extraction/blob/master/figure1.png)

主要思想：
根据是否为仇恨罪标记每篇文章的任务被定义为多实例学习（MIL）问题。我们通过使用文章所有句子中嵌入的信息来确定文章是否报道了仇恨犯罪。在一组带注释的文章上测试了模型之后，我们将被训练过的模型应用于联邦调查局没有报道过的城市，并对这些城市中仇恨犯罪的发生频率进行了下界估计。

</p></blockquote></details>



### 阅读理解式事件抽取


#### 2019
 <details/>
<summary/>
  <a href="https://arxiv.org/abs/1907.11692">Neural Cross-Lingual Event Detection with Minimal Parallel Resources, EMNLP2019(<a href="https://github.com/pytorch/fairseq">Github</a>)</summary><blockquote><p align="justify">
The scarcity in annotated data poses a great challenge for event detection (ED). Cross-lingual ED aims to tackle this challenge by transferring knowledge between different languages to boost performance. However, previous cross-lingual methods for ED demonstrated a heavy dependency on parallel resources, which might limit their applicability. In this paper, we propose a new method for cross-lingual ED, demonstrating a minimal dependency on parallel resources. Specifically, to construct a lexical mapping between different languages, we devise a context-dependent translation method; to treat the word order difference problem, we propose a shared syntactic order event detector for multilingual co-training. The efficiency of our method is studied through extensive experiments on two standard datasets. Empirical results indicate that our method is effective in 1) performing cross-lingual transfer concerning different directions and 2) tackling the extremely annotation-poor scenario.
</p></blockquote></details>





## Shallow Learning Models(浅层学习模型)
[:arrow_up:](#table-of-contents)

#### 2017




## Data（数据）
[:arrow_up:](#table-of-contents)

#### English Corpus

<details/>
<summary/> <a href="https://catalog.ldc.upenn.edu/LDC2006T06">ACE2005 English Corpus</a></summary><blockquote><p align="justify">
ACE 2005 Multilingual Training Corpus contains the complete set of English, Arabic and Chinese training data for the 2005 Automatic Content Extraction (ACE) technology evaluation. The corpus consists of data of various types annotated for entities, relations and events by the Linguistic Data Consortium (LDC) with support from the ACE Program and additional assistance from LDC.
</p></blockquote></details>


## Future Research Challenges（未来研究挑战）
[:arrow_up:](#table-of-contents)



#### 数据层面



#### 模型层面



#### 性能评估层面




## Tools and Repos（工具与资源）
[:arrow_up:](#table-of-contents)


<details>
<summary><a href="https://github.com/Tencent/NeuralNLP-NeuralClassifier">NeuralClassifier</a></summary><blockquote><p align="justify">
腾讯的开源NLP项目
</p></blockquote></details>

