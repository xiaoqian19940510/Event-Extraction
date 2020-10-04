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

Closed-domain event extraction uses predefined event schema to discover and extract desired events of particular type from text. An event schema contains several event types and their corresponding event structures. We use the ACE terminologies to introduce an event structure as follows:
1

<details/>
<summary/>
<a >Event mention</summary><blockquote><p align="justify">
 a phrase or sentence describing an event, including a trigger and several arguments.
</p></blockquote></details>


<details/>
<summary/>
<a >Event trigger</summary><blockquote><p align="justify">
the main word that most clearly expresses an event occurrence, typically a verb or a noun.
</p></blockquote></details>



<details/>
<summary/>
<a >Event argument</summary><blockquote><p align="justify">
an entity mention, temporal expression or value that serves as a participant or attribute with a specific role in an event.
</p></blockquote></details>


<details/>
<summary/>
<a >Argument role</summary><blockquote><p align="justify">
the relationship between an argument to the event in which it participants.
</p></blockquote></details>




D.Ahn [the stages of event extraction] first proposed to divide the ACE event extraction task into four subtasks: trigger detection, event/trigger type identification, event argument detection, and argument role identification. 

<details/>
<summary/>
<a >Event mention</summary><blockquote><p align="justify">
a phrase or sentence within which an event is described, including a trigger and arguments.
</p></blockquote></details>


<details/>
<summary/>
<a >Event trigger</summary><blockquote><p align="justify">
the main word that most clearly expresses the occurrence of an event (An ACE event trigger is typically a verb or a noun).
</p></blockquote></details>

<details/>
<summary/>
<a >Event argument</summary><blockquote><p align="justify">
an entity mention, temporal expression or value (e.g. Job-Title) that is involved in an event (viz., participants).
</p></blockquote></details>

<details/>
<summary/>
<a >Argument role</summary><blockquote><p align="justify">
the relationship between an argument to the event in which it participates.
</p></blockquote></details>



### Open domain

Without predefined event schemas, open-domain event extraction aims at detecting events from texts and in most cases, also clustering similar events via extracted event key-words. Event keywords refer to those words/phrases mostly describing an event, and sometimes keywords are further divided into triggers and arguments.



<details/>
<summary/>
<a >Story segmentation</summary><blockquote><p align="justify">
detecting the boundaries of a story from news articles.
</p></blockquote></details>


<details/>
<summary/>
<a >First story detection</summary><blockquote><p align="justify">
detecting the story that discuss anew topic in the stream of news.
</p></blockquote></details>


<details/>
<summary/>
<a >Topic detection</summary><blockquote><p align="justify">
grouping the stories based on the topics they discuss.
</p></blockquote></details>

<details/>
<summary/>
<a >Topic tracking</summary><blockquote><p align="justify">
detecting stories that discuss a previously known topic.
</p></blockquote></details>


<details/>
<summary/>
<a >Story link detection</summary><blockquote><p align="justify">
deciding whether a pair of stories discuss the same topic.
</p></blockquote></details>


The first two tasks mainly focus on event detection; and the rest three tasks are for event clustering. While the relation between the five tasks is evident, each requires a distinct evaluation process and encourages different approaches to address the particular problem.





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
One common application of text mining is event extraction,which encompasses deducing specific knowledge concerning incidents re-ferred to in texts. Event extraction can be applied to various types ofwritten text, e.g., (online) news messages, blogs, and manuscripts. Thisliterature survey reviews text mining techniques that are employed forvarious event extraction purposes. It provides general guidelines on howto choose a particular event extraction technique depending on the user,the available content, and the scenario of use.
</p></blockquote></details>


<details/>
<summary/>
<a href="https://arxiv.org/pdf/2008.00364.pdf">A Survey of Event Extraction from Text，2019</a> by<i>Wei Xiang, Bang Wang </a></summary><blockquote><p align="justify">
事件提取的任务定义、数据源和性能评估，还为其解决方案方法提供了分类。在每个解决方案组中，提供了最具代表性的方法的详细分析，特别是它们的起源、基础、优势和弱点。最后，对未来的研究方向进行了展望。
</p></blockquote></details>



<details/>
<summary/>
<a href="https://arxiv.org/pdf/2008.00364.pdf">A Survey of Textual Event Extraction from Social Networks，2017</a> by<i>Mohamed Mejri, Jalel Akaichi </a></summary><blockquote><p align="justify">
。
</p></blockquote></details>



<details/>
<summary/>
<a href="https://arxiv.org/pdf/2008.00364.pdf">A Survey of event extraction methods from text for decision support systems，2016</a> by<i>Frederik Hogenboom, Flavius Frasincar, Uzay Kaymak, Franciska de Jong, Emiel Caron </a></summary><blockquote><p align="justify">
。
</p></blockquote></details>


<details/>
<summary/>
<a href="https://arxiv.org/pdf/2008.00364.pdf">A Survey of Textual Event Extraction from Social Networks，2014</a> by<i>Vera DanilovaMikhail AlexandrovXavier Blanco </a></summary><blockquote><p align="justify">
。
</p></blockquote></details>

### 事件检测综述

<details/>
<summary/>
<a href="https://arxiv.org/pdf/2008.00364.pdf">A survey on multi-modal social event detection，2020</a> by<i>Han Zhou, Hongpeng Yin, Hengyi Zheng, Yanxia Li</a></summary><blockquote><p align="justify">
回顾了事件特征学习和事件推理这两种研究方法。特别是，学习事件特征是必要的前提，因为它能够将社交媒体数据转换为计算机友好的数字形式。事件推理的目的是决定一个样本是否属于一个社会事件。然后，介绍了该社区的几个公共数据集，并给出了比较结果。在本文的最后，本文对如何促进多模式社会事件检测的发展进行了一般性的讨论。
</p></blockquote></details>


<details/>
<summary/>
<a href="https://arxiv.org/pdf/2008.00364.pdf">Review on event detection techniques in social multimedia，2016</a> by<i>Han Zhou, Hongpeng Yin, Hengyi Zheng, Yanxia Li</a></summary><blockquote><p align="justify">
。
</p></blockquote></details>





## Deep Learning Models（深度学习模型）
[:arrow_up:](#table-of-contents)


### 事件抽取

#### 2016
 <details/>
<summary/>
  <a href="https://www.aclweb.org/anthology/P16-1116" >RBPB Regularization Based Pattern Balancing Method for Event Extraction,ACL2016 </a> by<i> Sha, Lei and Liu, Jing and Lin, Chin-Yew and Li, Sujian and Chang, Baobao and Sui, Zhifang
(<a href="https://link.zhihu.com/?target=https%3A//github.com/231sm/Low_Resource_KBP">Github</a>)</summary><blockquote><p align="justify">
动机：在最近的工作中，当确定事件类型（触发器分类）时，大多数方法要么是仅基于模式（pattern），要么是仅基于特征。此外，以往的工作在识别和文类论元的时候，忽略了论元之间的关系，只是孤立的考虑每个候选论元。
 
 ![image](https://github.com/xiaoqian19940510/text-classification-surveys/blob/master/picture1.png)
 
 ![image](https://github.com/xiaoqian19940510/Event-Extraction/blob/master/figure1.png)
 
主要思想：在本文中，我们同时使用‘模式’和‘特征’来识别和分类‘事件触发器’。 此外，我们使用正则化方法对候选自变量之间的关系进行建模，以提高自变量识别的性能。 我们的方法称为基于正则化的模式平衡方法。

数据集：ACE2005

</p></blockquote></details>



#### 2020
 <details/>
<summary/>
  <a >Event Extraction as Definitation Comprehension, arxiv 2020(<a href="https://link.zhihu.com/?target=https%3A//github.com/231sm/Low_Resource_KBP">Github</a>)</summary><blockquote><p align="justify">
动机：提出一种新颖的事件提取方法，为模型提供带有漂白语句的模型。漂白语句是指基于注释准则、描述事件发生的通常情况的机器可读的自然语言句子。实验结果表明，模型能够提取封闭本体下的事件，并且只需阅读新的漂白语句即可将其推广到未知的事件类型。
 
主要思想：提出了一种新的事件抽取方法，该方法考虑了通过漂白语句的注释准则；提出了一个多跨度的选择模型，该模型演示了事件抽取方法的可行性以及零样本或少样本设置的可行性。

数据集：ACE 2005

</p></blockquote></details>



 <details/>
<summary/>
  <a>Open-domain Event Extraction and Embedding for Natural Gas Market Prediction, arxiv 2020 (<a href="https://github.com/">Github</a>)</summary><blockquote><p align="justify">
动机：以前的方法大多数都将价格视为可推断的时间序列，那些分析价格和新闻之间的关系的方法是根据公共新闻数据集相应地修正其价格数据、手动注释标题或使用现成的工具。与现成的工具相比，我们的事件提取方法不仅可以检测现象的发生，还可以由公共来源检测变化的归因和特征。
 
主要思想：依靠公共新闻API的标题，我们提出一种方法来过滤不相关的标题并初步进行事件抽取。价格和文本均被反馈到3D卷积神经网络，以学习事件与市场动向之间的相关性。
数据集：NYTf、FT、TG
</p></blockquote></details>

#### 2019
 <details/>
<summary/>
  <a>One for All: Neural Joint Modeling of Entities and Events, AAAI2019(<a href="https://github.com/pytorch/fairseq">Github</a>)</summary><blockquote><p align="justify">

</p></blockquote></details>

 <details/>
<summary/>
  <a>Doc2EDAG: An End-to-End Document-level Framework for Chinese Financial Event Extraction(<a href="https://github.com/zihangdai/xlnet">Github</a>)</summary><blockquote><p align="justify">
任务:与其他研究不同，该任务被定义为：事件框架填充：也就是论元检测+识别
 
不同点有：不需要触发词检测;文档级的抽取;论元有重叠

动机:解码论元需要一定顺序，先后有关

主要思想:发布数据集，具有特性：arguments-scattering and multi-event,先对事件是否触发进行预测；然后，按照一定顺序先后来分别解码论元

数据集:ten years (2008-2018) Chinese financial announcements：ChFinAnn;Crawling from http://www.cninfo.com.cn/new/index
</p></blockquote></details>



 <details/>
<summary/>
  <a>HMEAE: Hierarchical Modular Event Argument Extraction, EMNLP 2019 short(<a href="https://github.com/pytorch/fairseq">Github</a>)</summary><blockquote><p align="justify">
任务:事件角色分类
 
动机:论元的类型（如PERSON）会给论元之间的关联带来影响

数据集:ACE 2005；TAC KBP 2016
</p></blockquote></details>


 <details/>
<summary/>
  <a>Joint Event Extraction Based on Hierarchical Event Schemas From FrameNet, EMNLP 2019 short(<a href="https://github.com/pytorch/fairseq">Github</a>)</summary><blockquote><p align="justify">
动机：事件抽取对于许多实际应用非常有用，例如新闻摘要和信息检索。但是目前很流行的自动上下文抽取（ACE）事件抽取程序仅定义了非常有限且粗糙的事件模式，这可能不适合实际应用。 FrameNet是一种语言语料库，它定义了完整的语义框架和框架间的关系。由于FrameNet中的框架与ACE中的事件架构共享高度相似的结构，并且许多框架实际上表达了事件，因此，我们建议基于FrameNet重新定义事件架构。

主要思想：（1）提取FrameNet中表示事件的所有框架，并利用框架与框架之间的关系建立事件模式的层次结构。（2）适当利用全局信息（例如事件间关系）和事件抽取必不可少的局部特征（例如词性标签和依赖项标签）。基于一种利用事件抽取结果的多文档摘要无监督抽取方法，我们实行了一种图排序。

数据集：ACE 2005，FrameNet 1.7 corpus
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
  <a href="https://arxiv.org/abs/1907.11692">Cross-lingual Structure Transfer for Relation and Event Extraction, EMNLP2019(<a href="https://github.com/pytorch/fairseq">Github</a>)</summary><blockquote><p align="justify">
Ananya Subburathinam, Di Lu, Heng Ji, Jonathan May, Shih-Fu Chang, Avirup Sil, Clare Voss
</p></blockquote></details>


 <details/>
<summary/>
  <a href="https://arxiv.org/abs/1907.11692"> A Hybrid Character Representatin for Chinese Event Detection, IJCNLP 2019(<a href="https://github.com/pytorch/fairseq">Github</a>)</summary><blockquote><p align="justify">
Xi Xiangyu ; Zhang Tong ; Ye Wei ; Zhang Jinglei ; Xie Rui ; Zhang Shikun
</p></blockquote></details>


 <details/>
<summary/>
  <a href="https://arxiv.org/abs/1907.11692">Event Detection with Trigger-Aware Lattice Neural Network, EMNLP 2019(<a href="https://github.com/pytorch/fairseq">Github</a>)</summary><blockquote><p align="justify">
Ning Ding, Ziran Li, Zhiyuan Liu, Haitao Zheng, Zibo Lin
</p></blockquote></details>

 <details/>
<summary/>
  <a href="https://arxiv.org/abs/1907.11692">Doc2EDAG: An End-to-End Document-level Framework for Chinese Financial Event Extraction, EMNLP 2019(<a href="https://github.com/pytorch/fairseq">Github</a>)</summary><blockquote><p align="justify">
Shun Zheng, Wei Cao, Wei Xu, Jiang Bian
</p></blockquote></details>


#### 2018
 <details/>
<summary/>
  <a href="https://arxiv.org/abs/1907.11692">DCFFE: A Document-level Chinese Financial Event Extraction System based on Automatically Labelled Training Data, ACL 2018(<a href="https://github.com/pytorch/fairseq">Github</a>)</summary><blockquote><p align="justify">
Yang, Hang and Chen, Yubo and Liu, Kang and Xiao, Yang and Zhao, Jun
</p></blockquote></details>

 <details/>
<summary/>
  <a href="https://arxiv.org/abs/1907.11692">Nugget Proposal Networks for Chinese Event Detection, ACL 2018(<a href="https://github.com/pytorch/fairseq">Github</a>)</summary><blockquote><p align="justify">
Lin, Hongyu and Lu, Yaojie and Han, Xianpei and Sun, Le
</p></blockquote></details>

#### 2016
 <details/>
<summary/>
  <a href="https://arxiv.org/abs/1907.11692">A convolution bilstm neural network model for chinese event extraction,NLPCC 2016(<a href="https://github.com/pytorch/fairseq">Github</a>)</summary><blockquote><p align="justify">

</p></blockquote></details>



### 半监督\远程监督事件抽取

#### 2019
 <details/>
<summary/>
  <a href="https://arxiv.org/abs/1907.11692">Neural Cross-Lingual Event Detection with Minimal Parallel Resources, EMNLP2019(<a href="https://github.com/pytorch/fairseq">Github</a>)</summary><blockquote><p align="justify">
The scarcity in annotated data poses a great challenge for event detection (ED). Cross-lingual ED aims to tackle this challenge by transferring knowledge between different languages to boost performance. However, previous cross-lingual methods for ED demonstrated a heavy dependency on parallel resources, which might limit their applicability. In this paper, we propose a new method for cross-lingual ED, demonstrating a minimal dependency on parallel resources. Specifically, to construct a lexical mapping between different languages, we devise a context-dependent translation method; to treat the word order difference problem, we propose a shared syntactic order event detector for multilingual co-training. The efficiency of our method is studied through extensive experiments on two standard datasets. Empirical results indicate that our method is effective in 1) performing cross-lingual transfer concerning different directions and 2) tackling the extremely annotation-poor scenario.
</p></blockquote></details>

### 开放域事件抽取

#### 2019
 <details/>
<summary/>
  <a href="https://arxiv.org/abs/1907.11692">Neural Cross-Lingual Event Detection with Minimal Parallel Resources, EMNLP2019(<a href="https://github.com/pytorch/fairseq">Github</a>)</summary><blockquote><p align="justify">
The scarcity in annotated data poses a great challenge for event detection (ED). Cross-lingual ED aims to tackle this challenge by transferring knowledge between different languages to boost performance. However, previous cross-lingual methods for ED demonstrated a heavy dependency on parallel resources, which might limit their applicability. In this paper, we propose a new method for cross-lingual ED, demonstrating a minimal dependency on parallel resources. Specifically, to construct a lexical mapping between different languages, we devise a context-dependent translation method; to treat the word order difference problem, we propose a shared syntactic order event detector for multilingual co-training. The efficiency of our method is studied through extensive experiments on two standard datasets. Empirical results indicate that our method is effective in 1) performing cross-lingual transfer concerning different directions and 2) tackling the extremely annotation-poor scenario.
</p></blockquote></details>


### 多语言事件抽取

#### 2019
 <details/>
<summary/>
  <a href="https://arxiv.org/abs/1907.11692">Neural Cross-Lingual Event Detection with Minimal Parallel Resources, EMNLP2019(<a href="https://github.com/pytorch/fairseq">Github</a>)</summary><blockquote><p align="justify">
The scarcity in annotated data poses a great challenge for event detection (ED). Cross-lingual ED aims to tackle this challenge by transferring knowledge between different languages to boost performance. However, previous cross-lingual methods for ED demonstrated a heavy dependency on parallel resources, which might limit their applicability. In this paper, we propose a new method for cross-lingual ED, demonstrating a minimal dependency on parallel resources. Specifically, to construct a lexical mapping between different languages, we devise a context-dependent translation method; to treat the word order difference problem, we propose a shared syntactic order event detector for multilingual co-training. The efficiency of our method is studied through extensive experiments on two standard datasets. Empirical results indicate that our method is effective in 1) performing cross-lingual transfer concerning different directions and 2) tackling the extremely annotation-poor scenario.
</p></blockquote></details>



### 数据生成

#### 2019
 <details/>
<summary/>
  <a href="https://arxiv.org/abs/1907.11692">Neural Cross-Lingual Event Detection with Minimal Parallel Resources, EMNLP2019(<a href="https://github.com/pytorch/fairseq">Github</a>)</summary><blockquote><p align="justify">
The scarcity in annotated data poses a great challenge for event detection (ED). Cross-lingual ED aims to tackle this challenge by transferring knowledge between different languages to boost performance. However, previous cross-lingual methods for ED demonstrated a heavy dependency on parallel resources, which might limit their applicability. In this paper, we propose a new method for cross-lingual ED, demonstrating a minimal dependency on parallel resources. Specifically, to construct a lexical mapping between different languages, we devise a context-dependent translation method; to treat the word order difference problem, we propose a shared syntactic order event detector for multilingual co-training. The efficiency of our method is studied through extensive experiments on two standard datasets. Empirical results indicate that our method is effective in 1) performing cross-lingual transfer concerning different directions and 2) tackling the extremely annotation-poor scenario.
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

