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


#### 2018
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

