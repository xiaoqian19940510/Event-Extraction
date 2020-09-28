# Event-Extraction（事件抽取资料综述总结）更新中...
近年来事件抽取方法总结，包括中文事件抽取、开放域事件抽取、事件数据生成、跨语言事件抽取、小样本事件抽取、零样本事件抽取等类型，DMCNN、FramNet、DLRNN、DBRNN、GCN、DAG-GRU、JMEE、PLMEE等方法

## 事件抽取的定义

### Closed-domain

Closed-domain event extraction uses predefined event schema to discover and extract desired events of particular type from text. An event schema contains several event types and their corresponding event structures. We use the ACE terminologies to introduce an event structure as follows:


<details/>
<summary/>
<a >Event mention</summary><blockquote><p align="justify">
 a phrase or sentence describing an event, including a trigger and several arguments.
</p></blockquote></details>


<details/>
<summary/>
<a href="https://arxiv.org/pdf/2008.00364.pdf">Event trigger:the main word that most clearly expresses an event occurrence, typically a verb or a noun.</summary><blockquote><p align="justify">
。
</p></blockquote></details>


Event mention:a phrase or sentence describing an event, including a trigger and several arguments.

Event trigger:the main word that most clearly expresses an event occurrence, typically a verb or a noun.

Event argument:an entity mention, temporal expression or value that serves as a participant or attribute with a specific role in an event.

Argument role:the relationship between an argument to the event in which it participants.

D.Ahn [the stages of event extraction] first proposed to divide the ACE event extraction task into four subtasks: trigger detection, event/trigger type identification, event argument detection, and argument role identification. 

Event mention: a phrase or sentence within which an event is described, including a trigger and arguments.

Event trigger: the main word that most clearly expresses the occurrence of an event (An ACE event trigger is typically a verb or a noun).

Event argument: an entity mention, temporal expression or value (e.g. Job-Title) that is involved in an event (viz., participants).

Argument role: the relationship between an argument to the event in which it participates.

### Open domain
Without predefined event schemas, open-domain event extraction aims at detecting events from texts and in most cases, also clustering similar events via extracted event key-words. Event keywords refer to those words/phrases mostly describing an event, and sometimes keywords are further divided into triggers and arguments.

Story segmentation: detecting the boundaries of a story from news articles.

First story detection: detecting the story that discuss anew topic in the stream of news.

Topic detection: grouping the stories based on the topics they discuss.

Topic tracking: detecting stories that discuss a previously known topic.

Story link detection: deciding whether a pair of stories discuss the same topic.

The first two tasks mainly focus on event detection; and the rest three tasks are for event clustering. While the relation between the five tasks is evident, each requires a distinct evaluation process and encourages different approaches to address the particular problem.


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
  <a href="https://transacl.org/ojs/index.php/tacl/article/view/1853">Meta-Learning with Dynamic-Memory-Based Prototypical Network for Few-Shot Event Detection, WSDM2020(<a href="https://link.zhihu.com/?target=https%3A//github.com/231sm/Low_Resource_KBP">Github</a>)</summary><blockquote><p align="justify">
Event detection (ED), a sub-task of event extraction, involves identifying triggers and categorizing event mentions. Existing methods primarily rely upon supervised learning and require large-scale labeled event datasets which are unfortunately not readily available in many real-life applications. In this paper, we consider and reformulate the ED task with limited labeled data as a Few-Shot Learning problem. We propose a Dynamic-Memory-Based Prototypical Network (DMB-PN), which exploits Dynamic Memory Network (DMN) to not only learn better prototypes for event types, but also produce more robust sentence encodings for event mentions. Differing from vanilla prototypical networks simply computing event prototypes by averaging, which only consume event mentions once, our model is more robust and is capable of distilling contextual information from event mentions for multiple times due to the multi-hop mechanism of DMNs. The experiments show that DMB-PN not only deals with sample scarcity better than a series of baseline models but also performs more robustly when the variety of event types is relatively large and the instance quantity is extremely small.

</p></blockquote></details>



 <details/>
<summary/>
  <a href="https://openreview.net/forum?id=H1eA7AEtvS">Exploiting the Matching Information in the Support Set for Few Shot Event Classification, PAKDD2020 (<a href="https://github.com/">Github</a>)</summary><blockquote><p align="justify">
The existing event classification (EC) work primarily focuseson the traditional supervised learning setting in which models are unableto extract event mentions of new/unseen event types. Few-shot learninghas not been investigated in this area although it enables EC models toextend their operation to unobserved event types. To fill in this gap, inthis work, we investigate event classification under the few-shot learningsetting. We propose a novel training method for this problem that exten-sively exploit the support set during the training process of a few-shotlearning model. In particular, in addition to matching the query exam-ple with those in the support set for training, we seek to further matchthe examples within the support set themselves. This method providesmore training signals for the models and can be applied to every metric-learning-based few-shot learning methods. Our extensive experiments ontwo benchmark EC datasets show that the proposed method can improvethe best reported few-shot learning models by up to 10% on accuracyfor event classification.
</p></blockquote></details>

#### 2019
 <details/>
<summary/>
  <a href="https://arxiv.org/abs/1907.11692">Neural Cross-Lingual Event Detection with Minimal Parallel Resources, EMNLP2019(<a href="https://github.com/pytorch/fairseq">Github</a>)</summary><blockquote><p align="justify">
The scarcity in annotated data poses a great challenge for event detection (ED). Cross-lingual ED aims to tackle this challenge by transferring knowledge between different languages to boost performance. However, previous cross-lingual methods for ED demonstrated a heavy dependency on parallel resources, which might limit their applicability. In this paper, we propose a new method for cross-lingual ED, demonstrating a minimal dependency on parallel resources. Specifically, to construct a lexical mapping between different languages, we devise a context-dependent translation method; to treat the word order difference problem, we propose a shared syntactic order event detector for multilingual co-training. The efficiency of our method is studied through extensive experiments on two standard datasets. Empirical results indicate that our method is effective in 1) performing cross-lingual transfer concerning different directions and 2) tackling the extremely annotation-poor scenario.
</p></blockquote></details>

 <details/>
<summary/>
  <a href="http://papers.nips.cc/paper/8812-xlnet-generalized-autoregressive-pretraining-for-language-understanding">Cross-lingual Structure Transfer for Relation and Event Extraction(<a href="https://github.com/zihangdai/xlnet">Github</a>)</summary><blockquote><p align="justify">
The identification of complex semantic structures such as events and entity relations, already a challenging Information Extraction task, is doubly difficult from sources written in under-resourced and under-annotated languages. We investigate the suitability of cross-lingual structure transfer techniques for these tasks. We exploit relation- and event-relevant language-universal features, leveraging both symbolic (including part-of-speech and dependency path) and distributional (including type representation and contextualized representation) information. By representing all entity mentions, event triggers, and contexts into this complex and structured multilingual common space, using graph convolutional networks, we can train a relation or event extractor from source language annotations and apply it to the target language. Extensive experiments on cross-lingual relation and event transfer among English, Chinese, and Arabic demonstrate that our approach achieves performance comparable to state-of-the-art supervised models trained on up to 3,000 manually annotated mentions: up to 62.6% F-score for Relation Extraction, and 63.1% F-score for Event Argument Role Labeling. The event argument role labeling model transferred from English to Chinese achieves similar performance as the model trained from Chinese. We thus find that language-universal symbolic and distributional representations are complementary for cross-lingual structure transfer.
</p></blockquote></details>




### 事件检测

#### 2019
 <details/>
<summary/>
  <a href="https://arxiv.org/abs/1907.11692">Neural Cross-Lingual Event Detection with Minimal Parallel Resources, EMNLP2019(<a href="https://github.com/pytorch/fairseq">Github</a>)</summary><blockquote><p align="justify">
The scarcity in annotated data poses a great challenge for event detection (ED). Cross-lingual ED aims to tackle this challenge by transferring knowledge between different languages to boost performance. However, previous cross-lingual methods for ED demonstrated a heavy dependency on parallel resources, which might limit their applicability. In this paper, we propose a new method for cross-lingual ED, demonstrating a minimal dependency on parallel resources. Specifically, to construct a lexical mapping between different languages, we devise a context-dependent translation method; to treat the word order difference problem, we propose a shared syntactic order event detector for multilingual co-training. The efficiency of our method is studied through extensive experiments on two standard datasets. Empirical results indicate that our method is effective in 1) performing cross-lingual transfer concerning different directions and 2) tackling the extremely annotation-poor scenario.
</p></blockquote></details>




###Few-shot or zero-shot

#### 2019
 <details/>
<summary/>
  <a href="https://arxiv.org/abs/1907.11692">Neural Cross-Lingual Event Detection with Minimal Parallel Resources, EMNLP2019(<a href="https://github.com/pytorch/fairseq">Github</a>)</summary><blockquote><p align="justify">
The scarcity in annotated data poses a great challenge for event detection (ED). Cross-lingual ED aims to tackle this challenge by transferring knowledge between different languages to boost performance. However, previous cross-lingual methods for ED demonstrated a heavy dependency on parallel resources, which might limit their applicability. In this paper, we propose a new method for cross-lingual ED, demonstrating a minimal dependency on parallel resources. Specifically, to construct a lexical mapping between different languages, we devise a context-dependent translation method; to treat the word order difference problem, we propose a shared syntactic order event detector for multilingual co-training. The efficiency of our method is studied through extensive experiments on two standard datasets. Empirical results indicate that our method is effective in 1) performing cross-lingual transfer concerning different directions and 2) tackling the extremely annotation-poor scenario.
</p></blockquote></details>


### 中文事件抽取


#### 2019
 <details/>
<summary/>
  <a href="https://arxiv.org/abs/1907.11692">Neural Cross-Lingual Event Detection with Minimal Parallel Resources, EMNLP2019(<a href="https://github.com/pytorch/fairseq">Github</a>)</summary><blockquote><p align="justify">
The scarcity in annotated data poses a great challenge for event detection (ED). Cross-lingual ED aims to tackle this challenge by transferring knowledge between different languages to boost performance. However, previous cross-lingual methods for ED demonstrated a heavy dependency on parallel resources, which might limit their applicability. In this paper, we propose a new method for cross-lingual ED, demonstrating a minimal dependency on parallel resources. Specifically, to construct a lexical mapping between different languages, we devise a context-dependent translation method; to treat the word order difference problem, we propose a shared syntactic order event detector for multilingual co-training. The efficiency of our method is studied through extensive experiments on two standard datasets. Empirical results indicate that our method is effective in 1) performing cross-lingual transfer concerning different directions and 2) tackling the extremely annotation-poor scenario.
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

