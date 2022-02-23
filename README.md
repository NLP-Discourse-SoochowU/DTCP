## Introduction

In this project, we publish a new corpus with discourse-level topic chains annotated on the 385 WSJ news articles
in RST-DT. For details, please refer to our paper: "Longyin Zhang, Xin Tan, Fang Kong and Guodong Zhou. **EDTC: A Corpus for Discourse-Level Topic Chain Parsing**. EMNLP2021-Findings."

Our latest research found that due to the size of the corpus, DTC parsing is still very [challenging](https://github.com/NLP-Discourse-SoochowU/DTCP/blob/main/data/extension.pdf). We also tried to apply the system pre-trained on EDTC to downstream NLP tasks, but the results were not ideal. As a preliminary exploration of DTC parsing, this research still has many imperfections, which are worthy of in-depth study by NLP researchers. For example:

* This article annotates sentences as the elementary discourse topic unit, but we found in real data that some sentences contain multiple topics. Of course, we also lack a theoretical basis to label each EDU as an elementary DTU. Therefore, how to segment the text into topical units still needs further research.
* The prototype of the topic chain we labeled is relatively simple, which is a 1-to-1 topic chain model. However, there are many n-to-n situations in the actual text, which requires a lot of manpower and energy to improve our annotation.

**We are eager to improve the quality of topic chain labeling and will continue to work hard in the follow-up research. We also invite people with the same goals to participate in the discussion of the research, and cooperate to build an open source data labeling environment and a larger scale DTC data set. Any questions or suggestions, please send an email to zzlynx@outlook.com (Zhang Longyin).**

#### DTC Corpus
We annotated the data in a traditional way: reading the paper WSJ news and manually labeling topic chains on the news paper.

![image](https://github.com/NLP-Discourse-SoochowU/DTCP/blob/main/data/corpus/papers.jpg)

In this published corpus, we only present the annotation of DTC chains, and we will provide a more informative version in the future.

#### DTC Parser

1. We provide our newest parser in "upd_parser" where we present the codes and the pre-trained DTC parsing model for downstream applications.

2. Some data is too big, and we failed to upload it to Github, the "data" package can be downloaded at https://pan.baidu.com/s/1PPd1T1WR6-vJFn1-M-sdSA, and the passcode is lynx. You need to download the "stanfordcorenlp" by yourself.

3. Run the following command to train and save your parser:
   ```
       python main.py
   ```
   Run the following command to evaluate topic chains:
   ```
       python eval.py
   ```

<b>-- License</b>
```
   Copyright (c) 2019, Soochow University NLP research group. All rights reserved.
   Redistribution and use in source and binary forms, with or without modification,
   are permitted provided that the following conditions are met:
   1. Redistributions of source code must retain the above copyright notice, this
      list of conditions and the following disclaimer.
   2. Redistributions in binary form must reproduce the above copyright notice, this
      list of conditions and the following disclaimer in the documentation and/or other
      materials provided with the distribution.
```
