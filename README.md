## Introduction

All the data and codes are coming soon, EoE.

In this project, we publish a new corpus with discourse-level topic chains annotated on the 385 WSJ news articles
in RST-DT. For details, please refer to our paper: "Longyin Zhang, Xin Tan, Fang Kong and Guodong Zhou. **EDTC: A Corpus for Discourse-Level Topic Chain Parsing**. EMNLP2021-Findings."

Any questions, send e-mails to zzlynx@outlook.com (Longyin Zhang).


#### DTC Corpus
We annotated the data in a traditional way: reading the paper WSJ news and manually labeling it on the news paper.

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
