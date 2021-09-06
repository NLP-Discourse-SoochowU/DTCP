## Introduction

In this project, we publish a new corpus with discourse-level topic chains annotated on the 385 WSJ news articles
in RST-DT. For details, please refer to our paper: "Longyin Zhang, Xin Tan, Fang Kong and Guodong Zhou. **Adversarial Learning for Discourse Rhetorical Structure Parsing**. EMNLP2021-Findings."

Any questions, just send e-mails to zzlynx@outlook.com (Longyin Zhang).


#### DTC Corpus
We annotated the data in a traditional way, that is, reading the paper WSJ news and manually labeling it on the news paper. The paper-style data contains some details of the thinking process, such as the notes of **TO** and **TE**, omission recovery, reference prompt, etc.

![image](https://github.com/NLP-Discourse-SoochowU/DTCP/blob/main/data/corpus/papers.jpg)

In this published corpus, we only present the annotation of DTC chains, and we'll provide a more complete version in the future. 

#### DTC Parser

1. The python environment is detailed in the txt file "env.txt".

2. Run the following command to train and save your own parser:
   ```
       python main.py
   ```
   For topic chain evaluation:
   ```
       python eval_chain.py
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
