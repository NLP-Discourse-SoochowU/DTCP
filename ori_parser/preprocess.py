# -*- coding: utf-8 -*-

"""
@Author: Lyzhang
@Date: 2020.1.29
@Description:
"""
import torch
import numpy as np
from path_config import *
from config import WORD_SIZE, SEED
from structs.document import Document
from utils.file_util import save_data, load_data
import progressbar
from transformers import *

p = progressbar.ProgressBar()
p_ = progressbar.ProgressBar()
pad_embedding = torch.zeros(1, WORD_SIZE)
np.random.seed(SEED)


def build_data_ids():
    """ Load sentences from source files and generate the data list with ids.
        (word_ids, pos_ids, syn_ids, tag_ids)
        No padding.
    """
    tokenizer_base = BertTokenizer.from_pretrained("bert-base-uncased")
    tokenizer_large = BertTokenizer.from_pretrained("bert-large-uncased")
    train_docs, dev_docs, test_docs = load_data(DOC_DATA_emnlp)
    train_list = gen_specific_instances(train_docs, tokenizer_base, tokenizer_large)
    dev_list = gen_specific_instances(dev_docs, tokenizer_base, tokenizer_large)
    test_list = gen_specific_instances(test_docs, tokenizer_base, tokenizer_large)
    data_set = (train_list, dev_list, test_list)
    save_data(data_set, EMNLP_DT)


def gen_specific_instances(data_set, tokenizer_base, tokenizer_large):
    data_set_ = []
    p_ = progressbar.ProgressBar()
    p_.start(len(data_set))
    p_idx = 1
    for doc in data_set:
        p_.update(p_idx)
        p_idx += 1
        sents_ids_base, sents_ids_large, link_idx, cls_idx_base, cls_idx_large = [], [], [0], [], []
        begin_base, begin_large = 0, 0
        # building
        for sent in doc.sentences:
            sent_text = sent.sent_text.strip().lower()
            base_ids = tokenizer_base.encode(sent_text)
            large_ids = tokenizer_large.encode(sent_text)
            sents_ids_base += base_ids
            sents_ids_large += large_ids
            link_id = int(sent.point_id) + 1
            link_idx.append(link_id)
            cls_idx_base.append(begin_base)
            cls_idx_large.append(begin_large)
            begin_base += len(base_ids)
            begin_large += len(large_ids)
        data_set_.append((sents_ids_base, sents_ids_large, link_idx, cls_idx_base, cls_idx_large))
    p_.finish()
    return data_set_


def docs_build(raw_path):
    docs = list()
    doc_sent = list()
    with open(raw_path, "r") as f:
        lines = f.readlines()
    for line in lines:
        line = line.strip()
        if len(line) == 0:
            continue
        if line.endswith(".out") and len(doc_sent) > 0:
            doc_name = doc_sent.pop(0)
            docs.append(Document((doc_name, doc_sent)))
            doc_sent = list()
        doc_sent.append(line)
    doc_name = doc_sent.pop(0)
    docs.append(Document((doc_name, doc_sent)))
    return docs


if __name__ == "__main__":
    train_docs = docs_build(TRAIN_RAW)
    dev_docs = docs_build(DEV_RAW)
    test_docs = docs_build(TEST_RAW_emnlp)
    save_data((train_docs, dev_docs, test_docs), DOC_DATA_emnlp)

    build_data_ids()
