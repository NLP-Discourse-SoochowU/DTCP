# -*- coding: utf-8 -*-

"""
@Author: Lyzhang
@Date: 2020.1.29
@Description:
"""
import torch
import numpy as np
from structs.document import Document
from path_config import *
from config import UNK, UNK_ID, PAD, PAD_ID, WORD_SIZE, SEED
from utils.file_util import save_data, load_data, write_iterate
import progressbar

p = progressbar.ProgressBar()
p_ = progressbar.ProgressBar()
pad_embedding = torch.zeros(1, WORD_SIZE)
np.random.seed(SEED)


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


def build_voc():
    """ word2ids, pos2ids save all these dictionaries.
    """
    data_sets_ = load_data(DOC_DATA)
    words_set = set()
    with open(GloVe_300, "r") as f:
        for line in f:
            tokens = line.split()
            words_set.add(tokens[0])
    # build word2ids
    word2ids, pos2ids, syn2ids, word2freq = dict(), dict(), dict(), dict()
    word2ids[PAD], word2ids[UNK] = 0, 1
    pos2ids[PAD], pos2ids[UNK] = 0, 1
    syn2ids[PAD], syn2ids[UNK] = 0, 1
    idx_1, idx_2, idx_3 = 2, 2, 2
    train_set, dev_set, test_set = data_sets_
    total_docs = train_set + dev_set + test_set
    corpus_words, corpus_pos, corpus_tags = [], [], []
    for doc in total_docs:
        for sent in doc.sentences:
            corpus_words += sent.tokens
            corpus_pos += sent.pos_tags
            corpus_tags += sent.syn_tags

    for word, pos_tag, syn_tag in zip(corpus_words, corpus_pos, corpus_tags):
        if word not in word2freq.keys():
            word2freq[word] = 1
            word2ids[word] = word2ids[UNK]
        elif (word not in word2ids.keys() or word2ids[word] == word2ids[UNK]) and word in words_set:
            # Only use words >= 2
            word2freq[word] += 1
            word2ids[word] = idx_1
            idx_1 += 1
        else:
            word2freq[word] += 1

        if pos_tag not in pos2ids.keys():
            pos2ids[pos_tag] = idx_2
            idx_2 += 1
        if syn_tag not in syn2ids.keys():
            syn2ids[syn_tag] = idx_3
            idx_3 += 1
    save_data(word2ids, WORD2IDS)
    save_data(pos2ids, POS2IDS)
    save_data(syn2ids, SYN2IDS)
    save_data(word2freq, WORD2FREQ)
    build_ids2word()
    build_ids2vec()


def build_ids2word():
    word2ids = load_data(WORD2IDS)
    ids2word = dict()
    for key_, val_ in zip(word2ids.keys(), word2ids.values()):
        if val_ == UNK_ID:
            ids2word[val_] = UNK
        else:
            ids2word[val_] = key_
    save_data(ids2word, IDS2WORD)


def build_ids2vec():
    word2ids = load_data(WORD2IDS)
    ids2vec = dict()
    with open(GloVe_300, "r") as f:
        for line in f:
            tokens = line.split()
            word = tokens[0]
            vec = np.array([[float(token) for token in tokens[1:]]])
            if tokens[0] in word2ids.keys() and word2ids[tokens[0]] != UNK_ID:
                ids2vec[word2ids[word]] = vec
    # dict2numpy
    embed = [np.zeros(shape=(300,), dtype=np.float32)]
    embed = np.append(embed, [np.random.uniform(-0.25, 0.25, 300)], axis=0)
    idx_valid = list(ids2vec.keys())
    idx_valid.sort()
    for idx in idx_valid:
        embed = np.append(embed, ids2vec[idx], axis=0)
    save_data(embed, IDS2VEC)


def build_data_ids():
    """ Load sentences from source files and generate the data list with ids.
        (word_ids, pos_ids, syn_ids, tag_ids)
        No padding.
    """
    train_docs_, dev_docs_, test_docs_ = load_data(DOC_DATA)
    word2ids, pos2ids, syn2ids = load_data(WORD2IDS), load_data(POS2IDS), load_data(SYN2IDS)
    train_list = gen_specific_instances(train_docs_, word2ids, pos2ids, syn2ids)
    dev_list = gen_specific_instances(dev_docs_, word2ids, pos2ids, syn2ids)
    test_list = gen_specific_instances(test_docs_, word2ids, pos2ids, syn2ids)
    data_set = (train_list, dev_list, test_list)
    save_data(data_set, TRAIN_DATA)


def gen_specific_instances(data_set, word2ids, pos2ids, syn2ids):
    """ Transform all data into ids.
        Build CH-ELMo embedding:
    """
    data_set_ = []
    p_ = progressbar.ProgressBar()
    p_.start(len(data_set))
    p_idx = 1
    for doc in data_set:
        # tmp_x = np.array(doc2x[doc.doc_name])
        p_.update(p_idx)
        p_idx += 1
        sents, sents_ids, link_ids, pos_tags, syn_tags, syn_points = [], [], [0], [], [], []
        sents_len = []
        # building
        for sent in doc.sentences:
            word_ids = list()
            for word in sent.tokens:
                if word in word2ids.keys():
                    word_ids.append(word2ids[word])
                else:
                    word_ids.append(UNK_ID)

            pos_ids = [pos2ids[pos_tag] for pos_tag in sent.pos_tags]
            syn_ids = [syn2ids[syn_tag] for syn_tag in sent.syn_tags]
            syn_point = sent.syn_point
            link_id = int(sent.point_id) + 1
            sents.append(sent.sent_text)
            sents_ids.append(word_ids)
            sents_len.append(len(word_ids))
            pos_tags.append(pos_ids)
            syn_tags.append(syn_ids)
            syn_points.append(syn_point)
            link_ids.append(link_id)
        pad_sents_ids, pad_pos_tags, pad_syn_tags, pad_syn_points = pad_all(sents_ids, pos_tags, syn_tags, syn_points)
        # print(pad_sents_ids.shape, pad_embeddings.shape)
        mask1, mask2, trans_arr, ref_arr = build_mask_(link_ids)
        # print(mask1.shape)
        link_ids = np.array(link_ids)
        trans_arr = np.array(trans_arr)
        ref_arr = np.array(ref_arr)
        data_set_.append((sents, pad_sents_ids, pad_pos_tags, pad_syn_tags, pad_syn_points, link_ids, trans_arr,
                          ref_arr, mask1, mask2))
    p_.finish()
    return data_set_


def build_mask_(link_ids):
    seq_num = len(link_ids)
    trans_arr = [0 for _ in range(seq_num)]
    ref_arr = [idx for idx in range(seq_num)]
    # mask1
    mask1 = [np.array([0 for _ in range(seq_num)])]
    tmp_mask_line = [0 for _ in range(seq_num)]
    for idx in range(seq_num - 1):
        link_id_ = link_ids[idx]
        if link_id_ > 0:
            tmp_mask_line[link_id_] = 1
            trans_arr[link_id_] = idx
        save_line = np.array([idx for idx in tmp_mask_line])
        mask1 = np.append(mask1, [save_line], axis=0)
    mask1 = torch.from_numpy(mask1).float()
    mask2 = torch.diag(torch.ones(seq_num))
    mask2[0, 0] = 0
    mask2[0, 1:] = 1
    for idx in range(1, seq_num):
        mask2[idx, :idx] = 1
        mask2[idx, 0] = 0
    return mask1, mask2, trans_arr, ref_arr


def pad_all(sents_ids, pos_tags, syn_tags, syn_points):
    count_ = 0
    max_sent_l = 0
    for sent_ids in sents_ids:
        if len(sent_ids) > max_sent_l:
            max_sent_l = len(sent_ids)
        count_ += len(sent_ids)
    pad_sents = None
    pad_pos_tags = None
    pad_syn_tags = None
    pad_syn_points = None
    for sent_ids, pos_tag, syn_tag, syn_point in zip(sents_ids, pos_tags, syn_tags, syn_points):
        sent_pad_len = max_sent_l - len(sent_ids)
        pad_sent_ids = np.array(sent_ids + [PAD_ID for _ in range(sent_pad_len)])

        pad_sents = pad_sent_ids[np.newaxis, :] if pad_sents is None else np.concatenate((pad_sents, [pad_sent_ids]), axis=0)
        pad_pos_tag = np.array(pos_tag + [PAD_ID for _ in range(sent_pad_len)])
        pad_pos_tags = pad_pos_tag[np.newaxis, :] if pad_pos_tags is None else np.concatenate((pad_pos_tags, [pad_pos_tag]), axis=0)
        pad_syn_tag = np.array(syn_tag + [PAD_ID for _ in range(sent_pad_len)])
        pad_syn_tags = pad_syn_tag[np.newaxis, :] if pad_syn_tags is None else np.concatenate((pad_syn_tags, [pad_syn_tag]), axis=0)
        pad_syn_point = np.array(syn_point + [PAD_ID for _ in range(sent_pad_len)])
        pad_syn_points = pad_syn_point[np.newaxis, :] if pad_syn_points is None else np.concatenate((pad_syn_points, [pad_syn_point])
                                                                                       , axis=0)
    return pad_sents, pad_pos_tags, pad_syn_tags, pad_syn_points


def build_names():
    train_docs, dev_docs, test_docs = load_data(DOC_DATA)
    dev_names, test_names = list(), list()
    for doc in dev_docs:
        dev_names.append(doc.doc_name)
    for doc in test_docs:
        test_names.append(doc.doc_name)
    save_data(dev_names, DEV_NAMEs)
    save_data(test_names, TEST_NAMEs)


def compare(mode="dev"):
    results = []
    if mode == "dev":
        predicted, targets = load_data(PRED_DEV)
        name_ = load_data(DEV_NAMEs)
    else:
        predicted, targets = load_data(PRED_TEST)
        name_ = load_data(TEST_NAMEs)
    for name, value, v2 in zip(name_, predicted, targets):
        res = name + ": " + str(value.cpu().numpy().tolist()) + ": " + str(v2.cpu().numpy().tolist())
        results.append(res)
    if mode == "dev":
        write_iterate(results, Compare_DEV)
    else:
        write_iterate(results, Compare_TEST)


def build_sentences():
    train_docs, dev_docs, test_docs = load_data(DOC_DATA)
    train_sents, dev_sents, test_sents = [], [], []
    for doc in train_docs:
        tmp_sent = []
        for sent in doc.sentences:
            sent = sent.sent_text
            tmp_sent.append(sent)
        train_sents.append(tmp_sent)
    for doc in dev_docs:
        tmp_sent = []
        for sent in doc.sentences:
            sent = sent.sent_text
            tmp_sent.append(sent)
        dev_sents.append(tmp_sent)
    for doc in test_docs:
        tmp_sent = []
        for sent in doc.sentences:
            sent = sent.sent_text
            tmp_sent.append(sent)
        test_sents.append(tmp_sent)
    dt_sents = (train_sents, dev_sents, test_sents)
    save_data(dt_sents, DOC_SENTs)


if __name__ == "__main__":
    print("begin ... ")
    train_docs = docs_build(TRAIN_RAW)
    dev_docs = docs_build(DEV_RAW)
    test_docs = docs_build(TEST_RAW)
    save_data((train_docs, dev_docs, test_docs), DOC_DATA)
    build_data_ids()
