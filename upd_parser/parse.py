# -*- coding: utf-8 -*-

"""
@Author: Lyzhang
@Date:
@Description:
"""
import random
from config import *
import numpy as np
import torch
from path_config import *
from utils.file_util import *
from stanfordcorenlp import StanfordCoreNLP
from transformers import *
import progressbar
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
nlp = StanfordCoreNLP(STANFORD_PATH)


class Document:
    def __init__(self, doc_info=None):
        doc_name, lines = doc_info
        self.doc_name = doc_name
        self.sentences = []
        self.init_doc(lines)

    def init_doc(self, lines=None):
        """ Enrich the document with sentence information
        """
        for line in lines:
            self.sentences.append(Sent(line))


class Sent:
    def __init__(self, sent_text):
        self.sent_text = sent_text
        self.sent_id = None
        self.point_id = None
        self.tokens = None
        self.pos_tags = None
        self.syn_tags = None
        self.syn_point = None
        self.init_all()

    def init_all(self):
        tok_pairs = nlp.pos_tag(self.sent_text)
        self.tokens = [pair[0] for pair in tok_pairs]
        self.pos_tags = [pair[1] for pair in tok_pairs]


def build_dt(source_dt, target_dt):
    print("loading...")
    docs = list()
    doc_sent = list()
    with open(source_dt, "r") as f:
        lines = f.readlines()
    p_ = progressbar.ProgressBar()
    p_.start(len(lines))
    p_idx = 1
    for line in lines:
        p_.update(p_idx)
        p_idx += 1
        line = line.strip()
        if len(line) == 0:
            if len(doc_sent) > 0:
                docs.append(Document(("god", doc_sent)))
                doc_sent = list()
        else:
            doc_sent.append(line)
    if len(doc_sent) > 0:
        docs.append(Document(("god", doc_sent)))
    p_.finish()
    word2ids, pos2ids = load_data(WORD2IDS), load_data(POS2IDS)
    doc_list = gen_specific_instances(docs, word2ids, pos2ids)
    save_data(doc_list, target_dt)


def gen_specific_instances(data_set, word2ids, pos2ids):
    data_set_ = []
    p_ = progressbar.ProgressBar()
    p_.start(len(data_set))
    p_idx = 1
    for doc in data_set:
        p_.update(p_idx)
        p_idx += 1
        sents, sents_ids, link_ids, pos_tags, syn_tags, syn_points = [], [], [0], [], [], []
        sents_len = []
        for sent in doc.sentences:
            word_ids = list()
            for word in sent.tokens:
                if word in word2ids.keys():
                    word_ids.append(word2ids[word])
                else:
                    word_ids.append(UNK_ID)
            pos_ids = [pos2ids[pos_tag] for pos_tag in sent.pos_tags]
            sents.append(sent.sent_text)
            sents_ids.append(word_ids)
            sents_len.append(len(word_ids))
            pos_tags.append(pos_ids)
            link_ids.append(1)
        pad_sents_ids, pad_pos_tags = pad_all(sents_ids, pos_tags)
        mask1, mask2, trans_arr, ref_arr = build_mask_(link_ids)
        link_ids = np.array(link_ids)
        trans_arr = np.array(trans_arr)
        ref_arr = np.array(ref_arr)
        data_set_.append((sents, pad_sents_ids, pad_pos_tags, None, None, link_ids, trans_arr, ref_arr, mask1, mask2))
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


def pad_all(sents_ids, pos_tags):
    count_ = 0
    max_sent_l = 0
    for sent_ids in sents_ids:
        if len(sent_ids) > max_sent_l:
            max_sent_l = len(sent_ids)
        count_ += len(sent_ids)
    pad_sents = None
    pad_pos_tags = None
    for sent_ids, pos_tag in zip(sents_ids, pos_tags):
        sent_pad_len = max_sent_l - len(sent_ids)
        pad_sent_ids = np.array(sent_ids + [PAD_ID for _ in range(sent_pad_len)])
        pad_sents = pad_sent_ids[np.newaxis, :] if pad_sents is None else np.concatenate((pad_sents, [pad_sent_ids]), axis=0)
        pad_pos_tag = np.array(pos_tag + [PAD_ID for _ in range(sent_pad_len)])
        pad_pos_tags = pad_pos_tag[np.newaxis, :] if pad_pos_tags is None else np.concatenate((pad_pos_tags, [pad_pos_tag]), axis=0)
    return pad_sents, pad_pos_tags


def do_parse(parse_dt_=None, glv=False):
    instances = load_data(parse_dt_)
    dt_iter = gen_batch_iter(instances)

    model_path = os.path.join(MODEL_SAVE, "DTCP_104.model" if glv else "DTCP_106.model")
    model = torch.load(model_path, map_location=lambda storage, loc: storage.cuda(CUDA_ID))
    model.cuda(CUDA_ID).eval()

    print("parsing...")
    parsed_docs = []
    for doc_item in dt_iter:
        inputs, target_ = doc_item
        _, _, mask2, _, _ = target_
        if glv:
            scores = model(inputs, mask2=mask2, mode_="test")
            predicts = torch.argmax(scores, dim=-1).squeeze(0)
        else:
            model_xl = torch.load(os.path.join(MODEL_SAVE, "DTCP_106.xl"),
                                  map_location=lambda storage, loc: storage.cuda(CUDA_ID))
            model_xl.cuda(CUDA_ID).eval()
            tokenizer = XLNetTokenizer.from_pretrained("xlnet-large-cased")
            scores = model(inputs, mask2=mask2, xl_m=model_xl, xl_t=tokenizer, mode_="test")
            predicts = torch.argmax(scores, dim=-1).squeeze(0)
        predicts = predicts.cpu()
        pred_ = predicts.int().numpy()
        parsed_docs.append(pred_)
        input(pred_[1:] - 1)


def gen_batch_iter(training_set, batch_s=BATCH_SIZE):
    num_docs = len(training_set)
    offset = 0
    while offset < num_docs:
        doc_ = training_set[offset]
        sents, word_ids, pos_tags, syn_tags, syn_points, link_ids, trans_arr, ref_arr, mask1, mask2 = doc_
        offset += batch_s
        word_inputs = torch.from_numpy(word_ids).long()
        pos_inputs = torch.from_numpy(pos_tags).long()
        targets = torch.from_numpy(link_ids).long()
        trans_arr = torch.from_numpy(trans_arr).long()
        ref_arr = torch.from_numpy(ref_arr).long()
        if USE_CUDA:
            word_inputs = word_inputs.cuda(CUDA_ID)
            pos_inputs = pos_inputs.cuda(CUDA_ID)
            targets = targets.cuda(CUDA_ID)
            mask1 = mask1.cuda(CUDA_ID)
            mask2 = mask2.cuda(CUDA_ID)
            trans_arr = trans_arr.cuda(CUDA_ID)
            ref_arr = ref_arr.cuda(CUDA_ID)
        yield (sents, word_inputs, pos_inputs, None, None), (targets, mask1, mask2, trans_arr, ref_arr)


if __name__ == "__main__":
    source = "data/samples/docs.tsv"
    target = "data/samples/docs.pkl"
    build_dt(source, target)

    # parse
    do_parse(parse_dt_=target)
