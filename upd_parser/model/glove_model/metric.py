# -*- coding: utf-8 -*-

"""
@Author: Lyzhang
@Date:
@Description:
"""
from path_config import SENT_ATTN, DOC_ATTN
from utils.data_iterator import *
from utils.file_util import save_data
import torch.nn.functional as func


def met(dataset, model, mode_="dev"):
    doc_weights = []
    predicts_all = list()
    targets_all = list()
    dev_batch_iter = gen_batch_iter(dataset)
    c_, g_, h_ = 0., 0., 0.
    for n_batch, (inputs, target) in enumerate(dev_batch_iter, start=1):
        scores = model(inputs, mask2=target[2], mode_="test")  # (batch_size, seq_len + 1, seq_len + 1)
        # weight save
        doc_attn = func.softmax(scores, dim=-1).squeeze(0)
        doc_weights.append(doc_attn)

        predicts = torch.argmax(scores, dim=-1).squeeze(0)
        predicts_all.append(predicts)
        targets_all.append(target[0])
        gold_ = target[0]
        if USE_CUDA:
            predicts = predicts.cpu()
            gold_ = gold_.cpu()
        pred_ = predicts.numpy()
        gold_ = gold_.numpy()
        link_idx = np.where(gold_ > 0)[0]
        pred_link = pred_[link_idx]
        gold_link = gold_[link_idx]
        c_ += np.sum(pred_link == gold_link)
        g_ += np.where(gold_ > 0)[0].shape[0]
        h_ += np.where(pred_ > 0)[0].shape[0]
    p_ = 0. if h_ == 0 else c_ / h_
    r_ = 0. if g_ == 0 else c_ / g_
    f_ = 0. if (g_ + h_) == 0 else (2 * c_) / (g_ + h_)
    if mode_ == "test":
        save_data(doc_weights, DOC_ATTN)
    return (p_, r_, f_), predicts_all, targets_all
