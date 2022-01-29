# -*- coding: utf-8 -*-

"""
@Author: Lyzhang
@Date:
@Description:
"""
from utils.data_iterator import *


def met(dataset, model):
    predicts_all = list()
    targets_all = list()
    dev_batch_iter = gen_batch_iter(dataset)
    c_, g_, h_ = 0., 0., 0.
    for n_batch, (inputs, target) in enumerate(dev_batch_iter, start=1):
        scores = model(inputs, mask2=target[2], mode_="test")  # (batch_size, seq_len + 1, seq_len + 1)
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
    # p r f
    p_ = 0. if h_ == 0 else c_ / h_
    r_ = 0. if g_ == 0 else c_ / g_
    f_ = 0. if (g_ + h_) == 0 else (2 * c_) / (g_ + h_)
    return (p_, r_, f_), predicts_all, targets_all


def met_(dataset, model):
    dev_batch_iter = gen_batch_iter(dataset)
    c_, g_ = 0., 0.
    for n_batch, (inputs, target) in enumerate(dev_batch_iter, start=1):
        scores = model(inputs, mask2=target[2], mode_="test")  # (batch_size, seq_len + 1, seq_len + 1)
        predicts = torch.argmax(scores, dim=-1).squeeze(0)
        gold_ = target[0]
        if USE_CUDA:
            predicts = predicts.cpu()
            gold_ = gold_.cpu()
        pred_ = predicts.numpy()
        gold_ = gold_.numpy()
        c_ += np.sum(pred_ == gold_)
        g_ += gold_.shape[0]
    acc = c_ / g_
    return acc
