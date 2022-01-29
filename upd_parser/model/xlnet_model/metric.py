# -*- coding: utf-8 -*-

"""
@Author: Lyzhang
@Date:
@Description:
"""
from utils.data_iterator import *


def met(dataset, model, mode_="dev", xl_m=None, xl_t=None):
    predicts_all = list()
    targets_all = list()
    dev_batch_iter = gen_batch_iter(dataset)
    c_, g_, h_ = 0., 0., 0.
    for n_batch, (inputs, target) in enumerate(dev_batch_iter, start=1):
        scores = model(inputs, mask2=target[2], mode_="test", xl_m=xl_m, xl_t=xl_t)
        predicts = torch.argmax(scores, dim=-1).squeeze(0)
        predicts = predicts.cpu().detach().numpy()
        predicts_all.append(predicts)
        gold_ = target[0].cpu().detach().numpy()
        targets_all.append(gold_)
        link_idx = np.where(gold_ > 0)[0]
        pred_link = predicts[link_idx]
        gold_link = gold_[link_idx]
        c_ += np.sum(pred_link == gold_link)
        g_ += np.where(gold_ > 0)[0].shape[0]
        h_ += np.where(predicts > 0)[0].shape[0]
    p_ = 0. if h_ == 0 else c_ / h_
    r_ = 0. if g_ == 0 else c_ / g_
    f_ = 0. if (g_ + h_) == 0 else (2 * c_) / (g_ + h_)
    return (p_, r_, f_), predicts_all, targets_all
