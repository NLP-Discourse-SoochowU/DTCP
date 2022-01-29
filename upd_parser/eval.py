# -*- coding: utf-8 -*-
"""
@Author: Lyzhang
@Date: 2020.2.8
@Description:
"""
import numpy as np
import random
import torch
from path_config import *
from utils.file_util import load_data
torch.manual_seed(7)
np.random.seed(7)
random.seed(7)


def get_chains(arr_):
    chains = []
    flag_li = [0 for _ in range(len(arr_))]
    for idx in range(1, len(arr_)):
        if flag_li[idx] != 0:
            continue
        flag_li[idx] = 1
        if arr_[idx] == 0:
            continue
        else:
            tmp_idx = idx
            tmp_chain = str(tmp_idx) + " "
            len_ = 1
            while arr_[tmp_idx] != 0 and arr_[tmp_idx] > tmp_idx:
                tmp_idx = int(arr_[tmp_idx])
                flag_li[tmp_idx] = 1
                tmp_chain += (str(tmp_idx) + " ")
                len_ += 1
            chains.append(tmp_chain)
    return chains


def chain_prf_gold(path_):
    """ In this evaluation, we hold the view that each chain is composed of several sub-chains. For example, the chain
        (1, 2, 3, 4, 5, 6) can be split to (1, 2, 3) and (4, 5, 6). When the chain (4, 5, 6) is a gold chain in our
        annotation, we should take the sub-chain in (1, 2, 3, 4, 5, 6) as a good prediction.
        Although one may think of our evaluation strange to some extent, that's a good way for performance comparison in
        future studies.
    """
    c_, g_, h_ = 0., 0., 0.
    predicted, targets = load_data(path_)
    idx = 0
    name_ = load_data(TEST_NAMEs)
    for name, value, v2 in zip(name_, predicted, targets):
        idx += 1
        pred_li, gold_li = value.tolist(), v2.tolist()
        pred_chains = get_chains(pred_li)
        gold_chains = get_chains(gold_li)
        g_ += len(gold_chains)
        for chain_g in gold_chains:
            for chain in pred_chains:
                if chain_g in chain:
                    pred_chains.remove(chain)
                    other_sub_chains = chain.split(chain_g)
                    pred_chains += other_sub_chains
                    pred_chains.append(chain_g)
                    c_ += 1.
                    break
        h_ += len(pred_chains)
    # p r f
    p_ = 0. if h_ == 0 else c_ / h_
    r_ = 0. if g_ == 0 else c_ / g_
    f_ = 0. if (g_ + h_) == 0 else (2 * c_) / (g_ + h_)
    print(p_, r_, f_)


if __name__ == "__main__":
    chain_prf_gold(PRED_TEST + "105.pkl")
    chain_prf_gold(PRED_TEST + "106.pkl")
