# -*- coding: utf-8 -*-

"""
@Author: Lyzhang
@Date:
@Description:
"""
import torch
import numpy as np
from config import *
np.random.seed(SEED)


def gen_batch_iter(training_set, batch_s=BATCH_SIZE):
    num_docs = len(training_set)
    offset = 0
    while offset < num_docs:
        doc_ = training_set[offset]
        sents, word_ids, pos_tags, syn_tags, syn_points, link_ids, trans_arr, ref_arr, mask1, mask2 = doc_
        offset += batch_s
        word_inputs = torch.from_numpy(word_ids).long()
        pos_inputs = torch.from_numpy(pos_tags).long()
        syn_inputs = torch.from_numpy(pos_tags).long()
        syn_points = torch.from_numpy(syn_points).long()
        targets = torch.from_numpy(link_ids).long()
        trans_arr = torch.from_numpy(trans_arr).long()
        ref_arr = torch.from_numpy(ref_arr).long()
        if USE_CUDA:
            word_inputs = word_inputs.cuda(CUDA_ID)
            pos_inputs = pos_inputs.cuda(CUDA_ID)
            syn_inputs = syn_inputs.cuda(CUDA_ID)
            syn_points = syn_points.cuda(CUDA_ID)
            targets = targets.cuda(CUDA_ID)
            mask1 = mask1.cuda(CUDA_ID)
            mask2 = mask2.cuda(CUDA_ID)
            trans_arr = trans_arr.cuda(CUDA_ID)
            ref_arr = ref_arr.cuda(CUDA_ID)
        yield (sents, word_inputs, pos_inputs, syn_inputs, syn_points), (targets, mask1, mask2, trans_arr, ref_arr)
