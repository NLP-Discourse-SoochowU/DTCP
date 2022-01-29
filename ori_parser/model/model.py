# -*- coding: utf-8 -*-

"""
@Author: Lyzhang
@Date:
@Description:
"""
import random
from config import *
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as func
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)


class DTCP(nn.Module):
    def __init__(self):
        super(DTCP, self).__init__()
        self.ctx_fnn = nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE)
        self.drop_out = nn.Dropout(0.7)
        self.softmax = nn.LogSoftmax(dim=-1)
        self.dep_encoder = MaskedGRU(WORD_SIZE, HIDDEN_SIZE // 2, bidirectional=True)
        self.dep_decoder = nn.GRU(HIDDEN_SIZE, HIDDEN_SIZE, bidirectional=False, batch_first=True)
        self.bia_attn = QEAttention(WORD_SIZE, WORD_SIZE)
        self.layer_norm = nn.LayerNorm(WORD_SIZE)
        self.relu = nn.ReLU()
        self.root_hidden = torch.rand(BATCH_SIZE, 1, WORD_SIZE)
        nn.init.xavier_normal_(self.root_hidden)
        if USE_CUDA:
            self.root_hidden = self.root_hidden.cuda(CUDA_ID)

    def loss(self, batch_dt, model):
        in_words, in_masks, target, target_mask, in_words_cls, in_words_cls_mask = batch_dt
        score_ = self(batch_dt, model)
        batch_size, e_num, _ = score_.size()
        score_ = score_.view(batch_size * e_num, -1)
        target = target.view(-1)
        target_mask = target_mask.view(-1)
        loss_ = func.nll_loss(score_, target, reduction="none")
        loss_ = (loss_ * target_mask).sum() / target_mask.sum()
        return loss_

    def predict(self, batch_dt, model):
        in_words, in_masks, target, target_mask, in_words_cls, in_words_cls_mask = batch_dt
        score_ = self(batch_dt, model)
        predicted_out = score_.argmax(-1).view(-1).float()
        target_mask = target_mask.view(-1)
        link_points = predicted_out * target_mask
        return link_points, target.view(-1)

    def forward(self, batch_dt, model):
        in_words, in_masks, target, target_mask, in_words_cls, in_words_cls_mask = batch_dt
        batch_size, doc_len = in_words.size()
        txt_len = in_words.size(1)
        k = (txt_len // CHUNK_SIZE) + 1
        outputs = None
        chunk_edu_ids = torch.chunk(in_words, k, 1)
        for tmp_chunk in chunk_edu_ids:
            tmp_out = model(tmp_chunk)[0]
            outputs = tmp_out if outputs is None else torch.cat((outputs, tmp_out), 1)
        outputs = outputs.view(batch_size, doc_len, -1)
        result_e = None
        for batch_idx in range(batch_size):
            tmp_e_out = torch.index_select(outputs[batch_idx], 0, in_words_cls[batch_idx])
            tmp_e_out = tmp_e_out.unsqueeze(0)
            result_e = tmp_e_out if result_e is None else torch.cat((result_e, tmp_e_out), 0)
        result_e = torch.cat((self.root_hidden, result_e), dim=1)
        result_e = self.layer_norm(result_e)
        dep_attn_rst = self.bia_attn(result_e, result_e, target_mask)
        dep_score = dep_attn_rst.log_softmax(dim=2)
        return dep_score


class QEAttention(nn.Module):
    def __init__(self, encoder_size, decoder_size):
        super(QEAttention, self).__init__()
        self.bi_linear = BiLinearAttention(encoder_size, decoder_size, 1, HIDDEN_SIZE)

    def forward(self, e_outputs, d_outputs, masks=None):
        bi_attn = self.bi_linear(e_outputs, d_outputs)
        attn = bi_attn.squeeze(-1)
        if masks is not None:
            batch, de_num, en_num = attn.size()
            masks = masks.unsqueeze(1).expand(batch, de_num, en_num)
            attn[masks == 0] = -1e8
        return attn


class BiLinearAttention(nn.Module):
    def __init__(self, encoder_size, decoder_size, num_labels, hidden_size):
        super(BiLinearAttention, self).__init__()
        self.e_mlp = nn.Linear(encoder_size, hidden_size)
        self.d_mlp = nn.Linear(decoder_size, hidden_size)
        self.U = nn.Parameter(torch.empty(num_labels, hidden_size, hidden_size, dtype=torch.float))
        nn.init.xavier_normal_(self.U)

    def forward(self, e_outputs, d_outputs):
        """ e_outputs: (batch, length_encoder, encoder_size)
            d_outputs: (batch, length_decoder, decoder_size)
            length_decoder = split_num
            length_encoder = split_num + 2
        """
        e_outputs = self.e_mlp(e_outputs)  # (batch, length_encoder, hidden)
        d_outputs = self.d_mlp(d_outputs)  # (batch, length_decoder, hidden)
        out_u = d_outputs.unsqueeze(1) @ self.U
        out_u = out_u @ e_outputs.unsqueeze(1).transpose(2, 3)
        out = out_u.permute(0, 2, 3, 1)
        return out


class MaskedGRU(nn.Module):
    def __init__(self, *args, **kwargs):
        super(MaskedGRU, self).__init__()
        self.rnn = nn.GRU(batch_first=True, *args, **kwargs)
        self.hidden_size = self.rnn.hidden_size

    def forward(self, padded, lengths, initial_state=None):
        zero_mask = lengths != 0
        lengths[lengths == 0] += 1  # in case zero length instance
        _, indices = lengths.sort(descending=True)
        _, rev_indices = indices.sort()
        padded_sorted = padded[indices]
        lengths_sorted = lengths[indices]
        padded_packed = pack_padded_sequence(padded_sorted, lengths_sorted, batch_first=True)
        self.rnn.flatten_parameters()
        outputs_sorted_packed, hidden_sorted = self.rnn(padded_packed, initial_state)
        outputs_sorted, _ = pad_packed_sequence(outputs_sorted_packed, batch_first=True)
        outputs = outputs_sorted[rev_indices]
        # [batch*edu, output_size]
        hidden = hidden_sorted.transpose(1, 0).contiguous().view(outputs.size(0), -1)[rev_indices]
        outputs = outputs * zero_mask.view(-1, 1, 1).float()
        hidden = hidden * zero_mask.view(-1, 1).float()
        return outputs, hidden
