# -*- coding: utf-8 -*-

"""
@Author: Lyzhang
@Date:
@Description: Dynamical weights assigner building.
"""
from config import *
import torch
import torch.nn as nn
from model.glove_model.gate_model import G_Model
torch.manual_seed(SEED)


class PointNet(nn.Module):
    def __init__(self):
        super(PointNet, self).__init__()
        rnn_hidden = HIDDEN_SIZE // 2
        self.nnW1 = nn.Linear(HIDDEN_SIZE, rnn_hidden, bias=False)
        self.nnW2 = nn.Linear(rnn_hidden, rnn_hidden, bias=False)
        self.nnV = nn.Linear(rnn_hidden, 1, bias=False)
        self.relu = nn.ReLU()
        self.info_drop = nn.Dropout(INFO_DROPOUT)
        self.g_model = G_Model()

    def dynamic_assign(self, en, de, mask_mul, mask_pad, seq_num):
        """ test period, decode one by one.
        """
        en = en.squeeze(0)  # (seq_num, hidden)
        en_ = en.clone()
        wd = self.nnW2(de).squeeze(0)  # (seq_num, hidden)
        # dynamic assigning.
        weights = None
        for idx_ in range(seq_num):
            # for the decoder out at the idx_
            de_ = wd[idx_]
            # weight computation
            we = self.nnW1(en_)
            att_weights = self.nnV(self.relu(we + de_)).squeeze(-1).unsqueeze(0)
            att_weights = att_weights.mul(mask_mul[idx_]) + mask_pad[idx_]
            weights = att_weights if weights is None else torch.cat((weights, att_weights), dim=0)
            # update attention weights
            tmp_idx = torch.argmax(att_weights, dim=-1)
            if tmp_idx > 0:
                gate_ = self.g_model(en[tmp_idx], en_[idx_])
                en_[idx_] = en_[idx_] + self.info_drop(gate_.mul(en[tmp_idx]).squeeze(0))
        return weights

    def forward(self, en, de, target, mask_):
        """ en: (batch_size, seq_len, rnn_hidden)
            de: (batch_size, seq_len, rnn_hidden)
        """
        _, seq_num, hidden = en.size()
        en = en.squeeze(0)
        en_l = en[target]
        # gate generate
        gates = self.g_model(en_l, en).unsqueeze(0).expand(seq_num, seq_num).unsqueeze(-1)
        en_l = en_l.unsqueeze(0).expand(seq_num, seq_num, hidden)
        # (batch, seq_len, seq_len, hidden)
        mask_ = mask_.unsqueeze(-1)
        ex_en = en + self.info_drop(mask_.mul(gates.mul(en_l))).unsqueeze(0)
        we = self.nnW1(ex_en)
        de = de.unsqueeze(2)  # (batch, seq_num, 1, hidden)
        wd = self.nnW2(de)
        # (batch, seq_num, seq_num)
        att_weights = self.nnV(self.relu(we + wd)).squeeze(-1)
        return att_weights


class BiliNet(nn.Module):
    def __init__(self):
        super(BiliNet, self).__init__()
        rnn_hidden = HIDDEN_SIZE // 2
        self.nnW1 = nn.Linear(HIDDEN_SIZE, rnn_hidden, bias=False)
        self.nnW2 = nn.Linear(HIDDEN_SIZE + HIDDEN_SIZE, rnn_hidden, bias=False)
        self.u1 = nn.Parameter(torch.empty(rnn_hidden, rnn_hidden, dtype=torch.float))
        nn.init.xavier_normal_(self.u1)
        self.info_drop = nn.Dropout(INFO_DROPOUT)
        self.g_model = G_Model()

    def dynamic_assign(self, en, de, mask_mul, mask_pad, seq_num):
        """ test period
        """
        en = en.squeeze(0)  # (seq_num, hidden)
        en_ = en.clone()
        de = de.squeeze(0)
        weights = None
        for idx_ in range(seq_num):
            # specific decoder output (hidden, 1)
            de_ = torch.cat((de[idx_], en_[idx_]), -1)
            de_ = self.nnW2(de_).squeeze(0).unsqueeze(-1)  # (seq_num, hidden)
            # weight computation，
            we = self.nnW1(en_)
            att_weights = we.matmul(self.u1).matmul(de_).squeeze(-1).unsqueeze(0)  # (seq_num, 1)
            att_weights = att_weights.mul(mask_mul[idx_]) + mask_pad[idx_]
            weights = att_weights if weights is None else torch.cat((weights, att_weights), dim=0)
            # update attention weights
            tmp_idx = torch.argmax(att_weights, dim=-1)
            if tmp_idx > 0:
                if USE_GATE:
                    gate_ = self.g_model(en[tmp_idx], en_[idx_])
                    en_[idx_] = en_[idx_] + self.info_drop(gate_.mul(en[tmp_idx]).squeeze(0))
                else:
                    en_[idx_] = en_[idx_] + self.info_drop(en[tmp_idx].squeeze(0))
        return weights

    def forward(self, en, de, trans_arr, ref_arr, mask_):
        """ en: (batch_size, seq_len, rnn_hidden)
            de: (batch_size, seq_len, rnn_hidden)
        """
        _, seq_num, hidden = en.size()
        en = en.squeeze(0)
        en_l = en[trans_arr]
        # gate generate
        gates = self.g_model(en_l, en).unsqueeze(0).expand(seq_num, seq_num).unsqueeze(-1)
        en_l = en_l.unsqueeze(0).expand(seq_num, seq_num, hidden)
        mask_ = mask_.unsqueeze(-1)
        if USE_GATE:
            ex_en = en + self.info_drop(mask_.mul(gates.mul(en_l))).unsqueeze(0)
        else:
            ex_en = en + self.info_drop(mask_.mul(en_l)).unsqueeze(0)
        ref_arr = ref_arr.unsqueeze(-1).unsqueeze(-1).expand(ref_arr.size(0), 1, HIDDEN_SIZE)
        de_ref = ex_en.squeeze(0).gather(1, ref_arr).squeeze(1).unsqueeze(0)
        de = torch.cat((de, de_ref), -1)
        we = self.nnW1(ex_en).squeeze(0)  # # (seq_len, seq_len, hidden)
        wd = self.nnW2(de).squeeze(0).unsqueeze(-1)  # (seq_num, hidden, 1)
        # (batch, seq_num, seq_num)
        att_weights = we.matmul(self.u1).bmm(wd).squeeze(-1).unsqueeze(0)
        return att_weights


class BiaNet(nn.Module):
    def __init__(self):
        super(BiaNet, self).__init__()
        rnn_hidden = HIDDEN_SIZE // 2
        self.nnW1 = nn.Linear(HIDDEN_SIZE, rnn_hidden, bias=False)
        self.nnW2 = nn.Linear(HIDDEN_SIZE + HIDDEN_SIZE, rnn_hidden, bias=False)
        self.u1 = nn.Parameter(torch.empty(rnn_hidden, rnn_hidden, dtype=torch.float))
        self.u2 = nn.Parameter(torch.empty(rnn_hidden, 1, dtype=torch.float))
        self.b = nn.Parameter(torch.zeros(1, 1, 1, dtype=torch.float))
        nn.init.xavier_normal_(self.u1)
        nn.init.xavier_normal_(self.u2)
        nn.init.xavier_normal_(self.b)
        self.info_drop = nn.Dropout(INFO_DROPOUT)
        self.g_model = G_Model()

    def dynamic_assign(self, en, de, mask_mul, mask_pad, seq_num):
        """ test period
        """
        en = en.squeeze(0)  # (seq_num, hidden)
        en_ = en.clone()
        de = de.squeeze(0)
        weights = None
        for idx_ in range(seq_num):
            # specific decoder output (hidden, 1)
            de_ = torch.cat((de[idx_], en_[idx_]), -1)
            de_ = self.nnW2(de_).squeeze(0).unsqueeze(-1)  # (seq_num, hidden)
            # weight computation，
            we = self.nnW1(en_)
            part_a = we.matmul(self.u1).matmul(de_)  # (seq_num, 1)
            de_ = de_.permute(1, 0)
            part_b = (we + de_).matmul(self.u2)  # (seq_num, 1)
            att_weights = (part_a + part_b + self.b.squeeze(0)).squeeze(-1).unsqueeze(0)
            att_weights = att_weights.mul(mask_mul[idx_]) + mask_pad[idx_]
            weights = att_weights if weights is None else torch.cat((weights, att_weights), dim=0)
            # update attention weights
            tmp_idx = torch.argmax(att_weights, dim=-1)
            if tmp_idx > 0:
                gate_ = self.g_model(en[tmp_idx], en_[idx_])
                en_[idx_] = en_[idx_] + self.info_drop(gate_.mul(en[tmp_idx]).squeeze(0))
        return weights

    def forward(self, en, de, trans_arr, ref_arr, mask_):
        """ en: (batch_size, seq_len, rnn_hidden)
            de: (batch_size, seq_len, rnn_hidden)
        """
        _, seq_num, hidden = en.size()
        en = en.squeeze(0)
        en_l = en[trans_arr]
        # gate generate
        gates = self.g_model(en_l, en).unsqueeze(0).expand(seq_num, seq_num).unsqueeze(-1)
        en_l = en_l.unsqueeze(0).expand(seq_num, seq_num, hidden)
        mask_ = mask_.unsqueeze(-1)
        ex_en = en + self.info_drop(mask_.mul(gates.mul(en_l))).unsqueeze(0)
        ref_arr = ref_arr.unsqueeze(-1).unsqueeze(-1).expand(ref_arr.size(0), 1, HIDDEN_SIZE)
        de_ref = ex_en.squeeze(0).gather(1, ref_arr).squeeze(1).unsqueeze(0)
        de = torch.cat((de, de_ref), -1)
        we = self.nnW1(ex_en).squeeze(0)  # # (seq_len, seq_len, hidden)
        wd = self.nnW2(de).squeeze(0).unsqueeze(-1)  # (seq_num, hidden, 1)
        # (batch, seq_num, seq_num)
        part_a = we.matmul(self.u1).bmm(wd)
        wd = wd.permute(0, 2, 1)
        part_b = (we + wd).matmul(self.u2)
        att_weights = (part_a + part_b + self.b).squeeze(-1).unsqueeze(0)
        return att_weights
