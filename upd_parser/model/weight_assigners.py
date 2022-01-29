# -*- coding: utf-8 -*-

"""
@Author: Lyzhang
@Date:
@Description:
"""
from config import *
import torch
import torch.nn as nn
torch.manual_seed(SEED)


class PointNet(nn.Module):
    def __init__(self, nnw2_in=None):
        super(PointNet, self).__init__()
        self.nnSELU = nn.SELU()
        rnn_hidden = HIDDEN_SIZE // 2
        self.nnW1 = nn.Linear(HIDDEN_SIZE, rnn_hidden, bias=False)
        self.nnW2 = nn.Linear(rnn_hidden, rnn_hidden, bias=False) if nnw2_in is None \
            else nn.Linear(nnw2_in, rnn_hidden, bias=False)
        self.nnV = nn.Linear(rnn_hidden, 1, bias=False)

    def forward(self, en, de):
        """ en: (batch_size, seq_len, rnn_hidden)
            de: (batch_size, seq_len, rnn_hidden)
        """
        batch, seq_num, hidden = en.size()
        we = self.nnW1(en).unsqueeze(2)  # (batch, seq_num, 1, hidden)
        we = we.permute(2, 0, 1, 3)  # (1, batch, seq_num, hidden)
        # (batch, seq_num, seq_num, hidden)
        ex_we = we.expand(seq_num, batch, seq_num, HIDDEN_SIZE // 2).permute(1, 0, 2, 3)

        de = de.unsqueeze(2)  # (batch, seq_num, 1, hidden)
        wd = self.nnW2(de)
        # [batch, seq_num + 1, seq_num + 1]
        observed_h = ex_we + wd
        att_weights = self.nnV(self.nnSELU(observed_h)).squeeze(-1)
        return att_weights


class BiliNet(nn.Module):
    def __init__(self, nnw2_in=None):
        super(BiliNet, self).__init__()
        rnn_hidden = HIDDEN_SIZE // 2
        self.nnW1 = nn.Linear(HIDDEN_SIZE, rnn_hidden, bias=False)
        self.nnW2 = nn.Linear(rnn_hidden, rnn_hidden, bias=False) if nnw2_in is None \
            else nn.Linear(nnw2_in, rnn_hidden, bias=False)
        self.u1 = nn.Parameter(torch.empty(rnn_hidden, rnn_hidden, dtype=torch.float))
        nn.init.xavier_normal_(self.u1)

    def forward(self, en, de):
        """ en: (batch_size, seq_len, rnn_hidden)
            de: (batch_size, seq_len, rnn_hidden)
        """
        we = self.nnW1(en)  # (batch, seq_num, hidden)
        wd = self.nnW2(de)  # (batch, seq_num, hidden)
        # (batch, seq_num + 1, seq_num + 1)
        att_weights = we.matmul(self.u1).bmm(wd.transpose(1, 2)).transpose(1, 2)
        return att_weights


class BiaNet(nn.Module):
    def __init__(self, nnw2_in=None):
        super(BiaNet, self).__init__()
        rnn_hidden = HIDDEN_SIZE // 2
        self.nnW1 = nn.Linear(HIDDEN_SIZE, rnn_hidden, bias=False)
        self.nnW2 = nn.Linear(rnn_hidden, rnn_hidden, bias=False)if nnw2_in is None \
            else nn.Linear(nnw2_in, rnn_hidden, bias=False)
        self.u1 = nn.Parameter(torch.empty(rnn_hidden, rnn_hidden, dtype=torch.float))
        self.u2 = nn.Parameter(torch.empty(rnn_hidden, 1, dtype=torch.float))
        self.b = nn.Parameter(torch.zeros(1, 1, 1, dtype=torch.float))
        nn.init.xavier_normal_(self.u1)
        nn.init.xavier_normal_(self.u2)
        nn.init.xavier_normal_(self.b)

    def forward(self, en, de):
        """ en: (batch_size, seq_len, rnn_hidden)
            de: (batch_size, seq_len, rnn_hidden)
        """
        we = self.nnW1(en)  # (batch, seq_num, hidden)
        wd = self.nnW2(de)  # (batch, seq_num, hidden)
        # (batch, seq_num, seq_num)
        part_a = we.matmul(self.u1).bmm(wd.transpose(1, 2)).transpose(1, 2)
        part_b = (we + wd).matmul(self.u2).squeeze(-1).unsqueeze(1)  # (batch, 1<per decoder>, length_encoder)
        att_weights = part_a + part_b + self.b
        return att_weights
