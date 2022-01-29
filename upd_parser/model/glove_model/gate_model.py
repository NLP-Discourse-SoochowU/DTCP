# utf-8

"""
    Author: Lyzhang
    Date: 2018.8.15
    Description:
"""
import torch.nn as nn
from config import *


class G_Model(nn.Module):
    def __init__(self):
        super(G_Model, self).__init__()
        self.linear_pre = nn.Linear(HIDDEN_SIZE, 1)
        self.linear_tmp = nn.Linear(HIDDEN_SIZE, 1)
        self.sig = nn.Sigmoid()

    def forward(self, pre_rt, tmp_rt):
        """ pre_rt: (seq_num, seq_num, hidden)
            tmp_rt: (seq_num, seq_num, hidden)
        """
        gates = self.sig((self.linear_pre(pre_rt) + self.linear_tmp(tmp_rt)).squeeze(-1))
        return gates
