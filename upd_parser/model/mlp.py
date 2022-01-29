# -*- coding: utf-8 -*-

"""
@Author: lyzhang
@Date:
@Description:
"""
import torch
from config import *
import torch.nn as nn
torch.manual_seed(SEED)


class MLP(nn.Module):
    def __init__(self, input_size=None, output_size=None, hidden_size=None):
        nn.Module.__init__(self)
        # linear probability
        self.linear_logits = nn.Linear(input_size, output_size)
        # input to first hidden
        self.input_linear = nn.Linear(input_size, hidden_size)
        self.input_dropout = nn.Dropout(p=DROPOUT)
        # multi hidden layers
        self.linears = nn.ModuleList([nn.Linear(hidden_size, hidden_size) for _ in range(MLP_Layers - 2)])
        self.dropouts = nn.ModuleList([nn.Dropout(p=DROPOUT) for _ in range(MLP_Layers - 2)])
        self.activations = nn.ModuleList([nn.ReLU() for _ in range(MLP_Layers - 2)])
        # probabilities
        self.logits = nn.Linear(hidden_size, output_size, bias=False)

    def forward(self, input_values):
        if MLP_Layers == 1:
            output = self.linear_logits(input_values)
        else:
            hidden = self.input_linear(input_values)
            hidden = self.input_dropout(hidden)
            # hidden layers
            for linear, dropout in zip(self.linears, self.dropouts):
                hidden = linear(hidden)
                hidden = dropout(hidden)
            output = self.logits(hidden)
        output = torch.tanh(output)
        return output
