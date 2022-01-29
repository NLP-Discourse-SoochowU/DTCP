# -*- coding: utf-8 -*-

"""
@Author: Lyzhang
@Date:
@Description:
"""
from config import *
import torch
import torch.nn as nn


class BiAffineAttention(nn.Module):
    def __init__(self, encoder_size, decoder_size, num_labels=1, hidden_size=HIDDEN_SIZE):
        super(BiAffineAttention, self).__init__()
        self.encoder_size = encoder_size
        self.decoder_size = decoder_size
        self.num_labels = num_labels
        self.hidden_size = hidden_size
        self.e_mlp = nn.Sequential(
            nn.Linear(encoder_size, hidden_size),
            nn.ReLU()
        )
        self.d_mlp = nn.Sequential(
            nn.Linear(decoder_size, hidden_size),
            nn.ReLU()
        )
        self.W_e = nn.Parameter(torch.empty(num_labels, hidden_size, dtype=torch.float))
        self.W_d = nn.Parameter(torch.empty(num_labels, hidden_size, dtype=torch.float))
        self.U = nn.Parameter(torch.empty(num_labels, hidden_size, hidden_size, dtype=torch.float))
        self.b = nn.Parameter(torch.zeros(num_labels, 1, 1, dtype=torch.float))
        nn.init.xavier_normal_(self.W_e)
        nn.init.xavier_normal_(self.W_d)
        nn.init.xavier_normal_(self.U)

    def forward(self, e_outputs, d_outputs):
        """ :param e_outputs: (batch, length_encoder, encoder_size)
            :param d_outputs: (batch, length_decoder, decoder_size)
            encoder_size == decoder_size = HIDDEN_SIZE
        """
        e_outputs = self.e_mlp(e_outputs)  # (batch, length_encoder, hidden_size)
        d_outputs = self.d_mlp(d_outputs)  # (batch, length_encoder, hidden_size)
        out_e = (self.W_e @ e_outputs.transpose(1, 2)).unsqueeze(2)  # (batch, num_labels, 1, length_encoder)
        out_d = (self.W_d @ d_outputs.transpose(1, 2)).unsqueeze(3)  # (batch, num_labels, length_decoder, 1)
        # [batch, 1, length_decoder, hidden_size] @ [num_labels, hidden_size, hidden_size]
        # [batch, num_labels, length_decoder, hidden_size]
        out_u = d_outputs.unsqueeze(1) @ self.U
        # [batch, num_labels, length_decoder, hidden_size] * [batch, 1, hidden_size, length_encoder]
        # [batch, num_labels, length_decoder, length_encoder]
        out_u = out_u @ e_outputs.unsqueeze(1).transpose(2, 3)
        # [batch, length_decoder, length_encoder, num_labels]
        out = (out_e + out_d + out_u + self.b).permute(0, 2, 3, 1)
        return out
