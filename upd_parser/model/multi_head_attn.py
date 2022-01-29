# -*- coding: utf-8 -*-

"""
@Author: Lyzhang
@Date:
@Description:
"""
from config import HEAD_NUM, DROPOUT, HIDDEN_SIZE
import math
import torch
import torch.nn as nn
import torch.nn.functional as func


class MultiHeadAttention(nn.Module):
    def __init__(self, input_size, head_num=HEAD_NUM, hidden_size=HIDDEN_SIZE, dropout=DROPOUT):
        super().__init__()
        self.h = head_num
        self.d_k = hidden_size // head_num
        self.q_linear = nn.Linear(input_size, hidden_size)
        self.k_linear = nn.Linear(input_size, hidden_size)
        self.v_linear = nn.Linear(input_size, hidden_size)
        self.o_linear = nn.Linear(self.d_k*self.h, input_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):
        batch = q.size(0)  # (batch, sequence_len, hidden)

        # Linear transform to h heads.
        q = self.q_linear(q).view(batch, -1, self.h, self.d_k)
        k = self.k_linear(k).view(batch, -1, self.h, self.d_k)
        v = self.v_linear(v).view(batch, -1, self.h, self.d_k)

        # Transpose to (batch, h, sl, d_k)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Calculate attention weights.
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(-1).float()
            scores = scores.masked_fill(mask == 0, -1e8)
        scores = func.softmax(scores, dim=-1)
        if mask is not None:
            scores = scores * mask
        scores = self.dropout(scores)
        attn = scores @ v

        # Concatenate heads and put through final linear layer
        concat = attn.transpose(1, 2).contiguous().view(batch, -1, self.d_k*self.h)
        output = self.o_linear(concat)
        if mask is not None:
            mask = (mask.squeeze(1).sum(-1, keepdim=True) > 0).float()
            output = output * mask
        output = torch.sum(output, dim=1)
        return output


class SelfAttn(nn.Module):
    def __init__(self, in_size):
        super().__init__()
        self.attn_query = nn.Parameter(torch.randn(in_size))
        self.attn_key = nn.Sequential(
            nn.Linear(in_size, in_size),
            nn.Tanh()
        )

    def forward(self, in_, mask=None):
        keys = self.attn_key(in_)
        attn = func.softmax(keys.matmul(self.attn_query), 1).unsqueeze(-1)
        output = (in_ * attn).sum(1)
        return output, attn


class LexicalAttn(nn.Module):
    def __init__(self, in_size):
        super().__init__()
        self.attn_query = nn.Parameter(torch.randn(in_size))
        self.attn_key = nn.Sequential(
            nn.Linear(in_size, in_size),
            nn.Tanh()
        )

    def forward(self, in_, mask=None):
        keys = self.attn_key(in_)
        attn = func.softmax(keys.matmul(self.attn_query), 1).unsqueeze(-1)
        output = in_ * attn
        return output, attn
