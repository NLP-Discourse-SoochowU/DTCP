# -*- coding: utf-8 -*-

"""
@Author: Lyzhang
@Date:
@Description:
"""
import torch
import torch.nn as nn
import torch.nn.functional as func
from config import *
from model.multi_head_attn import LexicalAttn
from model.glove_model.weight_assigner_dyg import PointNet, BiliNet, BiaNet
torch.manual_seed(SEED)


class Model(nn.Module):
    def __init__(self, word_emb=None):
        super(Model, self).__init__()
        # random 6992
        self.word_emb = nn.Embedding(word_emb.shape[0], 300)
        self.word_emb.weight.data.copy_(torch.from_numpy(word_emb))
        self.word_emb.weight.requires_grad = True if EMBED_LEARN else False
        self.pos_emb = nn.Embedding(POS_TAG_NUM, POS_TAG_SIZE)
        self.pos_emb.weight.requires_grad = True
        rnn_hidden = HIDDEN_SIZE // 2
        in_size = WORD_SIZE + POS_TAG_SIZE if USE_POS else WORD_SIZE
        if RNN_TYPE == "LSTM":
            self.sent_encode = nn.LSTM(in_size, rnn_hidden, num_layers=RNN_LAYER,
                                       dropout=DROPOUT, bidirectional=True, batch_first=True)
            self.context_encode = nn.LSTM(HIDDEN_SIZE, rnn_hidden, num_layers=RNN_LAYER,
                                          dropout=DROPOUT, bidirectional=True, batch_first=True)
            self.decoder = nn.LSTM(HIDDEN_SIZE, HIDDEN_SIZE, batch_first=True)
        else:
            self.sent_encode = nn.GRU(in_size, rnn_hidden, num_layers=RNN_LAYER,
                                      dropout=DROPOUT, bidirectional=True, batch_first=True)
            self.context_encode = nn.GRU(HIDDEN_SIZE, rnn_hidden, num_layers=RNN_LAYER,
                                         dropout=DROPOUT, bidirectional=True, batch_first=True)
            self.decoder = nn.GRU(HIDDEN_SIZE, HIDDEN_SIZE, batch_first=True)
        # (BATCH_SIZE, HIDDEN_SIZE)
        self.root_hidden = torch.rand(1, 1, HIDDEN_SIZE)
        nn.init.xavier_normal_(self.root_hidden)
        self.pointer = PointNet()
        self.bili_assigner = BiliNet()
        self.bia_assigner = BiaNet()
        self.dropout = nn.Dropout(0.1)
        self.sent_norm = nn.LayerNorm(HIDDEN_SIZE)
        self.lx_attn = LexicalAttn(in_size=in_size)
        if USE_CUDA:
            self.root_hidden = self.root_hidden.cuda(CUDA_ID)

    def weight_assign(self, en, de, trans_arr, ref_arr, mask1, mask2):
        """ en: (batch_size, seq_len, rnn_hidden)
            de: (batch_size, seq_len, rnn_hidden)
            target: (batch_size, seq_len)
            mask_: (batch_size, seq_len, seq_len) Dynamical & batch ens' representation learning.
        """
        if ATTN_TYPE == 0:
            att_weights = self.pointer(en, de, trans_arr, ref_arr, mask1)
        elif ATTN_TYPE == 1:
            att_weights = self.bili_assigner(en, de, trans_arr, ref_arr, mask1)
        else:
            att_weights = self.bia_assigner(en, de, trans_arr, ref_arr, mask1)
        mask_mul = 1 - mask2
        mask_pad = mask2 * SMOOTH_VAL  # [batch, seq_len + 1, seq_len + 1]
        att_weights = att_weights.mul(mask_mul) + mask_pad
        return att_weights

    def dynamic_weight_assign(self, en, de, mask2):
        """ dynamic batch encoding learning.
        """
        _, seq_num, _ = en.size()
        mask_mul = 1 - mask2
        mask_pad = mask2 * SMOOTH_VAL
        if ATTN_TYPE == 0:
            weights = self.pointer.dynamic_assign(en, de, mask_mul, mask_pad, seq_num)
        elif ATTN_TYPE == 1:
            weights = self.bili_assigner.dynamic_assign(en, de, mask_mul, mask_pad, seq_num)
        else:
            weights = self.bia_assigner.dynamic_assign(en, de, mask_mul, mask_pad, seq_num)
        return weights

    def sentence_encode(self, inputs):
        word_inputs, pos_inputs, sents = inputs
        word_reps = self.word_emb(word_inputs)  # (batch_size, seq_len, embed_size)
        pos_reps = self.pos_emb(pos_inputs)
        token_reps = torch.cat((word_reps, pos_reps), dim=-1) if USE_POS else word_reps
        if USE_LEXICAL_ATTN:
            token_reps, attn_weights = self.lx_attn(token_reps)
        else:
            attn_weights = None
        # GRU: h (layer_num * direct_num, seq_num, hidden_size/2)
        # LSTM: (h, c)
        hidden = self.sent_encode(token_reps)[1]
        hidden = hidden.permute(1, 0, 2)  # (seq_num, layer_num * direct_num, rnn_hidden)
        seq_num = hidden.size(0)
        hidden = hidden.contiguous().view(seq_num, -1).unsqueeze(0)
        # hidden = hidden.sum(1).unsqueeze(0)  # (seq_num, rnn_hidden)
        hidden = torch.cat((self.root_hidden, hidden), dim=1)  # (1, seq_len, rnn_hidden)
        if LAYER_NORM_USE:
            hidden = self.sent_norm(hidden)
        # ctx_hidden: (1, seq_len + 1, hidden_size), out_: (layer_num * direct_num, seq_num, hidden_size/2)
        ctx_hidden, out_ = self.context_encode(hidden)
        # ctx_hidden = hidden + self.dropout(ctx_hidden)
        out_ = out_.permute(1, 0, 2).unsqueeze(0).view(1, 1, -1)
        return ctx_hidden, out_, hidden, attn_weights

    def forward(self, inputs, trans_arr=None, ref_arr=None, mask1=None, mask2=None, mode_="train"):
        sents, word_inputs, pos_inputs, syn_inputs, syn_points = inputs
        ctx_hidden, out_, hidden, attn_weights = self.sentence_encode((word_inputs, pos_inputs, sents))
        de_hidden, _ = self.decoder(hidden, out_)
        link_weights = self.weight_assign(ctx_hidden, de_hidden, trans_arr, ref_arr, mask1, mask2) if mode_ == "train" \
            else self.dynamic_weight_assign(ctx_hidden, de_hidden, mask2)
        return link_weights

    def gen_loss(self, inputs, target):
        targets, mask1, mask2, trans_arr, ref_arr = target
        # (batch_size, seq_len + 1, seq_len + 1)
        link_weights = self(inputs, trans_arr=trans_arr, ref_arr=ref_arr, mask1=mask1, mask2=mask2)
        pred_prop = func.log_softmax(link_weights, dim=-1).squeeze(0)
        loss_ = func.nll_loss(pred_prop, targets)
        return loss_
