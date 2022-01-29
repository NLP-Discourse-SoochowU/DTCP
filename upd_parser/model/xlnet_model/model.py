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
from model.xlnet_model.weight_assigner_dyg import PointNet, BiliNet, BiaNet
torch.manual_seed(SEED)


class Model(nn.Module):
    def __init__(self, word_emb=None):
        super(Model, self).__init__()
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
        self.xl_dense = nn.Linear(WORD_SIZE, HIDDEN_SIZE)
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
        mask_pad = mask2 * SMOOTH_VAL  # (seq_len, seq_len)
        if ATTN_TYPE == 0:
            weights = self.pointer.dynamic_assign(en, de, mask_mul, mask_pad, seq_num)
        elif ATTN_TYPE == 1:
            weights = self.bili_assigner.dynamic_assign(en, de, mask_mul, mask_pad, seq_num)
        else:
            weights = self.bia_assigner.dynamic_assign(en, de, mask_mul, mask_pad, seq_num)
        return weights

    def sentence_encode(self, inputs, xl_m=None, xl_t=None):
        word_inputs, pos_inputs, sents = inputs
        sent_rep_idx = []
        b_idx = -1
        result_sent_ids = []
        for sent in sents:
            encoded_ids = xl_t.encode(sent, add_special_tokens=True)
            result_sent_ids += encoded_ids
            b_idx += len(encoded_ids)
            sent_rep_idx.append(b_idx)
        result_sent_ids = torch.tensor([result_sent_ids])  # (1, 6)
        if USE_CUDA:
            result_sent_ids = result_sent_ids.cuda(CUDA_ID)
        txt_len = result_sent_ids.size(1)
        k = (txt_len // CHUNK_SIZE) + 1
        outputs = None
        chunk_edu_ids = torch.chunk(result_sent_ids, k, 1)
        for tmp_chunk in chunk_edu_ids:
            tmp_out = xl_m(tmp_chunk)[0].squeeze()
            outputs = tmp_out if outputs is None else torch.cat((outputs, tmp_out), 0)
        if outputs is None:
            input("wrong")
        sent_rep_idx = torch.Tensor(sent_rep_idx).long()
        if USE_CUDA:
            sent_rep_idx = sent_rep_idx.cuda(CUDA_ID)
        hidden = torch.index_select(outputs, 0, sent_rep_idx)
        hidden = self.xl_dense(hidden).unsqueeze(0)
        hidden = torch.cat((self.root_hidden, hidden), dim=1)  # (1, seq_len, rnn_hidden)
        if LAYER_NORM_USE:
            hidden = self.sent_norm(hidden)
        ctx_hidden, out_ = self.context_encode(hidden)
        out_ = out_.permute(1, 0, 2).unsqueeze(0).view(1, 1, -1)
        return ctx_hidden, out_, hidden

    def forward(self, inputs, trans_arr=None, ref_arr=None, mask1=None, mask2=None, mode_="train", xl_m=None, xl_t=None):
        sents, word_inputs, pos_inputs, syn_inputs, syn_points = inputs
        ctx_hidden, out_, hidden = self.sentence_encode((word_inputs, pos_inputs, sents), xl_m, xl_t)
        de_hidden, _ = self.decoder(hidden, out_)
        link_weights = self.weight_assign(ctx_hidden, de_hidden, trans_arr, ref_arr, mask1, mask2) if mode_ == "train" \
            else self.dynamic_weight_assign(ctx_hidden, de_hidden, mask2)
        return link_weights

    def gen_loss(self, inputs, target, xl_m=None, xl_t=None):
        targets, mask1, mask2, trans_arr, ref_arr = target
        link_weights = self(inputs, trans_arr=trans_arr, ref_arr=ref_arr, mask1=mask1, mask2=mask2, xl_m=xl_m, xl_t=xl_t)
        pred_prop = func.log_softmax(link_weights, dim=-1).squeeze(0)
        loss_ = func.nll_loss(pred_prop, targets)
        return loss_
