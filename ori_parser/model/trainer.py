# -*- coding: utf-8 -*-

"""
@Author: Lyzhang
@Date: 2020.11.27
@Description:
"""
import random
import numpy as np
import progressbar
import torch
import torch.optim as optim
from config import *
from model.model import DTCP
from utils.file_util import *
from path_config import *
from transformers import *
from datetime import datetime

random.seed(SEED)
torch.manual_seed(SEED)
np.random.seed(SEED)


class Trainer:
    def __init__(self):
        set_all = load_data(EMNLP_DT)
        train_set, dev_set, test_set = set_all
        self.train_set = train_set
        self.dev_set = dev_set
        self.test_set = test_set
        self.model = DTCP()
        self.loss_all = []
        self.dev_max = 0.
        self.te_max = 0.
        self.report_qeh = 0.
        self.report_qer = 0.
        self.report_qw = 0.
        self.report_qfw = 0.
        if USE_BERT_base:
            self.tr_model = BertModel.from_pretrained('bert-base-uncased')
        else:
            self.tr_model = BertModel.from_pretrained('bert-large-uncased')
        if EMBED_LEARN is False:
            self.tr_model.eval()
        if CUDA_ID >= 0:
            self.model.cuda(CUDA_ID)
            self.tr_model.cuda(CUDA_ID)
        print(str(self.model))

    def train(self, test_desc=""):
        """ The main process of training and evaluating.
        """
        log_ite = 1
        log_file = os.path.join(LOG_ALL, "set_" + str(SET) + ".log")
        n_iter, log_loss_all = 0, 0.
        optimizer = optim.Adam([
            {'params': self.model.parameters()},
            {'params': self.tr_model.parameters(), 'lr': LR}], lr=LR, weight_decay=L2)
        optimizer.zero_grad()
        if PARAM_ANA:
            total_trainable_params = sum(
                p.numel() for p in self.model.parameters() if p.requires_grad)
            print(f'{total_trainable_params:,} training parameters.')
            total_trainable_params_tr = sum(
                p.numel() for p in self.tr_model.parameters() if p.requires_grad)
            print(f'{total_trainable_params_tr:,} training parameters.')
            input()

        p = progressbar.ProgressBar()
        early_stop_k = 0
        last_time = None
        for n_epoch in range(1, N_EPOCH + 1):
            if last_time is None:
                last_time = datetime.now()
            else:
                temp_time = datetime.now()
                secs = (temp_time - last_time).seconds
                input(secs)
            p.start(LOG_EVE)
            batch_iter = self.gen_batch_iter()
            for n_batch, batch_disc in enumerate(batch_iter, start=1):
                self.model.train()
                if EMBED_LEARN:
                    self.tr_model.train()
                n_iter += 1
                p.update((n_iter % LOG_EVE))
                loss_ = self.model.loss(batch_disc, self.tr_model)
                loss_all = loss_
                loss_all.backward()
                log_loss_all += loss_all.item()
                if n_iter % UPDATE_EVE == 0:
                    optimizer.step()
                    optimizer.zero_grad()
                if n_iter % LOG_EVE == 0:
                    p.finish()
                    print_("\niter: " + str(n_iter) + ", epoch: " + str(n_epoch) + ", loss_all: " +
                           str(log_loss_all / LOG_EVE), log_file)
                    self.loss_all.append(log_loss_all)
                    log_loss_all = 0.
                    p.start(LOG_EVE)
                if n_iter % EVA_EVE == 0:
                    score, _, _ = self.evaluate("dev")
                    p_, r_, f_ = score
                    print_("==== Dev: (P) " + str(p_) + " (R) " + str(r_) + " (F) " + str(f_) + " ====", log_file)
                    if f_ > self.dev_max:
                        self.dev_max = f_
                        score_t, predicts, targets = self.evaluate("test")
                        p_t, r_t, f_t = score_t
                        if f_t > self.te_max:
                            self.te_max = f_t
                            print_("==== Test: (P) " + str(p_t) + " (R) " + str(r_t) + " (F) " + str(f_t) + " ====",
                                   log_file)
                            log_ite += 1
                            self.save_model()
                            self.save_test(predicts, targets)
                        early_stop_k = 0.
                    else:
                        early_stop_k += 1
                        if early_stop_k >= EARLY_STOP_ALL and n_epoch > LOWER_BOUND:
                            print_("early stop!", log_file)
                            return

    @staticmethod
    def save_test(predicts, targets):
        save_data((predicts, targets), PRED_TEST + str(SET) + ".pkl")

    def save_model(self):
        if SAVE_MODEL:
            torch.save(self.model, os.path.join(MODEL_SAVE, "DTCP_" + str(SET) + ".model"))

    @staticmethod
    def report_max_t(score_, n_epoch=0):
        p_, r_, f_ = score_
        check_str = "---" + "VERSION: " + str(VERSION) + ", SET: " + str(SET) + ", EPOCH: " + str(n_epoch) + \
                    "---\n" + "TEST: " + str(p_) + "(P), " + str(r_) + "(R), " + str(f_) + "(F)\n"
        return check_str

    def evaluate(self, type_="dev"):
        self.model.eval()
        self.tr_model.eval()
        score, predicts, targets = self.met_micro(type_)
        return score, predicts, targets

    def met_micro(self, type_="dev"):
        predicts_all = list()
        targets_all = list()
        dev_batch_iter = self.gen_batch_iter(type_)
        c_, g_, h_ = 0., 0., 0.
        for n_batch, batch_disc in enumerate(dev_batch_iter, start=1):
            predicts, targets = self.model.predict(batch_disc, self.tr_model)  # (batch_size, seq_len + 1, seq_len + 1)
            predicts_all.append(predicts)
            targets_all.append(targets)
            if USE_CUDA:
                predicts = predicts.cpu()
                targets = targets.cpu()
            pred_ = predicts.int().numpy()
            gold_ = targets.numpy()
            link_idx = np.where(gold_ > 0)[0]
            pred_link = pred_[link_idx]
            gold_link = gold_[link_idx]
            c_ += np.sum(pred_link == gold_link)
            g_ += np.where(gold_ > 0)[0].shape[0]
            h_ += np.where(pred_ > 0)[0].shape[0]
        p_ = 0. if h_ == 0 else c_ / h_
        r_ = 0. if g_ == 0 else c_ / g_
        f_ = 0. if (g_ + h_) == 0 else (2 * c_) / (g_ + h_)
        return (p_, r_, f_), predicts_all, targets_all

    def met_macro(self, type_="dev"):
        dataset = self.dev_set if type_ == "dev" else self.test_set
        predicts_all = list()
        targets_all = list()
        dev_batch_iter = self.gen_batch_iter(dataset)
        f_all, p_all, r_all, doc_all = 0., 0., 0., 0.
        for n_batch, batch_disc in enumerate(dev_batch_iter, start=1):
            predicts, targets = self.model.predict(batch_disc, self.tr_model)  # (batch_size, seq_len + 1, seq_len + 1)
            predicts_all.append(predicts)
            targets_all.append(targets)
            if USE_CUDA:
                predicts = predicts.cpu()
                targets = targets.cpu()
            pred_ = predicts.int().numpy()
            gold_ = targets.numpy()
            link_idx = np.where(gold_ > 0)[0]
            pred_link = pred_[link_idx]
            gold_link = gold_[link_idx]
            c_ = np.sum(pred_link == gold_link)
            g_ = np.where(gold_ > 0)[0].shape[0]
            h_ = np.where(pred_ > 0)[0].shape[0]
            p_ = 0. if h_ == 0 else c_ / h_
            r_ = 0. if g_ == 0 else c_ / g_
            f_ = 0. if (g_ + h_) == 0 else (2 * c_) / (g_ + h_)
            p_all += p_
            r_all += r_
            f_all += f_
            doc_all += 1
        macro_p = p_all / doc_all
        macro_r = r_all / doc_all
        macro_f = f_all / doc_all
        return (macro_p, macro_r, macro_f), predicts_all, targets_all

    def report(self, epoch, iter_, test_desc, log_file):
        info_str = "Epoch: " + str(epoch) + "   iteration: " + str(iter_) + "  desc: " + test_desc
        print_(str_=info_str, log_file=log_file)
        print_(str_="Results on Test E-HIT: " + str(self.report_qeh) + " == E-RST: " + str(self.report_qer) +
                    " == Exact-W: " + str(self.report_qw) + " == F1-W: " + str(self.report_qfw), log_file=log_file)

    def gen_batch_iter(self, type_="train"):
        if type_ == "train":
            instances = self.train_set
        elif type_ == "dev":
            instances = self.dev_set
        else:
            instances = self.test_set
        if type_ == "train":
            random_instances = np.random.permutation(instances)
        else:
            random_instances = instances
        num_instances = len(instances)
        offset = 0
        while offset < num_instances:
            batch = random_instances[offset: min(num_instances, offset + BATCH_SIZE)]
            num_batch = len(batch)
            if num_batch < BATCH_SIZE:
                offset += BATCH_SIZE
                continue
            max_word_num = 0
            max_cls_num = 0
            for disc in batch:
                sents_ids_base, sents_ids_large, link_idx, cls_base, cls_large = disc
                in_ids = sents_ids_base if USE_BERT_base else sents_ids_large
                cls_ids = cls_base if USE_BERT_base else cls_large
                max_word_num = max_word_num if max_word_num >= len(in_ids) else len(in_ids)
                max_cls_num = max_cls_num if max_cls_num >= len(cls_ids) else len(cls_ids)
            in_words = np.zeros([num_batch, max_word_num], dtype=np.long)
            in_words_cls = np.zeros([num_batch, max_cls_num], dtype=np.long)
            in_words_cls_mask = np.zeros([num_batch, max_cls_num], dtype=np.long)
            in_masks = np.zeros([num_batch, max_word_num], dtype=np.float)
            target = np.zeros([num_batch, max_cls_num + 1], dtype=np.long)
            target_mask = np.zeros([num_batch, max_cls_num + 1], dtype=np.float)
            for batch_idx, disc in enumerate(batch):
                sents_ids_base, sents_ids_large, link_idx, cls_base, cls_large = disc
                in_ids = sents_ids_base if USE_BERT_base else sents_ids_large
                cls_ids = cls_base if USE_BERT_base else cls_large
                word_len = len(in_ids)
                in_words[batch_idx][:word_len] = in_ids
                in_masks[batch_idx][:word_len] = 1
                target_len = len(link_idx)
                target[batch_idx][:target_len] = link_idx
                target_mask[batch_idx][:target_len] = 1
                cls_len = len(cls_ids)
                in_words_cls[batch_idx][:cls_len] = cls_ids
                in_words_cls_mask[batch_idx][:cls_len] = 1
            in_words = torch.from_numpy(in_words).long()
            in_masks = torch.from_numpy(in_masks).float()
            target = torch.from_numpy(target).long()
            target_mask = torch.from_numpy(target_mask).float()
            in_words_cls = torch.from_numpy(in_words_cls).long()
            in_words_cls_mask = torch.from_numpy(in_words_cls_mask).float()
            if USE_CUDA:
                in_words = in_words.cuda(CUDA_ID)
                in_masks = in_masks.cuda(CUDA_ID)
                target = target.cuda(CUDA_ID)
                target_mask = target_mask.cuda(CUDA_ID)
                in_words_cls = in_words_cls.cuda(CUDA_ID)
                in_words_cls_mask = in_words_cls_mask.cuda(CUDA_ID)
            yield (in_words, in_masks, target, target_mask, in_words_cls, in_words_cls_mask)
            offset += BATCH_SIZE
