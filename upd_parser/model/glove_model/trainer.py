# -*- coding: utf-8 -*-

"""
@Author: Lyzhang
@Date:
@Description:
"""
from model.glove_model.metric import met
import random
import numpy as np
import torch
import torch.optim as optim
from config import *
from model.glove_model.model import Model
from path_config import *
from utils.data_iterator import gen_batch_iter
from utils.file_util import *

random.seed(SEED)
torch.random.manual_seed(SEED)
np.random.seed(SEED)


class Trainer:
    def __init__(self):
        self.log_file = os.path.join(LOG_ALL, "set_" + str(SET) + ".log")
        self.train_set, self.dev_set, self.test_set = load_data(TRAIN_DATA)
        word_emb = load_data(IDS2VEC)
        self.model = Model(word_emb)
        if USE_CUDA:
            self.model.cuda(CUDA_ID)

    @staticmethod
    def get_lr(optimizer):
        for param_group in optimizer.param_groups:
            return param_group['lr']

    def train(self):
        """ Train and evaluate
        """
        log_file = os.path.join(LOG_ALL, "set_" + str(SET) + ".log")
        log_ite = 1
        optimizer = optim.Adam(self.model.parameters(), lr=LR, weight_decay=L2)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, "max", factor=0.1, patience=7) if LR_DECAY else None
        step, best_score, best_score_, loss_ = 0, 0., 0., 0.
        for epoch in range(1, N_EPOCH):
            batch_iter = gen_batch_iter(self.train_set)
            for n_batch, (inputs, target) in enumerate(batch_iter, start=1):
                step += 1
                self.model.train()
                optimizer.zero_grad()
                loss_ = loss_ + self.model.gen_loss(inputs, target)
                if n_batch > 0 and n_batch % LOG_EVE == 0:
                    loss_ = loss_ / float(LOG_EVE * BATCH_SIZE)
                    loss_.backward()
                    optimizer.step()
                    self.report_each(self.get_lr(optimizer), epoch, n_batch, loss_.item())
                    loss_ = 0.
                    if n_batch % EVA_EVE == 0:
                        score, _, _ = self.evaluate(self.dev_set)
                        p_, r_, f_ = score
                        print("-- evaluation on the dev: (P) " + str(p_) + " (R) " + str(r_) + " (F) " + str(f_))
                        if LR_DECAY:
                            scheduler.step(f_, epoch)
                        if f_ > best_score:
                            best_score = f_
                            score, predicts, targets = self.evaluate(self.test_set, mode_="test")
                            f_t = score[2]
                            if f_t > best_score_:
                                best_score_ = f_t
                                log_str = self.report_max_t(score, n_epoch=epoch)
                                log_eve(ite=log_ite, str_=log_str, log_file=log_file)
                                log_ite += 1
                                self.save_model((predicts, targets), self.model)

    def evaluate(self, dataset, mode_="dev"):
        self.model.eval()
        score, predicts, targets = met(dataset, self.model, mode_)
        return score, predicts, targets

    @staticmethod
    def report_max_t(score_, n_epoch=0):
        p_, r_, f_ = score_
        check_str = "---" + "VERSION: " + str(VERSION) + ", SET: " + str(SET) + ", EPOCH: " + str(n_epoch) + \
                    "---\n" + "TEST: " + str(p_) + "(P), " + str(r_) + "(R), " + str(f_) + "(F)" + "\n"
        return check_str

    @staticmethod
    def report_each(lr=0., epoch=0, n_batch=0, loss_=0.):
        print("VERSION: " + str(VERSION) + ", SET: " + str(SET) + " -- lr %f, epoch %d, batch %d, loss %.4f" %
              (lr, epoch, n_batch, loss_))

    @staticmethod
    def save_model(dt, model):
        if SAVE_MODEL:
            save_data(dt, PRED_TEST + str(SET) + ".pkl")
            torch.save(model, os.path.join(MODEL_SAVE, "DTCP_" + str(SET) + ".model"))
