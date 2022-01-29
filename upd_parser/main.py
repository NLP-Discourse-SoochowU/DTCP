# -*- coding: utf-8 -*-

"""
@Author: Lyzhang
@Date:
@Description:
"""
from config import VERSION
from model.glove_model.trainer import Trainer as gl_trainer
from model.xlnet_model.trainer import Trainer as xl_trainer

if __name__ == "__main__":
    if VERSION == 7:
        trainer = gl_trainer()
    else:
        trainer = xl_trainer()
    trainer.train()
