# -*- coding: utf-8 -*-

"""
@Author: Lyzhang
@Date: 2020.1.29
@Description:
"""
from path_config import *
from stanfordcorenlp import StanfordCoreNLP
nlp = StanfordCoreNLP(STANFORD_PATH)


class Sent:
    def __init__(self, sent_text):
        self.sent_text = sent_text
        self.sent_id = None
        self.point_id = None
        self.tokens = None
        self.pos_tags = None
        self.syn_tags = None
        self.syn_point = None
        self.init_all()

    def init_all(self):
        """ To initialize the sentence information for batch learning.
            sentence tokenize
            tag analysis
            syntax analysis Auto
        """
        splits = self.sent_text.strip().split(" ")
        self.sent_id = int(splits[0])
        self.point_id = int(splits[2])
        sent_text_ = " ".join(splits[4:])
        tok_pairs = nlp.pos_tag(sent_text_)
        self.tokens = [pair[0] for pair in tok_pairs]
        self.pos_tags = [pair[1] for pair in tok_pairs]

        dep = nlp.dependency_parse(sent_text_)

        deps = len(dep)
        syn_dict = dict()
        flag = False
        flag2 = False
        base_ = 0
        for idx, dep_ in enumerate(dep):
            # update the dep information for modifying
            if flag and dep_[1] == 0:
                flag2 = True
                base_ = idx
                syn_dict[dep_[2] - 1 + base_] = (dep_[0], dep_[1] - 1)
            else:
                syn_dict[dep_[2] - 1 + base_] = (dep_[0], dep_[1] - 1 + base_)
            flag = True
        self.syn_tags = [syn_dict[idx][0] for idx in range(deps)]
        self.syn_point = [syn_dict[idx][1] for idx in range(deps)]
        self.sent_text = sent_text_
