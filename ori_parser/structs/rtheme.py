# -*- coding: utf-8 -*-

"""
@Author: Lyzhang
@Date:
@Description:
"""
import pkuseg
from util.ltp import LTPParser


class RTheme:
    def __init__(self, link_id=None, ref_ids=None, txt=None, cored_txt=None, tag_id=None, txt_type=None, tag_rt=None,
                 location=None, key_type=None, zero_type=None, link_type=None, use_time=None):
        self.id = tag_id
        self.type_ = txt_type
        self.position = tag_rt
        self.location = location
        self.key = key_type
        self.rtype = zero_type
        self.link_id = link_id
        self.link_type = link_type
        self.use_time = use_time
        self.z_type = None
        self.z_link = None
        self.z_position = None
        self.ref_id = ref_ids
        self.txt = txt
        self.cored_txt = cored_txt
        self.words = []
        self.cored_words = None
        self.pos_tags = None
        self.dep_arc = None
        self.init_attr()

    def init_attr(self):
        seg = pkuseg.pkuseg()
        self.cored_words = seg.cut(self.cored_txt)
        print(self.cored_words)
        flag_ = False
        for word in self.cored_words:
            if word == "<N" or word == "<S" or word == "<C":
                flag_ = True
                continue
            if word == ">" and flag_:
                input("aaa")
                flag_ = False
                continue
            self.words.append(word)
        with LTPParser() as parser:
            dep_arc, pos_tags = parser.parse(self.cored_words)
            self.dep_arc = dep_arc
            self.pos_tags = pos_tags

