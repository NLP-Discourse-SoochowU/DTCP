# -*- coding: utf-8 -*-

"""
@Author: Lyzhang
@Date:
@Description:
"""
from structs.sentence import Sent


class Document:
    def __init__(self, doc_info=None):
        doc_name, lines = doc_info
        self.doc_name = doc_name
        self.sentences = []
        self.init_doc(lines)

    def init_doc(self, lines=None):
        """ Enrich the document with sentence information
        """
        for line in lines:
            self.sentences.append(Sent(line))
