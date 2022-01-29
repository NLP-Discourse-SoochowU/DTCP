# -*- coding: utf-8 -*-

"""
@Author: Lyzhang
@Date:
@Description:
"""
import re
from structs.rt_ import RT


class EDTU:
    def __init__(self, edtu_text, edtu_sent_id):
        self.edu_id = None
        self.sent_id = edtu_sent_id
        self.theme = None
        self.rheme = None
        self.text = None
        self.init_tr(edtu_text)

    def init_tr(self, tr_txt):
        theme_link_id = t_tag_id = t_txt_type = t_location = t_key_type = t_zero_type = t_link_type = \
            t_use_time = rheme_link_id = r_tag_id = r_txt_type = r_location = r_key_type = r_zero_type = \
            r_link_type = r_use_time = None
        t_tag_rt = r_tag_rt = "None"
        print("================================")
        theme_txt, theme_cored_txt = "", ""
        rheme_txt, rheme_cored_txt = "", ""
        theme_ref_ids, rheme_ref_ids = [], []
        useless_flag = False
        useless2_flag = False
        print("tr_txt: ", tr_txt)
        if len(tr_txt) == 1:
            self.text = RT(cored_txt=tr_txt[0])
            # print("A: Sent and no RTs")
        else:
            for txt in tr_txt:
                if txt.startswith("</MTS>"):
                    continue
                else:
                    # print("tmp_line: ", txt)
                    txt_obj = re.search('(<MTS ID=.+?>)(.+)', txt)
                    if txt_obj is None:
                        tmp_tag = txt
                        tmp_txt = ""
                    else:
                        tmp_tag = txt_obj.group(1)
                        tmp_txt = txt_obj.group(2)
                    print("tmp_tag: ", tmp_tag)
                    print("tmp_txt: ", tmp_txt)
                    tag_trg = re.search('<MTS +ID="([.\d]+)".+?TYPE="([a-zA-Z]+?)" +POSITION="([a-zA-Z]+?)" +'
                                        'LOCATION="([a-zA-Z]+?)" +KEY="([a-zA-Z]+?)" +RTYPE="([a-zA-Z]+?)" +'
                                        'LINKID="([.\d]+)".+?LINKTYPE="([a-zA-Z]+?)".+?USETIME="(\d+)".*?>', tmp_tag)
                    tag_id, txt_type, tag_rt, location, key_type, zero_type, link_id, link_type, use_time = \
                        tag_trg.group(1), tag_trg.group(2), tag_trg.group(3), tag_trg.group(4), tag_trg.group(5), \
                        tag_trg.group(6), tag_trg.group(7), tag_trg.group(8), tag_trg.group(9)
                    if "." in tag_id:
                        if tag_rt == "Theme":
                            if key_type == "Satellite":
                                theme_cored_txt += " <S "
                            elif key_type == "Nucleus":
                                theme_cored_txt += " <N "
                            elif key_type == "Complex":
                                theme_cored_txt += " <C "
                            else:
                                input("wrong 4")
                            for tok in tmp_txt:
                                if tok == "<":
                                    useless2_flag = True
                                if not useless2_flag:
                                    theme_cored_txt += tok
                                    theme_txt += tok
                                if tok == ">":
                                    useless2_flag = False
                                    theme_cored_txt += " > "
                            if not theme_link_id == link_id.split(".")[0]:
                                theme_ref_ids.append(link_id.split(".")[0])
                        elif tag_rt == "Rheme":
                            if key_type == "Nucleus":
                                rheme_cored_txt += " <N "
                            elif key_type == "Satellite":
                                rheme_cored_txt += " <S "
                            elif key_type == "Complex":
                                rheme_cored_txt += " <C "
                            else:
                                input("wrong 5")
                            for tok in tmp_txt:
                                if tok == "<":
                                    useless2_flag = True
                                if not useless2_flag:
                                    rheme_cored_txt += tok
                                    rheme_txt += tok
                                if tok == ">":
                                    useless2_flag = False
                                    rheme_cored_txt += " > "
                            if not rheme_link_id == link_id.split(".")[0]:
                                rheme_ref_ids.append(link_id.split(".")[0])
                        else:
                            input("wrong 3")
                    else:
                        if tag_rt == "Theme":
                            theme_link_id = link_id.split(".")[0]
                            t_tag_id, t_txt_type, t_tag_rt, t_location, t_key_type, t_zero_type, t_link_type, t_use_time = \
                                tag_id, txt_type, tag_rt, location, key_type, zero_type, link_type, use_time
                        else:
                            rheme_link_id = link_id.split(".")[0]
                            r_tag_id, r_txt_type, r_tag_rt, r_location, r_key_type, r_zero_type, r_link_type, r_use_time = \
                                tag_id, txt_type, tag_rt, location, key_type, zero_type, link_type, use_time
                        for tok in tmp_txt:
                            if tok == "<":
                                useless_flag = True
                            elif tok == ">":
                                useless_flag = False
                            elif not useless_flag:
                                if tag_rt == "Theme":
                                    theme_txt += tok
                                    theme_cored_txt += tok
                                elif tag_rt == "Rheme":
                                    rheme_txt += tok
                                    rheme_cored_txt += tok
                                else:
                                    input("wrong 2")
                            else:
                                pass

            if "null" in theme_txt:
                theme_txt.replace("null", "null ")
            if "null" in rheme_txt:
                rheme_txt.replace("null", "null ")
            self.theme = RT(theme_link_id, theme_ref_ids, theme_txt, theme_cored_txt, t_tag_id, t_txt_type, t_tag_rt,
                            t_location, t_key_type, t_zero_type, t_link_type, t_use_time)
            self.rheme = RT(rheme_link_id, rheme_ref_ids, rheme_txt, rheme_cored_txt, r_tag_id, r_txt_type, r_tag_rt,
                            r_location, r_key_type, r_zero_type, r_link_type, r_use_time)
