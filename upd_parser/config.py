"""
@Author: Lyzhang
@Date: 2020.1.29
@Description:
"""
VERSION, SET = 7, 104
USE_CUDA, CUDA_ID = True, 0

INFO_DROPOUT, LR_DECAY, LAYER_NORM_USE = 0.33, False, False
USE_GATE = True
USE_XLNet_Base, USE_XLNet_Large, EMBED_LEARN = False, False, True
RNN_TYPE, MLP_Layers, ATTN_TYPE, HIDDEN_SIZE = "GRU", 2, 1, 128  # 0: PointNet, 1: BiliNet, 2: BiaNet
USE_POS, USE_LEXICAL_ATTN = True, True
SIM_TYPE = 0  # Cosine, os, jaac, pearson, JS_Div
SAVE_MODEL = True
ENC_STR = 2
CHUNK_SIZE = 768


LR = 0.001
THR = 0.83
# THR = 0.965
RNN_LAYER = 1  # both r-theme and context encoding.
Dist_Penalty, P_W, P_r = False, 1., 0.05  # P_r
HEAD_NUM = 1
L2 = 1e-5
SEED = 2
DROPOUT = 0.2
N_EPOCH = 20
BATCH_SIZE = 1  # doc num
LOG_EVE = 8
EVA_EVE = 32
PRINT_EVE = 256
PAD, PAD_ID = "<PAD>", 0
UNK, UNK_ID = "<UNK>", 1

if USE_XLNet_Base:
    WORD_SIZE = 768
elif USE_XLNet_Large:
    WORD_SIZE = 1024
else:
    WORD_SIZE = 300

POS_TAG_NUM, POS_TAG_SIZE = 47, 32
SYN_TAG_NUM, SYN_TAG_SIZE = 42, 32
sync2ids = {"head": 0, "dep": 1, "self": 2}
label2ids = {"NULL": 0, "TT": 1, "RT": 2, "TR": 3, "RR": 4}
SMOOTH_VAL = -1e2
