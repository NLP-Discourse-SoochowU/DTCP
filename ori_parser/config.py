"""
@Author: Lyzhang
@Date: 2020.1.29
@Description:
"""
VERSION, SET = 1000, 104
USE_CUDA, CUDA_ID = True, 0
USE_BERT_base, USE_BERT_large, EMBED_LEARN = True, False, True
BATCH_SIZE, N_EPOCH, LOG_EVE, EVA_EVE, PRINT_EVE, UPDATE_EVE = 1, 20, 20, 100, 384, 1
EARLY_STOP_ALL, LOWER_BOUND = 1, 20
LR = 0.00001
CHUNK_SIZE = 64
HIDDEN_SIZE = 256

# ========================================================

INFO_DROPOUT, LR_DECAY, LAYER_NORM_USE = 0.33, False, False
USE_GATE = False
USE_ELMo, USE_BERT = False, False
RNN_TYPE, MLP_Layers, ATTN_TYPE = "GRU", 2, 1  # 0: PointNet, 1: BiliNet, 2: BiaNet
USE_POS, USE_LEXICAL_ATTN = True, True
SIM_TYPE = 0  # Cosine, os, jaac, pearson, JS_Div
SAVE_MODEL = True
PARAM_ANA = True
TRAIN_XLNET = False
ENC_STR = 2
THR = 0.83
# THR = 0.965
RNN_LAYER = 1  # both r-theme and context encoding.
Dist_Penalty, P_W, P_r = False, 1., 0.05  # P_r 0.1~0.5
HEAD_NUM = 4
L2 = 1e-5
SEED = 2
DROPOUT = 0.2
PAD, PAD_ID = "<PAD>", 0
UNK, UNK_ID = "<UNK>", 1
if USE_BERT_base:
    WORD_SIZE = 768
else:
    WORD_SIZE = 1024
POS_TAG_NUM, POS_TAG_SIZE = 47, 32
SYN_TAG_NUM, SYN_TAG_SIZE = 42, 32
sync2ids = {"head": 0, "dep": 1, "self": 2}
label2ids = {"NULL": 0, "TT": 1, "RT": 2, "TR": 3, "RR": 4}
SMOOTH_VAL = -1e2
