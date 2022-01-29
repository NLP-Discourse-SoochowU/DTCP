#-*-coding:utf-8 -*-
import numpy as np
import torch
from scipy.spatial import distance
from scipy.spatial.distance import pdist


def o_s(x, y):
    z=np.vstack([x,y])
    d2=pdist(z)
    return d2


def jacc(x, y):
    X = np.vstack([x, y])
    d2 = pdist(X, 'jaccard')
    return d2


def jsd(x, y):
    X = np.vstack([x, y])
    d2 = pdist(X, 'jensenshannon')
    return d2


def pearson(x, y):
    X = np.vstack([x, y])
    d2 = np.corrcoef(X)[0][1]
    return d2


def feature_normalize(data):
    data_p = torch.nn.functional.softmax(data)
    data_p = data_p.numpy()
    # max_ = np.max(data)
    # min_ = np.min(data)
    # z_data = data - min_
    # total = np.sum(z_data)
    return data_p
