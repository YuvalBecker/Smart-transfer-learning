import numpy as np
from collections import defaultdict
from scipy import stats
import matplotlib.pyplot as plt
import os
import torch
import scipy
import torchvision
def kl(p, q):
    # Kl divergence metric
    p = np.abs(np.asarray(p, dtype=np.float) + 1e-15)
    q = np.abs(np.asarray(q, dtype=np.float) + 1e-15)
    p = p / np.sum(p)
    q = q / np.sum(q)

    return np.sum(np.where(p != 0, p * np.log(p / q), 0))

def smoothed_hist_kl_distance(a, b, nbins=20):
    ahist, bhist = (np.histogram(a, bins=nbins)[0],
                    np.histogram(b, bins=nbins)[0])
    return kl(ahist, bhist)
