import numpy as np
from scipy.spatial.distance import cosine, dice, jaccard
from scipy.stats import pearsonr

def cosine_similarity(a, b):
    return 1 - cosine(a, b)

def dice_similarity(a, b):
    return 1 - dice(a, b)

def jaccard_similarity(a, b):
    return 1 - jaccard(a, b)

def pearson_similarity(a, b):
    corr, _ = pearsonr(a, b)
    return corr
