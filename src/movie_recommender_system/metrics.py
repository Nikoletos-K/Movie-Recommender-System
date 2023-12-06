import numpy as np

def cosine_similarity(vector1: np.array, vector2: np.array):
    dot_product = np.dot(vector1, vector2)
    magnitude1 = np.linalg.norm(vector1)
    magnitude2 = np.linalg.norm(vector2)

    if magnitude1 == 0 or magnitude2 == 0:
        return 0.0
    else:
        similarity = dot_product / (magnitude1 * magnitude2)

        return similarity

def dice_similarity(vector1: np.array, vector2: np.array):
    intersection = np.sum(vector1 * vector2)
    sum_vector1 = np.sum(vector1)
    sum_vector2 = np.sum(vector2)

    if sum_vector1 == 0 and sum_vector2 == 0:
        return 1.0
    elif sum_vector1 == 0 or sum_vector2 == 0:
        return 0.0
    similarity = (2 * intersection) / (sum_vector1 + sum_vector2)

    return similarity

def jaccard_similarity(vector1: np.array, vector2: np.array):
    set1 = set(vector1.nonzero()[0])
    set2 = set(vector2.nonzero()[0])
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))

    if union == 0:
        return 0.0
    else:
        return float(intersection) / union

def pearson_similarity(vector1: np.array, vector2: np.array):
    common_indices = np.nonzero((vector1 != 0) & (vector2 != 0))[0]

    if len(common_indices) == 0:
        return 0.0

    v1_common = vector1[common_indices]
    v2_common = vector2[common_indices]

    mean1 = np.mean(v1_common)
    mean2 = np.mean(v2_common)

    numerator = np.sum((v1_common - mean1) * (v2_common - mean2))
    denominator_x = np.sum((v1_common - mean1)**2)
    denominator_y = np.sum((v2_common - mean2)**2)

    if denominator_x == 0 or denominator_y == 0:
        return 0.0

    correlation = numerator / (np.sqrt(denominator_x) * np.sqrt(denominator_y))

    return correlation