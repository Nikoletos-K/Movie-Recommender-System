import numpy as np

def cosine_similarity(vector1, vector2):
    vector1 = [item for item in vector1 if item is not None]
    vector2 = [item for item in vector2 if item is not None]

    # Ensure both vectors have the same length
    min_length = min(len(vector1), len(vector2))
    vector1 = vector1[:min_length]
    vector2 = vector2[:min_length]

    # Create NumPy arrays
    array1 = np.array(vector1)
    array2 = np.array(vector2)

    # Calculate cosine similarity
    dot_product = np.dot(array1, array2)
    norm1 = np.linalg.norm(array1)
    norm2 = np.linalg.norm(array2)

    similarity = dot_product / (norm1 * norm2) if norm1 != 0 and norm2 != 0 else 0

    return similarity

def dice_similarity(vector1, vector2):
    set1 = set(vector1)
    set2 = set(vector2)
    
    intersection_size = len(set1.intersection(set2))
    sum_sizes = len(set1) + len(set2)
    
    similarity = 2 * intersection_size / sum_sizes if sum_sizes != 0 else 0
    return similarity

def jaccard_similarity(vector1, vector2):
    set1 = set(vector1)
    set2 = set(vector2)
    
    intersection_size = len(set1.intersection(set2))
    union_size = len(set1.union(set2))
    
    similarity = intersection_size / union_size if union_size != 0 else 0
    return similarity

def pearson_similarity(vector1, vector2):
    # Remove None values
    vector1 = [item for item in vector1 if item is not None]
    vector2 = [item for item in vector2 if item is not None]

    # Create NumPy arrays
    array1 = np.array(vector1)
    array2 = np.array(vector2)

    # Calculate Pearson correlation coefficient
    correlation = np.corrcoef(array1, array2)[0, 1] if len(array1) == len(array2) and len(array1) > 0 else 0

    return correlation