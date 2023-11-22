import numpy as np

def cosine_similarity(vector1, vector2):

     # Replace None values with -1
    vector1 = [val if val is not None else -1 for val in vector1]
    vector2 = [val if val is not None else -1 for val in vector2]

    # Convert the vectors to numpy arrays
    array1 = np.array(vector1)
    array2 = np.array(vector2)
    
    # Calculate the dot product and magnitudes
    dot_product = np.dot(array1, array2)
    magnitude1 = np.linalg.norm(array1)
    magnitude2 = np.linalg.norm(array2)

    # Avoid division by zero
    if magnitude1 == 0 or magnitude2 == 0:
        return 0.0
    else:
        # Calculate cosine similarity
        similarity = dot_product / (magnitude1 * magnitude2)
        return similarity

def dice_similarity(vector1, vector2):
# Replace None values with 0
    vector1 = [1 if val is not None and val != 0 else 0 for val in vector1]
    vector2 = [1 if val is not None and val != 0 else 0 for val in vector2]

    # Calculate the intersection and sum of elements for each vector
    intersection = sum(val1 * val2 for val1, val2 in zip(vector1, vector2))
    sum_vector1 = sum(val1 for val1 in vector1)
    sum_vector2 = sum(val2 for val2 in vector2)

    # Avoid division by zero
    if sum_vector1 == 0 and sum_vector2 == 0:
        return 1.0
    elif sum_vector1 == 0 or sum_vector2 == 0:
        return 0.0

    # Calculate Dice similarity
    similarity = (2 * intersection) / (sum_vector1 + sum_vector2)
    return similarity

def jaccard_similarity(vector1, vector2):
    # Convert the vectors to sets of non-zero indices
    set1 = set(i for i, value in enumerate(vector1) if value != 0)
    set2 = set(i for i, value in enumerate(vector2) if value != 0)

    # Calculate the intersection and union of the sets
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))

    # Calculate Jaccard similarity
    if union == 0:
        return 0.0
    else:
        return float(intersection) / union

def pearson_similarity(vector1, vector2):
    # Replace None values with NaN
    vector1 = [val if val is not None else float('nan') for val in vector1]
    vector2 = [val if val is not None else float('nan') for val in vector2]

    # Calculate mean of non-NaN values for each vector
    mean1 = sum(val for val in vector1 if not np.isnan(val)) / (len(vector1) - vector1.count(float('nan')))
    mean2 = sum(val for val in vector2 if not np.isnan(val)) / (len(vector2) - vector2.count(float('nan')))

    # Calculate the numerator and denominators of the Pearson correlation formula
    numerator = sum((vector1[i] - mean1) * (vector2[i] - mean2) for i in range(len(vector1)) if not np.isnan(vector1[i]) and not np.isnan(vector2[i]))
    denominator_x = sum((vector1[i] - mean1)**2 for i in range(len(vector1)) if not np.isnan(vector1[i]))
    denominator_y = sum((vector2[i] - mean2)**2 for i in range(len(vector2)) if not np.isnan(vector2[i]))

    # Avoid division by zero
    if denominator_x == 0 or denominator_y == 0:
        return 0.0

    # Calculate Pearson correlation
    correlation = numerator / (denominator_x**0.5 * denominator_y**0.5)
    return correlation