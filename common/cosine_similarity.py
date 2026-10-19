import math
import numpy as np


def cosine_similarity_raw(a, b):
    """Cosine similarity of two equal-length lists of numbers, without NumPy.

    Same result as np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12).
    """
    if len(a) != len(b):
        raise ValueError("Both vectors need the same number of values.")

    # dot product: multiply matching numbers, then add them up
    dot_product = 0.0
    for i in range(len(a)):
        dot_product += a[i] * b[i]

    # length of a: square every number, add them up, take the square root
    sum_of_squares_a = 0.0
    for value in a:
        sum_of_squares_a += value * value
    length_a = math.sqrt(sum_of_squares_a)

    # length of b, the same way
    sum_of_squares_b = 0.0
    for value in b:
        sum_of_squares_b += value * value
    length_b = math.sqrt(sum_of_squares_b)

    # 1e-12 avoids dividing by zero when a vector is all zeros
    return dot_product / (length_a * length_b + 1e-12)


def cosine_similarity_numpy(a, b):
    dot_product = np.dot(a, b)
    length_a = np.linalg.norm(a)
    length_b = np.linalg.norm(b)
    # 1e-12 avoids dividing by zero
    return dot_product / (length_a * length_b + 1e-12)


if __name__ == "__main__":
    print(cosine_similarity_raw([1, 0], [1, 0]))   # same direction -> 0.9999999999989999
    print(cosine_similarity_raw([1, 0], [0, 1]))   # right angle -> 0.0
    print(cosine_similarity_raw([1, 0], [-1, 0]))  # opposite -> -0.9999999999989999
    print(cosine_similarity_raw([3, 4], [6, 8]))   # same direction, different length -> 0.99999999999998
