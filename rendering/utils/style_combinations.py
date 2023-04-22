"""
Utility functions to compute model style combinations.
"""
import math
import random

import numpy as np
from tqdm import tqdm


def pack_list(src_list, T, shuffle=False):
    """
    Pack a list of lists into a list of lists of length T.
    """
    N = len(src_list)
    # copy the nested list
    nested_list = [list[:] for list in src_list]
    nested_list.sort(key=len)

    col_list = [[] for _ in range(N)]
    for k in tqdm(range(T)):
        for n in range(N):
            if k < len(nested_list[n]):
                col_list[n].append(nested_list[n][k])
                nested_list[n][k] = None
            else:
                # pop element from the last row, last column
                # if there is no element, go to the n-1 row, etc.
                alt_idx = list(range(N - 1, -1, -1))
                if shuffle:
                    random.shuffle(alt_idx)
                for m in alt_idx:
                    if len(nested_list[m]) > k and nested_list[m][-1] is not None:
                        col_list[n].append(nested_list[m].pop())
                        break

    return list(map(list, zip(*col_list)))


def max_combs(part_map, mat_categories):
    """
    Get the number of combinations for each model.
    """
    # Enumerate all materials on a given model.
    model_comb = {}
    for model_id, part_mats in part_map.items():
        model_comb[model_id] = 1
        for _, mats in part_mats.items():
            parts_comb = sum([len(mat_categories[mat]) for mat in mats])
            model_comb[model_id] *= parts_comb

    # Clip the number of combinations to 10**24
    max_comb = 10**124
    for model_id, comb in model_comb.items():
        if comb > max_comb:
            model_comb[model_id] = max_comb

    return model_comb


def compute_adjust_gap(comb_dict, T):
    """
    Return the gap between the average number
    of combinations and the target number of combinations.
    """
    # Compute number of models above TARGET_COMB
    above_comb = [v for v in comb_dict.values() if v > T]
    above_comb = [min(v, T) for v in above_comb]
    above_ids = [k for k, v in comb_dict.items() if v > T]

    # Compute number of models below TARGET_COMB
    below_comb = [v for v in comb_dict.values() if v <= T]
    below_ids = [k for k, v in comb_dict.items() if v <= T]

    k = len(below_comb)
    j = len(above_comb)
    to_add = (T * k - sum(below_comb)) / j
    # new_avg = (sum(above_comb) + sum(below_comb) + to_add*j)/len(model_comb_fine)
    # new_avg, to_add, (math.ceil(to_add)*j + sum(above_comb) + sum(below_comb)) - T*len(model_comb_fine)
    return (below_ids, below_comb), (above_ids, above_comb), to_add


def adjust_above_combs(l_below, l_above, T, to_add):
    """
    Adjust the number of combinations for models with
    more than T combinations, to compensate for models
    with less than T combinations.
    """
    l_above = [k + to_add for k in l_above]

    # Use a list comprehension to convert each float to an int and scale it
    int_lst = l_below + [math.floor(num) for num in l_above]

    # Adjust up to k elements of the list to make sure the sum is still N
    N = T * (len(l_below) + len(l_above))
    remainder = N - sum(int_lst)

    k = len(l_above)
    if remainder > k:
        per_adj = remainder // k
        last_adj = remainder - per_adj * k

        int_lst = [math.floor(num) + per_adj for num in l_above[:-1]]
        int_lst[-1] = int_lst[-1] + last_adj
    else:
        int_lst = [math.floor(num) + 1 for num in l_above[:remainder]] + [
            math.floor(num) for num in l_above[remainder:]
        ]

    return int_lst


def adjusted_combs(model_comb, T):
    """
    Return a dictionary mapping model_id
    to the adjusted number of combinations to use for that model.
    """
    below_map, above_map, to_add = compute_adjust_gap(model_comb, T)
    below_ids, below_comb = below_map
    above_ids, above_comb = above_map

    adjusted_above_comb = adjust_above_combs(below_comb, above_comb, T, to_add)
    adjusted_comb = adjusted_above_comb + below_comb

    # Make a dict mapping model_id to adjusted number of combinations
    return {
        m_id: adjusted_comb
        for m_id, adjusted_comb in zip(above_ids + below_ids, adjusted_comb)
    }


def check_combs_dict(adjusted_dict, full_dict, T):
    """
    Verify that the adjusted combinations to not exceed the real number possible in full_dict.
    """
    # Iterate over each model
    for model_id, comb in adjusted_dict.items():
        # Get the number of combinations for the model in full_dict
        full_comb = full_dict[model_id]
        # Check if the adjusted number of combinations is greater than the real number
        if comb > full_comb:
            # If so, return False
            return False

    # Check that the mean is equal to T, and that the sum is equal to T*n_models
    if np.mean(list(adjusted_dict.values())) != T:
        return False
    if sum(adjusted_dict.values()) != T * len(adjusted_dict):
        return False

    # If all models pass the check, return True
    return True
