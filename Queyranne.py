# -*- Some useful packages -*-
from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import numpy.linalg as la
import copy
from collections import Sequence
from itertools import chain, count
from scipy.linalg import block_diag
from typing import Any

# -*- Some useful preliminary functions -*-
def trinv(matrix):
    tri = np.trace(la.inv(matrix))
    return tri

def permutation_matrix(n):
    rr = range(n)
    np.random.shuffle(rr)
    P = np.take(np.eye(n), rr, axis=0)
    return  P

def select_mat(matrix, index_row, index_column):
    # sort the row/cols lists
    index_row.sort()
    index_column.sort()
    index_row = list(set(index_row))
    index_column = list(set(index_column))
    S = np.transpose(np.transpose(matrix[index_row])[index_column])
    return S

def diff(first, second):
        second = set(second)
        return [item for item in first if item not in second]

def ismember(a, B):
    response = False
    index = 0
    while response == False and index < len(B):
        b = B[index]
        if b == a:
            return True
        else:
            index = index+1

    return response

def partition(collection):
    if len(collection) == 1:
        yield [ collection ]
        return
    first = collection[0]
    for smaller in partition(collection[1:]):
        # insert `first` in each of the subpartition's subsets
        for n, subset in enumerate(smaller):
            yield smaller[:n] + [[ first ] + subset]  + smaller[n+1:]
        # put `first` in its own subset
        yield [ [ first ] ] + smaller

def select_from_list(list, indices):
    list_select = [list[i] for i in indices]
    return list_select


# Finding a pendent-pair of a supermodular system(V,f):
# For a given sets W and Q, find the most far element
# in Q to W
# W and Q should be list of lists to account for fused elements

def Find_Far_Element(SS, F, WW, QQ):

    # Find the most far element to WW in QQ

    u = QQ[0]  # a list not an index
    W_cp = copy.copy(WW)
    W_cp.append(u)
    dist_max = F(SS, W_cp) - F(SS, u)
    elt_far = u
    Q_ = copy.copy(QQ)
    Q_.remove(u)
    for elt in Q_:
        W_cp = copy.copy(WW)
        W_cp.append(elt)
        dist_elt = F(SS, W_cp) - F(SS, elt)
        if dist_elt > dist_max:
            dist_max = dist_elt
            elt_far = elt

    return elt_far

# ----- Finding a pendent pair is a fundamental step in Queyranne's algorithm ----- #
def PENDENT_PAIR(SS, VV, F):

    # V is the set of all points including fused pairs
    # The size of V goes from n to 2
    # Start with a random element in V

    V_ = copy.copy(VV)
    rnd_pattern = np.random.permutation(len(V_))
    #x = V_[rnd_pattern[0]]
    x = V_[0]
    if type(x) == list:
        W = x
    else:
        W = [x]
    Q = copy.copy(V_)
    Q.remove(x)
    V_.remove(x)
    for i in range(len(V_)):
        elt_far = Find_Far_Element(SS, F, W, Q)
        W.append(elt_far)
        Q.remove(elt_far)

    return W[-2], W[-1]

def tr_inv(SS, set):
    """

    :rtype: float
    """
    if type(set) == int:
        LIST = [set]
    else:
        LIST = []
        for i in range(len(set)):
            if type(set[i]) == list:
                LIST.extend(set[i])
            else:
                LIST.append(set[i])

    return trinv(select_mat(SS, LIST, LIST))

def log_det(SS, set):
    """

    :rtype: float
    """
    if type(set) == int:
        LIST = [set]
    else:
        LIST = []
        for i in range(len(set)):
            if type(set[i]) == list:
                LIST.extend(set[i])
            else:
                LIST.append(set[i])

    return -np.log(la.det(select_mat(SS, LIST, LIST)))

def fuse(A, B):
    if type(A) == int and type(B) == int:
        f = [A, B]
    elif type(A) == int and type(B) == list:
        f = [A] + B
    elif type(A) == list and type(B) == int:
        f = A + [B]
    elif type(A) == list and type(B) == list:
        f = A + B

    return f

#-*- Full implementation of Queyranne's algorithm -*-

def QUEYRANNE(SS, F):
    # type: (matrix, function) -> (list, float, list)

    dim, _ = SS.shape
    V = range(dim)  # is the space of points which is updated at each step we find a pendent pair
    C = []  # set of candidates updated at each step
    while len(V) >= 3:
        W = copy.deepcopy(V)
        a, b = PENDENT_PAIR(SS, W, F)  # find a pendent pair in (V,F)
        if type(b) == int:
            C.append([b])
        else:
            C.append(b)

        fus = fuse(a, b)  # fuse this pair as a list
        V.append(fus)
        if ismember(a, V) is True and ismember(b, V) is True:
            V.remove(a)
            V.remove(b)

    for subset in V:
        if type(subset) == int:
            C.append([subset])
        else:
            C.append(subset)


    #  Once we have the list of candidates, we return the best one
    max_value = -np.Inf
    subset_opt = []
    cluster_max = 0
    partition_value = 0
    for subset in C:
        cluster_value = F(SS, subset)
        subset_value = cluster_value + F(SS, diff(range(dim), subset))
        if subset_value > max_value:
            subset_opt = subset
            partition_value = subset_value
            cluster_max = cluster_value
            max_value = subset_value

    return subset_opt, partition_value, cluster_max
