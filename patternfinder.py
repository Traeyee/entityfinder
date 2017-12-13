# -*- coding: utf-8 -*-

"""
1. Build a frequecy matrix M (how to define frequency?)
2. Decompose M into sorted eigenvalues and eigenvectors (It's clear)
3. Estimate the parameter k (A clear computing procedure)
4. Build principal eigenspace with first k eigenvectors and get the projection of M in principal eigenspace
5. Segment the query
If the number of segmented parts does not equal to k, modify sigma goto step 5
"""

from compatitator import strdecode
from collections import Counter
import sys
import numpy as np
import re
sys.path.append("/Users/traeyee/Codes/common")
from ACAutomaton import ACAutomaton
from functions import eigence


ptn_stf = re.compile(u"(\[STUFF\])+")
ptn_tgt = re.compile(u"(\[TARGET\])+")
ptn_tgt_1 = re.compile(u"\[TARGET\]")
num_semantics = 3
flag_test = False


def generalizer(pattern0, list_pattern, dict_reverse_idx, dict_word_freq):
    len_seed = len(u"[TARGET]")

    list_pos_seed = list()
    pos = 0
    pattern1 = pattern0
    idx = pattern1.find(u"[TARGET]", pos)
    while idx != -1:
        list_pos_seed.append(idx)
        pattern1 = pattern1[: idx] + u"_" + pattern1[idx + len_seed:]
        pos = idx + 1
        idx = pattern1.find(u"[TARGET]", pos)

    # return pattern1

    dimension = len(pattern1)

    if dimension <= 1:
        return u""

    sum_freq = 0.0
    for i, w in enumerate(pattern1):
        if i in list_pos_seed:
            sum_freq += dict_word_freq[u"[TARGET]"]
        else:
            sum_freq += dict_word_freq[w]

    matrix = np.zeros((dimension, dimension))
    for i in range(dimension):
        for j in range(dimension):
            mij = 0.0
            if i == j:
                if i in list_pos_seed:
                    mij = dict_word_freq[u"[TARGET]"] / sum_freq
                else:
                    mij = dict_word_freq[pattern1[i]] / sum_freq
            elif i < j:
                cnt = 0
                substring = u""
                flag_empty = True
                pre_set = set()
                for k in range(i, j + 1):
                    if k in list_pos_seed:
                        substring += u"[TARGET]"
                    else:
                        substring += pattern1[k]
                        if flag_empty:
                            pre_set = set(dict_reverse_idx[pattern1[k]])
                            flag_empty = False
                        else:
                            pre_set &= dict_reverse_idx[pattern1[k]]
                # substring = stn[i: j + 1]
                # pre_set = set(dict_rev_idx[substring[0]])
                # for w in substring[1:]:
                #     pre_set &= dict_rev_idx[w]
                for idx in pre_set:
                    if substring in list_pattern[idx]:
                        cnt += 1

                mij = cnt / sum_freq
            else:
                mij = matrix[j][i]
            matrix[i][j] = mij

    # Normalize
    for i in range(dimension):
        for j in range(dimension):
            if i != j:
                matrix[i][j] = 2 * matrix[i][j] / (matrix[i][i] + matrix[j][j])

    # print "Matrix computation ends: " + stn.encode("utf-8")
    eigenvalues, eigenvectors = eigence(matrix)

    eig_val_indices = np.argsort(eigenvalues)  # 按特征值由小到大排序
    sum_eig_val = 0.0
    for val in eigenvalues:
        sum_eig_val += val
    threshold = ((dimension - 1) / float(dimension)) ** 2
    k = 0
    vals_k = 0.0
    for i in range(len(eigenvalues) - 1, -1, -1):
        if vals_k >= threshold:
            break
        vals_k += eigenvalues[eig_val_indices[i]]
        k += 1

    # eigenspace = eigenvectors[:, eig_val_indices[-1: -k-1: -1]]
    kept_indices = eig_val_indices[-1: -k-1: -1]  # 最大的k个特征值的索引

    if dimension != k:
        db = "db"

    str_pattern = ""
    if flag_test:
        cnt = 0
        principal_space = eigenvectors[:, kept_indices]
        sigma = np.float128(0.5)
        low_bound = np.float128(0.0)
        high_bound = np.float128(1.0)
        while True:
            cnt += 1
            list_is_added = [False] * k
            ufs = range(dimension)
            list_kept_part = list(list_pos_seed)
            for i in range(k):
                for j in range(i + 1, k):
                    ai = principal_space[i]
                    aj = principal_space[j]
                    if np.dot(ai, aj.T) / (np.sum(np.square(ai)) * np.sum(np.square(aj))) >= sigma:
                        if kept_indices[i] in list_pos_seed:
                            list_kept_part.append(kept_indices[j])
                        elif kept_indices[j] in list_pos_seed:
                            list_kept_part.append(kept_indices[i])

                        if i < ufs[j]:
                            ufs[j] = i
            # Count the number of ufs
            m = 0
            dict_weight = dict()
            for j in range(k):
                # Find the root
                dad = ufs[j]
                while dad != ufs[dad]:
                    dad = ufs[dad]
                if not list_is_added[dad]:
                    list_is_added[dad] = True
                    m += 1
                if dad not in dict_weight:
                    dict_weight[dad] = 0.0
                dict_weight[dad] += eigenvalues[kept_indices[j]]
                ufs[j] = dad
                list_is_added[j] = True
            # print pattern0
            # print "%f\t%f\t%f\t%d\t%d" % (low_bound, sigma, high_bound, m, k)
            if m < k:
                low_bound = sigma
                sigma = (sigma + high_bound) / 2
            elif m > k:
                high_bound = sigma
                sigma = (sigma + low_bound) / 2
            else:
                break

            if cnt >= 32:
                break
        list_kept_root = [t[0] for t in sorted(dict_weight.items(), key=lambda t: t[1], reverse=True)[: num_semantics]]
        for i in range(k):
            if ufs[i] in list_kept_root:
                list_kept_part.append(kept_indices[i])
        set_kept_part = set(list_kept_part)
        for i in range(dimension):
            if i in set_kept_part:
                if i in list_pos_seed:
                    str_pattern += u"[TARGET]"
                else:
                    str_pattern += pattern1[i]
            else:
                str_pattern += u"[STUFF]"
    else:
        for i in range(dimension):
            if i in kept_indices:
                if i in list_pos_seed:
                    str_pattern += u"[TARGET]"
                else:
                    str_pattern += pattern1[i]
            elif i - 1 in list_pos_seed or i + 1 in list_pos_seed:
                return u""
            else:
                str_pattern += u"[STUFF]"

    str_pattern = ptn_stf.sub(u".{1,32}", str_pattern)
    str_pattern = ptn_tgt.sub(u"[TARGET]", str_pattern)
    if u"[TARGET]" not in str_pattern:
        return u""
    return str_pattern


def patter_filter(pattern, dict_ptn, len_min=3, freq_min=3):
    if len(pattern) - len(u"[TARGET]") + 1 >= len_min and dict_ptn[pattern] >= freq_min:
        return pattern
    return u""


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        exit(1)

    list_db = list()
    with open(sys.argv[1], "r") as fin:
        for line in fin:
            list_db.append(line.strip().split()[0])
    fin.close()

    ac = ACAutomaton(list_db)

    list_stn = list()
    dict_rev_idx = dict()
    dict_word = Counter()
    cnt = 0
    for line in sys.stdin:
    # f_in = open("/Users/traeyee/Documents/Nutstore/DATA/sent.10", "r")
    # for line in f_in:
    #     line = line.split("\t")[0]

        stn = strdecode(line.strip())
        list_matched = list()
        hit = False
        for item in ac.search(stn):
            list_matched.append(item)
            hit = True
        if hit:
            for item in sorted(list_matched, key=lambda e: len(e), reverse=True):
                stn = stn.replace(item, u"[TARGET]")
            dict_word[u"[TARGET]"] += len(ptn_tgt_1.findall(stn))
            for w in stn.replace(u"[TARGET]", u""):
                dict_word[w] += 1

                if w not in dict_rev_idx:
                    dict_rev_idx[w] = list()
                dict_rev_idx[w].append(cnt)
            if stn == u"[TARGET]":
                db = "db"
            list_stn.append(stn)
            cnt += 1
    # for stn in list_stn:
    #     print stn
    # exit(0)

    # Unify the reverse index
    for w in dict_rev_idx:
        dict_rev_idx[w] = set(dict_rev_idx[w])

    counter_pattern = Counter()

    for stn0 in list_stn:
        # print "Matrix computation begins: " + stn.encode("utf-8")
        # Build the matrix
        ptn_rt = generalizer(stn0, list_stn, dict_rev_idx, dict_word)
        if len(ptn_rt) > 0:
            counter_pattern[ptn_rt] += 1
            # print ptn_rt.encode("utf-8")

    for t in sorted(counter_pattern.items(), key=lambda t: t[1], reverse=True):
        ptn_rt = patter_filter(t[0], counter_pattern)
        if len(ptn_rt) > 0:
            print("%s\t%d" % t).encode("utf-8")
