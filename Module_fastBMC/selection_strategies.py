from Module_fastBMC import metrics as ms

import copy
import multiprocessing as mp
import time




# Filter the matrix: eliminate redundant transcripts
def filter_pairs_greedy(pairs):
    used_trans = list()
    delete_col = list()
    for i in range(len(pairs)):
        p1, p2, key = pairs[i].split("/")
        if p1 in used_trans or p2 in used_trans:
            delete_col.append(pairs[i])
        else:
            used_trans.append(p1)
            used_trans.append(p2)
    for x in delete_col:
        pairs.remove(x)
    return pairs

# Adaptively filter pairs based on the candidate pair
def filter_pairs_adapt(pairs, cl):
    delete_pairs = list()
    idx = pairs.index(cl)
    for i in range(idx, len(pairs)):
        p1, p2, key = pairs[i].split("/")
        if p1 in cl or p2 in cl:
            delete_pairs.append(pairs[i])
    for x in delete_pairs:
        pairs.remove(x)
    return pairs


# Get N-best pairs based on a cost function
def NB(df, ndf_, cost, k, thresh, nbcpus):
    ndf = copy.deepcopy(ndf_)
    ndf.drop(['uncertain', 'LOOCVE'], inplace=True)

    # Sort pairs by cost and apply filtering
    pairs = sorted(cost.items(), key=lambda t: t[1])
    pairs = [pairs[i][0] for i in range(len(pairs))]
    pairs = filter_pairs_greedy(pairs)
    pairs = pairs[0:k]

    # Create a set of values for the selected pairs
    set = [ndf[p].values.tolist() for p in pairs]

    return pairs, ms.MVE(set, thresh)


# Test a candidate pair in forward or backward search
def test_candidate(cand_pairs, cls, ndf, mes, i):
    current_pair = cand_pairs[i]

    # Construct a set of values including the candidate pair
    candidate_set_pairs = [ndf[cl].values.tolist() for cl in cls if cl != current_pair]

    # Compute the measure (e.g., MVE) for the candidate set
    current_pair_ms = mes(candidate_set_pairs)

    return i, current_pair, current_pair_ms


# Forward Search to find k-best classifiers
def FS(df, ndf_, cost, k, nbcpus, jump=30):
    pool = mp.Pool(nbcpus)
    ndf = copy.deepcopy(ndf_)
    ndf.drop(['uncertain', 'LOOCV'], inplace=True)

    # Identify the classifier with the lowest cost as the starting point
    temp = min(cost.values())
    res = [key for key in cost.keys() if cost[key] == temp]
    initial_classifier = res[0]
    cls = [initial_classifier]

    pairs = sorted(cost.items(), key=lambda t: t[1])
    pairs = [pairs[i][0] for i in range(len(pairs))]

    ind = 1
    tot_ind = ind

    while len(cls) < k:
        if tot_ind + jump > len(pairs):
            pairs = [p for p in pairs if p not in cls]
            ind = 1
            tot_ind = ind

        cand_pairs = pairs[ind:ind + jump]

        # Test candidate pairs and select the best one
        vals = [(cand_pairs, cls, ndf, i) for i in range(len(cand_pairs))]
        res = pool.starmap(test_candidate, vals, max(1, len(vals) // nbcpus))
        res.sort(key=lambda x: x[2])

        i, best_cand, best_cand_ms = res[0]
        cls.append(best_cand)

        # Update the list of pairs by removing used pairs
        pairs = filter_pairs_adapt(pairs, best_cand)

    # Create a set of values for the selected classifiers
    set = [ndf[p].values.tolist() for p in cls]

    pool.close()
    pool.join()

    return cls, ms.MVE(set)


# Backward Search to find k-best classifiers
def BS(df, ndf_, cost, k, nbcpus, mes=ms.F2, end=30):
    pool = mp.Pool(nbcpus)
    ndf = copy.deepcopy(ndf_)
    ndf.drop(['uncertain', 'LOOCV'], inplace=True)

    # Sort pairs by cost and apply filtering
    pairs = sorted(cost.items(), key=lambda t: t[1])
    pairs = [pairs[i][0] for i in range(len(pairs))]
    pairs = filter_pairs_greedy(pairs)
    cls = pairs[:end]

    while len(cls) > k:
        cand_pairs = cls.copy()

        # Test candidate pairs and select the best one
        vals = [(cand_pairs, cls, ndf, i) for i in range(len(cand_pairs))]
        res = pool.starmap(test_candidate, vals, max(1, len(vals) // nbcpus))

        res.sort(key=lambda x: x[2])

        i, best_cand, best_cand_ms = res[0]
        cls.remove(best_cand)

    # Create a set of values for the selected classifiers
    set = [ndf[p].values.tolist() for p in cls]

    pool.close()
    pool.join()

    return cls, ms.MVE(set)
