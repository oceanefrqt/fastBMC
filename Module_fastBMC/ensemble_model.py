from Module_fastBMC import monotonic_classifier as mc
from Module_fastBMC import loocve_matrix as em

from Module_fastBMC.mappings import equiv_to_key, equiv_to_case
from Module_fastBMC.prediction_functions import proba_ensemble_model, pred_ensemble_model
from Module_fastBMC.metrics import accuracy_score, auc_score

import os
import multiprocessing as mp
import copy
import pandas as pd
import numpy as np
import time



def find_k_ensemble_model(df, ndf, cost, k, thresh, nbcpus, strat):
    """
    Find the best ensemble model contaning k pairs among different strategies.

    Parameters:
    - df: DataFrame containing patient data.
    - ndf: DataFrame containing new data.
    - cost: Cost matrix.
    - k: Number of classifiers in the ensemble.
    - thresh: Threshold for the selection algorithm.
    - nbcpus: Number of CPUs for parallel processing.
    - strat: List of selection strategies.

    Returns:
    - Tuple (selected_pairs, minimum_mve, selected_strategy).
    """
    results = []
    min_mve = float('inf')

    for strategy in strat:
        selected_pairs, mve = strategy(df, ndf, cost, k, thresh, nbcpus)
        results.append((mve, selected_pairs, strategy.__name__))
        min_mve = min(min_mve, mve)

    potential = [result for result in results if result[0] == min_mve]
    potential.sort(key=lambda x: x[2], reverse=True)

    return potential[0]


def prediction_pair(df, out, pair, funct):
    """
    Construct the model and predict the output for a single patient.

    Parameters:
    - df: DataFrame containing patient data.
    - out: DataFrame containing patient labels.
    - pair: Classifier pair string.
    - funct: Prediction function.

    Returns:
    - Prediction for the patient.
    """
    p1, p2, key = pair.split('/')
    key = int(key)
    rev, up = equiv_to_key[key]

    tr1, tr2 = df[p1].values.tolist(), df[p2].values.tolist()
    diag = df['target'].values.tolist()

    data = [((tr1[n], tr2[n]), 1, diag[n]) for n in range(len(diag))]
    out_p = (out[p1], out[p2])

    models = mc.compute_recursion(data, (rev, up, key))
    reg_err, bpr, bpb, r_p, b_p = models[key]
    pred = funct(out_p, bpr, bpb, rev, up)
    return pred


def create_and_predict_ensemble_model(df_, out, pairs, thresh, nbcpus, funct):
    """
    Create an ensemble model with the given classifiers and predict the output for a patient.

    Parameters:
    - df_: DataFrame containing patient data.
    - out: DataFrame containing left-out patient.
    - pairs: List of classifier pairs.
    - thresh: Threshold for the AUC score.
    - nbcpus: Number of CPUs for parallel processing.
    - funct: Prediction function.

    Returns:
    - Tuple (majority_vote_prediction, probability_of_class1).
    """
    pool = mp.Pool(nbcpus)
    df = copy.deepcopy(df_)

    vals = [(df, out, p, funct) for p in pairs]
    preds = pool.starmap(prediction_pair, vals, max(1, len(vals) // nbcpus))

    pool.close()
    del df

    proba = proba_ensemble_model(preds)
    pred = pred_ensemble_model(preds)

    return pred, proba

def k_misclassification(df, cls, nbcpus, funct, strat, min_k, max_k):
    """
    Compute the average AUC score for ensemble models containing k classifiers.

    Parameters:
    - df: DataFrame containing the dataset.
    - cls: List of classifier names.
    - nbcpus: Number of CPUs to use for parallel processing.
    - funct: Prediction function.
    - strat: List of strategies for finding pairs in the ensemble model.
    - min_k: Minimum number of classifiers in the ensemble.
    - max_k: Maximum number of classifiers in the ensemble.

    Returns:
    - k_auc: Dictionary with AUC scores for different values of k.
    - k_thresh: Dictionary with optimal thresholds for different values of k.
    """
    
    k_mis_acc = {k: list() for k in range(min_k, max_k + 1)}
    k_mis_auc = {k: list() for k in range(min_k, max_k + 1)}

    for _, out in df.iterrows():
        t1 = time.time()

        df_2 = df[df.index != out.name]
        
        ndf_err = em.error_matrix(df_2, cls, nbcpus, funct)
        cost = em.cost_classifiers(ndf_err)

        for k in range(min_k, max_k + 1):
            t0 = time.time()
            thresh = 0.5
            mve, pairs, algo = find_k_ensemble_model(df_2, ndf_err, cost, k, thresh, nbcpus, strat)
            pred, proba = create_and_predict_ensemble_model(df_2, out, pairs, thresh, nbcpus, funct)
            
            k_mis_acc[k].append(pred)
            if proba != -1:
                k_mis_auc[k].append([out['target'], proba])
            else:
                k_mis_auc[k].append([out['target'], -1])

    k_auc = {}
    k_thresh = {}
    k_acc = {}
    
    
    for k in range(min_k, max_k + 1):
        y_true = [k_mis_auc[k][l][0] for l in range(len(k_mis_auc[k]))]
        y_proba = [k_mis_auc[k][l][1] for l in range(len(k_mis_auc[k]))]
        y_pred = k_mis_acc[k]
        
        auc, ci, thresh = auc_score(y_true, y_proba, None)
        acc = accuracy_score(y_true, y_pred)
        
        k_auc[k] = auc
        k_thresh[k] = thresh
        k_acc[k] = acc

    return k_auc, k_thresh




def optimal_k_auc(k_auc):
    """
    Find the optimal k based on AUC scores.

    Parameters:
    - k_auc: Dictionary with AUC scores for different values of k.

    Returns:
    - Optimal k and its corresponding AUC score.
    """
    mini = max(k_auc.values())
    keys = [k for k in k_auc.keys() if k_auc[k] == mini]
    return min(keys), mini
                                                       
def optimal_k_acc(k_acc):
    """
    Find the optimal k based on accuracy.

    Parameters:
    - k_auc: Dictionary with accuracies for different values of k.

    Returns:
    - Optimal k and its corresponding accuracy
    """
    mini = max(k_acc.values())
    keys = [k for k in k_acc.keys() if k_acc[k] == mini]
    return min(keys), mini
