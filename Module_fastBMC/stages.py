from Module_fastBMC import loocve_matrix as em
from Module_fastBMC import ensemble_model as mem
from Module_fastBMC import dynamic_preselection as dpsl

from Module_fastBMC.metrics import accuracy_score, auc_score

import time
import sys


def stage0(df, nbcpus, m, funct):

    config = dpsl.all_configurations(df)

    Q, pairs = dpsl.algorithm_2(config, df, m, nbcpus, funct)
    return pairs


def stage_1(df, min_k, max_k, nbcpus, strat, funct):

    # Obtain pairs using stage0
    cls = stage0(df, nbcpus, max_k, funct)



    # Compute k_auc and k_thresh using optimal_k_aggregations
    k_auc, k_thresh = mem.k_misclassification(df, cls, nbcpus, funct, strat, min_k, max_k)


    # Get optimal k and corresponding threshold
    k_opt, err_k = mem.optimal_k_auc(k_auc)



    return k_opt


def stage_2(df, k_opt, thresh, auc_file, nbcpus, funct, strat):

    errors, y_true, y_pred, y_proba = list(), list(), list(), list()

    for index, out in df.iterrows():
        # Obtain pairs using stage0 for each point in df
        df_2 = df.drop([index])


        cls = stage0(df_2, nbcpus, k_opt, funct)


        # Log time for stage2 for each point
        ndf_err = em.error_matrix(df_2, cls, nbcpus, funct)
        cost = em.cost_classifiers(ndf_err)

        # Find k metamodel using optimal_k_aggregations
        mve, pairs, algo = mem.find_k_ensemble_model(df_2, ndf_err, cost, k_opt, thresh, nbcpus, strat)
        
        
        # Create and predict using metamodel
        pred, proba = mem.create_and_predict_ensemble_model(df_2, out, pairs, thresh, nbcpus, funct)

        

        # Log total time for stage2 for each point
        errors.append(abs(out['target'] - pred))
        y_proba.append(proba)
        y_true.append(out['target'])
        y_pred.append(pred)

    # Calculate accuracy, auc, and confidence interval
    acc = accuracy_score(y_true, y_pred)
    auc, ci, thresh = auc_score(y_true, y_proba, auc_file)
    


    return acc, auc, ci


def stage_3(df, k_opt, thresh, nbcpus, funct, strat):

    # Obtain pairs using stage0 for stage3
    cls = stage0(df, nbcpus, k_opt, funct)



    ndf_err = em.error_matrix(df, cls, nbcpus, funct)
    
    cost = em.cost_classifiers(ndf_err)

    # Find k metamodel using optimal_k_aggregations
    mve, pairs, algo = mem.find_k_ensemble_model(df, ndf_err, cost, k_opt, thresh, nbcpus, strat)

    return pairs, ndf_err
