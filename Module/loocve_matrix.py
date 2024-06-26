from Module import monotonic_classifier as mc
from Module.mappings import equiv_to_key, equiv_to_case
from Module.univariate_classifier import single_feature_LOOCVE

import os
import multiprocessing as mp
import copy
import pandas as pd
import numpy as np





### Useful functions for parallel


def vals_mp(pairs, df_2, out, funct):
    """
    Prepare a list of tuples containing input parameters for parallel processing.

    Parameters:
    - pairs: List of classifier pairs.
    - df_2: DataFrame without the current patient.
    - out: Current patient data.
    - funct: Prediction function.

    Returns:
    - List of tuples for parallel processing.
    """
    return [(p, df_2, out, funct) for p in pairs]


def monotonic_model_CE(p, df):
    """
    Monotonic model for classification error calculation.

    Parameters:
    - p: Classifier name.
    - df: DataFrame.

    Returns:
    - Tuple containing classification error and classifier name.
    """
    p1, p2, key = p.split('/')
    key = int(key)
    rev, up = equiv_to_key[key]
    tr1, tr2, diag = df[p1].values.tolist(), df[p2].values.tolist(), df['target'].values.tolist()
    data = [((tr1[n], tr2[n]), 1, diag[n]) for n in range(len(diag))]
    models = mc.compute_recursion(data, (rev, up, key))
    reg, bpr, bpb, pr, pb = models[key]
    return p, reg


def univariate_contribution(df, feature):
    """
    Calculate LOOCVE for a single feature.

    Parameters:
    - df: DataFrame with dataset.
    - feature: feature to evaluate

    Returns:
    - Tuple containing feature and error.
    """
    
    if feature != 'target':
        X = df[feature].array
        y = df['target'].array
        loocve = single_feature_LOOCVE(X,y)
        return feature, loocve
    else:
        return feature, np.nan
    
def coeff_pair(df,pair, mat):
    """
    Calculate coefficient of a pair : coefficient = min⁡(s1,s2)−L

    Parameters:
    - df: DataFrame with dataset.
    - pair: pair to evaluate
    - mat: Matrix containing the LOOCVE score

    Returns:
    - Tuple containing pair and coefficient.
    """
    
    
    feat1, feat2, key = pair.split('/')
    s1, s2 = univariate_contribution(df, feat1)[1], univariate_contribution(df, feat2)[1]
    return pair, ((min(s1,s2) - mat.at['LOOCVE', pair]) +1)/2

    
    
def single_error(p, df_2, out, funct):
    """
    Calculate error for a single patient and classifier.

    Parameters:
    - p: Classifier name.
    - df_2: DataFrame without the current patient.
    - out: Current patient data.
    - funct: Prediction function.

    Returns:
    - Tuple containing classifier name and error.
    """
    p1, p2, key = p.split('/')
    key = int(key)
    rev, up = equiv_to_key[key]
    diag = df_2['target'].values.tolist()
    tr1, tr2 = df_2[p1].values.tolist(), df_2[p2].values.tolist()
    data = [((tr1[n], tr2[n]), 1, diag[n]) for n in range(len(diag))]
    out_p = (out[p1], out[p2])
    models = mc.compute_recursion(data, (rev, up, key))
    reg_err, bpr, bpb, r_p, b_p = models[key]
    pred = funct(out_p, bpr, bpb, rev, up)

    if pred == -1:
        return p, -1  # if uncertain, we keep it like this
    else:
        return p, abs(1 - int(pred == out['target']))  # int(True) = 1, so if the pred is equal to the real label, error is 0





def error_matrix(df_, pairs, nbcpus, funct):
    """
    Calculate the error matrix for a DataFrame and a list of classifier pairs.

    Parameters:
    - df_: DataFrame containing patient data.
    - pairs: List of classifier pairs.
    - nbcpus: Number of CPUs for parallel processing.
    - funct: Prediction function.

    Returns:
    - Error matrix DataFrame.
    """
    try:
        nbcpus = int(os.getenv('OMP_NUM_THREADS'))
    except:
        pass
    pool = mp.Pool(nbcpus)

    df = copy.deepcopy(df_)

    mat_err = pd.DataFrame(columns=pairs + ['target'])

    for j, out in df.iterrows():
        df_2 = df.drop([j])
        #df_2.reset_index(drop=True, inplace=True)

        vals = vals_mp(pairs, df_2, out, funct)

        dico_err = dict(pool.starmap(single_error, vals, max(1, len(vals) // nbcpus)))

        #dico_err = {r[0] : r[1] for r in res}
        dico_err['target'] = out['target']
        dico_err_s = pd.Series(dico_err)
        dico_err_s.name = j
        mat_err = pd.concat((mat_err, dico_err_s.to_frame().T), axis=0)

    vals_re = [(c, df) for c in pairs]
    REd = dict(pool.starmap(monotonic_model_CE, vals_re, max(1,len(vals_re)//nbcpus)))
    

    #REd = {re[1] : re[0] for re in res_re}
    REd['target'] = np.nan
    re_s = pd.Series(REd)
    re_s.name = 'CE'

    mat_err_re = pd.concat((mat_err,re_s.to_frame().T), axis=0)



    unc = {col: mat_err[col].to_list().count(-1) for col in pairs}
    unc['target'] = np.nan
    unc_s = pd.Series(unc)
    unc_s.name = 'uncertain'

    mat_err_unc = pd.concat((mat_err_re,unc_s.to_frame().T), axis=0)



    cols = list(mat_err_unc.columns)
    cols.remove('target')
    rem = list()
    for col in cols:
        val = mat_err_unc.at['uncertain', col]
        if val > len(df)/3:
            rem.append(col)
    mat_err.drop(rem, axis=1, inplace=True)
    mat_err_unc.drop(rem, axis=1, inplace=True)

    err = {col: mat_err[col].to_list().count(1)/(mat_err[col].to_list().count(1) + mat_err[col].to_list().count(0)) for col in pairs if col not in rem}
    err['target'] = np.nan
    err_s = pd.Series(err)
    err_s.name = 'LOOCVE'
    
    mat_err_final = pd.concat((mat_err_unc,err_s.to_frame().T), axis=0)
    
    vals_coeff = [(df, pair, mat_err_final) for pair in pairs]
    res_coeff = dict(pool.starmap(coeff_pair, vals_coeff, max(1,len(vals_coeff)//nbcpus)))
    pool.close()
    
    a = min(res_coeff.values())
    b = max(res_coeff.values())
    res_coeff_scaled = {col:(res_coeff[col]-a)/(b-a) for col in res_coeff.keys()}
    
    
    mat_err_final = pd.concat([mat_err_final, pd.DataFrame(res_coeff_scaled, index=['coeff'])])
    
    dic_corrected = {col:1-(1-err[col])*res_coeff_scaled[col] for col in pairs if col not in rem}
    mat_err_final = pd.concat([mat_err_final, pd.DataFrame(dic_corrected, index=['LOOCVE_corr'])])

    mat_err_final.sort_index(axis=1, inplace=True)
    mat_err_final.sort_values(axis = 1, by=['LOOCVE', 'CE', 'uncertain'], inplace=True)
    #mat_err_final.reindex(sorted(mat_err_final.columns), axis=1)

    del df
    return mat_err_final




#### CONVERTING ERROR MATRIX TO PREDICTION MATRIX
def error_to_prediction(error_matrix, df):
    """
    Convert error matrix to prediction matrix.

    Parameters:
    - error_matrix: Error matrix with classifiers as columns.
    - df: DataFrame containing the dataset.

    Returns:
    - prediction_matrix: Dictionary with classifiers as keys and predicted labels as values.
    """
    true_labels = df['target'].values.tolist()

    prediction_matrix = {}
    for classifier in error_matrix.keys():
        if classifier != 'target':
            errors = error_matrix[classifier]
            predictions = []
            for i in range(len(errors)):
                if errors[i] == 0:
                    predictions.append(true_labels[i])
                elif errors[i] == 1:
                    predictions.append(int(abs(1 - true_labels[i])))
                elif errors[i] == -1:
                    predictions.append(-1)
            prediction_matrix[classifier] = predictions

    return prediction_matrix


### GET DICTIONARY RELATING CLASSIFIERS WITH ERROR SCORE
def cost_classifiers(error_matrix):
    """
    Get a dictionary relating classifiers with error scores.

    Parameters:
    - error_matrix: Error matrix with classifiers as columns.

    Returns:
    - cost: Dictionary with classifiers as keys and their error score as values.
    """
    classifiers = list(error_matrix.columns)
    classifiers.remove('target')
    cost = {classifier: error_matrix[classifier].loc[['LOOCVE']][0] for classifier in classifiers}
    return cost
