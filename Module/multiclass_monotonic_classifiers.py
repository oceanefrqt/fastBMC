from Module.monotonic_classifier import compute_recursion
from Module.mappings import equiv_to_key, equiv_to_case
from Module import dynamic_preselection as dp
from collections import defaultdict

import numpy as np
import multiprocessing as mp
import copy
import pandas as pd
from itertools import combinations
from copy import deepcopy

from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt

from sklearn.metrics import roc_auc_score, accuracy_score, roc_curve
from sklearn.metrics import auc as auc_sk


# CREATION OF THE MULTICLASS MONOTONIC MODEL
class Data:
    def __init__(self, data):
        self.data = data
        self.coord, self.weights, self.labels = zip(*self.data)
        self.classes = sorted(list(set(self.labels)))
        self.concordance_labels = {}
        self.concordance_coordinates = {}
        self.concordance_weights = {}
        
    def concordance(self):
        self.concordance_labels = defaultdict(set)
        self.concordance_coordinates = defaultdict(set)
        self.concordance_weights = defaultdict(set)
        
        for c, i in zip(self.labels, self.coord):
            self.concordance_labels[c].add(i)
            
        for c, i in zip(self.coord, self.labels):
            self.concordance_coordinates[c] = i
            
        for c, i in zip(self.coord, self.weights):
            self.concordance_weights[c] = i




class MultiClassMonotonicClassifier:
    def __init__(self, D):
        self.data = D
        self.seps = {c : {'a':[], 'pt':[]} for c in self.data.classes}
            
            
            
    def DAC(self, new_data, case, classes, k):

        sub1 = classes[:k+1]
        sub2 = classes[k+1:]


        if len(sub1) > 0 and len(sub2) > 0:

            z = [classes[k], classes[k+1]]

            if len(new_data)>0:

                m = compute_recursion(new_data, z, case)
                re, bpr, bpb, r_p, b_p = m[case[2]]




                if len(sub1) == 1:
                    self.seps[classes[k]]['pt'] = b_p
                    self.seps[classes[k]]['a'] = bpb
                if len(sub2) == 1:
                    self.seps[classes[k+1]]['pt'] = r_p
                    self.seps[classes[k+1]]['a'] = bpr


                if len(sub1)%2 == 1:
                    mid1 = len(sub1)//2
                else:
                    mid1 = len(sub1)//2 - 1

                if len(sub2)%2 == 1:
                    mid2 = len(sub2)//2
                else:
                    mid2 = len(sub2)//2 - 1

                data1 = [((p),self.data.concordance_weights[p],self.data.concordance_coordinates[p]) for p in b_p]
                data2 = [((p),self.data.concordance_weights[p],self.data.concordance_coordinates[p]) for p in r_p]

                if len(sub1) > 1:
                    self.DAC(data1, case, sorted(sub1), mid1)

                if len(sub2) > 1:
                    self.DAC(data2, case, sorted(sub2), mid2)



    def clean_seps(self):
        for c in self.seps.keys():
            self.seps[c]['a'] = list(set(self.seps[c]['a']))
            self.seps[c]['pt'] = list(set(self.seps[c]['pt']))

                
# COMPUTATION OF THE ERROR MATRIX
                
def vals_mp(pairs, df_2, out):
    vals = list()
    for p in pairs:
        vals.append((p, df_2, out))
    return vals



def monotonic_model_CE_multiclass(p, df):
    """
    Parameters:
    - p: pair of features.
    - df: dataframe with data.
    
    Returns:
    Tuple with the pair and the classification error (not l1-norm).
    """
    p1, p2, key = p.split('/')
    key = int(key)
    rev, up = equiv_to_key[key]
    
    diag = df['target'].values.tolist()
    tr1, tr2 = df[p1].values.tolist(), df[p2].values.tolist()
    data = [((tr1[n], tr2[n] ), 1, diag[n]) for n in range(len(diag))]
    
    D = Data(data)
    D.concordance()

    mcmc = MultiClassMonotonicClassifier(D)
    if len(set(diag))%2 == 1:
        mid = len(set(diag))//2
    else:
        mid = len(set(diag))//2 - 1       


    mcmc.DAC(D.data, (rev, up, key), D.classes, mid)

    mcmc.clean_seps()
    
    
    CE = 0
    for c1 in D.classes:
        real_c = D.concordance_labels[c1]
        pred_c = mcmc.seps[c1]['pt']
        CE += len(set(real_c) - set(pred_c))
    
    return (p, CE)



def monotonic_model_l1E_multiclass(p, df):
    """
    Parameters:
    - p: pair of features.
    - df: dataframe with data.
    
    Returns:
    Tuple with the pair and the l1 error.
    """
    p1, p2, key = p.split('/')
    key = int(key)
    rev, up = equiv_to_key[key]
    
    diag = df['target'].values.tolist()
    tr1, tr2 = df[p1].values.tolist(), df[p2].values.tolist()
    data = [((tr1[n], tr2[n] ), 1, diag[n]) for n in range(len(diag))]

    D = Data(data)
    D.concordance()

    mcmc = MultiClassMonotonicClassifier(D)
    if len(set(diag))%2 == 1:
        mid = len(set(diag))//2
    else:
        mid = len(set(diag))//2 - 1       


    mcmc.DAC(D.data, (rev, up, key), D.classes, mid)

    mcmc.clean_seps()
    

    l1E = 0
    for c1 in D.classes:
        real_c = D.concordance_labels[c1]
        for c2 in D.classes:
            pred_c = mcmc.seps[c2]['pt']
            nb = len(set(real_c).intersection(set(pred_c)))
            l1E += nb*abs(c1-c2)
    
    return (p, l1E)


def pred_multiclass(out, seps, key):
    """
    Parameters:
    - out: left-out sample.
    - seps: dictionary containing the separation points for each class.
    - key: key of the configuration.
    
    Returns:
    Predicted class.
    """
    
    classes = sorted(list(seps.keys()), reverse=True)
    
    for c in classes:
        flag = False
        
        for pt_front in seps[c]['a']:
            if key == 2:
                if pt_front[0] >= out[0] and pt_front[1] <= out[1]:
                    flag = True
            elif key == 1:
                if pt_front[0] <= out[0] and pt_front[1] <= out[1]:
                    flag = True
            elif key == 3:
                if pt_front[0] >= out[0] and pt_front[1] >= out[1]:
                    flag = True
            elif key == 4:
                if pt_front[0] <= out[0] and pt_front[1] >= out[1]:
                    flag = True
        if flag == True:
            return c
    
    return classes[-1]

def monotonic_model_LOOCVE_multiclass(p, df):
    
    """
    Parameters:
    - p: pair of features.
    - df: dataframe with data.
    - out: left-out sample

    
    Returns:
    Tuple with the pair and the LOOCVE.
    """

    err = 0

    
    for j, out in df.iterrows():
        df_2 = df.drop([j])
        
        res = LOOCVE_multiclass(p, df_2, out)
        
        err += res[1]
    
    return (p, err)





def LOOCVE_multiclass(p, df, out):
    
    """
    Parameters:
    - p: pair of features.
    - df: dataframe with data.
    - out: left-out sample

    
    Returns:
    Tuple with the pair and the error of the prediction.
    """

    p1, p2, key = p.split('/')
    key = int(key)
    rev, up = equiv_to_key[key]

    diag = df['target'].values.tolist()

    tr1, tr2 = df[p1].values.tolist(), df[p2].values.tolist()

    data = [((tr1[n], tr2[n] ), 1, diag[n]) for n in range(len(diag))]
    out_p = (out[p1], out[p2])


    D = Data(data)
    D.concordance()

    mcmc = MultiClassMonotonicClassifier(D)
    if len(set(diag))%2 == 1:
        mid = len(set(diag))//2
    else:
        mid = len(set(diag))//2 - 1       


    mcmc.DAC(D.data, (rev, up, key), D.classes, mid)

    mcmc.clean_seps()
    
    pred = pred_multiclass(out_p, mcmc.seps, key)
    
    if abs(out['target']-pred) == 0:
        err = 0
    else:
        err = 1
    
    return (p, err)
            
    
def monotonic_model_l1CVE_multiclass(p, df):
    
    """
    Parameters:
    - p: pair of features.
    - df: dataframe with data.
    - out: left-out sample

    
    Returns:
    Tuple with the pair and the LOOCVE.
    """

    err = 0

    
    for j, out in df.iterrows():
        df_2 = df.drop([j])
        
        res = l1_error_multiclass(p, df_2, out)
        
        err += res[1]
    
    return (p, err)

       
        
    



def l1_error_multiclass(p, df, out):
    
    """
    Parameters:
    - p: pair of features.
    - df: dataframe with data.
    - out: left-out sample

    
    Returns:
    Tuple with the pair and the l1-norm.
    """

    p1, p2, key = p.split('/')
    key = int(key)
    rev, up = equiv_to_key[key]

    diag = df['target'].values.tolist()

    tr1, tr2 = df[p1].values.tolist(), df[p2].values.tolist()

    data = [((tr1[n], tr2[n] ), 1, diag[n]) for n in range(len(diag))]
    out_p = (out[p1], out[p2])


    D = Data(data)
    D.concordance()

    mcmc = MultiClassMonotonicClassifier(D)
    if len(set(diag))%2 == 1:
        mid = len(set(diag))//2
    else:
        mid = len(set(diag))//2 - 1       


    mcmc.DAC(D.data, (rev, up, key), D.classes, mid)

    mcmc.clean_seps()
    
    pred = pred_multiclass(out_p, mcmc.seps, key)
    return (p, abs(out['target']-pred))



def error_matrix_multiclass_old(df_, pairs, nbcpus):
    """
    Parameters:
    - df: dataframe with data.
    - pairs: pairs to evaluate
    - nbcpus: nb of cpus to use for the multiprocessing

    Returns:
    - Error matrix DataFrame.
    """

    pool = mp.Pool(nbcpus)

    df = copy.deepcopy(df_)

    index = list()

    mat_err = pd.DataFrame(columns = pairs + ['target']) #Dataframe with possible classifiers as columns
    
    for j, out in df.iterrows():
        df_2 = df.drop([j])
        
        vals = vals_mp(pairs, df_2, out)
        dico_err = dict(pool.starmap(l1_error_multiclass, vals, max(1, len(vals) // nbcpus)))
        
        dico_err['target'] = out['target']
        dico_err_s = pd.Series(dico_err)
        dico_err_s.name = j
        mat_err = pd.concat((mat_err, dico_err_s.to_frame().T), axis=0)


    vals_re = [(c, df) for c in pairs]
    REd = dict(pool.starmap(monotonic_model_CE_multiclass, vals_re, max(1,len(vals)//nbcpus)))
    REd['target'] = np.nan
    re_s = pd.Series(REd)
    re_s.name = 'CE'

    mat_err_re = pd.concat((mat_err,re_s.to_frame().T), axis=0)
    
    l1d = dict(pool.starmap(monotonic_model_l1E_multiclass, vals_re, max(1,len(vals)//nbcpus)))
    l1d['target'] = np.nan
    l1_s = pd.Series(l1d)
    l1_s.name = 'l1E'

    mat_err_re = pd.concat((mat_err_re,l1_s.to_frame().T), axis=0)

    err = {col: np.mean(mat_err_re[col][:-1]) for col in pairs}
    err['target'] = np.nan
    err_s = pd.Series(err)
    err_s.name = 'l1CVE'
    
    mat_err_l1 = pd.concat((mat_err_re,err_s.to_frame().T), axis=0)
    
    err = {col: np.count_nonzero(mat_err[col])/ len(mat_err[col]) for col in pairs}
    err['target'] = np.nan
    err_s = pd.Series(err)
    err_s.name = 'LOOCVE'

    mat_err_final = pd.concat((mat_err_l1,err_s.to_frame().T), axis=0)


    mat_err_final.sort_values(axis = 1, by=['l1CVE','LOOCVE',  'l1E', 'CE'], inplace=True)

    del df
    return mat_err_final


def error_matrix_multiclass(df_, pairs, nbcpus):
    """
    Parameters:
    - df: dataframe with data.
    - pairs: pairs to evaluate
    - nbcpus: nb of cpus to use for the multiprocessing

    Returns:
    - Error matrix DataFrame.
    """

    pool = mp.Pool(nbcpus)

    df = copy.deepcopy(df_)

    index = list()

    mat_err = pd.DataFrame(columns = pairs + ['target']) #Dataframe with possible classifiers as columns
    
    for j, out in df.iterrows():
        df_2 = df.drop([j])
        
        vals = vals_mp(pairs, df_2, out)
        dico_err = dict(pool.starmap(l1_error_multiclass, vals, max(1, len(vals) // nbcpus)))
        
        dico_err['target'] = out['target']
        dico_err_s = pd.Series(dico_err)
        dico_err_s.name = j
        mat_err = pd.concat((mat_err, dico_err_s.to_frame().T), axis=0)

    vals_re = [(c, df) for c in pairs]
    
    l1d = dict(pool.starmap(monotonic_model_l1E_multiclass, vals_re, max(1,len(vals)//nbcpus)))
    l1d['target'] = np.nan
    l1_s = pd.Series(l1d)
    l1_s.name = 'l1E'

    mat_err_re = pd.concat((mat_err,l1_s.to_frame().T), axis=0)

    err = {col: np.mean(mat_err_re[col][:-1]) for col in pairs}
    err['target'] = np.nan
    err_s = pd.Series(err)
    err_s.name = 'l1CVE'
    
    mat_err_final = pd.concat((mat_err_re,err_s.to_frame().T), axis=0)


    mat_err_final.sort_values(axis = 1, by=['l1CVE' ,'l1E'], inplace=True)

    del df
    return mat_err_final


### SELECT BEST K

def find_k_ensemble_model(df, ndf, k,nbcpus):
    """
    Find the best ensemble model contaning k pairs among different strategies.

    Parameters:
    - df: DataFrame containing patient data.
    - ndf: DataFrame containing error matrix.
    - k: Number of classifiers in the ensemble.
    - nbcpus: Number of CPUs for parallel processing.

    Returns:
    - List selected_pairs.
    """
    
    ndf.sort_values(axis = 1, by=['l1CVE',  'l1E'], inplace=True)
    
    pairs = ndf.columns.tolist()
    pairs.remove('target')
    
    best_pairs = list()
    feat_seen = list()
    count = 0
    
    while len(best_pairs) < k and count < len(pairs):
        p1, p2, key = pairs[count].split('/')
        if p1 not in feat_seen and p2 not in feat_seen:
            best_pairs.append(pairs[count])
            feat_seen.append(p1)
            feat_seen.append(p2)
        
        count +=1 

    return best_pairs

def prediction_multiclass(df, out, p):
    
    """
    Parameters:
    - p: pair of features.
    - df: dataframe with data.
    - out: left-out sample

    
    Returns:
    Tuple with the pair and the l1-norm.
    """

    p1, p2, key = p.split('/')
    key = int(key)
    rev, up = equiv_to_key[key]

    diag = df['target'].values.tolist()

    tr1, tr2 = df[p1].values.tolist(), df[p2].values.tolist()

    data = [((tr1[n], tr2[n] ), 1, diag[n]) for n in range(len(diag))]
    out_p = (out[p1], out[p2])


    D = Data(data)
    D.concordance()

    mcmc = MultiClassMonotonicClassifier(D)
    if len(set(diag))%2 == 1:
        mid = len(set(diag))//2
    else:
        mid = len(set(diag))//2 - 1       


    mcmc.DAC(D.data, (rev, up, key), D.classes, mid)

    mcmc.clean_seps()
    


    pred = pred_multiclass(out_p, mcmc.seps, key)

    return pred


def majority_vote_pred_and_proba(preds, classes):
    proba = list()
    pred = None
    best_proba = 0
    for c in sorted(classes):
        proba.append(preds.count(c)/len(preds))
        if preds.count(c)/len(preds) >= best_proba:
            pred = c
            best_proba = preds.count(c)/len(preds)
    print('proba', proba)
    return pred, proba
        
    



def create_and_predict_ensemble_model(df_, out, pairs, nbcpus):
    """
    Create an ensemble model with the given classifiers and predict the output for a patient.

    Parameters:
    - df_: DataFrame containing patient data.
    - out: DataFrame containing left-out patient.
    - pairs: List of classifier pairs.
    - classes: list of classes 
    - nbcpus: Number of CPUs for parallel processing.


    Returns:
    - Tuple (majority_vote_prediction, probability_of_class1).
    """
    pool = mp.Pool(nbcpus)
    df = copy.deepcopy(df_)

    vals = [(df, out, p) for p in pairs]
    preds = pool.starmap(prediction_multiclass, vals, max(1, len(vals) // nbcpus))

    pool.close()
    pool.join()
    del df

    classes = set(df_['target'])
    return majority_vote_pred_and_proba(preds, classes)


def custom_auc_score(y_true, y_proba, pair_list):
    
    pair_scores = []
    pair_scores_weighted = []
    mean_tpr = dict()
    fpr_grid = np.linspace(0.0, 1.0, 1000)

    sum_weights = 0
    
    index = {s: sorted(list(set(y_true))).index(s) for s in sorted(set(y_true)) }


    for ix, (label_a, label_b) in enumerate(pair_list):

        a_true = list()
        b_true = list()
        a_proba = list()
        b_proba = list()


        for idx in range(len(y_true)):
            if y_true[idx] == label_a:
                a_true.append(1)
                a_proba.append(y_proba[idx][index[label_a]])
                b_true.append(0)
                b_proba.append(y_proba[idx][index[label_b]])
            if y_true[idx] == label_b:
                a_true.append(0)
                a_proba.append(y_proba[idx][index[label_a]])
                b_true.append(1)
                b_proba.append(y_proba[idx][index[label_b]])




        fpr_a, tpr_a, _ = roc_curve(a_true, a_proba)
        fpr_b, tpr_b, _ = roc_curve(b_true, b_proba)

        mean_tpr[ix] = np.zeros_like(fpr_grid)
        mean_tpr[ix] += np.interp(fpr_grid, fpr_a, tpr_a)
        mean_tpr[ix] += np.interp(fpr_grid, fpr_b, tpr_b)
        mean_tpr[ix] /= 2
        mean_score = auc_sk(fpr_grid, mean_tpr[ix])

        weight = abs(label_a - label_b)
        sum_weights += weight

        pair_scores.append(mean_score)
        pair_scores_weighted.append(mean_score*weight)
        
    return np.average(pair_scores), np.sum(pair_scores_weighted)/sum_weights

def custom_acc_score(y_true, y_pred):
    return 1 - np.count_nonzero(np.asarray(y_true)-np.asarray(y_pred))/len(y_pred), 1 - np.sum(abs(np.asarray(y_true)-np.asarray(y_pred)))/len(y_pred)
    



def k_misclassification(df,nbcpus, min_k, max_k):
    """
    Compute the average AUC score for ensemble models containing k classifiers.

    Parameters:
    - df: DataFrame containing the dataset.
    - cls: List of classifier names.
    - nbcpus: Number of CPUs to use for parallel processing.
    - min_k: Minimum number of classifiers in the ensemble.
    - max_k: Maximum number of classifiers in the ensemble.

    Returns:
    - k_auc: Dictionary with AUC scores for different values of k.
    - k_thresh: Dictionary with optimal thresholds for different values of k.
    """
    
    k_mis_acc = {k: list() for k in range(min_k, max_k + 1)}
    k_mis_auc = {k: list() for k in range(min_k, max_k + 1)}

    for _, out in df.iterrows():

        df_2 = df[df.index != out.name]
        
        confs = all_configurations(df_2)
        cls = preselection_multiclass(confs, df_2, max_k, nbcpus)
        
        ndf_err = error_matrix_multiclass(df_2, cls, nbcpus)

        for k in range(min_k, max_k + 1):

            thresh = 0.5
            pairs = find_k_ensemble_model(df_2, ndf_err, k,nbcpus)
            pred, proba = create_and_predict_ensemble_model(df_2, out, pairs, nbcpus)
            
            k_mis_acc[k].append(pred)
            if proba != -1:
                k_mis_auc[k].append([out['target'], proba])
            else:
                k_mis_auc[k].append([out['target'], -1])

    k_auc = {}
    #k_acc = {}
    
    classes = sorted(list(set(df['target'])))
    
    pair_list = list(combinations(set(df['target']),2))

    
    
    for k in range(min_k, max_k + 1):
        y_true = [k_mis_auc[k][l][0] for l in range(len(k_mis_auc[k]))]
        y_proba = [k_mis_auc[k][l][1] for l in range(len(k_mis_auc[k]))]
        y_pred = k_mis_acc[k]
        
        #auc = roc_auc_score(y_true, y_proba, multi_class='ovo')
        auc, auc_weighted = custom_auc_score(y_true, y_proba, pair_list)
        acc, acc_weighted = custom_acc_score(y_true, y_pred)
        
        k_auc[k] = auc_weighted
        #k_acc[k] = acc

    return k_auc, optimal_k_auc(k_auc)

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
    return min(keys)


### ESTIMATE PERFORMANCE

def estimate_perf(df, k_opt, nbcpus):

    errors, y_true, y_pred, y_proba = list(), list(), list(), list()

    for index, out in df.iterrows():

        df_2 = df.drop([index])
        
        confs = all_configurations(df_2)
        cls = preselection_multiclass(confs, df_2, k_opt, nbcpus)

        ndf_err = error_matrix_multiclass(df_2, cls, nbcpus)
        pairs = find_k_ensemble_model(df_2, ndf_err, k_opt,nbcpus)
        
        pred, proba = create_and_predict_ensemble_model(df_2, out, pairs, nbcpus)
        
        errors.append(abs(out['target'] - pred))
        y_proba.append(proba)
        y_true.append(out['target'])
        y_pred.append(pred)

    #auc = roc_auc_score(y_true, y_proba, multi_class='ovo')
    
    classes = sorted(list(set(df['target'])))
    
    pair_list = list(combinations(set(df['target']),2))
    print(pair_list)
    auc, auc_weighted = custom_auc_score(y_true, y_proba, pair_list)
    acc, acc_weighted = custom_acc_score(y_true, y_pred)
    
    print('y_true = ', y_true)
    print('y_proba = ', y_proba)
    print('y_pred = ', y_pred)
    
    return acc, auc, acc_weighted, auc_weighted

### FINAL ENSEMBLE MODEL

def ensemble_model(df, k_opt, nbcpus):
    
    confs = all_configurations(df)
    cls = preselection_multiclass(confs, df, k_opt, nbcpus)
    
    ndf_err = error_matrix_multiclass(df, cls, nbcpus)
    pairs = find_k_ensemble_model(df, ndf_err, k_opt, nbcpus)
    return pairs

### PRESELECTION

def H_df(df, cls, nbcpus):
    pool = mp.Pool(nbcpus)
    vals = [(c, df) for c in cls]
    res = pool.starmap(monotonic_model_l1E_multiclass, vals, max(1, len(vals) // nbcpus))
    pool.close()
    return sorted(res)

def H_dict(H):
    Hd = dict()
    for h in H:
        kh = h[1]
        if kh not in Hd.keys():
            Hd[kh] = list()
        Hd[kh].append(h[0])
    return Hd


def Q_df(df, cls, nbcpus): # HERE THERE COULD BE AN ERROR: MULTIPLE LOOP OF POOL
    pool = mp.Pool(nbcpus)
    vals = [(c, df) for c in cls]
    S = sorted(pool.starmap(monotonic_model_l1CVE_multiclass, vals, max(1, len(vals) // nbcpus)))
    pool.close()
    return S


def Q_dict(Q):
    Qd = dict()
    G = dict()
    for q in Q:
        kq = q[1]
        g1, g2, g3 = q[0].split('/')
        if kq not in Qd.keys():
            Qd[kq] = set()
        if kq not in G.keys():
            G[kq] = set()
        Qd[kq].add(q[0])
        G[kq].add(g1)
        G[kq].add(g2)


    return Qd, G


def preselection_multiclass(cls, df, m, nbcpus):

    H = H_df(df, cls, nbcpus)

    Hd = H_dict(H)

    Q = dict() #For each strat of LOOCV, we have a list of pairs

    count = 0

    for h_key in sorted(Hd.keys()):

        pairs = Hd[h_key]
        Q_ = Q_df(df, pairs, nbcpus) #Compute the LOOCVE of the pairs with RE = h_key
        Qd, Gd = Q_dict(Q_)

        Q = dp.update_dict(Q, Qd)# Update Q to get the pairs grouped according to their LOOCVE


        if dp.check_disjoint_pairs_naive(Q,m):
            break


    a = max(Q.keys()) #Highest value of LOOCV in Q
    Hd = dp.supp_H_above_a(Hd, a)
    Hd = dp.supp_H_below_a(Hd, h_key)
    


    for h_key in sorted(Hd.keys()):

        
        a = max(Q.keys())
        if h_key <= a:

            pairs = Hd[h_key]
            Q_ = Q_df(df, pairs, nbcpus)
            Qd, Gd = Q_dict(Q_)

            Qd = dp.supp_H_above_a(Qd, a)

            Q = dp.update_dict(Q, Qd)

            Q_ = deepcopy(Q)
            del Q_[a]

            while dp.check_disjoint_pairs_naive(Q_, m):
                

                Q = Q_
                a = max(Q.keys())

                Q_ = deepcopy(Q)
                del Q_[a]
                
    pairs = list()
    for key in Q.keys():
        pairs += Q[key]

    return pairs


def all_configurations(df):
    transcripts = list(df.columns)
    transcripts.remove('target')

    configurations = list()
    for i in range(len(transcripts)):
        for j in range(i+1, len(transcripts)):
            for key in range(1,5):
                configurations.append('/'.join([transcripts[i], transcripts[j], str(key)]))
    return configurations




### VISUALIZATION


def visualization_multiclass(df, pairs, cm, label_dot):
    
    for p in pairs:
    
        p1, p2, key = p.split('/')

        key = int(key)
        rev, up = equiv_to_key[key]
        tr1 = df[p1].values.tolist()
        tr2 = df[p2].values.tolist()
        diag = df['target'].values.tolist()
        data = [((tr1[n], tr2[n] ), 1, diag[n]) for n in range(len(diag))]

        classes = sorted(list(set(diag)))

        colormap = plt.cm.viridis
        num_colors = len(classes)
        discrete_cmap = ListedColormap(colormap(np.linspace(0, 1, num_colors)))
        colors = discrete_cmap.colors


        D = Data(data)
        D.concordance()

        mcmc = MultiClassMonotonicClassifier(D)
        if len(set(diag))%2 == 1:
            mid = len(set(diag))//2
        else:
            mid = len(set(diag))//2 - 1       


        mcmc.DAC(D.data, (rev, up, key), D.classes, mid)

        mcmc.clean_seps()
        
        seps = mcmc.seps


        X,Y = list(), list()


        for c in seps.keys():
            for pt in seps[c]['pt']:
                X.append(pt[0])
                Y.append(pt[1])


        Xv, Yv = np.meshgrid(sorted(X), sorted(Y, reverse=True))
        Z = np.zeros((len(Y), len(X)))


        for c in sorted(seps.keys(), reverse=True):
            for (x,y) in seps[c]['pt']:

                if key == 1:

                    for i in range(len(X)):
                        for j in range(len(Y)):
                            if x== Xv[j,i] and y == Yv[j,i]:

                                Z[j,i] = c

                                for l in range(0,j+1):
                                    for m in range(i, len(X)):
                                        if Z[l,m] == 0:
                                            Z[l,m] =c
                elif key == 3:

                    for i in range(len(X)):
                        for j in range(len(Y)):
                            if x== Xv[j,i] and y == Yv[j,i]:
                                Z[j,i] = c

                                for l in range(j, len(Y)):
                                    for m in range(0, i+1):
                                        if Z[l,m] == 0:
                                            Z[l,m] =c

                elif key == 4:

                    for i in range(len(X)):
                        for j in range(len(Y)):
                            if x== Xv[j,i] and y == Yv[j,i]:
                                Z[j,i] = c

                                for l in range(j, len(Y)):
                                    for m in range(i, len(X)):
                                        if Z[l,m] == 0:
                                            Z[l,m] =c

                elif key == 2:

                    for i in range(len(X)):
                        for j in range(len(Y)):
                            if x== Xv[j,i] and y == Yv[j,i]:
                                Z[j,i] = c
                                #print(Z)
                                for l in range(0,j+1):
                                    for m in range(0, i+1):
                                        if Z[l,m] == 0:
                                            Z[l,m] =c

        plt.figure(figsize=(5,5))

        plt.xlabel(p1)
        plt.ylabel(p2)

        try:

            CE = cm.at['CE', p]
            LOOCVE = cm.at['LOOCVE', p]
            l1CVE = cm.at['l1CVE', p]

            plt.title('CE = {} & L1CVE = {}, LOOCVE= {}'.format(round(CE/len(data),2), round(l1CVE,2), round(LOOCVE,2)))
        except:
            pass

        plt.contourf(sorted(X),sorted(Y, reverse=True),Z, alpha=0.4)

        for d in data:
            plt.scatter(d[0][0], d[0][1], c=colors[d[2]-1], edgecolors='k')

        if label_dot is not None:
            labels = df[label_dot ].values.tolist()

            for mm in range(len(diag)):
                plt.annotate(labels[mm], (tr1[mm], tr2[mm]), zorder=6)

        plt.show()


def visualization_multiclass_2(df, pairs, cm, label_dot):
    
    for p in pairs:
    
        p1, p2, key = p.split('/')

        key = int(key)
        rev, up = equiv_to_key[key]
        tr1 = df[p1].values.tolist()
        tr2 = df[p2].values.tolist()
        diag = df['target'].values.tolist()
        data = [((tr1[n], tr2[n] ), 1, diag[n]) for n in range(len(diag))]

        classes = sorted(list(set(diag)))

        colormap = plt.cm.viridis
        num_colors = len(classes)
        discrete_cmap = ListedColormap(colormap(np.linspace(0, 1, num_colors)))
        colors = discrete_cmap.colors


        D = Data(data)
        D.concordance()

        mcmc = MultiClassMonotonicClassifier(D)
        if len(set(diag))%2 == 1:
            mid = len(set(diag))//2
        else:
            mid = len(set(diag))//2 - 1       


        mcmc.DAC(D.data, (rev, up, key), D.classes, mid)

        mcmc.clean_seps()
        
        seps = mcmc.seps


        X,Y = list(), list()
        for c in seps.keys():
            for pt in seps[c]['pt']:
                X.append(pt[0])
                Y.append(pt[1])
                
        minX = min(X)
        minY = min(Y)
        maxX = max(X)
        maxY = max(Y)
        
        step = 0.01
        
        X_ = np.arange(minX - step, maxX + step, step)
        Y_ = np.arange(minY - step, maxY + step, step)


        Xv, Yv = np.meshgrid(sorted(X_), sorted(Y_, reverse=True))
        Z = np.zeros((len(Y_), len(X_)))
        



        for c in sorted(seps.keys(), reverse=True):
            for (x,y) in seps[c]['pt']:
                x = round(x,2)
                y = round(y,2)
                


                if key == 1:

                    for i in range(len(X_)):
                        for j in range(len(Y_)):
                            if x== round(Xv[j,i],2) and y == round(Yv[j,i],2):

                                Z[j,i] = c

                                for l in range(0,j+1):
                                    for m in range(i, len(X_)):
                                        if Z[l,m] == 0:
                                            Z[l,m] =c
                elif key == 3:

                    for i in range(len(X_)):
                        for j in range(len(Y_)):
                            if x== round(Xv[j,i],2) and y == round(Yv[j,i],2):
                                Z[j,i] = c

                                for l in range(j, len(Y_)):
                                    for m in range(0, i+1):
                                        if Z[l,m] == 0:
                                            Z[l,m] =c

                elif key == 4:

                    for i in range(len(X_)):
                        for j in range(len(Y_)):
                            if x== Xv[j,i] and y == Yv[j,i]:
                                Z[j,i] = c

                                for l in range(j, len(Y_)):
                                    for m in range(i, len(X_)):
                                        if Z[l,m] == 0:
                                            Z[l,m] =c

                elif key == 2:

                    for i in range(len(X_)):
                        for j in range(len(Y_)):
                            if x== Xv[j,i] and y == Yv[j,i]:
                                Z[j,i] = c

                                for l in range(0,j+1):
                                    for m in range(0, i+1):
                                        if Z[l,m] == 0:
                                            Z[l,m] =c

        plt.figure(figsize=(5,5))

        plt.xlabel(p1)
        plt.ylabel(p2)

        try:

            CE = cm.at['CE', p]
            LOOCVE = cm.at['LOOCVE', p]
            l1CVE = cm.at['l1CVE', p]

            plt.title('CE = {} & L1CVE = {}, LOOCVE= {}'.format(round(CE/len(data),2), round(l1CVE,2), round(LOOCVE,2)))
        except:
            pass

        plt.imshow(np.flipud(Z), origin='lower', alpha= 0.4)

        for d in data:
            plt.scatter(d[0][0]*100, d[0][1]*100, c=colors[d[2]-1], edgecolors='k')

        if label_dot is not None:
            labels = df[label_dot ].values.tolist()

            for mm in range(len(diag)):
                plt.annotate(labels[mm], (tr1[mm], tr2[mm]), zorder=6)

        plt.show()
        
        
def visualization_multiclass_3(df, pairs, cm, label_dot, step):
    
    for p in pairs:
    
        p1, p2, key = p.split('/')

        key = int(key)
        rev, up = equiv_to_key[key]
        tr1 = df[p1].values.tolist()
        tr2 = df[p2].values.tolist()
        diag = df['target'].values.tolist()
        data = [((tr1[n], tr2[n] ), 1, diag[n]) for n in range(len(diag))]

        classes = sorted(list(set(diag)))

        colormap = plt.cm.viridis
        num_colors = len(classes)
        discrete_cmap = ListedColormap(colormap(np.linspace(0, 1, num_colors)))
        colors = discrete_cmap.colors


        D = Data(data)
        D.concordance()

        mcmc = MultiClassMonotonicClassifier(D)
        if len(set(diag))%2 == 1:
            mid = len(set(diag))//2
        else:
            mid = len(set(diag))//2 - 1       


        mcmc.DAC(D.data, (rev, up, key), D.classes, mid)

        mcmc.clean_seps()
        
        seps = mcmc.seps
        print(seps)

        X,Y = list(), list()
        for c in seps.keys():
            for pt in seps[c]['pt']:
                X.append(pt[0])
                Y.append(pt[1])
                
        minX = min(X)
        minY = min(Y)
        maxX = max(X)
        maxY = max(Y)
        
        
        X_ = np.arange(minX - step, maxX + step, step)
        Y_ = np.arange(minY - step, maxY + step, step)


        Xv, Yv = np.meshgrid(sorted(X_), sorted(Y_, reverse=True))
        Z = np.zeros((len(Y_), len(X_)))
        



        for c in sorted(seps.keys(), reverse=True):
            for (x,y) in seps[c]['pt']:
                x = round(x,2)
                y = round(y,2)
                


                if key == 1:

                    for i in range(len(X_)):
                        for j in range(len(Y_)):
                            if x== round(Xv[j,i],2) and y == round(Yv[j,i],2):

                                Z[j,i] = c

                                for l in range(0,j+1):
                                    for m in range(i, len(X_)):
                                        if Z[l,m] == 0:
                                            Z[l,m] =c
                elif key == 3:

                    for i in range(len(X_)):
                        for j in range(len(Y_)):
                            if x== round(Xv[j,i],2) and y == round(Yv[j,i],2):
                                Z[j,i] = c

                                for l in range(j, len(Y_)):
                                    for m in range(0, i+1):
                                        if Z[l,m] == 0:
                                            Z[l,m] =c

                elif key == 4:

                    for i in range(len(X_)):
                        for j in range(len(Y_)):
                            if x== round(Xv[j,i],2) and y == round(Yv[j,i],2):
                                Z[j,i] = c

                                for l in range(j, len(Y_)):
                                    for m in range(i, len(X_)):
                                        if Z[l,m] == 0:
                                            Z[l,m] =c

                elif key == 2:

                    for i in range(len(X_)):
                        for j in range(len(Y_)):
                            if x== round(Xv[j,i],2) and y == round(Yv[j,i],2):
                                Z[j,i] = c

                                for l in range(0,j+1):
                                    for m in range(0, i+1):
                                        if Z[l,m] == 0:
                                            Z[l,m] =c

        plt.figure(figsize=(5,5))

        plt.xlabel(p1)
        plt.ylabel(p2)

        try:

            CE = cm.at['CE', p]
            LOOCVE = cm.at['LOOCVE', p]
            l1CVE = cm.at['l1CVE', p]

            fig.title('CE = {} & L1CVE = {}, LOOCVE= {}'.format(round(CE/len(data),2), round(l1CVE,2), round(LOOCVE,2)))
        except:
            pass

        #plt.imshow(np.flipud(Z), origin='lower', alpha= 0.4)
        plt.contourf(sorted(X_),sorted(Y_),np.flipud(Z), alpha=0.4)

        for d in data:
            plt.scatter(d[0][0], d[0][1], c=colors[d[2]-1], edgecolors='k')

        if label_dot is not None:
            labels = df[label_dot ].values.tolist()

            for mm in range(len(diag)):
                plt.annotate(labels[mm], (tr1[mm], tr2[mm]), zorder=6)

                




        