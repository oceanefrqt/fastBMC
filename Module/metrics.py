import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn
import pandas as pd
import math


#### CALCULATE TRUE POSITIVE RATE (TPR) AND FALSE POSITIVE RATE (FPR)
def perf_metrics(y_true, y_hat, threshold):
    
    y_true = np.array(y_true)
    y_hat = np.array(y_hat)
    
    tp = sum((y_hat >= threshold) & (y_true == 1))
    fp = sum((y_hat >= threshold) & (y_true == 0))
    tn = sum((y_hat < threshold) & (y_true == 0))
    fn = sum((y_hat < threshold) & (y_true == 1))
    
    assert (tp + fn) != 0, "True positive rate can't be computed, no true positive and false negative"
    assert (tn + fp) != 0, "False positive rate can't be computed, no false positive and true negative"

    tpr = tp / (tp + fn)
    fpr = fp / (tn + fp)
    return [fpr, tpr]


#### CALCULATE AREA UNDER THE ROC CURVE (AUC)
def auc_score(y_true, y_proba, auc_file=None):
    assert len(set(y_true)) != 1, "Can't compute auc score with only one class"

    thresholds = np.arange(0.0, 1.01, 0.001)

    roc_points = [perf_metrics(y_true, y_proba, threshold) for threshold in thresholds]

    #fpr_array = np.array([[point1[0], point2[0]] for point1, point2 in zip(roc_points[:-1], roc_points[1:])])
    #tpr_array = np.array([[point1[1], point2[1]] for point1, point2 in zip(roc_points[:-1], roc_points[1:])])
    
    fpr_array = []
    tpr_array = []
    
    fpr = []
    tpr = []
    
    for i in range(len(roc_points)-1):
        point1 = roc_points[i];
        point2 = roc_points[i+1]
        tpr_array.append([point1[0], point2[0]])
        fpr_array.append([point1[1], point2[1]])
        
        fpr.append(point1[0])
        tpr.append(point1[1])
        
    tpr = np.array(tpr)
    fpr = np.array(fpr)
    
    tpr_array = np.array(tpr_array)
    fpr_array = np.array(fpr_array)

    #fpr = [point1[0] for point1 in roc_points[:-1]]
    #tpr = [point1[1] for point1 in roc_points[:-1]]

    auc3 = sum(np.trapz(tpr_array, fpr_array)) + 1
    assert 0 <= auc3 <= 1, "AUC score is not in [0,1]"

    gmeans = np.sqrt(tpr * (1 - fpr))

    CI = confidence_interval(auc3, y_true)

    if auc_file is not None:
        plt.plot(tpr_array, fpr_array, color='darkorange', label='ROC curve')
        plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.0])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('AUC={}, CI=[{};{}] '.format(round(auc3, 3), CI[0], CI[1]))
        plt.legend(loc="lower right")
        name = auc_file + '/auc.pdf'
        plt.savefig(name.replace('//', '/'), bbox_inches='tight')

    ix = np.argmax(gmeans)
    thresh = thresholds[ix]
    return auc3, CI, thresh


#### CALCULATE CONFIDENCE INTERVAL FOR AUC
def confidence_interval(auc, labels):
    labels = list(labels)
    N1 = labels.count(1)
    N2 = labels.count(0)
    Q1 = auc / (2 - auc)
    Q2 = (2 * (auc**2)) / (1 + auc)
    SE = np.sqrt((auc * (1 - auc) + (N1 - 1) * (Q1 - auc**2) + (N2 - 1) * (Q2 - auc**2)) / (N1 * N2))
    low_b = auc - 1.96 * SE
    high_b = auc + 1.96 * SE
    if low_b < 0:
        low_b = 0
    if high_b > 1:
        high_b = 1
    return [round(low_b, 3), round(high_b, 3)]


#### CALCULATE CONFUSION MATRIX AND ACCURACY
def confusion_matrix(y_real, y_pred, conf_mat_file=None):
    tp = sum((y_real == 1) & (y_pred == 1))
    fp = sum((y_real == 0) & (y_pred == 1))
    tn = sum((y_real == 0) & (y_pred == 0))
    fn = sum((y_real == 1) & (y_pred == 0))

    conf_mat = [[tn, fn], [fp, tp]]
    accuracy = (tp + tn) / (tp + tn + fp + fn)

    if conf_mat_file is not None:
        df_cm = pd.DataFrame(conf_mat, ['True 0', 'True 1'], ['Pred 0', 'Pred 1'])
        plt.figure(figsize=(5, 5))
        sn.set(font_scale=1.4)
        sn.heatmap(df_cm, annot=True, annot_kws={"size": 16})
        plt.xlabel('True labels')
        plt.ylabel('Output labels')
        plt.title(f'Confusion matrix, Accuracy={accuracy:.3f}')
        plt.legend()
        plt.savefig(conf_mat_file)

    return conf_mat, accuracy


#### CALCULATE ACCURACY
def accuracy_score(y_real, y_pred):
    accuracy = sum(np.array(y_real) == np.array(y_pred)) / len(y_real)
    return accuracy






#cls = classifier
# boolean prediction : 0 if the prediction is correct, 1 if it's wrong
#pred_xi : array of the boolean prediction of xi over the M classifiers
#set (size MxN) is a list of the boolean prediction of the M classifiers over the N patients

def m_xi(pred_xi):
    #number of classifiers producing error for the input sample xi
    #pred_xi is the boolean prediction of xi over the M classifiers
    return sum(pred_xi)

def error_rate_clj(pred_clj):
    # error rate of jth classifier
    #pred_clj is the boolean prediction of the N patients by classifier j
    return sum(pred_clj)/len(pred_clj)

def ensemble_mean_error_rate(set):
    #average error rate over the M classifiers
    #set (size MxN) is a list of the boolean prediction of the M classifiers over the N patients
    M = len(set)
    e = 0
    for i in range(M):
        e += error_rate_clj(set[i])
    return e/M



def yi_MV_1(pred_xi, thresh):
    #pred_xi is the boolean prediction of xi over the M classifiers
    #majority boolean prediction error in favor of the wrong pred 1
    M = len(pred_xi)
    if m_xi(pred_xi) >= thresh*M:
        #if the nb of misclassification is greater than or equal to the nb of classifiers
        #then the preditcion made with majority voting is wrong
        return 1
    else:
        return 0

def yi_MV_0(pred_xi, thresh):
    #pred_xi is the boolean prediction of xi over the M classifiers
    #majority boolean prediction error in favor of correct pred 0
    M = len(pred_xi)
    if m_xi(pred_xi) > M*thresh:
        #if the nb of misclassification is greater than the nb of classifiers
        #then the preditcion made with majority voting is wrong
        return 1
    else:
        return 0

def MVE(set, thresh, meth = yi_MV_1):
    #majority voting error rate
    M = len(set) #Nb of classifiers
    N = len(set[0]) #Nb of patients
    mve = 0
    for i in range(N): #For each patient i
        #construction of pred_xi
        pred_xi = list()
        for j in range(M):#For each classifier j
            pred_xi.append(set[j][i]) #We add the misclassification error of patient i for the classifier j

        yi_mv = meth(pred_xi, thresh) #Whereas the patient i was misclassified or not according to the ensemble

        mve += yi_mv
    return mve/N



def D2_ij(pred1, pred2):
    #disagreement measure for 2 pairs
    #pred1 and pred2 (size N) are the output of classifiers for the N patients
    N = len(pred1)
    D2 = 0
    for i in range(N):
        if pred1[i] != pred2[i]:
            D2 += 1
    return D2/N

def D2(set):
    #average disagreement measure over all pairs of a set
    #set (size MxN) is a list of the boolean prediction of the M classifiers over the N patients
    M = len(set)
    D2 = 0
    for i in range(M):
        for j in range(i+1, M):
            if i != j:
                D2 += D2_ij(set[i], set[j])

    return (2*D2)/(M*(M-1))


def F2_ij(pred1, pred2):
    #double fault measure
    #pred1 and pred2 (size N) are the output of classifiers for the N patients
    N = len(pred1)
    N11 = 0
    for i in range(N):
        if pred1[i] == 1 and pred2[i] == 1:
            #double fault
            N11 +=1
    return N11/N

def F2(set):
    #average double fault measure over all pairs of a set
    #set (size MxN) is a list of the boolean prediction of the M classifiers over the N patients
    M = len(set)
    F2 = 0
    for i in range(M):
        for j in range(i+1, M):
            if i != j:
                F2 += F2_ij(set[i], set[j])
    return (2*F2)/(M*(M-1))

def entropy(set):
    #set (size MxN) is a list of the boolean prediction of the M classifiers over the N patients
    #The entropy measure reaches its maximum (EN 1⁄4 1) for the highest disagreement, which is the case of observing M/2 votes with identical value (0 or 1) and
    #M M/2 with the alternative value. The lowest entropy (EN 1⁄4 0) is observed if all classifier outputs are
    #identical.
    M = len(set)
    N = len(set[0])

    EN = 0
    for i in range(N):
        #construction of pred_xi
        pred_xi = list()
        for j in range(M):
            pred_xi.append(set[j][i])

        EN += min(m_xi(pred_xi), M-m_xi(pred_xi)) / (M-math.ceil(M/2))
    return EN/N
