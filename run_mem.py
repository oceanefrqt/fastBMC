#!/usr/bin/env python

from Module import stages as st
from Module import monotonic_classifier as mc
from Module.metrics import accuracy_score, auc_score
from Module.mappings import equiv_to_key, equiv_to_case
from Module.prediction_functions import predict_fav_class1
from Module import dynamic_preselection as dpsl
from Module import selection_strategies as ss
from Module.prediction_functions import proba_ensemble_model, pred_ensemble_model


import numpy as np
import pandas as pd
import sys
import argparse
import os

def parse_args():
    parser=argparse.ArgumentParser(description="fastBMC")
    parser.add_argument("dataset", help="dataset in csv format")
    parser.add_argument("--target", help="name of the column of the classes", default='target')
    parser.add_argument("--nbcpus", help="number of cpus to use to do calculation", type=int, default=1)
    parser.add_argument("--m", help="number of minimum disjoint pairs for preselection", type=int, default=30)
    parser.add_argument("--favoring", help="class to favor in the model between 0 (control case) and 1 (cohort study)", default = 1, type=int, choices=[0,1])
    parser.add_argument("--outdir", help="directory for out files", default='./')
    #parser.add_argument("--selection", help="selection strategy to use for the ensemble")
    args=parser.parse_args()
    return args


def verify_nb_classes_dataset(df, target):
    if target in df.columns:
        tt = set(df[target])
        if len(tt) != 2:
            print('Expected number of classes: 2. There are {} classes in the dataset.\n'.format(len(tt)))
            return False
        else:
            return True
    else:
        print("Can't find the class labels column. Configure the name of the labels column with argument --target.\n")


def modify_label_classes_dataset(df, target):
    df.rename({target:'target'}, axis=1, inplace=True)
    if not set(df['target']) == {0,1}:
        tt = list(set(df['target']))
        df[target] = df['target'].map({tt[0]:0,tt[1]:1})
    return df


def main():
    inputs=parse_args()

    strat = [ss.NB]
    if inputs.favoring == 1:
        funct = predict_fav_class1

    else:
        funct= predict_fav_class0

    try:
        df = pd.read_csv(inputs.dataset,index_col=0, low_memory=False)
        df.reset_index(drop=True, inplace=True)
    except:
        print("Can't open the file {}. Check the format.\n".format(inputs.dataset))
        
    if not os.path.isdir(inputs.outdir):
        os.makedirs(inputs.outdir)
        print("Outdir created")


    if not verify_nb_classes_dataset(df, inputs.target):
        sys.exit()
        
    df = modify_label_classes_dataset(df, inputs.target)

    
    k_opt, thresh = st.stage_1(df, 1, inputs.m, inputs.nbcpus, strat, funct)

    print('Between 2 and {}, the minimal optimal number of classifiers for the ensemble model is {}. \n'.format(inputs.m, k_opt))

    acc, auc, ci = st.stage_2(df, k_opt, thresh, inputs.outdir, inputs.nbcpus, funct, strat)

    print('The estimated performance of the monotonic ensemble model is AUC = {} +/- {}. \n'.format(auc, ci))

    pairs, cm = st.stage_3(df, k_opt, thresh, inputs.nbcpus, funct, strat)

    print('The ensemble model is made of the following pairs : ')
    for pair in pairs:
        print('- {} \n'.format(pair[:-2]))

    if inputs.favoring == 1:
        sr.show_results_pos(df, pairs, inputs.nbcpus, inputs.outdir, cm)
    else:
        sr.show_results_neg(df, pairs, inputs.nbcpus, inputs.outdir, cm)

    output_file = inputs.outdir + '/output.txt'
    f = open(output_file.replace('//', '/'), 'w')
    for pair in pairs:
        f.write('{} \n'.format(pair[:-2]))
    f.close()








if __name__ == "__main__":
    main()
