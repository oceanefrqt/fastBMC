import pandas as pd
import numpy as np
import heapq as hq
from copy import deepcopy
import multiprocessing as mp
import time
import sys
import math

from random import shuffle

from Module import monotonic_classifier as mc

from Module.mappings import equiv_to_key, equiv_to_case
from Module.prediction_functions import proba_ensemble_model, pred_ensemble_model
from Module.metrics import accuracy_score, auc_score

def monotonic_model_CE(p, df):
    p1, p2, key = p.split('/')
    key = int(key)
    rev, up = equiv_to_key[key]
    tr1, tr2, diag = df[p1].tolist(), df[p2].tolist(), df['target'].tolist()
    data = [((tr1[n], tr2[n]), 1, diag[n]) for n in range(len(diag))]
    m = mc.compute_recursion(data, (rev, up, key))
    reg, bpr, bpb, pr, pb = m[key]
    return reg, p

def H_df(df, cls, nbcpus):
    pool = mp.Pool(nbcpus)
    vals = [(c, df) for c in cls]
    res = pool.starmap(monotonic_model_CE, vals, max(1, len(vals) // nbcpus))
    pool.close()
    return sorted(res)

def monotonic_model_LOOCVE(p, df, maxi, funct):
    p1, p2, key = p.split('/')
    key = int(key)
    rev, up = equiv_to_key[key]
    tr1, tr2, diag = df[p1].tolist(), df[p2].tolist(), df['target'].tolist()
    data = [((tr1[n], tr2[n]), 1, diag[n]) for n in range(len(diag))]

    err = 0

    for d in data:
        data_bis = deepcopy(data)
        data_bis.remove(d)
        m = mc.compute_recursion(data_bis, (rev, up, key))
        target, out = d[2], d[0]
        reg, bpr, bpb, rps, bps = m[key]
        pred = funct(out, bpr, bpb, rev, up)
        err += abs(target - pred)
        if err > maxi:
            return maxi + 1, p

    return err, p

def Q_df(df, cls, nbcpus, maxi, funct):
    pool = mp.Pool(nbcpus)
    vals = [(c, df, maxi, funct) for c in cls]
    S = sorted(pool.starmap(monotonic_model_LOOCVE, vals, max(1, len(vals) // nbcpus)))
    pool.close()
    return S


def heapify(arr, n, i):
    # Find the largest among root, left child and right child
    largest = i
    l = 2 * i + 1
    r = 2 * i + 2

    if l < n and arr[i][0] < arr[l][0]:
        largest = l

    if r < n and arr[largest][0] < arr[r][0]:
        largest = r

    # Swap and continue heapifying if root is not largest
    if largest != i:
        arr[i], arr[largest] = arr[largest], arr[i]
        heapify(arr, n, largest)


# Function to insert an element into the tree
def insert(array, newNum):
    size = len(array)+1
    if size == 0:
        array.append(newNum)
    else:
        array.append(newNum)
        for i in range((size // 2) - 1, -1, -1):
            heapify(array, size, i)


# Function to delete an element from the tree
def deleteNode(array, num):
    size = len(array)
    i = 0
    for i in range(0, size):
        if num == array[i][0]:
            break


    array[i], array[size - 1] = array[size - 1], array[i]

    array.remove(array[size - 1])


    for i in range((len(array) // 2) - 1, -1, -1):
        heapify(array, len(array), i)


def deleteNodeNum(array, num):
    size = len(array)
    i = 0
    for i in range(0, size):
        if num == array[i][0]:
            break

    array[i], array[size - 1] = array[size - 1], array[i]
    array.remove(array[size - 1])
    for i in range((len(array) // 2) - 1, -1, -1):
        heapify(array, len(array), i)

def deleteNodeName(array, name):
    size = len(array)
    i = 0
    for i in range(0, size):
        if name == array[i][1]:
            break

    array[i], array[size - 1] = array[size - 1], array[i]
    array.remove(array[size - 1])

    for i in range((len(array) // 2) - 1, -1, -1):
        heapify(array, len(array), i)



def keepPairs(Q, pair, q):
    g1, g2, g3 = pair.split('/')
    r1 = lookGene(Q, g1)
    r2 = lookGene(Q, g2)

    f1 = True
    f2 = True

    pairs = list()

    if r1[0]:
        if q >= r1[1][0]:
            f1 = False
        pairs.append(r1[1])

    if r2[0]:
        if q >= r2[1][0]:
            f2 = False
        pairs.append(r2[1])

    if f1 and f2:
        return True, pairs
    else:
        return False, pairs

def lookGene(Q, gene):
    flag = False
    pair_with_gene = None
    for el in Q:
        p1, p2, p3 = el[1].split('/')
        if gene == p1 or gene == p2:
            flag = True
            pair_with_gene = el
            break
    return flag, pair_with_gene


def suppH(H, num):
    j = 0
    while H[j][0] < num:
        j+=1
    return H[:j]




def nb_de_genes(G):
    s = set()
    for k in G.keys():
        s = s.union(G[k])
    return len(s)


def H_dict(H):
    Hd = dict()
    for h in H:
        kh = h[0]
        if kh not in Hd.keys():
            Hd[kh] = list()
        Hd[kh].append(h[1])
    return Hd


def Q_dict(Q):
    Qd = dict()
    G = dict()
    for q in Q:
        kq = q[0]
        g1, g2, g3 = q[1].split('/')
        if kq not in Qd.keys():
            Qd[kq] = set()
        if kq not in G.keys():
            G[kq] = set()
        Qd[kq].add(q[1])
        G[kq].add(g1)
        G[kq].add(g2)


    return Qd, G

def update_dict(G, G_):
    for key in G_.keys():
        if key not in G.keys():
            G[key] = G_[key]
        else:
            G[key] = G[key].union(G_[key])
    return G


def supp_H_above_a(H, a):
    S = [k for k in H.keys() if k > a]
    for s in S:
        del H[s]
    return H

def supp_H_below_a(H, a):
    S = [k for k in H.keys() if k <= a]
    for s in S:
        del H[s]
    return H



def check_disjoint_pairs_shuffled(Q, param):
    # List to store disjoint pairs
    dis_p = list()
    
    # List to keep track of genes already added
    genes = list()
    
    # Flag to indicate whether the condition is met
    flag = False
    
    # Combine all pairs from different score levels into a single list and shuffle
    pairs = [p for k in sorted(Q.keys()) for p in Q[k]]
    shuffle(pairs)

    # Iterate through the shuffled pairs
    for p in pairs:
        g1, g2, g3 = p.split('/')
        
        # Check if both genes g1 and g2 are not already in the list of genes
        if g1 not in genes and g2 not in genes:
            dis_p.append(p)
            genes.append(g1)
            genes.append(g2)
            
            # Check if the number of disjoint pairs exceeds the specified parameter
            if len(dis_p) >= param:
                return True
    return False


def check_disjoint_pairs_naive(Q, param):
    # List to store disjoint pairs
    dis_p = list()
    
    # List to keep track of genes already added
    genes = list()
  
    # Iterate through sorted keys in Q (keys represent different scores or levels)
    for k in sorted(Q.keys()):
        pairs = Q[k]

        # Iterate through pairs in the current score level
        for p in pairs:
            # Split the pair into three genes
            g1, g2, g3 = p.split('/')
            
            # Check if both genes g1 and g2 are not already in the list of genes
            if g1 not in genes and g2 not in genes:
                dis_p.append(p)
                genes.append(g1)
                genes.append(g2)
                
                # Check if the number of disjoint pairs exceeds the specified parameter
                if len(dis_p) >= param:
                    return True

    # If the loop completes without finding enough disjoint pairs, return False
    return False


def check_disjoint_pairs(Q, param, nbcpus):
    # Check disjoint pairs using the naive method
    if check_disjoint_pairs_naive(Q, param):
        return True
    
    # Calculate the total number of pairs in Q
    len_Q = sum(len(pairs) for pairs in Q.values())
    
    # Use multiprocessing to check disjoint pairs with the shuffled method
    pool = mp.Pool(nbcpus)
    vals = [(Q, param) for _ in range(math.ceil(math.sqrt(len_Q)))] 
    res = pool.starmap(check_disjoint_pairs_shuffled, vals, max(1, len(vals)//nbcpus))
    pool.close()

    # Check if any of the multiprocessing results or the naive result is True
    if any(res):
        return True
    
    return False

    

### Preselection based on the number of genes
def algorithm_1(cls, df, k, nbcpus, funct):

    t0 = time.time()
    H = H_df(df, cls, nbcpus)
    t1 = time.time()



    Hd = H_dict(H)

    Q = dict() #For each strat of LOOCV, we have a list of pairs
    G = dict() #List of genes in Q

    count = 0



    t2 = time.time()


    for h_key in sorted(Hd.keys()):

        pairs = Hd[h_key]
        Q_ = Q_df(df, pairs, nbcpus, max(Hd.keys()), funct)
        Qd, Gd = Q_dict(Q_)


        G = update_dict(G, Gd)
        Q = update_dict(Q, Qd)


        if nb_de_genes(G) >=k:
            break


    a = max(Q.keys())
    G_ = deepcopy(G)
    del G_[a]
    while nb_de_genes(G) > k and nb_de_genes(G_) >= k:
        G = G_
        del Q[a]
        a = max(Q.keys())
        G_ = deepcopy(G)
        del G_[a]




    a = max(Q.keys()) #Highest value of LOOCV in Q
    Hd = supp_H_above_a(Hd, a)
    Hd = supp_H_below_a(Hd, h_key)




    for h_key in sorted(Hd.keys()):
        a = max(Q.keys())
        if h_key <= a:

            pairs = Hd[h_key]
            Q_ = Q_df(df, pairs, nbcpus, a, funct)
            Qd, Gd = Q_dict(Q_)

            Qd = supp_H_above_a(Qd, a)



            Gd = supp_H_above_a(Gd, a)



            G = update_dict(G, Gd)
            Q = update_dict(Q, Qd)


            G_ = deepcopy(G)
            del G_[a]
            while nb_de_genes(G) > k and nb_de_genes(G_) >= k:
                G = G_
                del Q[a]
                a = max(Q.keys())
                G_ = deepcopy(G)
                del G_[a]

    pairs = list()
    for key in Q.keys():
        pairs += Q[key]


    return Q, pairs




## Preselection based on the number of pairs

def algorithm_2(cls, df, m, nbcpus, funct):

    H = H_df(df, cls, nbcpus)



    Hd = H_dict(H)

    Q = dict() #For each strat of LOOCV, we have a list of pairs


    count = 0


    

    for h_key in sorted(Hd.keys()):

        pairs = Hd[h_key]
        Q_ = Q_df(df, pairs, nbcpus, max(Hd.keys()), funct) #Compute the LOOCVE of the pairs with RE = h_key
        Qd, Gd = Q_dict(Q_)

        Q = update_dict(Q, Qd)# Update Q to get the pairs grouped according to their LOOCVE


        if check_disjoint_pairs_naive(Q,m):
            break


    a = max(Q.keys()) #Highest value of LOOCV in Q
    Hd = supp_H_above_a(Hd, a)
    Hd = supp_H_below_a(Hd, h_key)
    


    for h_key in sorted(Hd.keys()):

        
        a = max(Q.keys())
        if h_key <= a:

            pairs = Hd[h_key]
            Q_ = Q_df(df, pairs, nbcpus, a, funct)
            Qd, Gd = Q_dict(Q_)

            Qd = supp_H_above_a(Qd, a)

            Q = update_dict(Q, Qd)

            Q_ = deepcopy(Q)
            del Q_[a]

            while check_disjoint_pairs_naive(Q_, m):
                

                Q = Q_
                a = max(Q.keys())

                Q_ = deepcopy(Q)
                del Q_[a]
                
               

    pairs = list()
    for key in Q.keys():
        pairs += Q[key]



    return Q, pairs


def all_configurations(df):
    transcripts = list(df.columns)
    transcripts.remove('target')

    configurations = list()
    for i in range(len(transcripts)):
        for j in range(i+1, len(transcripts)):
            for key in range(1,5):
                configurations.append('/'.join([transcripts[i], transcripts[j], str(key)]))
    return configurations

