from Module_fastBMC.mappings import equiv_to_key, equiv_to_case

import numpy as np
from math import ceil, log, pow
import pandas as pd
import os
import multiprocessing as mp
import copy
import time


## UTILITY FUNCTIONS

def err(v, w, z):
    # Error function defined in the paper
    return w * abs(v - z)


def next_power_of_two(x):
    return int(pow(2, ceil(log(x, 2))))


def index_leaves(A, k):
    # Construct a dictionary linking each leaf index to the index of the corresponding node in the tree
    # A is an array containing all the values of the nodes
    # k is the number of leaves in the tree
    p_lev = next_power_of_two(k) - k
    lev = k - p_lev
    ind_leaves = {}
    for i in range(1, lev+1):
        ind_leaves[i] = (len(A)-lev) + i
    for i in range(1, p_lev+1):
        ind_leaves[lev+i ] = (len(A)-k) + i
    return ind_leaves


def is_leaf(ind_leaves, num):
    # Confirm whether the node under study is a leaf
    return num in ind_leaves.values()

def find_leaves(ind_leaves, num, L):
    # Return a list with all the leaves below the node num
    if not is_leaf(ind_leaves, num):
        find_leaves(ind_leaves, 2*num, L)
        find_leaves(ind_leaves, 2*num +1, L)
    else:
        L.append(num)

    
def Z_(H, A, ind_leaves):
    # Show Z: this one won't appear in the final program. It's just to check the tree for the moment
    Z = list()
    for i in range(1, len(H) + 1):
        v_i = ind_leaves[i]
        Z.append(int(compute_Z(A, v_i)))
    return Z

def is_A_balanced(A):
    return all(x == 0 or y == 0 for x, y in zip(A[1::2], A[2::2]))





## INITIALIZATION

def initialization(data, rev):
    # rev enables testing increasing and decreasing isotonicity
    # X is an array with sorted data
    # H is a sorted list of all column values
    # A is the array initialized at zero
    # ind_leaves is the dictionary linking each leaf index to the index of the corresponding node in the tree
    
    # Sort the data based on the specified sorting order
    X = sorted(data, reverse=rev)
    
    # Extract unique column values and sort them
    H = sorted(list(set([X[i][0][1] for i in range(len(X))])))
    
    # Initialize the array A with zeros
    A = np.zeros(2 * len(H) + 1)
    
    # Add infinity as the last element in H to handle edge cases
    H.append(float('inf'))
    
    # Create the dictionary linking each leaf index to the index of the corresponding node in the tree
    ind_leaves = index_leaves(A, len(H))
    
    return X, H, A, ind_leaves



## STEP 1: Compute Z and Z(g,c_g)

def compute_Z(A, v):
    # Compute Z by going up the tree from leaf v to the root (Z(g, h) = sum of z_v on the path to the root)
    Z = A[v - 1]
    while True:
        if v == 1:
            break
        v = v // 2
        Z += A[v - 1]
    return Z

#These three functions help for updating the tree when Z(g,c_g) is computed (it can change a lot in the path to the root)

def degenerate_interval(A, v, ind_leaves):
    # Handle the case where a specific interval becomes degenerate
    p = v // 2
    mod = v % 2
    while True:
        if p == 1:
            break
        if A[p-1] != 0:
            add_value(A, p, v, ind_leaves)
            A[p-1] = 0
        p = p // 2
        mod = p % 2

        
def add_value(A, p, v, ind_leaves):
    # Adds the value of A[p-1] to each leaf in the subtree rooted at p, excluding v
    L = list()
    find_leaves(ind_leaves, p, L)
    for l in L:
        if l != v:
            A[l-1] += A[p-1]
            
            
            
def rebalance(A, v, ind_leaves):
    # Rebalance the tree by redistributing values in A based on certain conditions
    if v != 0:
        p = v // 2
        mod = v % 2
        w = 2 * p + (1 - mod)
        if A[v-1] != 0 and A[w-1] != 0:
            delta = min(A[v-1], A[w-1])
            A[p-1] += delta
            A[v-1] -= delta
            A[w-1] -= delta
    else:
        mini = float('inf')
        for i in ind_leaves.values():
            Z = compute_Z(A, i)
            if Z < mini:
                mini = Z
        A[0] = mini
        
        
def step1(ind_cg, A, nb_leaves, ind_leaves, err1, S, H):
    mini = float('inf')
    h = 0
    c = 1
    for i in range(ind_cg, nb_leaves+1):
        v_i = ind_leaves[i]
        Z = compute_Z(A, v_i)

        if Z < mini:
            mini = Z
            ind = i
            c = 1
            h = H[i-1]
        elif Z == mini:
            c += 1
    if c > 1:
        while compute_Z(A, ind_leaves[ind]) == mini and ind < nb_leaves:
            ind += 1
        if ind == nb_leaves:
            h = H[ind-1]  # value of the leaves that give minimum Z(g,h)
        else:
            h = H[ind-2]
    if len(S) == 0:
        h = H[0]
    S.append(h)

    deg = compute_Z(A, ind_leaves[ind_cg]) - A[ind_leaves[ind_cg]-1]

    if deg <= mini + err1:
        A[ind_leaves[ind_cg]-1] = mini + err1 - deg
    else:
        A[ind_leaves[ind_cg]-1] = mini + err1
        degenerate_interval(A, ind_leaves[ind_cg], ind_leaves)


## STEP 2 : Update right and left

def v_has_right(v, A):
    return 2 * v + 1 <= len(A)

def update_right(v, A, val):
    # Update right child branch by adding val
    A[2 * v] += val
    
            
def update_all_right(v, A, err0):
    p = v//2
    while True:
        if v_has_right(p, A) and 2*p+1 !=v:
            update_right(p, A, err0)

        if p == 1: #end when root has been updated
            break
        v = p
        p = p//2

def v_has_left(v, A):
    return 2 * v <= len(A)

def update_left(v, A, val):
    # Update left child branch by adding val
    A[2 * v - 1] += val
    
            
def update_all_left(v, A, err1):
    p = v//2
    while True:
        if v_has_left(p, A) and 2*p != v:
            update_left(p, A, err1)

        if p == 1: #end when root has been updated
            break
        v = p
        p = p//2
            
def step2(A, v, err0, err1):
    # Add the error in left and right intervals
    update_all_right(v, A, err0)
    update_all_left(v, A, err1)
    
    
## High-level wrapper that orchestrates the steps of the recursive algorithm

def recursion(A, H, c_g, ind_leaves, err0, err1, nb_leaves, S):
    # Check if c_g is in H
    if c_g in H:
        # Find the index of c_g in H
        ind_cg = H.index(c_g) + 1
        v_cg = ind_leaves[ind_cg]

        # Perform STEP 1
        step1(ind_cg, A, nb_leaves, ind_leaves, err1, S, H)

        # Perform STEP 2
        step2(A, v_cg, err0, err1)
    else:
        print(f"{c_g} is not in the list H.")
        
        
        
## TRACEBACK TO FIND DECISION BOUNDARY

def find_h_Z_min(A, ind_leaves):

    #Finds the index of the leaf with the minimum value in the binary tree represented by A.

    #Parameters:
    #- A: array representing the binary tree
    #- ind_leaves: dictionary linking each leaf index to the index of the corresponding node in the tree

    #Returns:
    #- Index of the leaf with the minimum value or None if the tree is empty.

    p = 1
    while True:
        v = 2 * p
        w = 2 * p + 1
        if A[v - 1] == 0:
            p = v
        else:
            p = w

        if is_leaf(ind_leaves, p):
            return p
    return None

def search_key(dico, val):
    l = [c for c,v in dico.items() if v==val]
    if len(l) != 0:
        return l[0]
    else:
        return -1

def find_highest_point(X, hh):
    # Find the point with the highest rows
    for x in X:
        if x[0][1] == hh:
            return x
        
def find_index_point(X, xy):
    for x in X:
        if xy in x:
            return X.index(x)
    return None


def traceback(A, X, H, ind_leaves, S, up):
    
    #Traceback the breaking points in the regression. It corresponds to the breaking points of red (class 1).

    #Parameters:
    #- A: array representing the binary tree
    #- X: sorted data array
    #- H: sorted list of all column values
    #- ind_leaves: dictionary linking each leaf index to the index of the corresponding node in the tree
    #- S: list of values from the previous steps
    #- up: boolean indicating whether it's an increasing or decreasing isotonicity

    #Returns:
    #- List of breaking points for the decision boundary.
   
    b = search_key(ind_leaves, find_h_Z_min(A, ind_leaves))
    h = H[b - 1]
    breakpoint = list()
    for i in range(len(X) - 1, -1, -1):
        x = X[i]
        xy, w, lab = x
        cg = xy[1]
        if h == cg:
            h = S[i]
            breakpoint.append(xy)

    if X[0][2] == int(up) and X[0][0][1] == H[-2]:
        breakpoint.append(X[0][0])

    hx = find_highest_point(X, H[-2])
    id_hx = X.index(hx)
    if len(breakpoint) != 0:
        id_hbp = find_index_point(X, breakpoint[0])
        if hx[2] == int(up) and id_hx < id_hbp:
            breakpoint.append(hx[0])

    return breakpoint


def labels_point(X, bpr, rev, up):
    
    #Classify points into red (class 1) and blue (class 0) based on red breakpoints and isotonicty direction.

    #Parameters:
    #- X: sorted data array
    #- bpr: list of breakpoints
    #- rev: boolean indicating whether it's a reversed direction
    #- up: boolean indicating whether it's an increasing isotonicity

    #Returns:
    # - Lists of red points, blue points, and the regression error count.
    
    
    r_p = list()
    b_p = list()

    reg_err = 0
    for x in X:
        lab = x[2]
        x = x[0]

        if up and x in bpr:
            r_p.append(x)
            if lab == 0:
                reg_err +=1

        elif not up and x in bpr:
            b_p.append(x)
            if lab == 1:
                reg_err +=1

        else:
            if not rev and up: #CASE 1
                flag = 0
                for br in bpr:
                    if x[0] >= br[0] and x[1] >= br[1]:
                        flag = 1
                if flag == 0:
                    b_p.append(x)
                    if lab == 1:
                        reg_err +=1
                else:
                    r_p.append(x)
                    if lab == 0:
                        reg_err +=1

            if rev and up: #CASE 2
                flag = 0 #consider as blue by default
                for br in bpr:
                    if x[0] <= br[0] and x[1] >= br[1]:
                        flag = 1
                if flag == 0:
                    b_p.append(x)
                    if lab == 1:
                        reg_err +=1
                else:
                    r_p.append(x)
                    if lab == 0:
                        reg_err +=1

            if not rev and not up: #CASE 3
                flag = 1 #consider as red by default
                for br in bpr:
                    if x[0] >= br[0] and x[1] >= br[1]:
                        flag = 0
                if flag == 0:
                    b_p.append(x)
                    if lab == 1:
                        reg_err +=1
                else:
                    r_p.append(x)
                    if lab == 0:
                        reg_err +=1

            if rev and not up: #CASE 4
                flag = 1 #consider as red by default
                for br in bpr:
                    if  x[0] <= br[0] and x[1] >= br[1]:
                        flag =0
                if flag == 0:
                    b_p.append(x)
                    if lab == 1:
                        reg_err +=1
                else:
                    r_p.append(x)
                    if lab == 0:
                        reg_err +=1


    return r_p, b_p, reg_err



#previous functions give us the data labelled as 0 and 1. But if we want
#to predict new points, we must know the official separation of class 0 as well. As we prefer
#false positive rather than false negative, we draw the lines closest
#to "blue" points (class 0 points)

def breakpoint_b(X, b_p, rev, up):
    # Determine blue (class 0) breakpoints based on the direction of isotonicity.

    #Parameters:
    #- X: sorted data array
    #- b_p: list of blue points
    #- rev: boolean indicating whether it's a reversed direction
    #- up: boolean indicating whether it's an increasing isotonicity

    #Returns:
    #- List of blue breakpoints.
    
    
    
    bpb = list()
    b_ps = sorted(b_p)
    if not rev and up: #CASE 1
        while len(b_ps) != 0:
            maxi = b_ps[-1]
            x, y = maxi
            bpb.append(maxi)
            b_ps = [pt for pt in b_ps if pt[1] > y]
            b_ps = sorted(b_ps)


    elif rev and up: #CASE 2
        while len(b_ps) != 0:
            maxi = b_ps[0]
            x, y = maxi
            bpb.append(maxi)
            b_ps = [pt for pt in b_ps if pt[1] > y]
            b_ps = sorted(b_ps)

    elif not rev and not up: #CASE 3
        while len(b_ps) != 0:
            maxi = b_ps[0]
            x, y = maxi
            bpb.append(maxi)
            b_ps = [pt for pt in b_ps if pt[0] > x]
            b_ps = sorted(b_ps)

    elif rev and not up: #CASE 4
        while len(b_ps) != 0:
            maxi = b_ps[-1]
            x, y = maxi
            bpb.append(maxi)
            b_ps = [pt for pt in b_ps if pt[0] < x]
            b_ps = sorted(b_ps)
    return bpb



#Clean the list of border points

def clean_blue(bpr, rev, up):
    
    #Clean the list of blue (class 0) border points based on isotonicity direction.

    #Parameters:
    #- bpr: list of blue border points
    #- rev: boolean indicating whether it's a reversed direction
    #- up: boolean indicating whether it's an increasing isotonicity

    #Returns:
    #- Cleaned list of blue border points.
    
    
    bpr = sorted(bpr)
    nbpr = list()

    if rev and not up:
        for bp in bpr:
            itv = bpr[bpr.index(bp)+1:]
            flag = True
            for it in itv:
                if it[1] <= bp[1]:
                    flag=False
                    break
            if flag:
                nbpr.append(bp)

    elif rev and up:
        for bp in bpr:
            itv = bpr[:bpr.index(bp)]
            flag = True
            for it in itv:
                if it[1] >= bp[1]:
                    flag=False
                    break
            if flag:
                nbpr.append(bp)

    elif not rev and not up:

        for bp in bpr:
            itv = bpr[:bpr.index(bp)]
            flag = True
            for it in itv:
                if it[1] <= bp[1]:
                    flag=False
                    break
            if flag:
                nbpr.append(bp)

    elif not rev and up:

        for bp in bpr:
            itv = bpr[bpr.index(bp)+1:]
            flag = True
            for it in itv:
                if it[1] >= bp[1]:
                    flag=False
                    break
            if flag:
                nbpr.append(bp)

    assert len(nbpr) <= len(bpr)
    return nbpr


def clean_red(bpr, rev, up):
    
    #Clean the list of red (class 1) border points based on isotonicity direction.

    #Parameters:
    #- bpr: list of red border points
    #- rev: boolean indicating whether it's a reversed direction
    #- up: boolean indicating whether it's an increasing isotonicity

    #Returns:
    #- Cleaned list of red border points.
    
    bpr = sorted(bpr)
    nbpr = list()

    if rev and up:

        for bp in bpr:
            itv = bpr[bpr.index(bp)+1:]
            flag = True
            for it in itv:
                if it[1] <= bp[1]:
                    flag=False
                    break
            if flag:
                nbpr.append(bp)

    elif rev and not up:

        for bp in bpr:
            itv = bpr[:bpr.index(bp)]
            flag = True
            for it in itv:
                if it[1] >= bp[1]:
                    flag=False
                    break
            if flag:
                nbpr.append(bp)

    elif not rev and up:

        for bp in bpr:
            itv = bpr[:bpr.index(bp)]
            flag = True
            for it in itv:
                if it[1] <= bp[1]:
                    flag=False
                    break
            if flag:
                nbpr.append(bp)

    elif not rev and not up:

        for bp in bpr:
            itv = bpr[bpr.index(bp)+1:]
            flag = True
            for it in itv:
                if it[1] >= bp[1]:
                    flag=False
                    break
            if flag:
                nbpr.append(bp)

    assert len(nbpr) <= len(bpr)
    return nbpr

## 

def compute_model(data, rev, up):
    
    #Initialize and compute the model components for a given configuration.

    #Parameters:
    #- data: The input data.
    #- rev: Boolean indicating whether to test increasing or decreasing isotonicity.
    #- up: Boolean indicating whether to test for the upper or lower boundary.

    #Returns:
    #Tuple containing misclassification error, breakpoints for the red area (class 1), breakpoints for the blue area (class 0),
    #points labeled as red (class 1), and points labeled as blue (class 0).
    
    
    X, H, A, ind_leaves = initialization(data, rev)
    S = list()
    labs = [x[2] for x in X]
    nb_leaves = len(H)

    for i in range(len(X)):
        x = X[i]
        xy, w, lab = x
        cg = xy[1]
        err0, err1 = err(lab, w, abs(1 - int(up))), err(lab, w, int(up))

        recursion(A, H, cg, ind_leaves, err0, err1, nb_leaves, S)

    while not is_A_balanced(A):
        for v in range(len(A), -1, -1):
            rebalance(A, v, ind_leaves)

    if up:
        bpr = traceback(A, X, H, ind_leaves, S, up)
        r_p, b_p, reg_err = labels_point(X, bpr, rev, up)
        bpb = breakpoint_b(X, b_p, rev, up)
    else:
        bpb = traceback(A, X, H, ind_leaves, S, up)
        r_p, b_p, reg_err = labels_point(X, bpb, rev, up)
        bpr = breakpoint_b(X, r_p, rev, up)

    bpr = clean_red(bpr, rev, up)
    bpb = clean_blue(bpb, rev, up)

    return reg_err, bpr, bpb, r_p, b_p


def compute_recursion(data, case=None):
    
    #Main function to compute the recursion for various configurations.

    #Parameters:
    #- data: The input data.
    #- case: Optional configuration specifying whether to compute for specific cases.

    #Returns:
    #Dictionary containing misclassification error, breakpoints for the red area, breakpoints for the blue area,
    #points labeled as red, and points labeled as blue for each configuration.
    
    models = {}

    if case is None:
        for rev in [True, False]:
            for up in [True, False]:
                case_num = 2 * int(rev) + int(not up) + 1
                models[case_num] = compute_model(data, rev, up)
    else:
        rev, up = case[0], case[1]
        case_num = case[2]
        models[case_num] = compute_model(data, rev, up)

    return models


