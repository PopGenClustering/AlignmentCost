"""
Functions

@author: Xiran Liu 
"""

import numpy as np
from scipy import special

# import pandas as pd
# import os
# import sys
# # import matplotlib.pyplot as plt
# import time
# from itertools import product,combinations_with_replacement
# from collections import defaultdict


# Inverse of the digamma function 
def digamma_inv(y, n_iter=5):
    
    t_thre = -2.22
    x = np.exp(y) + 0.5 if y > t_thre else 1.0 / (-y+special.psi(1))

    for i_iter in range(n_iter):
        x = x-(special.psi(x)-y)/special.polygamma(1, x)

    return x

# Initial guess for Dirichlet MLE
def initial_guess(Q):
    Eq1 = np.mean(Q[:,0])
    Eq1sqr = np.mean(Q[:,0]**2)
    frac = Q.sum(axis=0)/Q.sum()
    if Q.shape[0]==1:
        return frac
    denom = (Eq1sqr-Eq1**2)
    if (Eq1sqr-Eq1**2)!=0:
        a = np.multiply(frac,(Eq1-Eq1sqr)/(Eq1sqr-Eq1**2))
    else:
        a = frac
    return a

# Fixed point method for Dirichlet MLE
def fixed_point(Q, a0, n_iter = 10):
    K = len(a0)
    N = Q.shape[0]
    if N==1:
        return a0
    logq_bar = np.sum(np.log(Q),axis=0)/N
    a = a0
    a_next = a
    for i_iter in range(n_iter):
        a_sum = np.sum(a)
        for k in range(K):
            a_next[k] = digamma_inv(special.psi(a_sum)+logq_bar[k])
        if np.sum(np.abs(a_next-a))<1e-3:
            a = a_next
            break
        a = a_next
    return a

# Alignment costs
def repdist(a,b):
    a0 = np.sum(a)
    b0 = np.sum(b)
    a_sub = a[:-1]
    b_sub = b[:-1]
    temp1 = np.sum(np.multiply(a_sub+1,a_sub))+np.sum(np.tril(np.outer(a,a),-1)[:-1,:])
    temp2 = np.sum(np.multiply(b_sub+1,b_sub))+np.sum(np.tril(np.outer(b,b),-1)[:-1,:])
    temp3 = np.sum(np.multiply(a_sub,b_sub))+np.sum(a_sub)*np.sum(b_sub)
    return 2*(temp1/((a0+1)*a0)+temp2/((b0+1)*b0)-temp3/(a0*b0))


def repdist0(a):
    a0 = np.sum(a)
    return 4*np.sum(np.tril(np.outer(a,a),-1))/((a0+1)*(a0**2))
    
def alignment_cost(a,b):
    a0 = np.sum(a)
    return 0.5*np.sum(np.multiply(a,a-b))*2/a0**2

# def cost_general(a,b):
#     return 0.5*(repdist(a,b)-repdist0(a))