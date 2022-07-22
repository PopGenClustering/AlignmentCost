"""
Functions

@author: Xiran Liu 
"""

import numpy as np
from scipy import special
import matplotlib.pyplot as plt
import seaborn as sns


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

# Helper function to compute the mean and variance of Dirichlet distribution
def dir_mean_var(a):
    a0 = np.sum(a)
    avg = a/a0
    var = np.multiply(a,a0-a)/(a0**2*(a0+1))
    return avg, var

# Alignment cost functions
# Eq.10
def repdist(a,b):
    a0 = np.sum(a)
    b0 = np.sum(b)
    a_sub = a[:-1]
    b_sub = b[:-1]
    temp1 = np.sum(np.multiply(a_sub+1,a_sub))+np.sum(np.tril(np.outer(a,a),-1)[:-1,:])
    temp2 = np.sum(np.multiply(b_sub+1,b_sub))+np.sum(np.tril(np.outer(b,b),-1)[:-1,:])
    temp3 = np.sum(np.multiply(a_sub,b_sub))+np.sum(a_sub)*np.sum(b_sub)
    return 2*(temp1/((a0+1)*a0)+temp2/((b0+1)*b0)-temp3/(a0*b0))
# Eq.11
def repdist0(a):
    a0 = np.sum(a)
    return 4*np.sum(np.tril(np.outer(a,a),-1))/((a0+1)*(a0**2))
# Eq.12
def alignment_cost(a,b):
    a0 = np.sum(a)
    return 0.5*np.sum(np.multiply(a,a-b))*2/a0**2


# plotting functions

def plot_heatmap(R, matrix, vmax, labelpad, labelsize, cmap, title, xlab, ylab, save_path):
    mask = np.zeros_like(matrix, dtype=bool)
    mask[np.triu_indices_from(mask,k=1)] = True
    cm = cmap

    fig, ax = plt.subplots(1,1, figsize=(7,6))

    sns.heatmap(matrix, cmap=cm, robust=True, square=True,
                linewidths=0.1, linecolor='white', vmin=0, vmax=vmax,
                annot=matrix, annot_kws={'fontsize': 13}, fmt='.2f',
                cbar_kws={'shrink': 0.8, 'aspect': 13},mask=mask)
    ax.set_xlabel(xlab, fontsize=20)
    ax.set_ylabel(ylab, fontsize=20)
    cbar = ax.collections[0].colorbar
    cbar.set_label(title, labelpad=labelpad, fontsize=labelsize)
    cbar.ax.tick_params(labelsize=15)

    ax.set_xticks([i+0.5 for i in range(R)])
    ax.set_xticklabels(range(1,R+1), fontsize=15)
    ax.set_yticks([i+0.5 for i in range(R)])
    ax.set_yticklabels(range(1,R+1), fontsize=15)

    pass
    fig.savefig(save_path, bbox_inches='tight', format='pdf', dpi=200, transparent=True) #svg

def plot_bar(R, N, K, popIDs, df_ind, colors, save_path):
    
    fig, axes = plt.subplots(R,1,figsize=(20,20))

    for rep in range(R):
        
        ax = axes[rep]
        df_rep = df_ind.loc[rep*N:((rep+1)*N-1)].reset_index(drop=True)
        Q_ind = df_rep[df_rep.columns[5:]].values

        pop_cnt = 0
        n_ind_list = []
        mid_ind_idx_list = []

        for p in popIDs:

            idx = np.array(df_rep[df_rep["popID"]==p].index)
            n_ind = len(idx)
            n_ind_list.append(n_ind)
            Q_pop = Q_ind[idx,:]
            Q_aug = np.hstack((np.zeros((n_ind,1)),Q_pop))

            for i in range(K):
                ax.bar(range(pop_cnt,pop_cnt+n_ind), Q_aug[:,(i+1)], bottom=np.sum(Q_aug[:,:(i+1)],axis=1), 
                       width=1.0, facecolor=colors[i], edgecolor='w', linewidth=0)
            ax.axvline(x=pop_cnt+n_ind, c="gray")
            mid_ind_idx_list.append((pop_cnt+pop_cnt+n_ind)//2)
            pop_cnt += n_ind

        ax.set_xticks([])
        ax.set_xlim([0,Q_ind.shape[0]])
        ax.set_ylim([0,1])
        ax.set_yticks([0,0.5,1])
        ax.set_yticklabels([0,0.5,1], fontsize=20) 
    pass
    fig.savefig(save_path, bbox_inches='tight', format='pdf', dpi=300, transparent=True) 