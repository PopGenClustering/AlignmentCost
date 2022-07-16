"""
Main Script

@author: Xiran Liu 
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import itertools
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import argparse
from AlignmentCost.func_utils import *


def main(args):

    param_file = args.param_file

    with open(param_file) as f:
        param_lines = f.read().splitlines() 

    for line in param_lines:
        if line:
            name = line.split(": ")[0]
            val = line.split(": ")[1]
            print(name,val)
            exec("{} = {}".format(name,val), globals())

    df_ind = pd.read_csv(input_file, delimiter=r"\s+", header = None)
    df_ind = df_ind.rename(columns={1:"indID",3:"popID"})
    N = df_ind["indID"].nunique()
    K = df_ind.shape[1]-5 # number of individuals
    
    popIDs = df_ind["popID"].unique()
    if N*R!=len(df_ind):
        sys.exit('ERROR: total number of rows in the file does not equal N*R. \nPlease check if all replicates contain memberships of the same individuals.')

    
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    

    # plot barcharts of the memberships of all replicates (Fig. 5 and 6 Panel A)
    if K==4:
        colors = [(1, 0.4, 0),(0, 0.6, 0.9),(1, 0.6, 0.9),(0.5, 0, 0.5)]
    else:
        cmap = matplotlib.cm.get_cmap('Spectral')
        colors = [cmap(i)[:3] for i in np.linspace(0,1,K)]
    plot_bar(R, N, K, popIDs, df_ind, colors=colors, save_path=os.path.join(output_path,"all_reps.pdf"))

    # load ground-truth permutation
    perm_results = np.loadtxt(perm_file).astype(int).tolist()

    # obtain permutation w.r.t. each of the replicates
    perm_wrt = [[[perm.index(i) for i in perm_results[k]] for perm in perm_results] for k in range(R)]
    
    # get number of individuals in each population 
    n_ind_list = []
    rep1 = 0 # (from first replicate)
    df_rep1 = df_ind.loc[rep1*N:(N-1)].reset_index(drop=True)
    for p in popIDs:
        idx = np.array(df_rep1[df_rep1["popID"]==p].index)
        n_ind = len(idx)
        n_ind_list.append(n_ind)

    # perform MLE of Dirichlet parameters
    a_list_all = []

    for r in range(R):
        df_rep = df_ind.loc[r*N:((r+1)*N-1)].reset_index(drop=True)
        Q_ind = df_rep[df_rep.columns[5:]].values

        a_list = []

        for p in popIDs:

            idx = np.array(df_rep[df_rep["popID"]==p].index)
            n_ind = len(idx)
            Q_pop = Q_ind[idx,:]
            Q_aug = np.hstack((np.zeros((n_ind,1)),Q_pop))

            a0 = initial_guess(Q_pop)
            a = fixed_point(Q_pop, a0) # estimated parameter
            a_list.append(a)
            
        a_list_all.append(a_list)

    
    # compute pairwise empirical and theoretical cost
    pw_cost_emp = np.zeros((R,R))
    pw_cost_the = np.zeros((R,R))

    for r1 in range(R):
        
        df_rep = df_ind.loc[r1*N:((r1+1)*N-1)].reset_index(drop=True)
        Q_ind = df_rep[df_rep.columns[5:]].values
        Q1 = Q_ind

        a_list1 = a_list_all[r1]

        for p in popIDs:

            idx = np.array(df_rep[df_rep["popID"]==p].index)
            n_ind = len(idx)
            Q_pop = Q_ind[idx,:]
            Q_aug = np.hstack((np.zeros((n_ind,1)),Q_pop))

        for r2 in range(R):

            df_rep = df_ind.loc[r2*N:((r2+1)*N-1)].reset_index(drop=True)
            Q_ind = df_rep[df_rep.columns[5:]].values
            Q2 = Q_ind

            a_list2 = a_list_all[r2]

            for p in popIDs:

                idx = np.array(df_rep[df_rep["popID"]==p].index)
                n_ind = len(idx)
                Q_pop = Q_ind[idx,:]
                Q_aug = np.hstack((np.zeros((n_ind,1)),Q_pop))
                
            # compute empirical cost
            A0 = np.sum((Q1-Q1)**2)
            A = np.sum((Q1-Q2)**2)
            C = (A-A0)/2
            pw_cost_emp[r1,r2] = C/N
            
            # compute theoretical cost
            mean_C_phi_list = [alignment_cost(a,a[np.array(perm_wrt[r1][r2])]) for a in a_list1]
            mean_total_C_phi = np.sum(np.multiply(mean_C_phi_list,n_ind_list))
            pw_cost_the[r1,r2] = mean_total_C_phi/N

    pass

    # average theoretical costs of rep1 w.r.t rep2 and rep2 w.r.t. rep1
    pw_cost_the = (pw_cost_the+pw_cost_the.T)/2
    
    # plot pairwise theoretical cost (Fig. 5 and 6 Panel B)
    cm = plt.get_cmap("YlOrBr")
    plot_heatmap(R=R, matrix=pw_cost_the, vmax=vmax, labelpad=-80, labelsize=20, cmap=cm, title='Theoretical cost', 
        xlab="Replicate", ylab="Replicate", save_path=os.path.join(output_path,"the_cost.pdf"))

    # plot pairwise empirical cost (Fig. 5 and 6 Panel C)
    cm = plt.get_cmap("YlOrBr")
    plot_heatmap(R=R, matrix=pw_cost_emp, vmax=vmax, labelpad=-80, labelsize=20, cmap=cm, title='Empirical cost', 
        xlab="Replicate", ylab="Replicate", save_path=os.path.join(output_path,"emp_cost.pdf"))

    # plot different between empirical and theoretical costs  (Fig. 5 and 6 Panel D)
    cm = plt.get_cmap("GnBu")
    abs_diff = np.abs(pw_cost_emp-pw_cost_the)
    rel_diff = np.divide(abs_diff, pw_cost_the, out=np.zeros_like(abs_diff), where=abs(pw_cost_the)>1e-6)

    plot_heatmap(R=R, matrix=rel_diff, vmax=np.ceil(np.max(rel_diff)*2)/2, labelpad=-90, labelsize=15, cmap=cm, 
        title='Relative difference between \n empirical and theoretical cost', 
        xlab="Replicate", ylab="Replicate", save_path=os.path.join(output_path,"cost_diff.pdf"))

    
    # plot the cost of all possible permutation w.r.t. the first replicate (Fig. 5 and 6 Panel E)
    permutations_of_index = np.array(list(itertools.permutations(range(0,K))))

    num_misaligned = []
    for perm in permutations_of_index:
        num_misaligned.append(np.sum((np.arange(K)-perm)!=0))

    a_list = a_list_all[0]
    perm_wrt1 = perm_wrt[0]

    the_C_wrt1_list = []
    the_C_wrt1_total_list = []
    for perm in permutations_of_index:
        C_wrt1 = []
        for a in a_list:
            C = alignment_cost(a,a[perm])
            C_wrt1.append(C)
        the_C_wrt1_list.append(C_wrt1)
        the_C_wrt1_total_list.append(np.dot(C_wrt1, n_ind_list)/N)

    sorted_idx = np.array(the_C_wrt1_total_list).argsort()
    the_C_wrt1_total_list.sort(reverse=False)

    permutations_of_index_sorted = permutations_of_index[sorted_idx]
    real_perm_idx_sorted = np.array([np.where(np.all(permutations_of_index_sorted==pm,axis=1))[0][0] for pm in perm_wrt1])

    cmap_misnum = plt.get_cmap("RdYlGn_r") #YlOrRd
    colors_misnum = cmap_misnum(np.linspace(0.3, 1, K+1))
    tick_colors = colors_misnum[np.array([num_misaligned[i] for i in sorted_idx])]
    if K==4:
        markers = {0:"D", 2:"o", 3:"s", 4:"^"}
    else:
        markers = {i:"o" for i in range(K+1)}
    tick_markers = [markers[num_misaligned[i]] for i in sorted_idx]
    costs = the_C_wrt1_total_list

    # plot 
    fig, ax = plt.subplots(figsize=(10,6))    

    ax.plot(range(len(costs)),costs,c="gray",alpha=0.5) 
    for i, c in enumerate(costs):
        ax.scatter(i,c,color=tick_colors[i],s=100,edgecolors="k", linewidths=1, zorder=3, marker=tick_markers[i],clip_on=False)

    for i, v in enumerate(costs):
        ax.text(i - .25, v+0.05, "{:0.3f}".format(v), color="k",rotation=70,size=17) 
            
    ax.set_xlabel("Permutation",size=22)
    ax.set_ylabel("Cost",size=22)
    ax.set_xticks(range(len(costs)))
    ax.set_xticklabels(["("+",".join([str(s) for s in permutations_of_index[i]])+")" for i in sorted_idx], 
                       rotation=90, fontsize=18)

    for ticklabel, tickcolor in zip(plt.gca().get_xticklabels(),tick_colors):
        ticklabel.set_color(tickcolor)
    ax.set_ylim([0,1])
    ax.tick_params(axis='y', labelsize=18)

    real_idx = []
    real_cost = []
    for i in range(len(costs)):
        if i in real_perm_idx_sorted:
            real_cost.append(np.mean(pw_cost_emp[0][np.where(real_perm_idx_sorted==i)[0]]))
            real_idx.append(i)
    ax.bar(real_idx,real_cost, color='tab:blue', alpha=0.5, width=0.4) 
    props = dict(boxstyle='round,pad=0.1', edgecolor='None', facecolor='white', alpha=0.7)
    if cost_vs_perm_label_above_bar: 
        for i in range(1,len(real_idx)):
            ax.text(real_idx[i]-0.5, real_cost[i]+0.4, "{:0.3f}".format(real_cost[i]), color="k",
                    rotation=70, size=18, bbox=props, verticalalignment='top')
            ax.axvline(real_idx[i], ymin=real_cost[i], ymax=real_cost[i]+0.35, color='tab:blue', 
                       linestyle=(0, (2, 3)), linewidth=2)
    else:
        for i in range(1,len(real_idx)):
            ax.text(real_idx[i]-0.5, real_cost[i]/2+0.05, "{:0.3f}".format(real_cost[i]), color="k",
                    rotation=70, size=18, bbox=props, verticalalignment='top')

    # number of misaligned clusters
    patches = []
    for i,c in enumerate(colors_misnum):
        if i!=1:
            circ = mlines.Line2D([], [], color=c, marker=markers[i], linestyle='None',mec='k', mew=1,
                              markersize=10, label="{}".format(i))
            patches.append(circ)

    patches2 = []
    patches2.append(mpatches.Patch(color='tab:blue', alpha=0.5, label='mean value of \nreal replicate(s)'))

    leg1 = plt.legend(handles=patches,title="number of \nmisaligned clusters",
                     fontsize=18, fancybox=True,ncol=len(patches), loc=2,
                     labelspacing = 0.3, borderpad=0.3,columnspacing = 0.5,handletextpad=0,
                     title_fontsize=18,bbox_to_anchor=(0,0,0.95,1))#1)loc=(0.6, 0.7)
    plt.gca().add_artist(leg1)
    leg1._legend_box.align = "center"
    leg1.get_title().set_horizontalalignment('center')

    leg2 = plt.legend(handles=patches2, bbox_to_anchor=(0,0,0.95,0.77),
               fontsize=18, fancybox=True,ncol=4,loc=2)
    pass

    fig.savefig(os.path.join(output_path,"cost_vs_perm_rep1.pdf"), bbox_inches='tight', format='pdf', dpi=300, transparent=True) 

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--param_file', type=str, required=True, help="path to the parameter file")

    args = parser.parse_args()
    main(args)