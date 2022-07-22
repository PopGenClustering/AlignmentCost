"""
Demo of computing alignment cost given membership matrix

@author: Xiran Liu 
"""
import numpy as np
from AlignmentCost.func_utils import *

# load membership matrix
# row: individual, column: membership for each cluster
# The example matrix contains memberships of 8 individuals in 4 clusters.
Q = np.array([[0.15,0.25,0.59,0.01],
    [0.16,0.22,0.55,0.07],
    [0.20,0.18,0.60,0.02],
    [0.16,0.22,0.61,0.01],
    [0.18,0.19,0.56,0.07],
    [0.22,0.25,0.50,0.03],
    [0.20,0.20,0.56,0.04],
    [0.15,0.22,0.57,0.06]])

# sanity check if the membership matrix is valid
assert(np.all(np.abs(Q.sum(axis=1)-1)<1e-6))

# use fixed point method for MLE of Dirichlet parameters
a0 = initial_guess(Q)
a = fixed_point(Q, a0)
print("a=({})".format(",".join(["{:.3f}".format(i) for i in a])))

# compute mean and variance of the empirical data
emp_mean = np.mean(Q,axis=0)
emp_var = np.var(Q,axis=0)
print("empirical mean: {}".format(" ".join(["{:.3f}".format(i) for i in emp_mean])))
print("empirical variance: {}".format(" ".join(["{:.3f}".format(i) for i in emp_var])))

# compute mean and variance of the estimated Dirichlet distribution
est_mean, est_var = dir_mean_var(a)
print("Dirichlet mean: {}".format(" ".join(["{:.3f}".format(i) for i in est_mean])))
print("Dirichlet mean: {}".format(" ".join(["{:.3f}".format(i) for i in est_var])))


# compute alignment cost for permutation pattern
permutation = np.array([0,1,2,3])
b = a[permutation]
cost = alignment_cost(a,b)
print("permutation: {}, cost: {:.3f}".format(" ".join([str(i+1) for i in permutation]),cost))

permutation = np.array([0,1,3,2])
b = a[permutation]
cost = alignment_cost(a,b)
print("permutation: {}, cost: {:.3f}".format(" ".join([str(i+1) for i in permutation]),cost))

permutation = np.array([1,0,2,3])
b = a[permutation]
cost = alignment_cost(a,b)
print("permutation: {}, cost: {:.3f}".format(" ".join([str(i+1) for i in permutation]),cost))