This directory contains the code files.

The file **func_utils.py**  contains several helper functions for data analysis and visualization, including
* ```fixed_point(Q,a,n_iter)``` which implements the fixed point algorithm for MLE of Dirichlet parameters from membership matrix Q,
* ```repdist(a,b)``` which implements Eq. 10 in the article,
* ```repdist0(a)``` which implements Eq. 11.
* and ```alignment_cost(a,b)``` which implements Eq. 12.

The file **main.py** contains the code we use to analyze the datasets and produce the plots in Fig. 5 and 6. It also serves as the main body of the Python package.

[*Last edited by Xiran Liu, 7/22/2022*]
