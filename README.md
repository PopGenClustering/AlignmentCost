Alignment Cost
------
#### The content of this GitHub repository is identical to the supplementary zip file provided along with the article **"A Dirichlet model of alignment cost in mixed-membership unsupervised clustering."**
It contains all datasets used as examples and the code (in a Python package **AlignmentCost**) to perform the empirical data analysis described in the article. 

[*Last edited by Xiran Liu, 9/12/2022*]

------

This README file describes the following:
  * [Contents of the folder](#contents)
  * [Installation instructions](#how-to-install) 
  * [Toy example on calculating the alignment cost from a membership matrix](#how-to-calculate-the-alignment-cost-from-a-membership-coefficient-matrix)
  * [How to reproduce the data analysis in the article](#how-to-run-the-data-analysis-in-the-article)
  * [How to analyze your own data](#how-to-run-the-analysis-on-your-own-data)


## Contents

* [AlignmentCost](AlignmentCost): the folder containing the code files.
* [demo](demo): the folder containing the demonstration of how to calculate the alignment cost given a membership matrix.
* [data](data): the folder containing the data analysis scripts used for generating Fig. 5 and 6 in the article.

#### [AlignmentCost](AlignmentCost)
The **AlignmentCost** folder contains the code files: *fund_utils.py* and *main.py*.
 * *func_utils.py* contains several helper functions for data analysis and visualization, including 
   * ```repdist(a,b)``` which implements Eq. 10 in the article,
   * ```repdist0(a)``` which implements Eq. 11,
   * and ```alignment_cost(a,b)``` which implements Eq. 12.
 * *main.py* contains the code we use to analyze the datasets and produce the plots in Fig. 5 and 6. It also serves as the main body of the Python package.

#### [demo](demo)
The **demo** folder contains the file *compute_alignment_cost_from_Q_demo.py*, which shows a toy example on calculating the alignment cost from a membership coefficient matrix. 

#### [data](data)
The **data** folder contains the input data files (in [data/input/](data/input)) and the scripts (in [data/scripts/](data/scripts)) used to generate the output figures (in [data/output/](data/output)).

1. **Input data files**: the **data/input** folder contains four data files taken from the Supplementary materials of *Fortier et al. 2020* that we used for the empirical anlaysis. 
  
    We **modified the ordering** of the populations and the ordering of the clusters in the raw *CLUMPP* data files so that the membership barplot patterns of the first replicates of both datasets are consistent with that of Fig. 2 in *Fortier et al. 2020*. Besides the reordering, there is no change to the data.

     * *all_loci_analysis_input_file.txt* and *all_codis_analysis_input_file.txt* are taken from files *CLUMPP/CLUMPP_input/all_indfile.txt* and *CLUMPP/CLUMPP_input/codis_indfile.txt* of the *Fortier et al. 2020* Supplementary Materials, and have been modified from *Fortier et al. 2020* by reordering. 
     * *all_loci_analysis_perm_file.txt* and *codis_loci_analysis_perm_file.txt* are taken from the middle parts of the *Fortier et al. 2020* files named *CLUMPP/CLUMPP_output/all_ind_miscfile.txt* and *CLUMPP/CLUMPP_output/codis_ind_miscfile.txt* files that contain the permutation information of the aligned replicates.

2. **Script files**: The **scripts** folder contains 
   * the parameter files used for the examples (more details appear below in a later section).
   * a sample bash script with the command line argument. 

3. **Output files**: The **output** folder will be automatically generated when running the examples provided. Figures will be output to this folder.

## How to Install

* Install [Python 3](https://www.python.org/downloads/).
* Open a terminal shell. E.g., [Git Bash](https://git-scm.com/downloads) shell on Windows, Bash Shell on Linux, or Mac Terminal on Mac.

**NOTE 1. Depending on your operating system, you may need to use the command ```python3``` instead of ```python``` and ```pip3``` instead of ```pip``` for all the following commands.** You can check this by running
  ```
  which python
  ```
  and
  ```
  which python3
  ```
  These commands will identify the location of executable Python on your system. You should use the one that returns you the path, not the one that is not found.
  
  Alternatively, if your system supports the command ```python3``` but you prefer using ```python```, you may choose to set the alias by
  ```
  alias python=python3
  ```
  then you can go ahead and use the ```python``` command. Same for the command ```pip```.

**NOTE 2. If you are using a Linux system, please install ```python3-matplotlib``` after installing Python 3 before installing the AlignmentCost package.** For example, if you are using Linux Red Hat, run
  ```
  sudo yum install python3-matplotlib
  ```

* Install the package:

  * Download and install the AlignmentCost package from GitHub:
 
    Download the repository **AlignmentCost** (https://github.com/PopGenClustering/AlignmentCost) to local (*click on the green **Code** button on the top right corner, then download the repository as a ZIP file.*)

    Unzip the file **AlignmentCost-main.zip**. 
    
    Navigate to the directory '''AlignmentCost-main''' (using the command ```cd```). You should now be under the main ```AlignmentCost-main/``` directory which contains the *README.md* file and the *setup.py* file. 
    
    Then, to install the package, type

    ```
    pip install -e .
    ```
    
  * Alternatively, you may install the package (without data and example scripts) directly from GitHub [**not recommended**]:

    ```
    pip install git+https://github.com/PopGenClustering/AlignmentCost
    ```
    
    This will install only the contents in the *AlignmentCost* folder but not the *data* folder and the *demo* folder. The package will be installed to your default path of site-packages for Python. You can view its location by running the following command:
    ```
    pip show AlignmentCost
    ```

* The *AlignmentCost.egg-info* folder will be automatically generated upon installation, and you can ignore this folder. You also don't need to change anything in the file *setup.py*.

## How to Use

> You may run the following toy example to check if the package is installed successfully.

**NOTE 3. You may run this part from any directory.**

### How to calculate the alignment cost from a membership coefficient matrix
 
  * Launch Python. 
  
    On Windows Git Bash, run
    ```
    python -i
    ```
    with the key "-i" for interactive prompt. You cursor should now appear on a line starting with ``>>>``.
  
  * Import packages.
  
    ```python
    import numpy as np
    from AlignmentCost.func_utils import *
    ```
    
  * Load the membership matrix. 
  
    Here we use a toy example of an 8x4 membership matrix. Each row stands for a individual and each column stands for a cluster. The entries of the matrix correspond to the membership coefficients of 8 individuals for each of 4 clusters.
    ```python
    Q = np.array([
    [0.15,0.25,0.59,0.01],
    [0.16,0.22,0.55,0.07],
    [0.20,0.18,0.60,0.02],
    [0.16,0.22,0.61,0.01],
    [0.18,0.19,0.56,0.07],
    [0.22,0.25,0.50,0.03],
    [0.20,0.20,0.56,0.04],
    [0.15,0.22,0.57,0.06]])
    ```
  
  * Use a fixed-point algorithm to perform MLE (maximum likelihood estimation) of the Dirichlet parameters.
    ```python
    a0 = initial_guess(Q)
    a = fixed_point(Q, a0)
    ```
    
  * Print the estimated parameters.
    ```python
    print("a=({})".format(",".join(["{:.3f}".format(i) for i in a])))
    ```
    You should get the following line:
    ```
    a=(41.716,50.887,133.328,7.592)
    ```
    This line represents an estimate of Dirichlet parameters $(a_1,a_2,a_3,a_4)$.
    
  * Verify the estimation by comparing the mean and variance of the empirical data and the estimated distribution.
    ```python
    emp_mean = np.mean(Q,axis=0)
    emp_var = np.var(Q,axis=0)
    print("empirical mean: {}".format(" ".join(["{:.3f}".format(i) for i in emp_mean])))
    print("empirical variance: {}".format(" ".join(["{:.3f}".format(i) for i in emp_var])))
    
    est_mean, est_var = dir_mean_var(a)
    print("Dirichlet mean: {}".format(" ".join(["{:.3f}".format(i) for i in est_mean])))
    print("Dirichlet variance: {}".format(" ".join(["{:.3f}".format(i) for i in est_var])))
    ```
     You should get the following lines:
    ```
    empirical mean: 0.177 0.216 0.568 0.039
    empirical variance: 0.001 0.001 0.001 0.001
    Dirichlet mean: 0.179 0.218 0.571 0.033
    Dirichlet variance: 0.001 0.001 0.001 0.000
    ```
    
  * Compute the alignment cost for a specific permutation pattern. 
  
    For instance, with the identity permutation, run
    ```python
    permutation = np.array([0,1,2,3])
    ```
    Note that Python uses zero-based indexing but we label clusters $1,2,\ldots,K$. Hence, Python's $(0,1,2,3)$ refers to clusters $(1,2,3,4)$, for example.
    
    Then run
    ```python
    b = a[permutation]
    cost = alignment_cost(a,b)
    print("permutation: {}, cost: {:.3f}".format(" ".join([str(i+1) for i in permutation]),cost))
    ```
    You should get the following line:
    ```
    permutation: 1 2 3 4, cost: 0.000
    ```
    For this toy example Q matrix, if you set the permutation to be `permutation = np.array([0,1,3,2])`, then after repeating the above commands, you will get `permutation: 1 2 4 3, cost: 0.290`. If you set the permutation to be `permutation = np.array([1,0,2,3])`, then you will get `permutation: 2 1 3 4, cost: 0.002`.

  * Quit Python.
    ```
    exit()
    ```

The above Python commands are provided in [compute_alignment_cost_from_Q_demo.py](compute_alignment_cost_from_Q_demo.py). You may run this file from any Python IDE, or simply run the command below:

```
python demo/compute_alignment_cost_from_Q_demo.py
```

------
> You may run the following parts to perform data analysis using the alignment cost.

**NOTE 4. Please run these parts from the directory ```AlignmentCost-main/```.**

### How to run the data analysis in the article 
You may reproduce the analysis on the datasets in Fig. 5 and 6 in the article. This involves running scripts on a parameter file.

  * For *STRUCTURE* replicates based on the full set of 791 loci (Fig. 5):
    ```
    python -m AlignmentCost.main --param_file data/scripts/all_loci_analysis_param.txt
    ```
    The parameter setting will be printed out to the command window. This command may take a few minutes to finish.
  
  * For *STRUCTURE* replicates based on 13 CODIS loci (Fig. 6):
    ```
    python -m AlignmentCost.main --param_file data/scripts/codis_loci_analysis_param.txt
    ```
    The parameter setting will be printed out to the command window. This command may take a few minutes to finish.
  
    These commands will generate 5 plots for each dataset, corresponding to the 5 panels in the manuscript figures:
    1. `all_replicates.pdf`: the stacked bar plots of all replicates (not aligned yet).
    2. `theoretical_cost.pdf`: a heatmap of pairwise theoretical cost between replicates.
    3. `empirical_cost.pdf`: a heatmap of pairwise empirical cost between replicates.
    4. `cost_difference.pdf`: a heatmap of the relative difference between the two costs.
    5. `cost_vs_perm_rep1.pdf`: a cost vs. permutation plot showing all the possible permutations of clusters with respect to the first replicate (rep1) and their theoretical alignment cost, together with bars showing the empirical costs from the rest of the replicates. For these empirical costs, the permutations (alignment) of the subsequent replicates w.r.t. the first one are those permutations that are present in the *CLUMPP* output file used as the input.

### How to run the analysis on your own data
  ```
  python -m AlignmentCost.main --param_file PATH_TO_YOUR_OWN_PARAMETER_FILE
  ```
  ```PATH_TO_YOUR_OWN_PARAMETER_FILE``` should contain information about your own data, where 
  1. ```input_file``` points to a **space-delimited** input file in the same format as the example file  [```data/input/all_loci_analysis_input_file.txt```](data/input/all_loci_analysis_input_file.txt). It should contain 5+K columns, where K is the number of clusters in the replicates. The 2nd column contains the individual identifier number and the 4th column contains the population identifier number. It follows the *CLUMPP* individual input file format. The path must be wrapped by quotes, as in the example parameter file located in ```data/scripts```.
  2. ```perm_file``` points to a **space-delimited** file containing the correct permutations (alignment) with respect to the first replicate. The file should be in the same format as the example file  [```data/input/all_loci_analysis_perm_file.txt```](data/input/all_loci_analysis_perm_file.txt). Each row corresponds to the permutation of one replicate, and each column corresponds to a cluster. The first row should always be ```1 2 ... K```. These rows correponding to the permutations can be extracted from the middle part of the output miscellaneous file *miscfile.txt* of *CLUMPP*. In the miscellaneous file, these rows can be found following the text *"the list of permutations of the clusters that produces that H value is (runs are listed sequentially on separate rows)"*. The path to the file containing these rows must be wrapped by quotes, as in the example parameter file located in ```data/scripts```. 
  3. ```output_path``` points to the directory where you want to save your output figures. The path must be wrapped by quotes, as in the example file.
  4. ```R``` is the number of replicates.
  5. ```vmax``` is a number to specify the upper limit of the colorbar when plotting the heatmap of costs (Panel B and C). You may leave it as the default value 1.0.
  6. ```cost_vs_perm_label_above_bar``` is either ```True``` or ```False```, depending on where you want the labels of the empirical costs to be in Panel E.
  * Note that the paths can either be relative or absolute.
  
