# fastBMC
==============================================

## Introduction

fastBMC\[[3]()\] is an extension of an established approach \[[2](https://academic.oup.com/jid/article/217/11/1690/4911472?login=true)\] to construct an ensemble classifier based on bivariate monotonic classifiers \[[1](https://link.springer.com/article/10.1007/s00453-012-9628-4)\]. It is implemented in Python and integrates a preselection phase reducing drastically the running time of the approach compared to the original one.

## Possible Use Cases

The fastBMC approach, which identifies an ensemble model made of bivariate monotonic classifiers, has been designed to apply to transcriptomic data for predicting binary outcomes. However, its applications extend beyond transcriptomics and can be used for any binary classification problem with continuous data.
- Transcriptomic analysis: fastBMC can be used to analyze transcriptomic data from microarrays or RNA-seq experiments to identify pairs of genes that are associated with a binary outcome, such as disease vs. healthy or treated vs. untreated.
- Biomarker discovery: By identifying pairs of genes that are strongly associated with a binary outcome, fastBMC can be used to discover novel biomarkers for diseases or conditions.



## Python and Libraries Versions
- python 3.9.1
- pandas 1.2.3
- numpy 1.19.2
- matplotlib 3.3.4
- multiprocessing
- psutil

This code uses the multiprocessing library to parallelize calculations. By default, the number of CPUs is set to 1, but it is possible to change this configuration with the parameters.

## How to Run the Code
To use this code, your data needs to be prepared and pretreated in advance. This code does not realize any pretreatment, normalization, etc.

### Data
Data should be presented in csv format, in the form of samples per row and features per column, with a column for classes. For the moment, this code only allows to work on data with two classes.

### Code
To run the approach with the default parameters, just run the following command in your shell:
```python
python3 run_mem.py <dataset>
```
with \<dataset\> corresponding to the dataset file and repository.


There are some optional parameters, such as the number of CPUs to use for the calculations, the maximum number of pairs for the ensemble model, the name of the label column in the dataset file, etc. By default, the outputs of the code are stored in the current repository, but it can be configured differently with the optional parameter --outdir . The ouput files comprise a txt file with the final pair classifiers for the ensemble model, a pdf file with the roc curve and pdf files visualizing the final pair classifiers.

For more information about parameters, run the command:
```python
python3 run_mem.py -h
```


### Example
In the Example subrepository, you can find a dataset and the output files that you would obtain by running the following command line:
```python
python3 run_mem.py sobar-72.csv  --nbcpus 12 --target ca_cervix
```
The dataset comes from the UCI Machine Learning Repository ([dataset](https://archive.ics.uci.edu/ml/datasets/Cervical+Cancer+Behavior+Risk)). It contains 19 attributes regarding cervical cancer behavior risk, with class label ca_cervix with 1 and 0 as values which means the respondent is with and without cervical cancer, respectively.

### Experiments
In the Experiments subrepository, you can find the code that was used to get the results for the publication [3].


In the Simulated Data, the create_simulated_data.py file enables to create the simulated data, you can select the directory for the data:
```python
python3 create_simulated_data.py -your_outdir/
```
Then, fastBMC_simulated_data.sh creates a dataframe containing five columns: the simulated dataset, the running time, the auc, the accuracy, and the f1 score, by using the run_fastBMC_simulated_data.py script. The bash script needs to be updated by selecting the directory containing the simulated dat that you want to run. The python script takse as parameters the same ones as run_mem.py. You can adjust your parameters by modifying the following line in the bash script:
```python
output=$(python3 run_fastBMC_simulated_data.py "$file" -your_parameters)
```
The values from the output dataframe are used to construct Figures 4 and 5 in [3].


In the Real Data, you can run the fastBMC.sh and naiveBMC.sh to get two CSV files respectively, containing three columns: the number of features as input, the running time, and the AUC performance. These values are used to construct Figures 6 and 7 in [3]. The two bash script need to be updated by taking as input_file the dataset that you want to run. run_fastBMC_time_auc.py and run_naiveBMC_time_auc.py takes as parameters the same ones as run_mem.py. You can also adjust your parameters by modifying the following line in the bash script:
```python
output=$(python3 run_fastBMC_time_auc.py "$temp_output" -your_parameters)
```


## References
[1] Q.F Stout. Isotonic Regression via Partitioning. Algorithmica 66, 93–112 (2013). https://doi.org/10.1007/s00453-012-9628-4

[2] I. Nikolayeva, P. Bost, I. Casademont, V. Duong, F. Koeth, M. Prot, U. Czerwinska, S. Ly, K. Bleakley, T. Cantaert, P. Dussart, P. Buchy, E. Simon-Lorière, A. Sakuntabhai, B. Schwikowski. A Blood RNA Signature Detecting Severe Disease in Young Dengue Patients at Hospital Arrival. The Journal of Infectious Diseases 217, 1690–1698 (2018). https://doi.org/10.1093/infdis/jiy086

[3] O. Fourquet, M.S. Krejca, C. Doerr, B. Schwikowski. Towards the Genome-scale Discovery of Bivariate Monotonic Classifiers. bioRxiv (2024). https://doi.org/10.1101/2023.02.22.529510
