# fastBMC

Author: Océane FOURQUET

fastBMC\[[3]()\] is an extension of an established approach \[[2](https://academic.oup.com/jid/article/217/11/1690/4911472?login=true)\] to construct an emseble classifer based on bivariate monotonic models \[[1](https://link.springer.com/article/10.1007/s00453-012-9628-4)\]. It is implemented in Python and integrates a preselection phase reducing drastically the running time of the approach compared to the original one.



## Python and Libraries Versions
- python 3.9.1
- pandas 1.2.3
- numpy 1.19.2
- matplotlib 3.3.4
- multiprocessing

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



## References
[1] Stout, Q.F. Isotonic Regression via Partitioning. Algorithmica 66, 93–112 (2013). https://doi.org/10.1007/s00453-012-9628-4

[2] Iryna Nikolayeva, Pierre Bost, Isabelle Casademont, Veasna Duong, Fanny Koeth, Matthieu Prot, Urszula Czerwinska, Sowath Ly, Kevin Bleakley, Tineke Cantaert, Philippe Dussart, Philippe Buchy, Etienne Simon-Lorière, Anavaj Sakuntabhai, Benno Schwikowski, A Blood RNA Signature Detecting Severe Disease in Young Dengue Patients at Hospital Arrival, The Journal of Infectious Diseases, Volume 217, Issue 11, Pages 1690–1698 (2018) https://doi.org/10.1093/infdis/jiy086

[3]
