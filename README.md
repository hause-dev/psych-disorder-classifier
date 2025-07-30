# Psychiatric Disorder Classifier
This repository contains jupyter notebooks with various machine learning algorithms trained on the datasets.
# Overview
Early detection and intervention are critical for prevention and effective treatment of psychiatric disorders. However, current diagnostic methods have limitations such as patient bias and interpretation. Using data from electroencephalograms (EEG), machine learning models were built and trained on different feature combinations. To address the curse of dimensionality resulting from high-dimensional spaces for a small amount of data, principal component analysis (PCA) was applied. 

This project is based on the data and methodology from:

Park SM, Jeong B, Oh DY, Choi C-H, Jung HY, Lee J-Y, Lee D and Choi J-S (2021) Identification of Major Psychiatric Disorders From Resting-State Electroencephalography Using a Machine Learning Approach. Front. Psychiatry 12:707581. doi: 10.3389/fpsyt.2021.707581

# Installation
This project can be installed using git via:

```
git clone https://github.com/hause-dev/psych-disorder-classifier.git
```

All libraries can be installed through conda using 'requirements.txt'

You can find the conda-forge installation instructions at https://conda-forge.org/download/

Create a conda environment using
```
conda create --name psych-classifier 
```
And activate the environment with
```
conda activate psych-classifier
```
Then install the module with
```
conda install --file requirements.txt
```
To set up the Jupyter environment, use

```
python -m ipykernel install --user --name=psych-classifier
jupyter lab
```
This will create a Jupyter kernel that matches the installed packages and then open the editor.


# Notebooks
The notebook folder contains multiple notebooks used to extract datasets, run models, and get results. 

`Data_Extraction.ipynb` - This notebook extracts specified sections from the original dataset and generates CSV files for the specified disorder.

`Model_Training.ipynb` - This notebook contains the main loop that runs every file in a specified folder through the machine learning models for every band (delta, theta, alpha, beta, highbeta, gamma, all) and inserts the results into a csv file. 

`DisordervsDisorder.ipynb` - This notebook compares specified disorders to other disorders and runs a loop through all the frequency bands and different machine learning models and prints out the results. 

`Finding_Best_Features.ipynb` - This notebook calculates the averages and standard deviations for each algorithm in the csv files generated from the Model_Training notebook. It also filters through the different disorders and finds the feature combinations that produced the best results for each algorithm and each disorder, then stores them in a csv file.

`Plotting_Results.ipynb` - This notebook takes in the files with averages and standard deviations, and plots the results in line graphs on matplotlib. 

The notebooks should be run in the order they are listed above. 


# Data
The Data folder contains the different feature combination data for each psychiatric disorder. All of the EEG data was sourced from `https://osf.io/8bsvr/`, an open repository.

# License
This project is licensed under the GNU General Public License v3.0
