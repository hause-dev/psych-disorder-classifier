# Psychiatric Disorder Classifier
This repository contains jupyter notebooks with various machine learning algorithms trained on the datasets.
# Overview
Early detection and intervention are critical for prevention and effective treatment of psychiatric disorders. However, current diagnostic methods have limitations such as patient bias and interpretation. Using data from electroencephalograms (EEG), machine learning models were built and trained on different feature combinations. To address the curse of dimensionality resulting from high-dimensional spaces for a small amount of data, principal component analysis was applied. 

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
The notebook folder contains multiple notebooks used to extract datasets and run models. `Training_loop.ipynb` is the main loop that runs a specified dataset through the machine learning models for every band (delta, theta, alpha, beta, highbeta, gamma) and inserts it into a csv file. `Data_with_Age_Education_IQ.ipynb` contains code for extracting certain sections of the original data. 


# Data
The Data folder contains the different feature combination data for each psychiatric disorder. All of the EEG data was sourced from `https://osf.io/8bsvr/`, an open repository.

# License
This project is licensed under the GNU General Public License v3.0
