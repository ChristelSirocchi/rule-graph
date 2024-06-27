# Enhancing Interpretability of Rule-Based Classifiers through Feature Graphs

This repository contains the code associated with the project "Enhancing Interpretability of Rule-Based Classifiers through Feature Graphs." In this project, we propose a comprehensive framework for estimating feature contributions in rule-based systems. Our contributions include a graph-based feature visualization strategy, a novel feature importance metric agnostic to rule-based predictors, and a distance metric for comparing rule sets based on feature contributions.

## Repository Structure

The folder contains all the code to replicate the experiments conducted and synthetic data generated. The files are organized as follows:

### Method Implementation

- **parsing_rules.py**
  - Contains parsing functions to parse the rules output from the following rule-based strategies to derive rules from data: decision tree, association rule mining, logic learning machines, and black-box models with rule extraction.

- **graph_building.py**
  - Contains functions to compute a variety of rule relevance and feature relevance metrics, and to generate the adjacency matrix for the feature graph, as proposed by our method.

### Notebooks

- **Synthetic_experiments_notebook.ipynb**
  - Contains the code to generate all synthetic experiments and to analyze them.

- **Benchmark_experiments_notebook.ipynb**
  - Contains the code to validate the method on benchmark datasets.

### Folders

- **synthetic**
  - Contains the datasets generated in the investigation.
    
- **LLM-outputs**
  - Contains the rules output from the logic learning machine, computed within the Rulex platform.

- **Feature-selection**
  - *_imps.csv files: feature importance scores computed according to permutation importance, Gini importance, average SHAP values, and the proposed method (with all using relevance or impurity as feature relevance criterion + relevace, support, lift, confidence or equal as rule relevance criterion);
  - *_scores.csv: accuracy of decision trees trained on the top k features (with k ranging from 2 to 12) according to each of the four considered feature importance metrics.

## Benchmark Datasets

Datasets Hill Valley, Hypothyroid, Pixel, and Tokyo were retrieved from the **Penn Machine Learning Benchmarks** (https://github.com/EpistasisLab/pmlb) while the remaining datasets were retrieved from the **UCI Machine Learning Repository** (https://github.com/uci-ml-repo/ucimlrepo), accessed through the respective Python wrappers.

| **Dataset**         | **#instances** | **#features** | **#binary** | **#categorical** | **#continuous** | **#classes** |
|---------------------|----------------|---------------|-------------|------------------|-----------------|--------------|
| Zoo                 | 101            | 17            | 16          | 1                | 0               | 7            |
| **Breast Tissue**   | 106            | 9             | 0           | 0                | 9               | 6            |
| Hepatitis           | 155            | 19            | 13          | 6                | 0               | 2            |
| BCW Prognostic      | 198            | 34            | 0           | 0                | 34              | 2            |
| SPECT Heart         | 267            | 22            | 22          | 0                | 0               | 2            |
| Breast Cancer       | 286            | 9             | 0           | 9                | 0               | 2            |
| BCW Diagnostic      | 569            | 30            | 0           | 0                | 30              | 2            |
| Balance Scale       | 625            | 4             | 0           | 4                | 0               | 3            |
| BCW Original        | 699            | 9             | 0           | 0                | 30              | 2            |
| **Pima Diabetes**   | 768            | 8             | 0           | 0                | 8               | 2            |
| Tokyo               | 959            | 44            | 0           | 2                | 42              | 2            |
| Hill Valley         | 1212           | 100           | 0           | 0                | 100             | 2            |
| Contraceptive       | 1473           | 9             | 0           | 2                | 7               | 3            |
| Car Evaluation      | 1728           | 6             | 0           | 6                | 0               | 4            |
| Pixel               | 2000           | 240           | 0           | 240              | 0               | 10           |
| Hypothyroid         | 3163           | 25            | 17          | 1                | 7               | 2            |
| Waveform            | 5000           | 21            | 0           | 0                | 21              | 3            |

