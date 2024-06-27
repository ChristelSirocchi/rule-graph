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

- **LLM-outputs**
  - Contains the rules output from the logic learning machine, computed within the Rulex platform.

- **synthetic**
  - Contains the datasets generated in the investigation.

## How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/ChristelSirocchi/rule-graph.git
   cd rule-graph
