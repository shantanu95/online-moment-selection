# Online Data Collection for Efficient Semiparametric Inference


### Overview

This repository contains the code and data for the paper: <br>
__Online Data Collection for Efficient Semiparametric Inference__ [[PDF]](https://arxiv.org/pdf/2411.03195)
<br>
Shantanu Gupta, Zachary Lipton, David Childers

### Setup

We use [`miniconda`](https://docs.anaconda.com/miniconda/) and [`poetry`](https://python-poetry.org/) for managing dependencies.
The following commands setup the Python dependencies.


```bash
# Install conda environment.
conda install --name oms --file environment.yml
conda activate oms

# Install poetry.
pip install poetry

# Install dependencies.
poetry install
```

To execute the results in parallel, we use [`ipyparallel`](https://ipyparallel.readthedocs.io/en/latest/).
The following command starts the clusters:
```bash
ipcluster start -n <num_engines>
```

### Code and Datasets

The following Jupyter notebooks contain the code for running the experiments in our paper:

* `nonlinear_iv_LATE_main.ipynb`: Code for the experiment for the nonlinear instrumental variable (IV) graph (Figure 3). 
* `jtpa_iv_LATE_main_MLP.ipynb`: Code for the experiment with the JTPA dataset (Figure 4a).
* `copd_data_main.ipynb`: Code for the experiment with the COPD dataset (Figure 4b).
* `linear_iv_LATE_main.ipynb`: Code for the experiment for the linear IV graph (Figure 7a).
* `linear_frontdoor_backdoor_main.ipynb`: Code for the experiment for the linear confounder-mediator graph (Figure 7b).
* `observational_two_covariates_main.ipynb`: Code for the experiment for combining two observational datasets (Figure 7c).

We also have the following additional notebooks:
* `jtpa_data_processing.ipynb`: Generates the `datasets/jtpa_processed.pkl` file used for our experiments. 
* `jtpa_IV_true_LATE.ipynb`: Compute the ground-truth LATE for the JTPA dataset.
* `copd_data_true_ATE.ipynb`: Compute the ground-truth ATE for the COPD dataset.

### Citation
If you find this code useful, please consider citing our work:
```bib
@misc{gupta2024onlinedatacollectionefficient,
      title={Online Data Collection for Efficient Semiparametric Inference}, 
      author={Shantanu Gupta and Zachary C. Lipton and David Childers},
      year={2024},
      eprint={2411.03195},
      archivePrefix={arXiv},
      primaryClass={stat.ML},
      url={https://arxiv.org/abs/2411.03195}, 
}
```