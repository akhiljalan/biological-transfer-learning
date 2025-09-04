>📋  A template README.md for code accompanying a Machine Learning paper

# Metabolic Network Estimation with Transfer Learning

This repository contains implementation code from the paper: 
```
Jalan, Akhil, et al. "Transfer learning for latent variable network models." Advances in Neural Information Processing Systems 37 (2024): 78797-78835.
```

## Requirements

To install requirements: create a new environment with Python=3.10.9 and the dependencies in `env.yml`.

```setup
conda env create -f env.yml
```

## Evaluation on Test Data 

To recreate the $n_q = 100$ setting in the paper's metabolic transfer experiments, run: 

```eval
python metabolic_network_estimation.py
```
