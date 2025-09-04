>📋  A template README.md for code accompanying a Machine Learning paper

# My Paper Title

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

To generate all experimental results, run the script: 

```eval
python ell2_run_all.py
```
