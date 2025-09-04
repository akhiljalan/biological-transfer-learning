# Metabolic Network Estimation with Transfer Learning

This repository contains implementation code from the paper: 
```
Jalan, Akhil, et al. "Transfer learning for latent variable network models." Advances in Neural Information Processing Systems 37 (2024): 78797-78835.
```

## Requirements

To install requirements: create a new environment with Python=3.10.9 and the dependencies in `requirements.txt`.


```setup
python3.10 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Evaluation on Test Data 

To recreate the $n_q = 100$ setting in the paper's metabolic transfer experiments, run: 

```eval
python metabolic_network_estimation.py
```
