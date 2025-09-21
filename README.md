# Adversarial Robustness of Graph Transformers

Code for the TMLR submission "Adversarial Robustness of Graph Transformers".

## Installation

Our setup runs with Python 3.11.7 including the following packages and versions:

* pytorch 2.1.2 (cuda12.1, cudnn8.9.2)
* pyg (pytorch geometric) 2.4.0
* torch-scatter 2.1.2
* torch-sparse 0.6.18
* lightning 2.1.3
* opt_einsum 3.3.0
* tensorboardx 2.6.2.2
* yacs 0.1.8
* numpy 1.26.3
* scipy 1.11.4
* pandas 2.1.4
* scikit-learn 1.6.1
* networkx 3.1
* ogb 1.3.6

## Other Notes

This codebase is based on the GPS implementation, which is built on the GraphGym structure of PyTorch Geometric:

* [GPS: General Powerful Scalable Graph Transformers](https://github.com/rampasek/GraphGPS)
* [PyTorch Geometric](https://github.com/pyg-team/pytorch_geometric/tree/master)
* [GraphGym (PyG)](https://pytorch-geometric.readthedocs.io/en/2.3.1/notes/graphgym.html)

Our attack implementations are based on the PRBCD implementation:

* [PRBCD (PyG)](https://pytorch-geometric.readthedocs.io/en/2.3.1/_modules/torch_geometric/contrib/nn/models/rbcd_attack.html)

We thank the authors for making their code publicly available.