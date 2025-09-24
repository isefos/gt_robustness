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
* seml 0.4.2


## Experiment Files

We use the [seml](https://github.com/TUM-DAML/seml/tree/master) framework to execute experiments and track results in a Mongo database. We refer to the [seml documentation](https://github.com/TUM-DAML/seml/blob/master/docs.md#seml-configure) for setup and configuration of your MongoDB. 

All our experiment configuration files are located in the `configs_seml/configs` directory. The file structure follows the pattern: `configs_seml/configs/<model>/<dataset>/<experiment>.yaml`. The options for each are:
- **Models**:
  - **GAT**
  - **GATv2**
  - **GCN**
  - **GPS** (local: GatedGCN, global: Transformer, PE: spectral/laplacian + deepset)
  - **GPS-GCN** (local: GCN, global: Transformer, PE: spectral/laplacian + deepset)
  - **Graphormer**
  - **GRIT**
  - **Polynormer**
  - **SAN**

- **Datasets**:
  - **CLUSTER** (for CLUSTER attacks we use either <ins>a</ins>rbitrary <ins>s</ins>ampling or <ins>c</ins>onstrained <ins>s</ins>ampling, so we append _as and _cs to the experiment names)
  - **reddit_threads**
  - **UPFD_gos_bert** (uses the BERT node features)
  - **UPFD_pol_bert** (uses the BERT node features)

- **Experiments**:
  - **hs**: Training / hyperparameter search.
  - **attack**: Our adaptive attack + random attack baseline.
  - **rand_pert**: Single random perturbation baseline.
  - **transfer**: Transfer attacks by applying precomputed perturbations (from other model adaptive attacks).
  - **ablations**: Ablations for different attack relaxation configurations.
  - **best_adv**: Adversarial training using the hyperparameters from the normal hyperparameter search.
  - **attack_adv** and **transfer_adv**: Attacking the adversarially trained models.

(Note that not all combinations are defined)

### Hyperparameter search ranges

The hyperparameter search and training configurations are defined in the `hs.yaml` files. We used a random search, sampling values for each hyperparameter from predefined ranges. The ranges are defined in the `random` block in `hs.yaml`. In this block, `samples` shows how many different configurations we ran. Additionally, each parameter is listed with the interval ranges and the sampling type (integer, uniform, log-uniform). Since the random seeds are also predetermined, runnning the hyperparameter search should result in running the same configurations again, ensuring complete reproducibility.

After the hyperparameter search, we selected the best model based on the validation metric. This best configuration (and model) is used in subsequent attack experiments. As a result, the best hyperparameters that we found in our search can be found in the other configuration files, such as `attack.yaml` in the `fixed` block.

### Order of experiments and customization

Notably, we separated the experiments for training and attacking. We first ran the hyperparameter seach to find a good model. We then saved the weights of this pre-trained model and load it again for the attack experiments. 

During training, model checkpoints can be automatically saved to disk. We copy the checkpoint of the best model to a known location: `path-to-model/best.ckpt`. In the attack configuration yaml file, we then point to the model path:
```
fixed:
  graphgym:
    pretrained:
      dir: 'path-to-model'
```
Additionally, the model configurations must correspond to the loaded model (will results in error during loading otherwise). Thus, **if you train and load your own models you must replace the respective configuration values** (everything below the line indicated by `# copied from pre-trained model` in the yaml files).

### Run the experiments

#### 1) Add experiments

To add an experiment to the database, use the following command:
```
seml <db_collection> add configs_seml/configs/<model>/<dataset>/<experiment>.yaml
```

As an example, we can create a database collection "a_gph_upfd_pol" (automatically gets created if it does not exist already) for experiments of the type "attack the Graphormer model on the UPFD politifact dataset". The configurations of these experiments are defined in the file `configs_seml/configs/Graphormer/UPFD_pol_bert/attack.yaml`. Note that seml adds 32 experiments from this single configuration file, one for each combination of random seed and attack budget defined in the `grid` block. To add the experiments to the database, use the `add` command:
```
seml a_gph_upfd_pol add configs_seml/configs/Graphormer/UPFD_pol_bert/attack.yaml
```

#### 2) Execute added experiments
To start the experiments, use the `start` command. By default, this will try to execute via SLURM. To run locally, add the `local` flag:
```
seml a_gph_upfd_pol start --local
```

## Other Notes

This codebase is based on the GPS implementation, which is built on the GraphGym structure of PyTorch Geometric:

* [GPS: General Powerful Scalable Graph Transformers](https://github.com/rampasek/GraphGPS)
* [PyTorch Geometric](https://github.com/pyg-team/pytorch_geometric/tree/master)
* [GraphGym (PyG)](https://pytorch-geometric.readthedocs.io/en/2.3.1/notes/graphgym.html)

Our attack implementations are based on the PRBCD implementation:

* [PRBCD (PyG)](https://pytorch-geometric.readthedocs.io/en/2.3.1/_modules/torch_geometric/contrib/nn/models/rbcd_attack.html)

We thank the authors for making their code publicly available.
