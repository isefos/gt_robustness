import numpy  # noqa, fixes mkl error
import graphgps  # noqa, register custom modules
import torch
torch.backends.cuda.matmul.allow_tf32 = True  # Default False in PyTorch 1.12+
torch.backends.cudnn.allow_tf32 = True  # Default True
import datetime
import os
import logging
from seml.experiment import Experiment
from torch_geometric.graphgym.config import cfg, set_cfg
from torch_geometric.graphgym.register import train_dict
from graphgps.attack.attack import prbcd_attack_dataset
from graphgps.attack.transfer import transfer_attack_dataset
from graphgps.attack.transfer_unit_test import transfer_unit_test
from graphgps.loader.dataset.robust_unittest import RobustnessUnitTest
from graphgps.run_utils import (
    convert_cfg_to_dict,
    convert_readonly_to_dict,
    setup_run,
    initialize_run,
    load_best_val_model,
)

    
def main_function(cfg):
    loaders, loggers, model, optimizer, scheduler = initialize_run(cfg)
    logging.info(f"[*] Starting now: {datetime.datetime.now()}, with seed={cfg.seed}, running on {cfg.accelerator}")
    # Train
    if cfg.pretrained.dir and not cfg.pretrained.finetune:
        training_results = None
    else:
        assert cfg.train.mode != 'standard', "Default train.mode not supported, use `custom` (or other specific mode)"
        if cfg.train.mode == "custom":
            training_results = train_dict[cfg.train.mode](loggers, loaders, model, optimizer, scheduler)
        elif cfg.train.mode == "adversarial":
            training_results = train_dict[cfg.train.mode](loggers, loaders, model, optimizer, scheduler)
        else:
            raise ValueError(f"Invalid training mode: train.mode=`{cfg.train.mode}`")

    # Robustness unit test
    rut_results = None
    if cfg.robustness_unit_test.enable:
        assert isinstance(loaders[0].dataset, RobustnessUnitTest), (
            "To run the robustness unit test, the model should be trained on a "
            "corresponding RobustnessUnitTest dataset,"
        )
        if cfg.robustness_unit_test.load_best_model:
            model = load_best_val_model(model, training_results)
        logging.info("Running robustness unit test")
        rut_results = transfer_unit_test(model, loaders[2])

    # Attack
    attack_results = None
    if cfg.attack.enable:
        if cfg.attack.load_best_model:
            model = load_best_val_model(model, training_results)
        if cfg.attack.get("transfer", {}).get("enable", False):
            attack_results = transfer_attack_dataset(model, loaders, cfg.attack.transfer.perturbation_path)
        else:
            attack_results = prbcd_attack_dataset(model, loaders)

    logging.info(f"[*] Finished now: {datetime.datetime.now()}")
    results = {"training": training_results, "attack": attack_results, "robustness_unit_test": rut_results}
    return results


ex = Experiment()

set_cfg(cfg)
cfg_dict = convert_cfg_to_dict(cfg)
ex.add_config({"graphgym": cfg_dict, "dims_per_head": 0, "dims_per_head_PE": 0})

os.makedirs("configs_seml/logs", exist_ok=True)


@ex.automain
def run(seed, graphgym, dims_per_head: int, dims_per_head_PE: int):
    graphgym = convert_readonly_to_dict(graphgym)
    setup_run(graphgym, dims_per_head, dims_per_head_PE, seed)
    results = main_function(cfg)
    results["run_dir"] = str(cfg.run_dir)
    results["num_params"] = cfg.params
    return results
