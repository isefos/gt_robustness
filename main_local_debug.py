"""
You can use this script to run experiments "locally" 
(directly with sacred, instead of through seml)
"""
from main import ex
import yaml
import torch


# for debugging:
torch.autograd.set_detect_anomaly(True, check_nan=True)

with open("configs_local_debug/example.yaml") as f:
    cfg = yaml.safe_load(f)

r = ex.run(config_updates=cfg)
