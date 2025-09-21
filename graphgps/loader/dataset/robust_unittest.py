import os.path as osp
from typing import Callable, List, Optional
import numpy as np
import torch
from torch_geometric.data import (
    Data,
    InMemoryDataset,
    download_url,
)
from torch_geometric.utils import to_undirected, coalesce


names = ["cora_ml", "citeseer"]
models = ["gcn", "jaccard_gcn", "svd_gcn", "rgcn", "pro_gnn", "gnn_guard", "grand", "soft_median_gdc"]
splits = [0, 1, 2, 3, 4]
scenarios = ["evasion", "poisoning"]


class RobustnessUnitTest(InMemoryDataset):
    r"""The Robust Unittest dataset from <https://www.cs.cit.tum.de/daml/are-gnn-defenses-robust/>
    <https://github.com/LoadingByte/are-gnn-defenses-robust/blob/master/unit_test/sketch.py>

    Includes Cora ML and Citeseer with different splits. Also includes precomputed perturbations.

    Args:
        root (str): Root directory where the dataset should be saved.
        name (str): The name of the graph set (:obj:`"cora_ml"`,
            :obj:`"citeseer"`).
        split (int): (0 to 4) Will load the dataset with the corresponding split.
            (default: :obj:`0`)
        pert_scenario (str, optional): None, "evasion", or "poisoning". Defines 
            which perturbation to load (if any). (default: :obj:`None`)
        transform (callable, optional): A function/transform that takes in an
            :obj:`torch_geometric.data.Data` object and returns a transformed
            version. The data object will be transformed before every access.
            (default: :obj:`None`)
        pre_transform (callable, optional): A function/transform that takes in
            an :obj:`torch_geometric.data.Data` object and returns a
            transformed version. The data object will be transformed before
            being saved to disk. (default: :obj:`None`)
        pre_filter (callable, optional): A function that takes in an
            :obj:`torch_geometric.data.Data` object and returns a boolean
            value, indicating whether the data object should be included in the
            final dataset. (default: :obj:`None`)
    """
    url = "https://github.com/LoadingByte/are-gnn-defenses-robust/raw/master/unit_test/unit_test.npz"

    def __init__(
        self,
        root: str,
        name: str,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        pre_filter: Optional[Callable] = None,
    ):
        self.name = name.lower()
        assert self.name in names
        super().__init__(root, transform, pre_transform, pre_filter)
        self.load_data()

    def load_data(self):
        self.data, self.slices, self.budgets = torch.load(self.processed_paths[0])

    @property
    def raw_dir(self) -> str:
        return osp.join(self.root, "raw")

    @property
    def processed_dir(self) -> str:
        return osp.join(self.root, self.name, "processed", "clean")

    @property
    def raw_file_names(self) -> List[str]:
        return "unit_test.npz"

    @property
    def processed_file_names(self):
        return "data.pt"
    
    def _get_budget_prefix(self, split: int, model: str, scenario: str) -> str:
        return f"{self.name}/perturbations/{scenario}/{model}/split_{split}/budget_"

    def download(self):
        download_url(self.url, self.raw_dir)

    def process(self):
        with np.load(self.raw_paths[0]) as f:
            budgets = self._read_budgets(f)
            data = self._read_data_clean(f)
        data = data if self.pre_transform is None else self.pre_transform(data)
        data, slices = self.collate([data])
        torch.save((data, slices, budgets), self.processed_paths[0])

    def _read_budgets(self, loader):
        budgets = []
        for split in splits:
            budgets.append({})
            for model in models:
                budgets[split][model] = {}
                for scenario in scenarios:
                    prefix = self._get_budget_prefix(split, model, scenario)
                    bs = []
                    for k in loader.keys():
                        if k.startswith(prefix):
                            bs.append(k[len(prefix):])
                    budgets[split][model][scenario] = sorted([int(b) for b in bs])
        return budgets

    def _read_data_clean(self, loader):
        prefix = f"{self.name}/dataset/"
        x_index = torch.from_numpy(loader[prefix + "features"]).to(torch.long)
        y = torch.from_numpy(loader[prefix + "labels"]).to(torch.long)
        N = y.size(0)
        D = int(x_index[:, 1].max()) + 1
        x = torch.zeros((N, D))
        x[x_index[:, 0], x_index[:, 1]] = 1
        edge_index = torch.from_numpy(loader[prefix + "adjacency"]).to(torch.long).T
        edge_index = to_undirected(edge_index, num_nodes=x.size(0))
        data = Data(x=x, edge_index=edge_index, y=y)
        train_mask = torch.zeros((N, 5), dtype=torch.bool)
        val_mask = torch.zeros((N, 5), dtype=torch.bool)
        test_mask = torch.zeros((N, 5), dtype=torch.bool)
        prefix = f"{self.name}/splits/"
        for s in splits:
            train_nodes = torch.from_numpy(loader[f"{prefix}{s}/train"]).to(torch.long)
            train_mask[train_nodes, s] = True
            val_nodes = torch.from_numpy(loader[f"{prefix}{s}/val"]).to(torch.long)
            val_mask[val_nodes, s] = True
            test_nodes = torch.from_numpy(loader[f"{prefix}{s}/test"]).to(torch.long)
            test_mask[test_nodes, s] = True
        data.train_mask = train_mask
        data.val_mask = val_mask
        data.test_mask = test_mask
        return data

    def __repr__(self) -> str:
        return (f"{self.name.capitalize()}()")


class RUTAttack(RobustnessUnitTest):
    r"""The same dataset but with the adversarial perturbations loaded.
    """
    models = models

    def __init__(
        self,
        root: str,
        name: str,
        split: int,
        scenario: str,
        model: str,
        budget: int,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        pre_filter: Optional[Callable] = None,
    ):
        self.name = name.lower()
        self.scenario = scenario
        self.split = split
        self.model = model
        assert self.name in names
        assert self.split in splits
        assert self.scenario in scenarios
        assert self.model in models
        self.budget = budget
        super().__init__(
            root=root,
            name=name,
            transform=transform,
            pre_transform=pre_transform,
            pre_filter=pre_filter,
        )
    
    def load_data(self):
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def processed_dir(self) -> str:
        return osp.join(self.root, self.name, "processed", "pert", self.model)

    @property
    def processed_file_names(self):
        return f"data_b{self.budget}.pt"
    
    @property
    def budget_prefix(self) -> str:
        prefix = self._get_budget_prefix(self.split, self.model, self.scenario)
        return f"{prefix}{self.budget:05}"

    def process(self):
        with np.load(self.raw_paths[0]) as f:
            data = self._read_data_pert(f)
        data = data if self.pre_transform is None else self.pre_transform(data)
        data, slices = self.collate([data])
        torch.save((data, slices), self.processed_paths[0])

    def _read_data_pert(self, loader):
        data = self._read_data_clean(loader)
        N = data.x.size(0)
        E_clean = data.edge_index.size(1)
        try:
            pert_edges = loader[self.budget_prefix]
        except KeyError:
            raise ValueError(f"budget={self.budget} is not one of the precomputed budgets!")
        pert_edge_index = torch.from_numpy(pert_edges).to(torch.long).T
        pert_edge_index = to_undirected(pert_edge_index, num_nodes=N)
        E_pert = pert_edge_index.size(1)
        assert E_pert == 2 * self.budget
        modified_edge_index = torch.cat((data.edge_index, pert_edge_index), dim=-1)
        modified_edge_weight = torch.ones(E_clean + E_pert)
        modified_edge_index, modified_edge_weight = coalesce(
            modified_edge_index,
            modified_edge_weight,
            num_nodes=N,
            reduce='sum',
        )
        removed_edges_mask = ~(modified_edge_weight == 2)
        modified_edge_index = modified_edge_index[:, removed_edges_mask]
        data.edge_index = modified_edge_index
        return data

    def __repr__(self) -> str:
        return (
            f"Attacked{self.name.capitalize()}("
            f"split={self.split}, scenario={self.scenario}, model={self.model}, budget={self.budget})"
        )
