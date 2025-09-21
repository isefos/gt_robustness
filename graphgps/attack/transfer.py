import torch
import logging
import json
from pathlib import Path
from yacs.config import CfgNode as CN
from torch_geometric.graphgym.config import cfg
from torch_geometric.data import Data, Batch
from torch_geometric.utils import to_undirected, coalesce
from graphgps.attack.preprocessing import forward_wrapper, remove_isolated_components
from graphgps.attack.dataset_attack import (
    get_total_dataset_graphs,
    get_attack_datasets,
)
from graphgps.attack.postprocessing import (
    get_accumulated_stat_keys,
    accumulate_output_stats,
    accumulate_output_stats_pert,
    get_output_stats,
    log_pert_output_stats,
    basic_edge_and_node_stats,
    log_and_accumulate_num_stats,
    get_summary_stats,
)
from graphgps.attack.attack import (
    apply_node_mask,
    get_attack_loader,
    get_attack_graph,
)


def transfer_attack_dataset(model, loaders, perturbation_path):
    """
    """
    perturbation_path = Path(perturbation_path)
    with open(perturbation_path / "attack_configs.yaml", "r") as f:
        attack_configs = CN.load_cfg(f)
    check_equal_configs(attack_configs, cfg)

    # extract perturbations from perturbation path
    perturbations, seeds, budgets, victim_nodes = [], [], [], None
    for child in perturbation_path.iterdir():
        if not child.is_dir():
            continue
        seed = int(child.name[1:])
        for pert_file in child.iterdir():
            try:
                budget = float(pert_file.name[15:-5])
                with open(pert_file, "r") as f:
                    perturbations.append(json.load(f))
                seeds.append(seed)
                budgets.append(budget)
            except ValueError:
                if not pert_file.name == "victim_nodes.json":
                    raise
                if not attack_configs.attack.local.enable:
                    raise Exception("found victim_nodes.json, but cfg is not local")
                if victim_nodes is None:
                    victim_nodes = dict()
                with open(pert_file, "r") as f:
                    victim_nodes[seed] = json.load(f)
    num_perturbations = len(perturbations)
     
    if cfg.dataset.task == "node" and (cfg.attack.node_injection.enable or cfg.attack.remove_isolated_components):
        raise NotImplementedError(
            "Need to handle the node mask (also to calculate attack success rate) "
            "with node injection or pruning away isolated components."
        )
    logging.info("Start of transfer attack:")
    model.eval()
    model.forward = forward_wrapper(model.forward)
    stat_keys = get_accumulated_stat_keys(with_random=False)
    all_stats = [{k: [] for k in sorted(stat_keys)} for _ in range(num_perturbations)]

    # PREPARE DATASETS
    dataset_to_attack, additional_injection_datasets, inject_nodes_from_attack_dataset = get_attack_datasets(loaders)
    total_attack_dataset_graph, attack_dataset_slices, total_additional_datasets_graph = None, None, None
    if cfg.attack.node_injection.enable:
        total_attack_dataset_graph, attack_dataset_slices, total_additional_datasets_graph = get_total_dataset_graphs(
            inject_nodes_from_attack_dataset=inject_nodes_from_attack_dataset,
            dataset_to_attack=dataset_to_attack,
            additional_injection_datasets=additional_injection_datasets,
            include_root_nodes=cfg.attack.node_injection.include_root_nodes,
        )
    clean_loader = get_attack_loader(dataset_to_attack)

    test_num_edges = []
    test_num_neighbors = []

    for i, clean_data in enumerate(clean_loader):
        clean_data: Batch
        clean_graph = clean_data.get_example(0)

        seen_seeds = set()
        seen_graph = False

        assert clean_data.num_graphs == 1
        for p, seed, stats in zip(perturbations, seeds, all_stats):
            perturbation = p.get(str(i), None)
            if perturbation is None:
                continue

            # for temporary analysis, do not commit, delete/ revert later!!!

            if not seen_graph:
                seen_graph = True
                test_num_edges.append(clean_graph.edge_index.size(1) // 2)

            if seed not in seen_seeds:
                seen_seeds.add(seed)
                for node_idx in victim_nodes[seed][str(i)]:
                    # TODO: find neighborhood size
                    n = (clean_graph.edge_index[1, :] == node_idx).sum()
                    test_num_neighbors.append(n)

            continue


            transfer_attack_graph(
                i,
                model,
                clean_data.get_example(0),
                stats,
                perturbation,
                total_attack_dataset_graph,
                attack_dataset_slices,
                total_additional_datasets_graph,
                victim_nodes[seed],
            )
    model.forward = model.forward.__wrapped__
    logging.info("End of transfer attack.")

    # summarize results
    summary_stats = []
    for stats in all_stats:
        summary_stats.append(get_summary_stats(stats, log=False))
    results = {"avg": summary_stats}
    if not cfg.attack.only_return_avg:
        results["all"] = all_stats
    results["seeds"] = seeds
    results["budgets"] = budgets
    return results


def check_equal_configs(configs_given: CN, currect_configs: CN) -> None:
    for k, value_given in configs_given.items():
        if isinstance(value_given, CN):
            check_equal_configs(value_given, currect_configs[k])
        else:
            value = currect_configs[k]
            if k == "name" and configs_given.get("format", "") == "PyG-UPFD":
                # for UPFD only the dataset politifact / gossipcop has to match, the features don't matter
                value = value.split("-")[0]
                value_given = value_given.split("-")[0]
            assert value == value_given, "The transfer attack configs are not compatible with the current ones."


def transfer_attack_graph(
    i: int,
    model,
    clean_data: Data,
    stats: dict[str, list],
    perturbation: list[list[int]] | list[list[list[int]]],
    total_attack_dataset_graph: Data | None,
    attack_dataset_slices: list[tuple[int, int]] | None,
    total_additional_datasets_graph: Data | None,
    victim_nodes: None | dict[str, list[int]],
):
    """
    """
    logging.info(f"Attacking graph {i}")
    num_nodes = clean_data.x.size(0)
    num_edges = clean_data.edge_index.size(1)
    node_mask = clean_data.get(f'{cfg.attack.split}_mask')
    if node_mask is not None:
        node_mask = node_mask.to(device=cfg.accelerator)
        assert not cfg.attack.prediction_level == "graph"

    # CHECK CLEAN GRAPH
    output_clean = model(clean_data.clone().to(device=cfg.accelerator), unmodified=True)

    # AUGMENT GRAPH (ONLY WHEN NODE INJECTION)
    attack_graph_data = get_attack_graph(
        graph_data=clean_data,
        total_attack_dataset_graph=total_attack_dataset_graph,
        attack_dataset_slice=None if attack_dataset_slices is None else attack_dataset_slices[i],
        total_additional_datasets_graph=total_additional_datasets_graph,
    )

    local_nodes = [None] if victim_nodes is None else victim_nodes[str(i)]

    # APPLY TRANSFER ATTACK

    for j, local_node in enumerate(local_nodes):
        if local_node is not None:
            p = perturbation[j]
            node_mask = torch.zeros(num_nodes, dtype=torch.bool, device=cfg.accelerator)
            node_mask[local_node] = True
        else:
            p = perturbation
            node_mask = None

        # CLEAN
        output_clean_masked = apply_node_mask(output_clean, node_mask)
        y_gt = apply_node_mask(clean_data.y.to(device=cfg.accelerator), node_mask)
        output_stats_clean = get_output_stats(y_gt, output_clean_masked)
        accumulate_output_stats(stats, output_stats_clean, mode="clean", random=False)

        # PERT
        pert_edge_index = get_perturbed_edge_index(attack_graph_data, p)
        data = Data(x=attack_graph_data.x.clone(), edge_index=pert_edge_index.clone())
        data, _ = remove_isolated_components(data)
        output_pert = model(data.to(device=cfg.accelerator), unmodified=True)
        output_pert = apply_node_mask(output_pert, node_mask)
        output_stats_pert = get_output_stats(y_gt, output_pert)
        log_pert_output_stats(output_stats_pert, output_stats_clean=output_stats_clean, random=False)
        _, num_stats = basic_edge_and_node_stats(clean_data.edge_index, pert_edge_index, num_edges, num_nodes)
        accumulate_output_stats_pert(stats, output_stats_pert, output_stats_clean, False)
        log_and_accumulate_num_stats(stats, num_stats, random=False)


def get_perturbed_edge_index(data: Data, perturbation):
    N = data.x.size(0)
    E_clean = data.edge_index.size(1)
    pert_edge_index = torch.tensor(perturbation, dtype=torch.long, device=data.x.device)
    if cfg.attack.is_undirected:
        pert_edge_index = to_undirected(pert_edge_index, num_nodes=N)
    E_pert = pert_edge_index.size(1)
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
    return modified_edge_index
