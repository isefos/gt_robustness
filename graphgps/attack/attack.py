import torch
import logging
import json
from collections import defaultdict
from pathlib import Path
from tensorboardX import SummaryWriter
from torch_geometric.graphgym.config import cfg
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data, Batch
from graphgps.attack.prbcd import PRBCDAttack
from graphgps.attack.prbcd_nia import PRBCDAttackNI
from graphgps.attack.preprocessing import forward_wrapper, remove_isolated_components
from graphgps.attack.dataset_attack import (
    get_total_dataset_graphs,
    get_augmented_graph,
    get_attack_datasets,
    get_local_attack_nodes,
)
from graphgps.attack.postprocessing import (
    get_accumulated_stat_keys,
    accumulate_output_stats,
    accumulate_output_stats_pert,
    get_output_stats,
    log_pert_output_stats,
    basic_edge_and_node_stats,
    zero_budget_edge_and_node_stats,
    log_and_accumulate_num_stats,
    get_summary_stats,
)
from graphgps.attack.nettack import Nettack
import numpy as np
from torch_sparse import SparseTensor


def prbcd_attack_dataset(model, loaders):
    """
    """
    if cfg.dataset.task == "node" and (cfg.attack.node_injection.enable or cfg.attack.remove_isolated_components):
        raise NotImplementedError(
            "Need to handle the node mask (also to calculate attack success rate) "
            "with node injection or pruning away isolated components."
        )
    logging.info("Start of attack:")
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    model.forward = forward_wrapper(model.forward)
    if cfg.attack.node_injection.enable:
        prbcd = PRBCDAttackNI(model)
    else:
        if cfg.attack.local.enable and cfg.attack.local.nettack:
            prbcd = None
        else:
            prbcd = PRBCDAttack(model)
    graph_idx = []
    stat_keys = get_accumulated_stat_keys()
    all_stats = {k: [] for k in sorted(stat_keys)}
    all_stats_zb = None
    if cfg.attack.minimum_budget < 1:
        all_stats_zb = {k: [] for k in sorted(stat_keys)}

    # create tensorboard directory
    tb_logdir = Path(cfg.run_dir) / "tb_attack_stats"
    tb_logdir.mkdir(parents=True)

    # PREPARE DATASETS
    dataset_to_attack, additional_injection_datasets, inject_nodes_from_attack_dataset = get_attack_datasets(loaders)
    # for local attack -> load victim nodes
    if cfg.attack.local.enable:
        perturbations: dict[int, list[list[list[int]]]] = defaultdict(list)
        victim_nodes: dict[int, list[int]] = {}
        local_attack_nodes = get_local_attack_nodes(dataset_to_attack, cfg.attack.local.num_victim_nodes)
        local_node_idx = []
    else:
        local_attack_nodes = None
        local_node_idx = None
        perturbations: dict[int, list[list[int]]] = {}
    total_attack_dataset_graph, attack_dataset_slices, total_additional_datasets_graph = None, None, None
    if cfg.attack.node_injection.enable:
        # TODO: attach a global index to all possible nodes, that can later be used to trace which nodes where added
        # how many times
        total_attack_dataset_graph, attack_dataset_slices, total_additional_datasets_graph = get_total_dataset_graphs(
            inject_nodes_from_attack_dataset=inject_nodes_from_attack_dataset,
            dataset_to_attack=dataset_to_attack,
            additional_injection_datasets=additional_injection_datasets,
            include_root_nodes=cfg.attack.node_injection.include_root_nodes,
        )
    clean_loader = get_attack_loader(dataset_to_attack)
    for i, clean_data in enumerate(clean_loader):
        if cfg.attack.num_attacked_graphs and i >= cfg.attack.num_attacked_graphs:
            break
        clean_data: Batch
        assert clean_data.num_graphs == 1
        if not cfg.attack.local.enable:
            graph_idx.append(i)
            tb_writer = SummaryWriter(tb_logdir / f"graph_{i}")
            attack_or_skip_graph(
                i,
                model,
                prbcd,
                clean_data.get_example(0),
                all_stats,
                all_stats_zb,
                perturbations,
                total_attack_dataset_graph,
                attack_dataset_slices,
                total_additional_datasets_graph,
                tb_writer,
            )
            tb_writer.close()
        else:
            victim_nodes[i] = local_attack_nodes[i]
            for victim_node_idx in local_attack_nodes[i]:
                if cfg.attack.local.nettack:
                    # TODO: init here with everyting it needs
                    #  - implement the interface to match prbcd args and methods
                    _attacker = Nettack(
                        adj = SparseTensor.from_edge_index(
                            edge_index=clean_data.edge_index.to(device=cfg.accelerator),
                            edge_attr=torch.ones((clean_data.edge_index.size(1)), device=cfg.accelerator),
                        ),
                        attr = clean_data.x.to(device=cfg.accelerator),
                        labels = clean_data.y.to(device=cfg.accelerator),
                        idx_attack = np.array([victim_node_idx]),
                        model = model,
                        device = cfg.accelerator,
                        data_device = cfg.accelerator,
                        make_undirected = True,
                        binary_attr = False,
                        loss_type = 'CE',
                    )
                else:
                    _attacker = prbcd
                graph_idx.append(i)
                local_node_idx.append(victim_node_idx)
                tb_writer = SummaryWriter(tb_logdir / f"graph_{i}_node_{victim_node_idx}")
                attack_or_skip_graph(
                    i,
                    model,
                    _attacker,
                    clean_data.get_example(0),
                    all_stats,
                    all_stats_zb,
                    perturbations,
                    total_attack_dataset_graph,
                    attack_dataset_slices,
                    total_additional_datasets_graph,
                    tb_writer,
                    victim_node_idx,
                )
                tb_writer.close()
    model.forward = model.forward.__wrapped__
    for param in model.parameters():
        param.requires_grad = True
    logging.info("End of attack.")

    # save perturbations
    pert_file = Path(cfg.run_dir) / f"perturbations_b{cfg.attack.e_budget}.json"
    with open(pert_file, "w") as f:
        json.dump(perturbations, f)
    if local_attack_nodes:
        victim_node_file = Path(cfg.run_dir) / f"victim_nodes.json"
        with open(victim_node_file, "w") as f:
            json.dump(victim_nodes, f)

    # summarize results
    summary_stats = get_summary_stats(all_stats)
    results = {"avg": summary_stats}
    if not cfg.attack.only_return_avg:
        results["all"] = all_stats

    if all_stats_zb is not None and len(all_stats_zb["budget"]) > len(all_stats["budget"]):
        summary_stats_zb = get_summary_stats(all_stats_zb, zb=True)
        results["avg_including_zero_budget"] = summary_stats_zb
        if not cfg.attack.only_return_avg:
            results["all_including_zero_budget"] = all_stats_zb
    return results


def attack_or_skip_graph(
    i: int,
    model,
    prbcd: PRBCDAttack | Nettack,
    clean_data: Data,
    all_stats: dict[str, list],
    all_stats_zb: None | dict[str, list],
    perturbations: dict[int, list[list[int]]] | dict[int, list[list[list[int]]]],
    total_attack_dataset_graph: Data | None,
    attack_dataset_slices: list[tuple[int, int]] | None,
    total_additional_datasets_graph: Data | None,
    tb_writer: SummaryWriter,
    local_victim_node: None | int = None,
):
    """
    """
    num_nodes = clean_data.x.size(0)
    num_edges = clean_data.edge_index.size(1)
    # currently only global attack (entire split), but could attack specific nodes by using this node mask
    node_mask = clean_data.get(f'{cfg.attack.split}_mask')
    if cfg.attack.local.enable:
        assert local_victim_node is not None
        node_mask = torch.zeros(num_nodes, dtype=torch.bool, device=cfg.accelerator)
        node_mask[local_victim_node] = True
    else:
        if node_mask is not None:
            node_mask = node_mask.to(device=cfg.accelerator)
            assert not cfg.attack.prediction_level == "graph"

    # CHECK CLEAN GRAPH
    output_clean = model(clean_data.clone().to(device=cfg.accelerator), unmodified=True)
    output_clean = apply_node_mask(output_clean, node_mask)
    y_gt = apply_node_mask(clean_data.y.to(device=cfg.accelerator), node_mask)
    output_stats_clean = get_output_stats(y_gt, output_clean)

    # SKIP SCENARIO 1 - INCORRECT GRAPH CLASSIFICATION
    if (
        cfg.attack.prediction_level == "graph"
        and cfg.attack.skip_incorrect_graph_classification
        and not output_stats_clean.get("correct", True)
    ):
        log_incorrect_graph_skip(all_stats, all_stats_zb)
        return

    # BUDGET DEFINITION
    budget_edges, global_budget = get_budget(num_edges)

    # SKIP SCENARIO 2 - NO BUDGET (SMALL GRAPH)
    if global_budget == 0:
        log_budget_skip(
            clean_data.edge_index,
            num_edges,
            num_nodes,
            budget_edges,
            output_stats_clean,
            all_stats_zb,
        )
        return

    # GRAPH WAS NOT SKIPPED - ATTACK
    if local_victim_node is not None:
        logging.info(f"Attacking graph {i} -- node {local_victim_node}")
    else:
        logging.info(f"Attacking graph {i}")
    accumulate_output_stats(all_stats, output_stats_clean, mode="clean", random=False)
    if all_stats_zb is not None:
        accumulate_output_stats(all_stats_zb, output_stats_clean, mode="clean", random=False)

    # AUGMENT GRAPH (ONLY WHEN NODE INJECTION)
    attack_graph_data = get_attack_graph(
        graph_data=clean_data,
        total_attack_dataset_graph=total_attack_dataset_graph,
        attack_dataset_slice=None if attack_dataset_slices is None else attack_dataset_slices[i],
        total_additional_datasets_graph=total_additional_datasets_graph,
    )

    if cfg.attack.only_random_baseline:
        random_flags = [True]
    elif cfg.attack.run_random_baseline:
        random_flags = [False, True]
    else:
        random_flags = [False]

    for is_random_attack in random_flags:
        # ATTACK
        pert_edge_index, perts = attack_single_graph(
            attack_graph_data=attack_graph_data,
            model=model,
            attack=prbcd,
            global_budget=global_budget,
            random_attack=is_random_attack,
            node_mask=node_mask,
            local_victim_node=local_victim_node,
            tb_writer=tb_writer,
            _model_forward_already_wrapped=True,
            _keep_forward_wrapped=True,
        )
        if not is_random_attack:
            if cfg.attack.local.enable:
                perturbations[i].append(perts.tolist())
            else:
                perturbations[i] = perts.tolist()
        log_used_budget(all_stats, all_stats_zb, global_budget, perts, is_random_attack)
        
        # CHECK OUTPUT
        data = Data(x=attack_graph_data.x.clone(), edge_index=pert_edge_index.cpu().clone())
        data, _ = remove_isolated_components(data)
        output_pert = model(data.to(device=cfg.accelerator), unmodified=True)
        output_pert = apply_node_mask(output_pert, node_mask)
        output_stats_pert = get_output_stats(y_gt, output_pert)
        log_pert_output_stats(output_stats_pert, output_stats_clean=output_stats_clean, random=is_random_attack)
        stats, num_stats = basic_edge_and_node_stats(clean_data.edge_index, pert_edge_index, num_edges, num_nodes)
        accumulate_output_stats_pert(all_stats, output_stats_pert, output_stats_clean, is_random_attack)
        log_and_accumulate_num_stats(all_stats, num_stats, random=is_random_attack)
        if all_stats_zb is not None:
            accumulate_output_stats_pert(all_stats_zb, output_stats_pert, output_stats_clean, is_random_attack, True)
            log_and_accumulate_num_stats(all_stats_zb, num_stats, random=is_random_attack, zero_budget=True)


def apply_node_mask(tensor_to_mask, mask):
    if mask is not None:
        return tensor_to_mask[mask]
    return tensor_to_mask


def get_budget(num_edges):
    # TODO: allow for other ways to define the budget
    budget_edges = num_edges // 2 if cfg.attack.is_undirected else num_edges
    global_budget = int(cfg.attack.e_budget * budget_edges)
    if cfg.attack.minimum_budget > global_budget:
        global_budget = cfg.attack.minimum_budget
        logging.info(
            f"Budget smaller than minimum, thus set to minimum: relative budget "
            f"effectively increased from {cfg.attack.e_budget} to "
            f"{cfg.attack.minimum_budget / budget_edges} for this graph."
        )
    return budget_edges, global_budget


def get_attack_graph(
    graph_data: Data,
    total_attack_dataset_graph: None | Data,
    attack_dataset_slice: None | tuple[int, int],
    total_additional_datasets_graph: None | Data,
) -> Data:
    """
    """
    if cfg.attack.node_injection.enable:
        graph_data_augmented = get_augmented_graph(
            graph=graph_data.clone(),
            total_attack_dataset_graph=total_attack_dataset_graph,
            attack_dataset_slice=attack_dataset_slice,
            total_additional_datasets_graph=total_additional_datasets_graph,
        )
        assert graph_data_augmented.edge_index.size(1) == graph_data.edge_index.size(1)
    else:
        graph_data_augmented = graph_data
    return graph_data_augmented


def attack_single_graph(
    attack_graph_data: Data,
    model: torch.nn.Module,
    attack: PRBCDAttack | Nettack,
    global_budget: int,
    random_attack: bool = False,
    node_mask: None | torch.Tensor = None,
    local_victim_node: None | int = None,
    tb_writer: None | SummaryWriter = None,
    _model_forward_already_wrapped: bool = False,
    _keep_forward_wrapped: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    """
    if not _model_forward_already_wrapped:
        model.forward = forward_wrapper(model.forward)

    if cfg.attack.local.enable and cfg.attack.local.nettack:
        assert not random_attack
        assert isinstance(attack, Nettack)

        direct = cfg.attack.local.sampling_direct_edge_weight > 0
        if direct:
            n_influencers = 0
        else:
            assert cfg.attack.is_undirected
            # TODO: make all other nodes influencer nodes
            n_influencers = (attack_graph_data.edge_index[0, :] == local_victim_node).sum().item()
        pert_edge_index, perts = attack.attack(
            n_perturbations=global_budget,
            node_idx=local_victim_node,
            direct=direct,
            n_influencers=n_influencers,
        )
    else:
        assert isinstance(attack, PRBCDAttack)
        attack_fun = attack.attack_random_baseline if random_attack else attack.attack
        pert_edge_index, perts = attack_fun(
            attack_graph_data.x.to(device=cfg.accelerator),
            attack_graph_data.edge_index.to(device=cfg.accelerator),
            attack_graph_data.y.to(device=cfg.accelerator),
            budget=global_budget,
            idx_attack=node_mask,
            local_victim_node=local_victim_node,
            tb_writer=tb_writer,
        )

    if not _keep_forward_wrapped:
        model.forward = model.forward.__wrapped__

    return pert_edge_index, perts


def get_attack_loader(dataset_to_attack):
    pw = cfg.num_workers > 0
    loader = DataLoader(
        dataset_to_attack,
        batch_size=1,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
        persistent_workers=pw
    )
    return loader


def log_incorrect_graph_skip(all_stats, all_stats_zb):
    logging.info("Skipping graph attack because it is already incorrectly classified by model.")
    # set correct_ to False (for the accuracy calculations)
    for k in all_stats:
        if k.startswith("correct"):
            all_stats[k].append(False)
        else:
            all_stats[k].append(None)
    if all_stats_zb is None:
        return
    for k in all_stats_zb:
        if k.startswith("correct"):
            all_stats_zb[k].append(False)
        else:
            all_stats_zb[k].append(None)


def log_budget_skip(edge_index, E, N, budget_edges, output_stats_clean, all_stats_zb):
    assert all_stats_zb is not None
    logging.info(
        f"Skipping graph attack because maximum budget is less than 1 "
        f"({cfg.attack.e_budget} of {budget_edges}), so cannot make perturbations."
    )
    # In this case we only accumulate the stats for the clean graph in the zero budget dict
    all_stats_zb["budget"].append(0)
    for k in ["budget_used", "budget_used_rel"]:
        all_stats_zb[k].append(0)
        all_stats_zb[k + "_random"].append(0)
    accumulate_output_stats(all_stats_zb, output_stats_clean, mode="clean", random=False)
    for random in [False, True]:
        accumulate_output_stats_pert(all_stats_zb, output_stats_clean, output_stats_clean, random, True)
    _, num_stats = zero_budget_edge_and_node_stats(edge_index, E, N)
    for random in [False, True]:
        log_and_accumulate_num_stats(all_stats_zb, num_stats, random, zero_budget=True)


def log_used_budget(all_stats, all_stats_zb, global_budget, perts, is_random_attack):
    all_stats["budget"].append(global_budget)
    if all_stats_zb is not None:
        all_stats_zb["budget"].append(global_budget)
    E_mod = perts.size(1)
    b_rel = E_mod / global_budget
    for key, value in zip(["budget_used", "budget_used_rel"], [E_mod, b_rel]):
        if is_random_attack:
            key += "_random"
        all_stats[key].append(value)
        if all_stats_zb is not None:
            all_stats_zb[key].append(value)
    m = "Random perturbation" if is_random_attack else "Perturbation"
    logging.info(f"{m} uses {100 * b_rel:.1f}% [{E_mod}/{global_budget}] of the given attack budget.")


def check_augmentation_correctness(model, loaders):
    """
    The attacked model's ooutput should be the same for the clean graph and the augmented one.
    Augmentation adds disconected new nodes, should be pruned away.
    Test was more important in previous code versions, when the augmentation would also permute the existing nodes.
    Now the extra disconnected nodes are always appended after the existing ones, so as long as the connected component
    pruning is used and works correctly, this 'test' should pass (does not require model node permutation invariance
    anymore).
    """
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
    for i, clean_data in enumerate(clean_loader):
        node_mask = clean_data.get(f'{cfg.attack.split}_mask')
        if node_mask is not None:
            node_mask = node_mask.to(device=cfg.accelerator)
            assert not cfg.attack.prediction_level == "graph"
        # check the prediction of the clean graph
        output_clean = model(clean_data.clone().to(device=cfg.accelerator), unmodified=True)
        output_clean = apply_node_mask(output_clean, node_mask)
        # get the graph to attack (potentially augmented)
        attack_graph_data = get_attack_graph(
            graph_data=clean_data.get_example(0),
            total_attack_dataset_graph=total_attack_dataset_graph,
            attack_dataset_slice=None if attack_dataset_slices is None else attack_dataset_slices[i],
            total_additional_datasets_graph=total_additional_datasets_graph,
        )
        attack_graph_data.edge_attr = torch.ones(attack_graph_data.edge_index.size(1))
        data = Batch.from_data_list([attack_graph_data.clone()])
        augmented_output = model(data)
        augmented_output = apply_node_mask(augmented_output, node_mask)
        assert torch.allclose(output_clean, augmented_output, atol=0.001, rtol=0.001)
