import torch
import numpy as np
import logging
from torch_geometric.utils import to_scipy_sparse_matrix, index_to_mask, subgraph
from scipy.sparse.csgraph import breadth_first_order, connected_components
from torch_geometric.graphgym.config import cfg
import torch.nn.functional as F
from itertools import compress


def get_accumulated_stat_keys(with_random: None | bool = None):
    if with_random is None:
        with_random = cfg.attack.run_random_baseline
    keys = [
        "budget_used",
        "budget_used_rel",
        "num_edges_added",
        "num_edges_removed",
        "num_nodes_added",
        "num_nodes_removed",
    ]
    if cfg.attack.remove_isolated_components:
        keys.extend([
            "num_edges_added_connected",
            "num_nodes_added_connected",
        ])
    if with_random:
        keys.extend([k + "_random" for k in keys])

    keys.extend(["budget", "num_edges_clean", "num_nodes_clean"])

    if cfg.dataset.task_type.startswith("classification"):
        asr = "attack_success_rate"
        asr_keys = [asr]
        if with_random:
            asr_keys.append(asr + "_random")
        keys.extend(asr_keys)

        if cfg.attack.prediction_level == "node":
            key_prefixes = [
                "correct_acc", "margin_mean", "margin_median", "margin_min", "margin_max",
                "margin_correct_mean", "margin_correct_median", "margin_correct_min", "margin_correct_max",
                "margin_wrong_mean", "margin_wrong_median", "margin_wrong_min", "margin_wrong_max",
            ]
        else:
            key_prefixes = ["correct", "margin", "margin_correct", "margin_wrong"]
        
        keys.extend([k + "_clean" for k in key_prefixes])
        keys.extend([k + "_pert" for k in key_prefixes])
        if with_random:
            keys.extend([k + "_pert_random" for k in key_prefixes])

    else:
        raise NotImplementedError
    
    return keys


def get_output_stats(y_gt, model_output):
    if cfg.attack.prediction_level == "node":
        return get_node_output_stats(y_gt, model_output)
    
    elif cfg.attack.prediction_level == "graph":
        return get_graph_output_stats(y_gt, model_output)
    
    else:
        raise NotImplementedError


def get_node_output_stats(y_gt, logits):

    if cfg.dataset.task_type.startswith("classification"):

        if cfg.dataset.task_type == "classification_binary":
            # logits (N, 1)
            prob_binary = torch.sigmoid(logits)
            class_idx_pred = (prob_binary[:, 0] > cfg.model.thresh).to(dtype=torch.long)
            probs = torch.cat([1 - prob_binary, prob_binary], dim=1)
        
        elif cfg.dataset.task_type == "classification":
            class_idx_pred = logits.argmax(dim=1)
            probs = logits.softmax(dim=1)

        num_classes = probs.size(1)
        y_correct_mask = F.one_hot(y_gt, num_classes).to(dtype=torch.bool)
        assert probs.shape == y_correct_mask.shape
        margin = probs[y_correct_mask] - probs[~y_correct_mask].reshape(-1, num_classes-1).max(dim=1)[0]
        correct: torch.Tensor = class_idx_pred == y_gt
        acc = correct.float().mean().item()
        margin_correct = None
        if correct.any():
            margin_correct = margin[correct]
        margin_wrong = None
        if not correct.all():
            margin_wrong = margin[~correct]
        output_stats = {
            "probs": probs.tolist(),
            "logits": logits.tolist(),
            "correct": correct.tolist(),
            "correct_acc": acc,
            "margin": margin.tolist(),
            "margin_mean": margin.mean().item(),
            "margin_median": margin.median().item(),
            "margin_min": margin.min().item(),
            "margin_max": margin.max().item(),
            "margin_correct_mean": None if margin_correct is None else margin_correct.mean().item(),
            "margin_correct_median": None if margin_correct is None else margin_correct.median().item(),
            "margin_correct_min": None if margin_correct is None else margin_correct.min().item(),
            "margin_correct_max": None if margin_correct is None else margin_correct.max().item(),
            "margin_wrong_mean": None if margin_wrong is None else margin_wrong.mean().item(),
            "margin_wrong_median": None if margin_wrong is None else margin_wrong.median().item(),
            "margin_wrong_min": None if margin_wrong is None else margin_wrong.min().item(),
            "margin_wrong_max": None if margin_wrong is None else margin_wrong.max().item(),
        }
        return output_stats

    else:
        raise NotImplementedError


def get_graph_output_stats(y_gt, model_output):

    if cfg.dataset.task_type.startswith("classification"):
        logits = model_output[0, :]
        y_gt = y_gt[0]
        class_index_gt = int(y_gt.item())
        class_index_pred: int
        probs: torch.Tensor

        if cfg.dataset.task_type == "classification_binary":
            prob_binary = torch.sigmoid(logits)
            class_index_pred = int(prob_binary > cfg.model.thresh)
            probs = torch.cat([1 - prob_binary, prob_binary], dim=0)
        
        elif cfg.dataset.task_type == "classification":
            class_index_pred = int(logits.argmax().item())
            probs = logits.softmax(dim=0)

        num_classes = probs.size(0)
        y_correct_mask = F.one_hot(y_gt, num_classes).flatten().to(dtype=torch.bool)
        margin = float((probs[y_correct_mask] - probs[~y_correct_mask].max()).item())
        correct = class_index_pred == class_index_gt
        output_stats = {
            "probs": probs.tolist(),
            "logits": logits.tolist(),
            "correct": correct,
            "margin": margin,
            "margin_correct": margin if correct else None,
            "margin_wrong": margin if not correct else None,
        }
        return output_stats

    else:
        raise NotImplementedError


def log_pert_output_stats(
    output_stats_pert,
    output_stats_clean,
    random=False,
):
    if cfg.attack.prediction_level == "graph":
        if cfg.dataset.task_type.startswith("classification"):
            log_graph_classification_output_stats(output_stats_pert, output_stats_clean, random)
        else:
            raise NotImplementedError

    elif cfg.attack.prediction_level == "node":
        if cfg.dataset.task_type.startswith("classification"):
            log_node_classification_output_stats(output_stats_pert, output_stats_clean, random)
        else:
            raise NotImplementedError
    
    else:
        raise NotImplementedError


def accumulate_output_stats_pert(
    accumulated_stats,
    output_stats_pert,
    output_stats_clean,
    random=False,
    no_asr_log=False,
):
    accumulate_output_stats(
        accumulated_stats,
        output_stats=output_stats_pert,
        mode="pert",
        random=random,
    )
    if cfg.dataset.task_type.startswith("classification"):
        asr_key = "attack_success_rate"
        if random:
            asr_key += "_random"
        if cfg.attack.prediction_level == "graph":
            asr = None
            if output_stats_clean["correct"]:
                asr = 0 if output_stats_pert["correct"] else 1
            accumulated_stats[asr_key].append(asr)
        if cfg.attack.prediction_level == "node":
            pert_c = list(compress(output_stats_pert["correct"], output_stats_clean["correct"]))
            asr = None
            if pert_c:
                asr = 1 - (sum(pert_c) / len(pert_c))
            accumulated_stats[asr_key].append(asr)
        if not no_asr_log and asr is not None:
            logging.info(f"Attack success rate: {asr:.3f}")
    

def accumulate_output_stats(
    accumulated_stats,
    output_stats,
    mode: str,
    random: bool,
):
    for key, stat in output_stats.items():
        k = key + "_" + mode
        if random:
            k = k + "_random"
        if k in accumulated_stats:
            accumulated_stats[k].append(stat)


def log_graph_classification_output_stats(
    output_stats_pert,
    output_stats_clean,
    random=False,
):
    for name, output_stats in zip(
        ["clean", "pert_rand" if random else "pert"],
        [output_stats_clean, output_stats_pert]
    ):
        correct_str = f"{str(output_stats['correct']):5}"
        margin = output_stats['margin']
        margin_str = f"{f'{margin:.4f}':>7}"
        prob_str = ", ".join((f"{p:.3f}" for p in output_stats['probs']))
        logit_str = ", ".join([f"{l:.3f}" for l in output_stats['logits']])
        logging.info(
            f"{name + ':':<10}   correct  (margin) [probs] <logits>:   "
            f"{correct_str} ({margin_str}) [{prob_str}] <{logit_str}>"
        )


def log_node_classification_output_stats(
    output_stats_pert,
    output_stats_clean,
    random=False,
):
    for name, output_stats in zip(
        ["clean", "pert_rand" if random else "pert"],
        [output_stats_clean, output_stats_pert]
    ):
        acc = f"{output_stats['correct_acc']:.4f}"
        margin_mean = f"{output_stats['margin_mean']:.4f}"
        margin_median = f"{output_stats['margin_median']:.4f}"
        margin_min = f"{output_stats['margin_min']:.4f}"
        margin_max = f"{output_stats['margin_max']:.4f}"
        logging.info(
            f"{name + ':':<10}   acc  (margin_mean) [margin_median] <margin_min> |margin_max|:   "
            f"{acc} ({margin_mean}) [{margin_median}] <{margin_min}> |{margin_max}|"
        )


def _get_edges_nodes_from_index(edge_index):
    edges = set()
    nodes = set()
    for edge in torch.split(edge_index, 1, dim=1):
        edge = tuple(int(n) for n in edge.squeeze().tolist())
        edges.add(edge)
        nodes.update(edge)
    return edges, nodes


def _get_edge_index_connected(edge_index):
    num_nodes = edge_index.max().item() + 1
    root_node = cfg.attack.root_node_idx
    adj = to_scipy_sparse_matrix(edge_index, num_nodes=num_nodes)
    if root_node is not None:
        bfs_order = breadth_first_order(adj, root_node, return_predecessors=False)
        subset_mask = index_to_mask(
            torch.tensor(bfs_order, dtype=torch.long, device=edge_index.device),
            size=num_nodes,
        )
    else:
        _, component = connected_components(adj, connection="weak")
        _, count = np.unique(component, return_counts=True)
        subset_np = np.in1d(component, count.argsort()[-1:])
        subset_mask = torch.from_numpy(subset_np)
        subset_mask = subset_mask.to(edge_index.device, torch.bool)
    edge_index_connected = subgraph(subset_mask, edge_index)[0]
    return edge_index_connected


def _get_stats_and_num_stats(
    edges, edges_pert, edges_pert_connected, edges_added, edges_added_connected, edges_removed,
    nodes, nodes_pert, nodes_pert_connected, nodes_added, nodes_added_connected, nodes_removed,
):
    stats = {
        "edges": {
            "clean": edges,
            "pert": edges_pert,
            "added": edges_added,
            "removed": edges_removed,
        },
        "nodes": {
            "clean": nodes,
            "pert": nodes_pert,
            "added": nodes_added,
            "removed": nodes_removed,
        },
    }
    if cfg.attack.remove_isolated_components:
        # add the connected stats (should be set to None in the other case anyway)
        connected_entries = [
            ("edges", "pert_connected", edges_pert_connected),
            ("edges", "added_connected", edges_added_connected),
            ("nodes", "pert_connected", nodes_pert_connected),
            ("nodes", "added_connected", nodes_added_connected),
        ]
        for (key1, key2, value) in connected_entries:
            if value is not None:
                stats[key1][key2] = value
    
    num_stats = {}
    for key1 in stats:
        num_key = "num_" + key1
        num_stats[num_key] = {}
        for key2, value in stats[key1].items():
            v = len(value)
            if cfg.attack.is_undirected and key1 == "edges":
                v = v // 2
            num_stats[num_key][key2] = v
    return stats, num_stats


def zero_budget_edge_and_node_stats(
    edge_index: torch.Tensor,
    num_edges: int,
    num_nodes: int,
) -> tuple[dict[str, dict[str, set[tuple[int, int]]]], dict[str, dict[str, int]]]:    
    edges, nodes = _get_edges_nodes_from_index(edge_index)
    if cfg.attack.remove_isolated_components:
        edge_index_pert_connected = _get_edge_index_connected(edge_index)
        edges_pert_connected, nodes_pert_connected = _get_edges_nodes_from_index(edge_index_pert_connected)
        edges_added_connected = set()
        nodes_added_connected = set()
    else:
        edges_pert_connected = None
        nodes_pert_connected = None
        edges_added_connected = None
        nodes_added_connected = None
    stats, num_stats = _get_stats_and_num_stats(
        edges, edges, edges_pert_connected, set(), edges_added_connected, set(),
        nodes, nodes, nodes_pert_connected, set(), nodes_added_connected, set(),
    )
    num_edges = num_edges // 2 if cfg.attack.is_undirected else num_edges
    assert num_edges == num_stats["num_edges"]["clean"]
    assert num_nodes == num_stats["num_nodes"]["clean"]
    return stats, num_stats


def basic_edge_and_node_stats(
    edge_index: torch.Tensor,
    edge_index_pert: torch.Tensor,
    num_edges_clean: int,
    num_nodes_clean: int,
) -> tuple[dict[str, dict[str, set[tuple[int, int]]]], dict[str, dict[str, int]]]:
    edges, nodes = _get_edges_nodes_from_index(edge_index)
    edges_pert, nodes_pert = _get_edges_nodes_from_index(edge_index_pert)

    edges_added = edges_pert - edges
    edges_removed = edges - edges_pert

    nodes_added = nodes_pert - nodes
    nodes_removed = nodes - nodes_pert

    if cfg.attack.remove_isolated_components:
        edge_index_pert_connected = _get_edge_index_connected(edge_index_pert)
        edges_pert_connected, nodes_pert_connected = _get_edges_nodes_from_index(edge_index_pert_connected)
        edges_added_connected = edges_pert_connected - edges
        nodes_added_connected = nodes_pert_connected - nodes
    else:
        edges_pert_connected = None
        nodes_pert_connected = None
        edges_added_connected = None
        nodes_added_connected = None

    stats, num_stats = _get_stats_and_num_stats(
        edges, edges_pert, edges_pert_connected, edges_added, edges_added_connected, edges_removed,
        nodes, nodes_pert, nodes_pert_connected, nodes_added, nodes_added_connected, nodes_removed,
    )

    num_edges_clean = num_edges_clean // 2 if cfg.attack.is_undirected else num_edges_clean

    assert num_edges_clean == num_stats["num_edges"]["clean"]
    assert num_nodes_clean == num_stats["num_nodes"]["clean"]

    return stats, num_stats


def log_and_accumulate_num_stats(accumulated_stats, num_stats, random=False, zero_budget=False):
    stat_keys = [
        ("num_edges", "clean"), ("num_edges", "added"), ("num_edges", "removed"),
        ("num_nodes", "clean"), ("num_nodes", "added"), ("num_nodes", "removed"),
    ]
    info_texts = [
        "Original number of edges", "Added edges", "Removed edges",
        "Original number of nodes", "Added nodes", "Removed nodes"
    ]
    if cfg.attack.remove_isolated_components:
        stat_keys.extend([("num_edges", "added_connected"), ("num_nodes", "added_connected")])
        info_texts.extend(["Added edges (connected)", "Added nodes (connected)"])

    for stat_key, info_text in zip(stat_keys, info_texts):
        current_stat = num_stats[stat_key[0]][stat_key[1]]
        if not zero_budget:
            logging.info(f"{info_text + ':':<26} {current_stat:>7}")
        acc_key = stat_key[0] + "_" + stat_key[1]
        if random:
            if stat_key[1] == "clean":
                continue
            acc_key += "_random"
        accumulated_stats[acc_key].append(current_stat)


def get_summary_stats(accumulated_stats, zb=False, log=True):
    summary_stats = {}
    if log:
        logging.info(
            f"Attack stats summary (averages over attacked graphs{'' if not zb else 'including zero budget'}):"
        )
    for key, current_stat in accumulated_stats.items():
        name = "avg_" + key
        filtered_stat = [s for s in current_stat if s is not None]
        if filtered_stat:
            try:
                avg = sum(filtered_stat) / len(filtered_stat)
            except TypeError:
                continue
            if log:
                logging.info(f"  {name + ':':<48} {f'{avg:.2f}':>8}")
        else:
            avg = None
            if log:
                logging.info(f"  {name + ':':<48} {'nan':>8}")
        summary_stats[name] = avg
    return summary_stats
