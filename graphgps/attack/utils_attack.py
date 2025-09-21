import torch


# Not used anymore, may be useful for quick debugging:


def get_reached_nodes(root: int, edge_index: torch.Tensor) -> set[int]:
    children_nodes = get_undirected_graph_node_children(edge_index)
    # DFS to find all reachable nodes:
    to_explore = [root]
    reached: set[int] = set((root, ))
    while to_explore:
        current_node = to_explore.pop()
        children = children_nodes[current_node]
        for child_node in children:
            if child_node in reached:
                continue
            to_explore.append(child_node)
        reached |= children
    return reached


def get_undirected_graph_node_children(edge_index: torch.Tensor) -> dict[int, set[int]]:
    children_nodes: dict[int, set[int]] = {}
    for i in range(edge_index.size(1)):
        edge_node_1 = int(edge_index[0, i])
        edge_node_2 = int(edge_index[1, i])
        if edge_node_1 not in children_nodes:
            children_nodes[edge_node_1] = set()
        if edge_node_2 not in children_nodes:
            children_nodes[edge_node_2] = set()
        # TODO: if edge weight is nonzero?
        children_nodes[edge_node_1].add(edge_node_2)
        # TODO: if undirected and edge weight is nonzero?
        children_nodes[edge_node_2].add(edge_node_1)
    return children_nodes


def check_if_tree(edge_index: torch.Tensor) -> bool:
    root = int(edge_index[0, 0])
    children_nodes = get_undirected_graph_node_children(edge_index)
    # DFS check wheather a child has already been visited from different parent
    to_explore: list[tuple[int, None | int]] = [(root, None)]
    reached: set[int] = set((root, ))
    is_tree = True
    while to_explore:
        current_node, parent_node = to_explore.pop()
        children = children_nodes[current_node]
        for child_node in children:
            if child_node == parent_node:
                continue
            if child_node in reached:
                is_tree = False
                break
            to_explore.append((child_node, current_node))
        if not is_tree:
            break
        reached |= children
    return is_tree
