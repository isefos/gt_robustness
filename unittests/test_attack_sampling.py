import unittest as ut
import torch
from graphgps.attack.sampling import (
    get_connected_sampling_fun,
    WeightedIndexSampler,
    WeightedIndexSampler2,
)
from graphgps.attack.prbcd import PRBCDAttack as P


class TestAttackSampling(ut.TestCase):

    def test_weighted_sampling(self):
        n = 10000
        max_index = 12
        w_idx = [7, 1, 5]
        weighted_idx = torch.tensor(w_idx, dtype=torch.int64)
        z_idx = [8, 9, 3]
        zero_idx = torch.tensor(z_idx, dtype=torch.int64)
        w = 3
        sampler = WeightedIndexSampler(
            weighted_idx=weighted_idx,
            zero_idx=zero_idx,
            weight=w,
            max_index=max_index,
            output_device=torch.device("cpu"),
        )
        s = sampler.sample(n)
        sample_values, sample_frequencies = torch.unique(s, sorted=True, return_counts=True)
        d = {i: 0.0 for i in range(max_index + 1)}
        for v, f in zip(list(sample_values), list(sample_frequencies)):
            d[int(v.item())] = ((f / n) * sampler.n_total).item()
        for v, relative_f in d.items():
            rounded = round(relative_f)
            error = abs(relative_f - rounded)
            if v in w_idx:
                self.assertEqual(rounded, w)
                self.assertTrue(error < 0.25)
            elif v in z_idx:
                self.assertEqual(rounded, 0)
                self.assertEqual(relative_f, 0)
                self.assertEqual(error, 0)
            else:
                self.assertEqual(rounded, 1)
                self.assertTrue(error < 0.25)

    def test_weighted_sampling2(self):
        n = 100000
        max_index = 20
        default_weight = 2
        d_true = {
            0: [8, 17],
            1: [9, 14, 16],
            2: [4, 5, 6, 7, 10, 12, 13, 15],
            3: [0, 1, 2, 3, 18, 19, 20],
        }
        weighted_idx = {
            0: torch.tensor([8, 17], dtype=torch.int64),
            1: torch.tensor([9, 14, 16], dtype=torch.int64),
            3: torch.tensor([0, 1, 2, 3, 17, 18, 19, 20], dtype=torch.int64),
        }
        sampler = WeightedIndexSampler2(
            weighted_idx=weighted_idx,
            default_weight=default_weight,
            max_index=max_index,
            output_device=torch.device("cpu"),
        )
        s = sampler.sample(n)
        sample_values, sample_frequencies = torch.unique(s, sorted=True, return_counts=True)
        d = {i: 0.0 for i in range(max_index + 1)}
        for v, f in zip(list(sample_values), list(sample_frequencies)):
            d[int(v.item())] = ((f / n) * sampler.n_transformed).item()
        for v, relative_f in d.items():
            rounded = round(relative_f)
            error = relative_f - rounded
            for w, w_idx in d_true.items():
                if v in w_idx:
                    assert (rounded == w)
                    assert (abs(error) < 0.25)
                    if w == 0:
                        assert (relative_f == 0)
                        assert (error == 0)

    def test_connected_sampler(self):
        n = 10000
        n_ex_nodes = 3
        n_new_nodes = 4
        n_all_nodes = n_ex_nodes + n_new_nodes
        settings = [
            (True, True),
            (True, False),
            (False, True),
            (False, False),
        ]
        wanted = [
            torch.tensor(
                [[0, 1, 1, 1, 1, 1, 1],
                 [0, 0, 1, 1, 1, 1, 1],
                 [0, 0, 0, 1, 1, 1, 1],
                 [0, 0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0, 0]]
            ),
            torch.tensor(
                [[1, 1, 1, 1, 1, 1, 1],
                 [1, 1, 1, 1, 1, 1, 1],
                 [1, 1, 1, 1, 1, 1, 1],
                 [1, 1, 1, 0, 0, 0, 0],
                 [1, 1, 1, 0, 0, 0, 0],
                 [1, 1, 1, 0, 0, 0, 0],
                 [1, 1, 1, 0, 0, 0, 0]]
            ),
            torch.tensor(
                [[0, 0, 0, 1, 1, 1, 1],
                 [0, 0, 0, 1, 1, 1, 1],
                 [0, 0, 0, 1, 1, 1, 1],
                 [0, 0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0, 0]]
            ),
            torch.tensor(
                [[0, 0, 0, 1, 1, 1, 1],
                 [0, 0, 0, 1, 1, 1, 1],
                 [0, 0, 0, 1, 1, 1, 1],
                 [1, 1, 1, 0, 0, 0, 0],
                 [1, 1, 1, 0, 0, 0, 0],
                 [1, 1, 1, 0, 0, 0, 0],
                 [1, 1, 1, 0, 0, 0, 0]]
            ),
        ]

        for (allow_existing_graph_pert, is_undirected), solution in zip(settings, wanted):
            sampling_fun = get_connected_sampling_fun(
                allow_existing_graph_pert=allow_existing_graph_pert,
                is_undirected=is_undirected,
                n_ex_edges=P._num_possible_edges(n_ex_nodes, is_undirected),
                n_ex_nodes=n_ex_nodes,
                n_new_nodes=n_new_nodes,
                device=torch.device("cpu"),
            )
            sample_idx = sampling_fun(n)
            sample_values, sample_frequencies = torch.unique(sample_idx, sorted=True, return_counts=True)
            if is_undirected:
                idx_2d = P._linear_to_triu_idx(n_all_nodes, sample_values)
            else:
                idx_2d = P._linear_to_full_idx(n_all_nodes, sample_values)
            rel_samples = torch.zeros((n_all_nodes, n_all_nodes))
            rel_samples[idx_2d[0, :], idx_2d[1, :]] = sample_frequencies / n
            rel_samples /= rel_samples[idx_2d[0, :], idx_2d[1, :]].mean()
            rel_samples_rounded = rel_samples.round()
            self.assertTrue(torch.all(rel_samples_rounded == solution))
            error = (rel_samples - rel_samples_rounded).abs()
            self.assertTrue(torch.all(error < 0.25))


if __name__ == '__main__':
    ut.main()
