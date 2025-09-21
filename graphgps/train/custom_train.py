import logging
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.graphgym.checkpoint import load_ckpt, save_ckpt, clean_ckpt
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.loss import compute_loss
from torch_geometric.graphgym.register import register_train
from torch_geometric.graphgym.utils.epoch import is_eval_epoch, is_ckpt_epoch
from torch_geometric.utils import degree

from graphgps.loss.subtoken_prediction_loss import subtoken_cross_entropy
from graphgps.utils import cfg_to_dict, flatten_dict, make_wandb_name


def train_epoch(logger, loader, model, optimizer, scheduler, batch_accumulation):
    model.train()
    optimizer.zero_grad()
    time_start = time.time()
    for iter, batch in enumerate(loader):
        #batch.split = 'train' -> commented to make homophily_regularization possible
        batch.to(torch.device(cfg.accelerator))
        if cfg.train.homophily_regularization > 0:
            original_edge_index = batch.edge_index.clone()
        pred, true = model(batch)

        if cfg.gnn.head == "node" and batch.get("train_mask") is not None:
            pred_full, true_full = pred, true
            mask = batch.train_mask
            pred, true = pred_full[mask], true_full[mask] if true_full is not None else None
        else:
            pred_full, true_full = None, None

        if cfg.dataset.name == 'ogbg-code2':
            loss, pred_score = subtoken_cross_entropy(pred, true)
            _true = true
            _pred = pred_score
        else:
            loss, pred_score = compute_loss(pred, true)
            _true = true.detach().to('cpu', non_blocking=True)
            _pred = pred_score.detach().to('cpu', non_blocking=True)
        
        # add homophily regularization if specified
        if pred_full is not None and cfg.train.homophily_regularization > 0:
            reg = homophily_regularization(
                pred_full, true_full, original_edge_index, batch.train_mask, batch.x.size(0),
            )
            loss += cfg.train.homophily_regularization * reg
        
        loss.backward()
        # Parameters update after accumulating gradients for given num. batches.
        if ((iter + 1) % batch_accumulation == 0) or (iter + 1 == len(loader)):
            if cfg.optim.clip_grad_norm:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.optim.clip_grad_norm_value)
            optimizer.step()
            optimizer.zero_grad()
        logger.update_stats(
            true=_true,
            pred=_pred,
            loss=loss.detach().cpu().item(),
            lr=scheduler.get_last_lr()[0],
            time_used=time.time() - time_start,
            params=cfg.params,
            dataset_name=cfg.dataset.name,
        )
        time_start = time.time()


def homophily_regularization(pred_full, true_full, edge_index, train_mask, N):
    pred_full = pred_full.squeeze(-1) if pred_full.ndim > 1 else pred_full
    true_full = true_full.squeeze(-1) if true_full.ndim > 1 else true_full
    multiclass = pred_full.ndim > 1 and true_full.ndim == 1
    # pseudo labels from the predictions of neighbors
    if multiclass:
        true_hom = pred_full.detach().max(dim=1)[1]
    else:
        true_hom = (torch.sigmoid(pred_full.detach()) > cfg.model.thresh).long()
    true_hom = true_hom[edge_index[0, :]]
    # set the actual training labels that we know
    label_mask = train_mask[edge_index[0, :]]
    label_hom = true_full[edge_index[0, :]]
    true_hom[label_mask] = label_hom[label_mask]

    if multiclass:
        reg = F.nll_loss(F.log_softmax(pred_full[edge_index[1, :]], dim=-1), true_hom, reduction="none")
    else:
        reg = F.binary_cross_entropy_with_logits(pred_full[edge_index[1, :]], true_hom.float(), reduction="none")

    # weight by inverse degree, such that all nodes get equal regularization
    deg = degree(edge_index[1, :], N)
    reg /= deg[edge_index[1, :]]
    # weight those higher where the label is a known label
    w = torch.ones_like(label_mask).float()
    w[label_mask] = cfg.train.homophily_regularization_gt_weight
    reg = reg * w
    # remove the training nodes from the regularization loss
    not_train_mask = (~train_mask)[edge_index[1, :]]
    reg = reg[not_train_mask]
    return reg.mean()


@torch.no_grad()
def eval_epoch(logger, loader, model, split='val'):
    model.eval()
    time_start = time.time()
    for batch in loader:
        batch.split = split
        batch.to(torch.device(cfg.accelerator))
        if cfg.gnn.head == 'inductive_edge':
            pred, true, extra_stats = model(batch)
        else:
            pred, true = model(batch)
            extra_stats = {}
        if cfg.dataset.name == 'ogbg-code2':
            loss, pred_score = subtoken_cross_entropy(pred, true)
            _true = true
            _pred = pred_score
        else:
            loss, pred_score = compute_loss(pred, true)
            _true = true.detach().to('cpu', non_blocking=True)
            _pred = pred_score.detach().to('cpu', non_blocking=True)
        logger.update_stats(
            true=_true,
            pred=_pred,
            loss=loss.detach().cpu().item(),
            lr=0, time_used=time.time() - time_start,
            params=cfg.params,
            dataset_name=cfg.dataset.name,
            **extra_stats,
        )
        time_start = time.time()


@register_train('custom')
def custom_train(loggers, loaders, model, optimizer, scheduler):
    """
    Customized training pipeline.

    Args:
        loggers: List of loggers
        loaders: List of loaders
        model: GNN model
        optimizer: PyTorch optimizer
        scheduler: PyTorch learning rate scheduler

    """
    start_epoch = 0
    max_epoch = cfg.optim.max_epoch
    if cfg.train.auto_resume:
        start_epoch = load_ckpt(model, optimizer, scheduler, cfg.train.epoch_resume)
    if start_epoch == max_epoch:
        logging.info('Checkpoint found, Task already done')
    else:
        logging.info('Start from epoch %s', start_epoch)

    early_stopping = cfg.optim.early_stopping
    patience = cfg.optim.early_stopping_patience
    best_val_loss = None
    patience_m = 1 + cfg.optim.early_stopping_delta_e
    patience_warmup = start_epoch + cfg.optim.early_stopping_warmup

    # for polynormer with local pre-training:
    if cfg.optim.num_local_epochs > 0:
        assert max_epoch > cfg.optim.num_local_epochs, (
            f"total number of epochs ({max_epoch}) must be larger than "
            f"the number of local pre-train epochs ({cfg.optim.num_local_epochs})"
        )
        logging.info('local-only pre-training for %s epochs', cfg.optim.num_local_epochs)
        logging.info('global training for %s epochs', max_epoch - cfg.optim.num_local_epochs)
        patience_warmup += cfg.optim.num_local_epochs
        model.model._global = False

    if cfg.wandb.use:
        run = init_wandb()

    num_splits = len(loggers)
    all_splits = [l.name for l in loggers]
    split_names = ['val', 'test']
    full_epoch_times = []
    perf = [[] for _ in range(num_splits)]
    for cur_epoch in range(start_epoch, max_epoch):

        if early_stopping and patience <= 0:
            logging.info('Early stopping because validation loss is not decreasing further')
            break

        # for polynormer, with from local pre-training to global
        if cur_epoch == cfg.optim.num_local_epochs:
            logging.info('Local-only pre-training done, starting global training')
            model.model._global = True

        # train epoch
        start_time = time.perf_counter()
        train_epoch(loggers[0], loaders[0], model, optimizer, scheduler, cfg.optim.batch_accumulation)
        perf[0].append(loggers[0].write_epoch(cur_epoch))

        if is_eval_epoch(cur_epoch):
            # eval
            for i in range(1, num_splits):
                eval_epoch(loggers[i], loaders[i], model, split=split_names[i - 1])
                perf[i].append(loggers[i].write_epoch(cur_epoch))
        else:
            # repeat log of previous eval
            for i in range(1, num_splits):
                perf[i].append(perf[i][-1])

        # lr scheduler
        if cfg.optim.scheduler == 'reduce_on_plateau':
            # based on val loss
            scheduler.step(perf[1][-1]['loss'])
        else:
            scheduler.step()

        # time
        full_epoch_times.append(time.perf_counter() - start_time)
        
        # checkpoint with regular frequency (if enabled)
        if cfg.train.enable_ckpt and not cfg.train.ckpt_best and is_ckpt_epoch(cur_epoch):
            save_ckpt(model, optimizer, scheduler, cur_epoch)

        if cfg.wandb.use:
            run.log(flatten_dict(perf), step=cur_epoch)

        # log current best stats on eval epoch
        if is_eval_epoch(cur_epoch):
            val_losses = np.array([vp['loss'] for vp in perf[1]])
            val_loss_cur_epoch = float(val_losses[-1])

            # patience is computed on validation loss
            if cur_epoch > patience_warmup:
                if best_val_loss is None:
                    best_val_loss = val_loss_cur_epoch
                elif val_loss_cur_epoch < best_val_loss:
                    best_val_loss = val_loss_cur_epoch
                elif patience_m * best_val_loss <= val_loss_cur_epoch:
                    patience -= 1

            best_epoch = int(val_losses.argmin())
            best_m = ["", "", ""]  # loss is reported anyway

            if cfg.metric_best != 'auto':
                # select again based on val perf of `cfg.metric_best`
                m = cfg.metric_best

                # also updated best_m with the metric values
                best_epoch = get_best_epoch(perf, val_losses, m, best_m)

                if cfg.wandb.use:
                    log_wandb_val_epoch(run, perf, m, cur_epoch, best_epoch, full_epoch_times, all_splits)

            # checkpoint and log
            checkpoint_and_log(model, optimizer, scheduler, cur_epoch, best_epoch, perf, full_epoch_times, best_m)

    logging.info(f"Avg time per epoch: {np.mean(full_epoch_times):.2f}s")
    logging.info(f"Total train loop time: {np.sum(full_epoch_times) / 3600:.2f}h")
    for logger in loggers:
        logger.close()
    if cfg.train.ckpt_clean:
        clean_ckpt()

    if cfg.wandb.use:
        run.finish()

    logging.info('Task done, results saved in %s', cfg.run_dir)
    # return the logged results per epoch (organized differently)
    results = _get_epoch_log_results(perf, best_epoch)
    return results


def get_best_epoch(perf, val_losses, m, best_m):
    val_metric = np.array([vp[m] for vp in perf[1]])
    if cfg.metric_agg == "argmax":
        metric_agg = "max"
    elif cfg.metric_agg == "argmin":
        metric_agg = "min"
    else:
        raise ValueError("cfg.metric_agg should be either 'argmax' or 'argmin'")
    best_val_metric = getattr(val_metric, metric_agg)()
    best_val_metric_epochs = np.arange(len(val_metric), dtype=int)[val_metric == best_val_metric]

    # use loss to decide when epochs have same best metric
    best_epoch = int(best_val_metric_epochs[val_losses[best_val_metric_epochs].argmin()])

    # save the metric to be reported
    if m in perf[0][best_epoch]:
        best_m[0] = f"train_{m}: {perf[0][best_epoch][m]:.4f}"
    else:
        # (for some datasets it is too expensive to compute the main metric on the training set)
        best_m[0] = f"train_{m}: {0:.4f}"
    best_m[1] =     f"val_{m}:   {perf[1][best_epoch][m]:.4f}"
    best_m[2] =     f"test_{m}:  {perf[2][best_epoch][m]:.4f}"
    return best_epoch


def init_wandb():
    try:
        import wandb
    except:
        raise ImportError('WandB is not installed.')
    if cfg.wandb.name == '':
        wandb_name = make_wandb_name(cfg)
    else:
        wandb_name = cfg.wandb.name
    run = wandb.init(entity=cfg.wandb.entity, project=cfg.wandb.project, name=wandb_name)
    run.config.update(cfg_to_dict(cfg))
    return run


def log_wandb_val_epoch(run, perf, m, cur_epoch, best_epoch, full_epoch_times, all_splits):
    bstats = {"best/epoch": best_epoch}
    for i, s in enumerate(all_splits):
        bstats[f"best/{s}_loss"] = perf[i][best_epoch]['loss']
        if m in perf[i][best_epoch]:
            bstats[f"best/{s}_{m}"] = perf[i][best_epoch][m]
            run.summary[f"best_{s}_perf"] = perf[i][best_epoch][m]
        for x in ['hits@1', 'hits@3', 'hits@10', 'mrr']:
            if x in perf[i][best_epoch]:
                bstats[f"best/{s}_{x}"] = perf[i][best_epoch][x]
    run.log(bstats, step=cur_epoch)
    run.summary["full_epoch_time_avg"] = np.mean(full_epoch_times)
    run.summary["full_epoch_time_sum"] = np.sum(full_epoch_times)


def checkpoint_and_log(model, optimizer, scheduler, cur_epoch, best_epoch, perf, full_epoch_times, best_m):
    # checkpoint the best epoch params (if enabled)
    if cfg.train.enable_ckpt and cfg.train.ckpt_best and best_epoch == cur_epoch:
        save_ckpt(model, optimizer, scheduler, cur_epoch)
        if cfg.train.ckpt_clean:  # delete old ckpt each time
            clean_ckpt()
    logging.info(get_log_str(cur_epoch, full_epoch_times, best_epoch, perf, best_m))
    if hasattr(model, 'trf_layers'):
        # log SAN's gamma parameter values if they are trainable
        for li, gtl in enumerate(model.trf_layers):
            if torch.is_tensor(gtl.attention.gamma) and gtl.attention.gamma.requires_grad:
                logging.info(f"    {gtl.__class__.__name__} {li}: gamma={gtl.attention.gamma.item()}")


def get_log_str(cur_epoch, full_epoch_times, best_epoch, perf, best_m) -> str:
    s = f"> Epoch {cur_epoch}: took {full_epoch_times[-1]:.1f}s (avg {np.mean(full_epoch_times):.1f}s)"
    s += f"\n  Best so far: epoch {best_epoch}"
    names = ["train", "val", "test", "val_adv"]
    for i in range(len(best_m)):
        s += f"\n    {best_m[i]},  {names[i].rjust(7)}_loss: {perf[i][best_epoch]['loss']:.4f}"
    return s


def _get_epoch_log_results(perf, best_epoch: int):
    results = {}
    for split, p in zip(["train", "val", "test"], perf):
        results[split] = {}
        for p_epoch in p:
            for k, v in p_epoch.items():
                if k in ["epoch", "eta", "eta_hours", "params"]:
                    continue
                if split in ["val", "test"] and k == "lr":
                    continue
                if k not in results[split]:
                    results[split][k] = []
                results[split][k].append(v)
    results["best_val_epoch"] = best_epoch
    results["best_val_" + cfg.metric_best] = perf[1][best_epoch].get(cfg.metric_best)
    results["best_val_train_" + cfg.metric_best] = perf[0][best_epoch].get(cfg.metric_best)
    results["best_val_test_" + cfg.metric_best] = perf[2][best_epoch].get(cfg.metric_best)
    return results


@register_train('inference-only')
def inference_only(loggers, loaders, model, optimizer=None, scheduler=None):
    """
    Customized pipeline to run inference only.

    Args:
        loggers: List of loggers
        loaders: List of loaders
        model: GNN model
        optimizer: Unused, exists just for API compatibility
        scheduler: Unused, exists just for API compatibility
    """
    num_splits = len(loggers)
    split_names = ['train', 'val', 'test']
    perf = [[] for _ in range(num_splits)]
    cur_epoch = 0
    start_time = time.perf_counter()

    for i in range(0, num_splits):
        eval_epoch(loggers[i], loaders[i], model,
                   split=split_names[i])
        perf[i].append(loggers[i].write_epoch(cur_epoch))

    best_epoch = 0
    best_train = best_val = best_test = ""
    if cfg.metric_best != 'auto':
        # Select again based on val perf of `cfg.metric_best`.
        m = cfg.metric_best
        if m in perf[0][best_epoch]:
            best_train = f"train_{m}: {perf[0][best_epoch][m]:.4f}"
        else:
            # Note: For some datasets it is too expensive to compute
            # the main metric on the training set.
            best_train = f"train_{m}: {0:.4f}"
        best_val = f"val_{m}: {perf[1][best_epoch][m]:.4f}"
        best_test = f"test_{m}: {perf[2][best_epoch][m]:.4f}"

    logging.info(
        f"> Inference | "
        f"train_loss: {perf[0][best_epoch]['loss']:.4f} {best_train}\t"
        f"val_loss: {perf[1][best_epoch]['loss']:.4f} {best_val}\t"
        f"test_loss: {perf[2][best_epoch]['loss']:.4f} {best_test}"
    )
    logging.info(f'Done! took: {time.perf_counter() - start_time:.2f}s')
    for logger in loggers:
        logger.close()


@register_train('PCQM4Mv2-inference')
def ogblsc_inference(loggers, loaders, model, optimizer=None, scheduler=None):
    """
    Customized pipeline to run inference on OGB-LSC PCQM4Mv2.

    Args:
        loggers: Unused, exists just for API compatibility
        loaders: List of loaders
        model: GNN model
        optimizer: Unused, exists just for API compatibility
        scheduler: Unused, exists just for API compatibility
    """
    from ogb.lsc import PCQM4Mv2Evaluator
    evaluator = PCQM4Mv2Evaluator()

    num_splits = 3
    split_names = ['valid', 'test-dev', 'test-challenge']
    assert len(loaders) == num_splits, "Expecting 3 particular splits."

    # Check PCQM4Mv2 prediction targets.
    logging.info(f"0 ({split_names[0]}): {len(loaders[0].dataset)}")
    assert(all([not torch.isnan(d.y)[0] for d in loaders[0].dataset]))
    logging.info(f"1 ({split_names[1]}): {len(loaders[1].dataset)}")
    assert(all([torch.isnan(d.y)[0] for d in loaders[1].dataset]))
    logging.info(f"2 ({split_names[2]}): {len(loaders[2].dataset)}")
    assert(all([torch.isnan(d.y)[0] for d in loaders[2].dataset]))

    model.eval()
    for i in range(num_splits):
        all_true = []
        all_pred = []
        for batch in loaders[i]:
            batch.to(torch.device(cfg.accelerator))
            pred, true = model(batch)
            all_true.append(true.detach().to('cpu', non_blocking=True))
            all_pred.append(pred.detach().to('cpu', non_blocking=True))
        all_true, all_pred = torch.cat(all_true), torch.cat(all_pred)

        if i == 0:
            input_dict = {'y_pred': all_pred.squeeze(),
                          'y_true': all_true.squeeze()}
            result_dict = evaluator.eval(input_dict)
            logging.info(f"{split_names[i]}: MAE = {result_dict['mae']}")  # Get MAE.
        else:
            input_dict = {'y_pred': all_pred.squeeze()}
            evaluator.save_test_submission(input_dict=input_dict,
                                           dir_path=cfg.run_dir,
                                           mode=split_names[i])


@register_train('log-attn-weights')
def log_attn_weights(loggers, loaders, model, optimizer=None, scheduler=None):
    """
    Customized pipeline to inference on the test set and log the attention
    weights in Transformer modules.

    Args:
        loggers: Unused, exists just for API compatibility
        loaders: List of loaders
        model (torch.nn.Module): GNN model
        optimizer: Unused, exists just for API compatibility
        scheduler: Unused, exists just for API compatibility
    """
    import os.path as osp
    from torch_geometric.loader.dataloader import DataLoader
    from torch_geometric.utils import unbatch, unbatch_edge_index

    start_time = time.perf_counter()

    # The last loader is a test set.
    l = loaders[-1]
    # To get a random sample, create a new loader that shuffles the test set.
    loader = DataLoader(l.dataset, batch_size=l.batch_size,
                        shuffle=True, num_workers=0)

    output = []
    # batch = next(iter(loader))  # Run one random batch.
    for b_index, batch in enumerate(loader):
        bsize = batch.batch.max().item() + 1  # Batch size.
        if len(output) >= 128:
            break
        print(f">> Batch {b_index}:")

        X_orig = unbatch(batch.x.cpu(), batch.batch.cpu())
        batch.to(torch.device(cfg.accelerator))
        model.eval()
        model(batch)

        # Unbatch to individual graphs.
        X = unbatch(batch.x.cpu(), batch.batch.cpu())
        edge_indices = unbatch_edge_index(batch.edge_index.cpu(),
                                          batch.batch.cpu())
        graphs = []
        for i in range(bsize):
            graphs.append({'num_nodes': len(X[i]),
                           'x_orig': X_orig[i],
                           'x_final': X[i],
                           'edge_index': edge_indices[i],
                           'attn_weights': []  # List with attn weights in layers from 0 to L-1.
                           })

        # Iterate through GPS layers and pull out stored attn weights.
        for l_i, (name, module) in enumerate(model.model.layers.named_children()):
            if hasattr(module, 'attn_weights'):
                print(l_i, name, module.attn_weights.shape)
                for g_i in range(bsize):
                    # Clip to the number of nodes in this graph.
                    # num_nodes = graphs[g_i]['num_nodes']
                    # aw = module.attn_weights[g_i, :num_nodes, :num_nodes]
                    aw = module.attn_weights[g_i]
                    graphs[g_i]['attn_weights'].append(aw.cpu())
        output += graphs

    logging.info(
        f"[*] Collected a total of {len(output)} graphs and their "
        f"attention weights for {len(output[0]['attn_weights'])} layers.")

    # Save the graphs and their attention stats.
    save_file = osp.join(cfg.run_dir, 'graph_attn_stats.pt')
    logging.info(f"Saving to file: {save_file}")
    torch.save(output, save_file)

    logging.info(f'Done! took: {time.perf_counter() - start_time:.2f}s')
