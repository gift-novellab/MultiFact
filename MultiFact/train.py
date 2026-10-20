from typing import Dict, List

import torch
from torch import Tensor
from tqdm import tqdm

from .model import MultiFactModel
from .utils import prepare_target


def _train_stage(
    model: MultiFactModel,
    P: Tensor,
    stage: int,
    max_iters: int,
) -> List[Dict[str, float]]:
    if max_iters == 0:
        return []

    if stage == 1:
        parameters = [model.Z_cell, model.A_cell]
        loss_fn = model.stage1_loss
    elif stage == 3:
        parameters = [
            model.Z_cell,
            model.A_cell,
        ]
        if model.has_condition:
            parameters.extend([model.Z_condition, model.A_condition])
        loss_fn = model.stage3_loss
    elif stage == 4:
        parameters = [
            model.Z_cell,
            model.A_cell,
            model.Z_batch,
            model.A_batch,
        ]
        if model.has_condition:
            parameters.extend([model.Z_condition, model.A_condition])
        loss_fn = model.stage4_loss
    elif stage == 5:
        model.Z_cell.requires_grad_(not model.config.stage5_freeze_u_cell)
        model.Z_batch.requires_grad_(not model.config.stage5_freeze_u_batch)
        parameters = []
        if not model.config.stage5_freeze_u_cell:
            parameters.append(model.Z_cell)
        if not model.config.stage5_freeze_u_batch:
            parameters.append(model.Z_batch)
        if model.has_condition:
            model.Z_condition.requires_grad_(True)
            parameters.append(model.Z_condition)
        loss_fn = model.stage5_loss
    else:
        raise ValueError("stage must be 1, 3, 4, or 5.")

    if not parameters:
        raise ValueError(
            "Stage 5 has no trainable parameters. Disable at least one of "
            "stage5_freeze_u_cell or stage5_freeze_u_batch, or provide "
            "condition labels so U_condition can be optimized."
        )

    optimizer = torch.optim.Adam(parameters, lr=model.config.lr)
    amp_enabled = model.config.use_amp and model.device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    history: List[Dict[str, float]] = []

    with tqdm(
        total=max_iters,
        disable=not model.config.verbose,
        desc=f"MultiFact stage {stage}",
    ) as pbar:
        for iteration in range(max_iters):
            optimizer.zero_grad()
            with torch.amp.autocast("cuda", enabled=amp_enabled):
                loss, pieces = loss_fn(P)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            if (
                iteration % model.config.history_interval == 0
                or iteration == max_iters - 1
            ):
                record = {"stage": float(stage), "iter": float(iteration)}
                record.update(
                    {
                        key: float(value.detach().cpu())
                        for key, value in pieces.items()
                    }
                )
                history.append(record)
            pbar.update(1)
    return history


def fit_multifact(
    model: MultiFactModel,
    X: Tensor,
) -> List[Dict[str, float]]:
    """Run cell learning, batch pruning, condition learning, then batch learning."""

    P = prepare_target(X, device=model.device)
    if P.shape != (model.n_genes, model.n_cells):
        raise ValueError(
            f"X implies target shape {tuple(P.shape)}, expected "
            f"{(model.n_genes, model.n_cells)}."
        )

    history: List[Dict[str, float]] = []
    history.extend(
        _train_stage(model, P, stage=1, max_iters=model.config.stage1_iters)
    )

    pruning_report = model.prune_batch_factors()
    history.append(
        {
            "stage": 2.0,
            "iter": 0.0,
            "batch_score_threshold": float(pruning_report["threshold"]),
            "n_factors_before": float(pruning_report["n_factors_before"]),
            "n_factors_after": float(pruning_report["n_factors_after"]),
            "n_factors_removed": float(
                pruning_report["n_factors_before"]
                - pruning_report["n_factors_after"]
            ),
        }
    )
    model.initialize_condition_factors()
    history.extend(
        _train_stage(model, P, stage=3, max_iters=model.config.stage3_iters)
    )

    model.initialize_batch_factors()
    history.extend(
        _train_stage(model, P, stage=4, max_iters=model.config.stage4_iters)
    )

    if model.config.stage5_iters > 0:
        model.freeze_factor_matrices()
        history.extend(
            _train_stage(model, P, stage=5, max_iters=model.config.stage5_iters)
        )

    model.history = history
    return history
