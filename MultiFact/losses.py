import torch
from torch import Tensor
import torch.nn.functional as F

from .utils import EPS


def fisher_rao_distance(input: Tensor, target: Tensor) -> Tensor:
    """Mean Fisher-Rao distance between corresponding distribution columns."""

    inner = torch.sum(
        torch.sqrt(input + EPS) * torch.sqrt(target + EPS),
        dim=0,
    )
    return torch.acos(inner.clamp(EPS, 1.0 - EPS)).mean()


def selected_factor_cross_entropy(
    logits: Tensor,
    labels: Tensor,
    cell_factor_ids: Tensor,
) -> tuple[Tensor, Tensor]:
    """Classify labels using only each cell's assigned cell-factor block.

    Args:
        logits: Raw logits with shape (cells, cell_factors, classes).
        labels: Integer labels with shape (cells,).
        cell_factor_ids: Assigned cell factor for every cell, shape (cells,).
    """

    if logits.ndim != 3:
        raise ValueError("logits must have shape (cells, factors, classes).")
    row = torch.arange(logits.shape[0], device=logits.device)
    selected_logits = logits[row, cell_factor_ids]
    loss = F.cross_entropy(selected_logits, labels)
    accuracy = (selected_logits.argmax(dim=1) == labels).float().mean()
    return loss, accuracy


def _median_rbf_gamma(X: Tensor) -> Tensor:
    if X.shape[0] < 2:
        return X.new_ones(())
    with torch.no_grad():
        dist2 = torch.pdist(X.float()).pow(2)
        positive = dist2[dist2 > EPS]
        if positive.numel() == 0:
            return X.new_ones(())
        median_dist2 = torch.median(positive)
        gamma = 1.0 / (2.0 * median_dist2.clamp_min(EPS))
    return gamma.to(device=X.device, dtype=X.dtype)


def _rbf_mean(X: Tensor, Y: Tensor, gamma: Tensor) -> Tensor:
    dist2 = torch.cdist(X, Y).pow(2)
    return torch.exp(-gamma * dist2).mean()


def _sample_group(
    X: Tensor,
    indices: Tensor,
    max_cells: int,
) -> Tensor:
    if indices.numel() > max_cells:
        selected = torch.randperm(indices.numel(), device=X.device)[:max_cells]
        indices = indices[selected]
    return X[indices]


def _mmd_between(X: Tensor, Y: Tensor, gamma: Tensor) -> Tensor:
    return _rbf_mean(X, X, gamma) + _rbf_mean(Y, Y, gamma) - 2.0 * _rbf_mean(
        X,
        Y,
        gamma,
    )


def sampled_batch_mmd_loss(
    X: Tensor,
    batch_labels: Tensor,
    max_cells_per_batch: int = 64,
) -> Tensor:
    """Average pairwise RBF MMD across batches using sampled cells.

    Each non-empty batch contributes at most ``max_cells_per_batch`` cells. The
    RBF bandwidth is estimated from the concatenated sampled cells by the median
    squared-distance heuristic.
    """

    if X.ndim != 2:
        raise ValueError("X must have shape (cells, features).")
    if batch_labels.shape != (X.shape[0],):
        raise ValueError("batch_labels must have shape (cells,).")
    if max_cells_per_batch <= 0:
        raise ValueError("max_cells_per_batch must be positive.")

    groups = []
    for label in torch.unique(batch_labels):
        indices = torch.nonzero(batch_labels == label, as_tuple=False).flatten()
        if indices.numel() < 2:
            continue
        groups.append(_sample_group(X, indices, max_cells_per_batch))

    if len(groups) < 2:
        return X.new_zeros(())

    gamma = _median_rbf_gamma(torch.cat(groups, dim=0))
    losses = []
    for i, Xi in enumerate(groups):
        for Xj in groups[i + 1:]:
            losses.append(_mmd_between(Xi, Xj, gamma))
    return torch.stack(losses).mean() if losses else X.new_zeros(())


def sampled_condition_batch_ref_mmd_loss(
    X: Tensor,
    condition_labels: Tensor,
    batch_labels: Tensor,
    max_cells_per_group: int = 64,
) -> Tensor:
    """Batch MMD within each condition against a reference batch.

    For every condition, cells are split by batch. The largest batch group within
    that condition is used as the reference, and every other valid batch group is
    compared against it. Each condition-batch group contributes at most
    ``max_cells_per_group`` sampled cells.
    """

    if X.ndim != 2:
        raise ValueError("X must have shape (cells, features).")
    if condition_labels.shape != (X.shape[0],):
        raise ValueError("condition_labels must have shape (cells,).")
    if batch_labels.shape != (X.shape[0],):
        raise ValueError("batch_labels must have shape (cells,).")
    if max_cells_per_group <= 0:
        raise ValueError("max_cells_per_group must be positive.")

    losses = []
    for condition in torch.unique(condition_labels):
        condition_mask = condition_labels == condition
        groups = []
        group_sizes = []
        for batch in torch.unique(batch_labels[condition_mask]):
            indices = torch.nonzero(
                condition_mask & (batch_labels == batch),
                as_tuple=False,
            ).flatten()
            if indices.numel() < 2:
                continue
            group_sizes.append(indices.numel())
            groups.append(_sample_group(X, indices, max_cells_per_group))

        if len(groups) < 2:
            continue

        gamma = _median_rbf_gamma(torch.cat(groups, dim=0))
        reference_index = max(range(len(group_sizes)), key=group_sizes.__getitem__)
        reference = groups[reference_index]
        for index, group in enumerate(groups):
            if index == reference_index:
                continue
            losses.append(_mmd_between(reference, group, gamma))

    return torch.stack(losses).mean() if losses else X.new_zeros(())


def sampled_joint_ref_mmd_loss(
    X: Tensor,
    condition_labels: Tensor,
    batch_labels: Tensor,
    max_cells_per_group: int = 64,
) -> Tensor:
    """MMD from one reference condition-batch group to all other groups.

    This follows the scDisInFact-style shared-factor alignment: cells are grouped
    by the joint ``(condition, batch)`` label, the largest valid joint group is
    used as the global reference, and every other valid joint group is compared
    against that reference.
    """

    if X.ndim != 2:
        raise ValueError("X must have shape (cells, features).")
    if condition_labels.shape != (X.shape[0],):
        raise ValueError("condition_labels must have shape (cells,).")
    if batch_labels.shape != (X.shape[0],):
        raise ValueError("batch_labels must have shape (cells,).")
    if max_cells_per_group <= 0:
        raise ValueError("max_cells_per_group must be positive.")

    groups = []
    group_sizes = []
    for condition in torch.unique(condition_labels):
        condition_mask = condition_labels == condition
        for batch in torch.unique(batch_labels[condition_mask]):
            indices = torch.nonzero(
                condition_mask & (batch_labels == batch),
                as_tuple=False,
            ).flatten()
            if indices.numel() < 2:
                continue
            group_sizes.append(indices.numel())
            groups.append(_sample_group(X, indices, max_cells_per_group))

    if len(groups) < 2:
        return X.new_zeros(())

    gamma = _median_rbf_gamma(torch.cat(groups, dim=0))
    reference_index = max(range(len(group_sizes)), key=group_sizes.__getitem__)
    reference = groups[reference_index]
    losses = [
        _mmd_between(reference, group, gamma)
        for index, group in enumerate(groups)
        if index != reference_index
    ]
    return torch.stack(losses).mean() if losses else X.new_zeros(())


def sampled_within_cluster_joint_ref_mmd_loss(
    X: Tensor,
    condition_labels: Tensor,
    batch_labels: Tensor,
    cluster_labels: Tensor,
    max_cells_per_group: int = 64,
) -> Tensor:
    """Average joint reference MMD within each hard cell-factor cluster.

    ``cluster_labels`` usually comes from ``argmax(U_cell)`` and matches the
    hard cell mask used for cell-type-specific condition and batch factors.
    Within each non-empty cluster, cells are grouped by joint
    ``(condition, batch)`` labels and aligned to the largest valid group.
    """

    if cluster_labels.shape != (X.shape[0],):
        raise ValueError("cluster_labels must have shape (cells,).")

    losses = []
    for cluster in torch.unique(cluster_labels):
        cluster_mask = cluster_labels == cluster
        if int(cluster_mask.sum().item()) < 4:
            continue
        valid_groups = 0
        for condition in torch.unique(condition_labels[cluster_mask]):
            condition_mask = cluster_mask & (condition_labels == condition)
            for batch in torch.unique(batch_labels[condition_mask]):
                group_size = torch.sum(condition_mask & (batch_labels == batch))
                if int(group_size.item()) >= 2:
                    valid_groups += 1
        if valid_groups < 2:
            continue
        losses.append(
            sampled_joint_ref_mmd_loss(
                X[cluster_mask],
                condition_labels[cluster_mask],
                batch_labels[cluster_mask],
                max_cells_per_group=max_cells_per_group,
            )
        )

    finite_losses = [loss for loss in losses if torch.isfinite(loss)]
    if not finite_losses:
        return X.new_zeros(())
    return torch.stack(finite_losses).mean()


def factor_overlap_loss(A_left: Tensor, A_right: Tensor) -> Tensor:
    """Penalize shared gene-weight profiles between two factor families.

    Columns are transformed with softmax over genes and then L2-normalized. The
    loss is the mean squared cosine overlap across all cross-family factor
    pairs.
    """

    if A_left.ndim != 2 or A_right.ndim != 2:
        raise ValueError("A_left and A_right must have shape (genes, factors).")
    if A_left.shape[0] != A_right.shape[0]:
        raise ValueError("A_left and A_right must have the same number of genes.")

    left = torch.softmax(A_left, dim=0)
    right = torch.softmax(A_right, dim=0)
    left = F.normalize(left, p=2, dim=0, eps=1e-12)
    right = F.normalize(right, p=2, dim=0, eps=1e-12)
    overlap = left.t() @ right
    return torch.mean(overlap ** 2)


def batch_eta_squared(U_cell: Tensor, batch_labels: Tensor) -> Tensor:
    """Return batch-associated variance fraction for every cell factor."""

    if U_cell.ndim != 2:
        raise ValueError("U_cell must have shape (cells, factors).")
    global_mean = U_cell.mean(dim=0)
    total_ss = torch.sum((U_cell - global_mean) ** 2, dim=0)
    between_ss = torch.zeros_like(total_ss)

    for label in torch.unique(batch_labels):
        group = U_cell[batch_labels == label]
        if group.numel() == 0:
            continue
        between_ss += group.shape[0] * (group.mean(dim=0) - global_mean) ** 2

    score = between_ss / total_ss.clamp_min(1e-12)
    return torch.where(total_ss > 1e-12, score, torch.zeros_like(score))
