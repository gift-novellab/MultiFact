from typing import Dict, Optional, Tuple

import torch
from torch import Tensor
from torch.nn import Parameter
import torch.nn.functional as F

from .config import MultiFactConfig
from .losses import (
    batch_eta_squared,
    factor_overlap_loss,
    fisher_rao_distance,
    sampled_condition_batch_ref_mmd_loss,
    sampled_joint_ref_mmd_loss,
    sampled_within_cluster_joint_ref_mmd_loss,
    selected_factor_cross_entropy,
)
from .utils import normalize_nonnegative


class MultiFactModel(torch.nn.Module):
    """Four-stage structured factorization for cell-specific covariate effects."""

    def __init__(
        self,
        n_cells: int,
        n_genes: int,
        condition_labels: Optional[Tensor] = None,
        batch_labels: Optional[Tensor] = None,
        n_conditions: Optional[int] = None,
        n_batches: Optional[int] = None,
        config: Optional[MultiFactConfig] = None,
        device: Optional[str] = None,
    ):
        super().__init__()
        self.config = config or MultiFactConfig()
        self.config.validate()
        self.device = torch.device(device) if device else torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.n_cells = int(n_cells)
        self.n_genes = int(n_genes)
        if batch_labels is None:
            raise ValueError("batch_labels is required.")

        self.has_condition = condition_labels is not None
        if condition_labels is None:
            condition_labels = torch.zeros(
                self.n_cells, dtype=torch.long, device=self.device
            )
        else:
            condition_labels = torch.as_tensor(
                condition_labels, dtype=torch.long, device=self.device
            )
        batch_labels = torch.as_tensor(
            batch_labels, dtype=torch.long, device=self.device
        )
        if condition_labels.shape != (self.n_cells,):
            raise ValueError("condition_labels must have shape (n_cells,).")
        if batch_labels.shape != (self.n_cells,):
            raise ValueError("batch_labels must have shape (n_cells,).")

        inferred_conditions = int(condition_labels.max().item()) + 1
        inferred_batches = int(batch_labels.max().item()) + 1
        self.n_conditions = (
            inferred_conditions if n_conditions is None else int(n_conditions)
        )
        self.n_batches = inferred_batches if n_batches is None else int(n_batches)
        if self.n_conditions != inferred_conditions:
            raise ValueError("n_conditions must match encoded condition labels.")
        if not self.has_condition and self.n_conditions != 1:
            raise ValueError("n_conditions must be 1 when condition_labels is None.")
        if self.n_batches != inferred_batches:
            raise ValueError("n_batches must match encoded batch labels.")
        self.shared_condition_factor_indices = [
            tuple(int(index) for index in group)
            for group in self.config.shared_condition_factor_indices
        ]
        for group in self.shared_condition_factor_indices:
            for condition_index in group:
                if condition_index >= self.n_conditions:
                    raise ValueError(
                        "shared_condition_factor_indices contains an index "
                        "outside the encoded condition range."
                    )
        self.n_shared_condition_factors = (
            len(self.shared_condition_factor_indices) if self.has_condition else 0
        )
        self.n_condition_factors = self.n_conditions + self.n_shared_condition_factors
        self.condition_factor_names = [
            f"condition_{index}" for index in range(self.n_conditions)
        ] + [
            "shared_condition_" + "_".join(str(index) for index in group)
            for group in self.shared_condition_factor_indices
        ]

        self.initial_k_cell = self.config.k_cell
        self.k_cell = self.initial_k_cell
        self.register_parameter(
            "Z_cell",
            Parameter(torch.randn(self.n_cells, self.k_cell, device=self.device)),
        )
        self.register_parameter(
            "A_cell",
            Parameter(torch.randn(self.n_genes, self.k_cell, device=self.device)),
        )
        self.register_parameter("Z_condition", None)
        self.register_parameter("A_condition", None)
        self.register_parameter("Z_batch", None)
        self.register_parameter("A_batch", None)

        self.register_buffer("condition_labels", condition_labels)
        self.register_buffer("batch_labels", batch_labels)
        observed_condition_mask = F.one_hot(condition_labels, self.n_conditions).float()
        self.register_buffer(
            "condition_mask",
            torch.cat(
                [
                    observed_condition_mask,
                    observed_condition_mask.new_zeros(
                        self.n_cells,
                        self.n_shared_condition_factors,
                    ),
                ],
                dim=1,
            ),
        )
        condition_reconstruction_mask = self.condition_mask.clone()
        for index in self.config.open_condition_factor_indices:
            condition_index = int(index)
            if condition_index >= self.n_conditions:
                raise ValueError(
                    "open_condition_factor_indices contains an index outside "
                    "the encoded condition range."
                )
            condition_reconstruction_mask[:, condition_index] = 1.0
        for shared_offset, group in enumerate(self.shared_condition_factor_indices):
            shared_index = self.n_conditions + shared_offset
            group_mask = torch.zeros(
                self.n_cells,
                dtype=condition_reconstruction_mask.dtype,
                device=self.device,
            )
            for condition_index in group:
                group_mask = group_mask + observed_condition_mask[:, condition_index]
            condition_reconstruction_mask[:, shared_index] = group_mask.clamp_max(1.0)
        self.register_buffer(
            "condition_reconstruction_mask",
            condition_reconstruction_mask,
        )
        self.register_buffer(
            "batch_mask",
            F.one_hot(batch_labels, self.n_batches).float(),
        )
        self.register_buffer(
            "original_factor_ids",
            torch.arange(self.k_cell, device=self.device),
        )

        self.pruning_report: Dict[str, object] = {}
        self.current_stage = 1

    @staticmethod
    def _positive(x: Tensor) -> Tensor:
        return F.softplus(x)

    def cell_embedding(self) -> Tensor:
        return self._positive(self.Z_cell)

    def cell_embedding_for_mmd(self) -> Tensor:
        return normalize_nonnegative(self.cell_embedding(), dim=1)

    def cell_factor_ids(self) -> Tensor:
        return self.cell_embedding().argmax(dim=1)

    def cell_gate(self) -> Tensor:
        return F.one_hot(
            self.cell_factor_ids(), num_classes=self.k_cell
        ).to(dtype=self.Z_cell.dtype)

    def condition_tensor(
        self,
        label_mask: bool = True,
        reconstruction_mask: bool = True,
    ) -> Tensor:
        if self.Z_condition is None:
            raise RuntimeError("Condition factors have not been initialized.")
        value = self._positive(self.Z_condition) * self.cell_gate()[:, :, None]
        if label_mask:
            mask = (
                self.condition_reconstruction_mask
                if reconstruction_mask
                else self.condition_mask
            )
            value = value * mask[:, None, :]
        return value

    def batch_tensor(self, label_mask: bool = True) -> Tensor:
        if self.Z_batch is None:
            raise RuntimeError("Batch factors have not been initialized.")
        value = self._positive(self.Z_batch) * self.cell_gate()[:, :, None]
        if label_mask:
            value = value * self.batch_mask[:, None, :]
        return value

    def _factor_matrix(self, stage: int) -> Tensor:
        blocks = [self.A_cell]
        if stage >= 3 and self.has_condition:
            if self.A_condition is None:
                raise RuntimeError("Condition gene factors are not initialized.")
            blocks.append(
                self.A_condition.reshape(
                    self.n_genes, self.k_cell * self.n_condition_factors
                )
            )
        if stage >= 4:
            if self.A_batch is None:
                raise RuntimeError("Batch gene factors are not initialized.")
            blocks.append(
                self.A_batch.reshape(self.n_genes, self.k_cell * self.n_batches)
            )
        return torch.cat(blocks, dim=1)

    def embedding(self, stage: int, label_mask: bool = True) -> Tensor:
        if stage not in (1, 3, 4, 5):
            raise ValueError("stage must be 1, 3, 4, or 5.")
        blocks = [self.cell_embedding()]
        if stage >= 3 and self.has_condition:
            blocks.append(
                self.condition_tensor(label_mask=label_mask).reshape(
                    self.n_cells, self.k_cell * self.n_condition_factors
                )
            )
        if stage >= 4:
            blocks.append(
                self.batch_tensor(label_mask=label_mask).reshape(
                    self.n_cells, self.k_cell * self.n_batches
                )
            )
        return torch.cat(blocks, dim=1)

    def normalized_factor_matrix(self, stage: int) -> Tensor:
        return torch.softmax(self._factor_matrix(stage), dim=0)

    def normalized_embedding(self, stage: int, label_mask: bool = True) -> Tensor:
        return normalize_nonnegative(self.embedding(stage, label_mask=label_mask), dim=1)

    def forward(self, stage: int, label_mask: bool = True) -> Tensor:
        return self.normalized_factor_matrix(stage) @ self.normalized_embedding(
            stage, label_mask=label_mask
        ).t()

    def reconstruction_loss(
        self,
        P: Tensor,
        stage: int,
        label_mask: bool = True,
    ) -> Tensor:
        return fisher_rao_distance(
            self.forward(stage, label_mask=label_mask),
            P.to(self.device),
        )

    def _zero_loss(self) -> Tensor:
        return self.Z_cell.new_zeros(())

    def condition_within_cell_factor_overlap_loss(self) -> Tensor:
        """Penalize overlap between condition gene factors sharing a cell factor."""

        if (
            not self.has_condition
            or self.A_condition is None
            or self.n_condition_factors <= 1
        ):
            return self._zero_loss()
        A_condition = torch.softmax(self.A_condition, dim=0)
        overlaps = []
        for left_condition in range(self.n_condition_factors):
            for right_condition in range(
                left_condition + 1,
                self.n_condition_factors,
            ):
                pair_overlap = (
                    A_condition[:, :, left_condition]
                    * A_condition[:, :, right_condition]
                ).sum(dim=0)
                overlaps.append(pair_overlap)
        if not overlaps:
            return self._zero_loss()
        return torch.cat(overlaps).mean()

    def cell_mmd_loss(self) -> Tensor:
        U_cell = self.cell_embedding_for_mmd()
        if self.config.cell_mmd_mode == "global":
            return sampled_joint_ref_mmd_loss(
                U_cell,
                self.condition_labels,
                self.batch_labels,
                max_cells_per_group=self.config.mmd_cells_per_batch,
            )
        if self.config.cell_mmd_mode == "within_cell_factor":
            return sampled_within_cluster_joint_ref_mmd_loss(
                U_cell,
                self.condition_labels,
                self.batch_labels,
                self.cell_factor_ids(),
                max_cells_per_group=self.config.mmd_cells_per_batch,
            )
        raise ValueError(f"Unknown cell_mmd_mode: {self.config.cell_mmd_mode!r}")

    def condition_entropy_loss(self) -> Tensor:
        if not self.has_condition or self.Z_condition is None:
            return self._zero_loss()
        if self.n_condition_factors <= 1:
            return self._zero_loss()
        U_condition = self.condition_tensor(label_mask=False)
        cell_ids = torch.arange(self.n_cells, device=self.device)
        selected = U_condition[cell_ids, self.cell_factor_ids(), :]
        probabilities = normalize_nonnegative(selected, dim=1)
        entropy = -(
            probabilities * torch.log(probabilities.clamp_min(1e-12))
        ).sum(dim=1)
        max_entropy = torch.log(
            self.Z_cell.new_tensor(float(self.n_condition_factors))
        ).clamp_min(1e-12)
        normalized_entropy = entropy / max_entropy
        return 1.0 - normalized_entropy.mean()

    def stage1_loss(self, P: Tensor) -> Tuple[Tensor, Dict[str, Tensor]]:
        loss_rec = self.reconstruction_loss(P, stage=1)
        return loss_rec, {"loss": loss_rec.detach(), "loss_rec": loss_rec.detach()}

    def stage3_loss(self, P: Tensor) -> Tuple[Tensor, Dict[str, Tensor]]:
        if self.has_condition and self.Z_condition is None:
            raise RuntimeError("Call initialize_condition_factors() first.")
        loss_rec = self.reconstruction_loss(P, stage=3)
        if self.has_condition:
            loss_condition, condition_accuracy = selected_factor_cross_entropy(
                self.Z_condition[:, :, : self.n_conditions],
                self.condition_labels,
                self.cell_factor_ids(),
            )
        else:
            loss_condition = self._zero_loss()
            condition_accuracy = self._zero_loss()
        loss_cell_batch_mmd = self.cell_mmd_loss()
        if self.has_condition:
            A_condition = self.A_condition.reshape(
                self.n_genes,
                self.k_cell * self.n_condition_factors,
            )
            loss_a_cell_condition_overlap = factor_overlap_loss(
                self.A_cell,
                A_condition,
            )
            loss_a_condition_within_cell_overlap = (
                self.condition_within_cell_factor_overlap_loss()
            )
        else:
            loss_a_cell_condition_overlap = self._zero_loss()
            loss_a_condition_within_cell_overlap = self._zero_loss()
        total = (
            loss_rec
            + self.config.lambda_condition_ce * loss_condition
            + self.config.lambda_cell_batch_mmd * loss_cell_batch_mmd
            + self.config.lambda_a_cell_condition_overlap
            * loss_a_cell_condition_overlap
            + self.config.lambda_a_condition_within_cell_overlap
            * loss_a_condition_within_cell_overlap
        )
        return total, {
            "loss": total.detach(),
            "loss_rec": loss_rec.detach(),
            "loss_condition_ce": loss_condition.detach(),
            "loss_cell_batch_mmd": loss_cell_batch_mmd.detach(),
            "loss_a_cell_condition_overlap": (
                loss_a_cell_condition_overlap.detach()
            ),
            "loss_a_condition_within_cell_overlap": (
                loss_a_condition_within_cell_overlap.detach()
            ),
            "condition_accuracy": condition_accuracy.detach(),
        }

    def stage4_loss(self, P: Tensor) -> Tuple[Tensor, Dict[str, Tensor]]:
        if self.Z_batch is None:
            raise RuntimeError("Call initialize_batch_factors() first.")
        loss_rec = self.reconstruction_loss(P, stage=4)
        if self.has_condition:
            loss_condition, condition_accuracy = selected_factor_cross_entropy(
                self.Z_condition[:, :, : self.n_conditions],
                self.condition_labels,
                self.cell_factor_ids(),
            )
        else:
            loss_condition = self._zero_loss()
            condition_accuracy = self._zero_loss()
        loss_batch, batch_accuracy = selected_factor_cross_entropy(
            self.Z_batch,
            self.batch_labels,
            self.cell_factor_ids(),
        )
        loss_cell_batch_mmd = self.cell_mmd_loss()
        A_batch = self.A_batch.reshape(
            self.n_genes,
            self.k_cell * self.n_batches,
        )
        loss_a_cell_batch_overlap = factor_overlap_loss(
            self.A_cell,
            A_batch,
        )
        if self.has_condition:
            U_condition = self.condition_tensor(
                label_mask=True,
                reconstruction_mask=False,
            ).reshape(self.n_cells, self.k_cell * self.n_condition_factors)
            loss_condition_batch_mmd = sampled_condition_batch_ref_mmd_loss(
                U_condition,
                self.condition_labels,
                self.batch_labels,
                max_cells_per_group=self.config.mmd_cells_per_batch,
            )
            A_condition = self.A_condition.reshape(
                self.n_genes,
                self.k_cell * self.n_condition_factors,
            )
            loss_a_cell_condition_overlap = factor_overlap_loss(
                self.A_cell,
                A_condition,
            )
            loss_a_condition_batch_overlap = factor_overlap_loss(
                A_condition,
                A_batch,
            )
            loss_a_condition_within_cell_overlap = (
                self.condition_within_cell_factor_overlap_loss()
            )
        else:
            loss_condition_batch_mmd = self._zero_loss()
            loss_a_cell_condition_overlap = self._zero_loss()
            loss_a_condition_batch_overlap = self._zero_loss()
            loss_a_condition_within_cell_overlap = self._zero_loss()
        total = (
            loss_rec
            + self.config.lambda_condition_ce * loss_condition
            + self.config.lambda_batch_ce * loss_batch
            + self.config.lambda_cell_batch_mmd * loss_cell_batch_mmd
            + self.config.lambda_condition_batch_mmd
            * loss_condition_batch_mmd
            + self.config.lambda_a_cell_condition_overlap
            * loss_a_cell_condition_overlap
            + self.config.lambda_a_cell_batch_overlap
            * loss_a_cell_batch_overlap
            + self.config.lambda_a_condition_batch_overlap
            * loss_a_condition_batch_overlap
            + self.config.lambda_a_condition_within_cell_overlap
            * loss_a_condition_within_cell_overlap
        )
        return total, {
            "loss": total.detach(),
            "loss_rec": loss_rec.detach(),
            "loss_condition_ce": loss_condition.detach(),
            "loss_batch_ce": loss_batch.detach(),
            "loss_cell_batch_mmd": loss_cell_batch_mmd.detach(),
            "loss_condition_batch_mmd": loss_condition_batch_mmd.detach(),
            "loss_a_cell_condition_overlap": (
                loss_a_cell_condition_overlap.detach()
            ),
            "loss_a_cell_batch_overlap": loss_a_cell_batch_overlap.detach(),
            "loss_a_condition_batch_overlap": (
                loss_a_condition_batch_overlap.detach()
            ),
            "loss_a_condition_within_cell_overlap": (
                loss_a_condition_within_cell_overlap.detach()
            ),
            "condition_accuracy": condition_accuracy.detach(),
            "batch_accuracy": batch_accuracy.detach(),
        }

    def stage5_loss(self, P: Tensor) -> Tuple[Tensor, Dict[str, Tensor]]:
        if self.Z_batch is None:
            raise RuntimeError("Call initialize_batch_factors() first.")
        loss_rec = self.reconstruction_loss(P, stage=5, label_mask=False)
        if self.config.stage5_use_cell_mmd:
            loss_cell_batch_mmd = self.cell_mmd_loss()
        else:
            loss_cell_batch_mmd = self._zero_loss()
        if self.has_condition and self.config.stage5_use_condition_batch_mmd:
            U_condition = self.condition_tensor(label_mask=False).reshape(
                self.n_cells,
                self.k_cell * self.n_condition_factors,
            )
            loss_condition_batch_mmd = sampled_condition_batch_ref_mmd_loss(
                U_condition,
                self.condition_labels,
                self.batch_labels,
                max_cells_per_group=self.config.mmd_cells_per_batch,
            )
        else:
            loss_condition_batch_mmd = self._zero_loss()
        loss_condition_entropy = self.condition_entropy_loss()
        total = (
            loss_rec
            + self.config.lambda_cell_batch_mmd * loss_cell_batch_mmd
            + self.config.lambda_stage5_condition_batch_mmd
            * loss_condition_batch_mmd
            + self.config.lambda_stage5_condition_entropy
            * loss_condition_entropy
        )
        return total, {
            "loss": total.detach(),
            "loss_rec": loss_rec.detach(),
            "loss_cell_batch_mmd": loss_cell_batch_mmd.detach(),
            "loss_condition_batch_mmd": loss_condition_batch_mmd.detach(),
            "loss_condition_entropy": loss_condition_entropy.detach(),
        }

    @torch.no_grad()
    def prune_batch_factors(
        self,
        threshold: Optional[float] = None,
        min_cell_factors: Optional[int] = None,
    ) -> Dict[str, object]:
        """Remove batch-associated U_cell columns and matching A_cell columns."""

        if self.current_stage != 1:
            raise RuntimeError("Batch pruning must be performed after stage 1.")
        threshold = (
            self.config.batch_score_threshold if threshold is None else threshold
        )
        min_keep = (
            self.config.min_cell_factors
            if min_cell_factors is None
            else int(min_cell_factors)
        )
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("threshold must be in [0, 1].")
        if not 1 <= min_keep <= self.k_cell:
            raise ValueError("min_cell_factors must be in [1, current k_cell].")

        U_cell = self.cell_embedding()
        scores = batch_eta_squared(U_cell, self.batch_labels)
        keep = scores < threshold
        if int(keep.sum()) < min_keep:
            keep = torch.zeros_like(keep)
            keep[torch.argsort(scores)[:min_keep]] = True

        removed = ~keep
        old_original_ids = self.original_factor_ids.clone()
        self.Z_cell = Parameter(self.Z_cell[:, keep].detach().clone())
        self.A_cell = Parameter(self.A_cell[:, keep].detach().clone())
        self.original_factor_ids = old_original_ids[keep]
        self.k_cell = int(keep.sum().item())

        self.current_stage = 2

        self.pruning_report = {
            "threshold": float(threshold),
            "scores": scores.detach().cpu().tolist(),
            "kept_original_factor_ids": old_original_ids[keep].cpu().tolist(),
            "removed_original_factor_ids": old_original_ids[removed].cpu().tolist(),
            "n_factors_before": int(scores.numel()),
            "n_factors_after": self.k_cell,
        }
        return self.pruning_report

    def initialize_condition_factors(self) -> None:
        if self.current_stage != 2:
            raise RuntimeError("Initialize condition factors after batch pruning.")
        if not self.has_condition:
            self.current_stage = 3
            return
        self.Z_condition = Parameter(
            torch.zeros(
                self.n_cells,
                self.k_cell,
                self.n_condition_factors,
                device=self.device,
            )
        )
        self.A_condition = Parameter(
            torch.randn(
                self.n_genes,
                self.k_cell,
                self.n_condition_factors,
                device=self.device,
            )
        )
        self.current_stage = 3

    def initialize_batch_factors(self) -> None:
        if self.current_stage != 3:
            raise RuntimeError("Initialize batch factors after stage 3.")
        self.Z_batch = Parameter(
            torch.zeros(
                self.n_cells,
                self.k_cell,
                self.n_batches,
                device=self.device,
            )
        )
        self.A_batch = Parameter(
            torch.randn(
                self.n_genes,
                self.k_cell,
                self.n_batches,
                device=self.device,
            )
        )
        self.current_stage = 4

    def freeze_factor_matrices(self) -> None:
        self.A_cell.requires_grad_(False)
        if self.A_condition is not None:
            self.A_condition.requires_grad_(False)
        if self.A_batch is not None:
            self.A_batch.requires_grad_(False)
        self.current_stage = 5

    @torch.no_grad()
    def export_embeddings(self) -> Dict[str, Tensor]:
        if self.current_stage >= 5 and self.Z_batch is not None:
            stage = 5
            label_mask = False
        elif self.Z_batch is not None:
            stage = 4
            label_mask = True
        elif self.Z_condition is not None:
            stage = 3
            label_mask = True
        else:
            stage = 1
            label_mask = True
        result = {
            "U_all": self.normalized_embedding(
                stage,
                label_mask=label_mask,
            ).detach().cpu(),
            "U_cell": self.cell_embedding().detach().cpu(),
            "A_all": self.normalized_factor_matrix(stage).detach().cpu(),
            "cell_factor_ids": self.cell_factor_ids().detach().cpu(),
            "original_factor_ids": self.original_factor_ids.detach().cpu(),
            "condition_factor_names": list(self.condition_factor_names),
        }
        if self.Z_condition is not None:
            result["U_condition"] = self.condition_tensor(
                label_mask=label_mask,
            ).detach().cpu()
        if self.Z_batch is not None:
            result["U_batch"] = self.batch_tensor(
                label_mask=label_mask,
            ).detach().cpu()
        return result
