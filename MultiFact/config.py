from dataclasses import asdict, dataclass, field


@dataclass
class MultiFactConfig:
    """Configuration for the four-stage MultiFact training procedure."""

    k_cell: int = 12
    stage1_iters: int = 1000
    stage3_iters: int = 1000
    stage4_iters: int = 1000
    stage5_iters: int = 0
    lr: float = 1e-3

    batch_score_threshold: float = 0.20
    min_cell_factors: int = 2

    lambda_condition_ce: float = 0.1
    lambda_batch_ce: float = 0.1
    lambda_cell_batch_mmd: float = 0.1
    lambda_condition_batch_mmd: float = 0.1
    lambda_stage5_condition_batch_mmd: float = 0.1
    lambda_stage5_condition_entropy: float = 0.0
    stage5_freeze_u_cell: bool = False
    stage5_freeze_u_batch: bool = False
    stage5_use_cell_mmd: bool = True
    stage5_use_condition_batch_mmd: bool = True
    cell_mmd_mode: str = "global"
    mmd_cells_per_batch: int = 64
    lambda_a_cell_condition_overlap: float = 0.1
    lambda_a_cell_batch_overlap: float = 0.1
    lambda_a_condition_batch_overlap: float = 0.1
    lambda_a_condition_within_cell_overlap: float = 0.0
    open_condition_factor_indices: list[int] = field(default_factory=list)
    shared_condition_factor_indices: list[list[int]] = field(default_factory=list)

    use_amp: bool = True
    verbose: bool = False
    history_interval: int = 10

    def validate(self) -> None:
        if self.k_cell <= 0:
            raise ValueError("k_cell must be positive.")
        if min(
            self.stage1_iters,
            self.stage3_iters,
            self.stage4_iters,
            self.stage5_iters,
        ) < 0:
            raise ValueError("Stage iteration counts must be non-negative.")
        if self.lr <= 0:
            raise ValueError("lr must be positive.")
        if not 0.0 <= self.batch_score_threshold <= 1.0:
            raise ValueError("batch_score_threshold must be in [0, 1].")
        if not 1 <= self.min_cell_factors <= self.k_cell:
            raise ValueError("min_cell_factors must be in [1, k_cell].")
        nonnegative_weights = (
            self.lambda_condition_ce,
            self.lambda_batch_ce,
            self.lambda_cell_batch_mmd,
            self.lambda_condition_batch_mmd,
            self.lambda_stage5_condition_batch_mmd,
            self.lambda_stage5_condition_entropy,
            self.lambda_a_cell_condition_overlap,
            self.lambda_a_cell_batch_overlap,
            self.lambda_a_condition_batch_overlap,
            self.lambda_a_condition_within_cell_overlap,
        )
        if any(weight < 0 for weight in nonnegative_weights):
            raise ValueError("Loss weights must be non-negative.")
        if any(int(index) < 0 for index in self.open_condition_factor_indices):
            raise ValueError("open_condition_factor_indices must be non-negative.")
        for group in self.shared_condition_factor_indices:
            if len(group) < 2:
                raise ValueError(
                    "Every shared condition factor group must contain at least "
                    "two condition indices."
                )
            if any(int(index) < 0 for index in group):
                raise ValueError(
                    "shared_condition_factor_indices must contain non-negative "
                    "condition indices."
                )
        if self.cell_mmd_mode not in {"global", "within_cell_factor"}:
            raise ValueError(
                "cell_mmd_mode must be 'global' or 'within_cell_factor'."
            )
        if self.mmd_cells_per_batch <= 0:
            raise ValueError("mmd_cells_per_batch must be positive.")
        if self.history_interval <= 0:
            raise ValueError("history_interval must be positive.")

    def to_dict(self):
        return asdict(self)
