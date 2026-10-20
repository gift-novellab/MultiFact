# MultiFact

`MultiFact` implements a four-stage single-cell factorization for disentangling
cell, condition, and batch factors in single-cell expression data.

## Four stages

1. Learn only `U_cell` and `A_cell` with Fisher-Rao reconstruction loss.
2. Score every `U_cell` column by the fraction of its variance explained by
   batch labels (eta-squared). Remove factors above the configured threshold and
   remove the matching columns from `A_cell`.
3. Continue training the retained `U_cell`, assign every cell dynamically to
   `argmax(U_cell)`, and add one condition factor per
   `(cell factor, condition)`. The selected cell-factor block predicts condition
   with cross entropy. Reconstruction uses `[U_cell, U_condition]`.
4. Add one batch factor per `(cell factor, batch)`. The selected cell-factor
   block predicts batch with cross entropy. Reconstruction uses
   `[U_cell, U_condition, U_batch]`. Both `U_cell` and `U_condition` remain
   trainable. Stage 4 can also align `U_cell` and `U_condition` across batches
   with sampled batch MMD losses.

The cell-factor mask is recomputed from the current `U_cell`:

```text
cell_factor_id[i] = argmax_k U_cell[i, k]
gate[i, k] = 1[k == cell_factor_id[i]]
```

For reconstruction, condition and batch factors are additionally masked by the
observed condition or batch label. Cross entropy uses the raw logits from the
cell's selected cell-factor block before the observed-label mask.

## Minimal usage

```python
import torch

from MultiFact import MultiFactConfig, MultiFactModel, fit_multifact

X = torch.rand(1000, 600)
condition = torch.randint(0, 3, (1000,))
batch = torch.randint(0, 2, (1000,))

config = MultiFactConfig(
    k_cell=12,
    stage1_iters=1000,
    stage3_iters=1000,
    stage4_iters=1000,
    stage5_iters=0,
    batch_score_threshold=0.20,
    min_cell_factors=2,
    lambda_condition_ce=0.1,
    lambda_batch_ce=0.1,
    lambda_cell_batch_mmd=0.1,
    lambda_condition_batch_mmd=0.1,
    lambda_stage5_condition_batch_mmd=0.1,
    lambda_stage5_condition_entropy=0.0,
    stage5_freeze_u_cell=False,
    stage5_freeze_u_batch=False,
    stage5_use_cell_mmd=True,
    stage5_use_condition_batch_mmd=True,
    cell_mmd_mode="global",
    mmd_cells_per_batch=64,
    lambda_a_cell_condition_overlap=0.1,
    lambda_a_cell_batch_overlap=0.1,
    lambda_a_condition_batch_overlap=0.1,
    lambda_a_condition_within_cell_overlap=0.0,
    open_condition_factor_indices=[],
    verbose=True,
)

model = MultiFactModel(
    n_cells=X.shape[0],
    n_genes=X.shape[1],
    condition_labels=condition,
    batch_labels=batch,
    config=config,
)
history = fit_multifact(model, X)

outputs = model.export_embeddings()
print(model.pruning_report)
print(outputs["U_cell"].shape)
print(outputs["U_condition"].shape)  # cells x retained_K x conditions
print(outputs["U_batch"].shape)      # cells x retained_K x batches
```

For data with batches but no condition labels, omit `condition_labels` or pass
`condition_labels=None`. MultiFact then uses a single internal dummy condition
label for the existing MMD routines, so joint `(condition, batch)` MMD naturally
reduces to batch-group MMD without changing the MMD implementation:

```python
model = MultiFactModel(
    n_cells=X.shape[0],
    n_genes=X.shape[1],
    batch_labels=batch,
    config=config,
)
```

## Important behavior

- `U_cell` remains trainable in stages 3 and 4. Cell-factor assignments and
  masks are recomputed from its current `argmax`.
- `U_condition` remains trainable in stage 4.
- If no condition labels are provided, condition-specific factors are skipped.
  Stage 3 still applies the configured `U_cell` MMD, and stage 4 trains
  batch-specific factors.
- `open_condition_factor_indices` can relax the reconstruction label mask for
  selected condition factors. For example, if condition index `0` is healthy
  and index `1` is disease, setting `open_condition_factor_indices=[0]` lets
  disease cells also use the healthy condition factor during reconstruction,
  while the disease condition factor remains masked away from healthy cells.
  The condition cross-entropy target still uses the original observed labels,
  and stage-4 `U_condition` batch MMD still uses the original condition mask.
- `shared_condition_factor_indices` adds extra condition factors shared by
  specified condition groups. For example, if the encoded condition order is
  `[HV, Sepsis, Covid_CRITC]`, setting
  `shared_condition_factor_indices=[[1, 2]]` adds one extra factor per cell
  factor that can be used by Sepsis and Covid cells during reconstruction, but
  is masked out for HV cells. Cross entropy still predicts only the original
  condition labels, so the shared factor is learned through reconstruction and
  regularization rather than as an additional class.
- `lambda_a_condition_within_cell_overlap` penalizes overlap between
  `A_condition[:, k, c1]` and `A_condition[:, k, c2]` only within the same cell
  factor `k`. This includes original condition factors and any configured
  shared condition factors, but does not compare factors across different cell
  factors.
- If `stage5_iters > 0`, MultiFact runs an additional refinement stage after
  stage 4. All gene-factor matrices `A_cell`, `A_condition`, and `A_batch` are
  frozen; only the cell-side factors are optimized. Reconstruction uses
  unmasked `U_condition` and `U_batch`, while cross-entropy losses are disabled.
  By default, this keeps the previous behavior: `U_cell`, `U_condition`, and
  `U_batch` are optimized, with `U_cell` MMD and `U_condition` batch MMD active.
  Set `stage5_freeze_u_cell=True` to keep cell-factor assignments fixed, and
  set `stage5_freeze_u_batch=True` if the refinement should not update
  batch-side usages. With condition labels present, setting both to `True`
  makes stage 5 optimize `U_condition` only.
  Set `stage5_use_cell_mmd=False` and
  `stage5_use_condition_batch_mmd=False` to make stage 5 use reconstruction
  loss only. Set `lambda_stage5_condition_entropy > 0` to add a stage-5
  entropy penalty on each active cell factor's condition-factor distribution;
  this discourages binary `U_condition` usage across condition factors.
- Stages 3 and 4 compute optional MMD loss on row-normalized `U_cell`.
  `cell_mmd_mode="global"` groups cells by joint `(condition, batch)` and
  compares every valid joint group to the largest reference group.
  `cell_mmd_mode="within_cell_factor"` first splits cells by the current
  `argmax(U_cell)` hard factor assignment, then computes that same joint MMD
  inside each factor. Stage 4 also computes MMD on active flattened
  `U_condition` within each condition: the largest batch group in that condition
  is used as the reference, and every other batch group in the same condition is
  compared with it. Each group contributes at most `mmd_cells_per_batch` cells
  per iteration.
- Gene-factor overlap losses penalize shared high-weight genes between
  `A_cell` and `A_condition`, between `A_cell` and `A_batch`, and between
  `A_condition` and `A_batch`.
- `A_cell` and `A_condition` remain trainable without an explicit modification
  cost.
- Batch-factor selection is association-based. If biological cell types are
  confounded with batch, a true cell-type factor can also receive a high batch
  score. Inspect `model.pruning_report` and tune `batch_score_threshold`.
- At least `min_cell_factors` are retained, choosing the factors with the lowest
  batch scores when the threshold would remove too many.
