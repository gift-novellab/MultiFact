from copy import deepcopy
import os
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import torch
import torch.nn.functional as F


def find_project_root(start: Path | None = None) -> Path:
    start = Path.cwd() if start is None else Path(start).resolve()
    for candidate in (start, *start.parents):
        if (candidate / "MultiFact" / "config.py").is_file():
            return candidate
        nested = candidate / "MultiFact"
        if (nested / "MultiFact" / "config.py").is_file():
            return nested
    raise FileNotFoundError("Could not find the MultiFact project root.")


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = find_project_root(SCRIPT_DIR)
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from MultiFact import MultiFactConfig, MultiFactModel, fit_multifact  # noqa: E402
from MultiFact.train import _train_stage  # noqa: E402
from MultiFact.utils import prepare_target  # noqa: E402


torch.manual_seed(0)
np.random.seed(0)

INPUT_H5AD = Path(os.environ.get("MULTIFACT_INPUT_H5AD", SCRIPT_DIR / "sim.h5ad"))
if not INPUT_H5AD.is_absolute():
    INPUT_H5AD = SCRIPT_DIR / INPUT_H5AD

OUTPUT_DIR = Path(os.environ.get("MULTIFACT_OUTPUT_DIR", SCRIPT_DIR / "MultiFact_results"))
if not OUTPUT_DIR.is_absolute():
    OUTPUT_DIR = SCRIPT_DIR / OUTPUT_DIR
CONDITION_KEY = "condition"
BATCH_KEY = "batch"
CELLTYPE_KEY = "cell_type"


def parse_n_train_genes(default=700):
    value = os.environ.get("MULTIFACT_N_TRAIN_GENES")
    if value is None or value == "":
        return default
    value = value.strip().lower()
    if value in {"all", "none", "full"}:
        return None
    return int(value)


N_TRAIN_GENES = parse_n_train_genes()
OUTPUT_NAME_SUFFIX = os.environ.get("MULTIFACT_OUTPUT_SUFFIX", "")

K_CELL = int(os.environ.get("MULTIFACT_K_CELL", "3"))
QUICK_RUN = False
QUICK_STAGE_ITERS = (20, 20, 20, 20)
FULL_STAGE_ITERS = (4000, 8000, 20000, 10000)
if os.environ.get("MULTIFACT_FULL_STAGE_ITERS"):
    FULL_STAGE_ITERS = tuple(
        int(value.strip())
        for value in os.environ["MULTIFACT_FULL_STAGE_ITERS"].split(",")
    )
    if len(FULL_STAGE_ITERS) != 4:
        raise ValueError("MULTIFACT_FULL_STAGE_ITERS must contain four comma-separated integers.")
STAGE_ITERS = QUICK_STAGE_ITERS if QUICK_RUN else FULL_STAGE_ITERS
LEARNING_RATE = 1e-3

ENTROPY_WEIGHTS = [1e-3]
SKIP_STAGE5_SWEEP = os.environ.get("MULTIFACT_SKIP_STAGE5_SWEEP", "0") in {"1", "true", "True"}
OUTPUT_BASENAME = os.environ.get("MULTIFACT_OUTPUT_BASENAME", INPUT_H5AD.stem)

# Relax the reconstruction condition mask for selected condition factors.
# Use labels from adata.obs[CONDITION_KEY], e.g. ["condition_0"],
# or encoded indices, e.g. [0].
OPEN_CONDITION_FACTOR_LABELS = []
OPEN_CONDITION_FACTOR_INDICES = []
SHARED_CONDITION_FACTOR_INDICES = []

BATCH_SCORE_THRESHOLD = 0.20
MIN_CELL_FACTORS = 2
LAMBDA_CONDITION_CE = 0.4
LAMBDA_BATCH_CE = 0.4
LAMBDA_CELL_BATCH_MMD = 1.0
LAMBDA_CONDITION_BATCH_MMD = 1.0
LAMBDA_STAGE5_CONDITION_BATCH_MMD = 1.0
CELL_MMD_MODE = "within_cell_factor"
MMD_CELLS_PER_BATCH = 128
LAMBDA_A_CELL_CONDITION_OVERLAP = 1.0
LAMBDA_A_CELL_BATCH_OVERLAP = 0.0
LAMBDA_A_CONDITION_BATCH_OVERLAP = 1.0
LAMBDA_A_CONDITION_WITHIN_CELL_OVERLAP = 1.0

STAGE5_FREEZE_U_CELL = True
STAGE5_FREEZE_U_BATCH = True
STAGE5_USE_CELL_MMD = False
STAGE5_USE_CONDITION_BATCH_MMD = False


def resolve_condition_factor_indices(labels, indices, categories):
    resolved = []
    for label in labels:
        label = str(label)
        if label not in categories:
            raise ValueError(
                f"Unknown open condition label {label!r}. "
                f"Available labels: {categories}"
            )
        resolved.append(categories.index(label))
    for index in indices:
        index = int(index)
        if index < 0 or index >= len(categories):
            raise ValueError(
                f"Open condition index {index} is outside "
                f"[0, {len(categories) - 1}]."
            )
        resolved.append(index)
    return list(dict.fromkeys(resolved))


def history_to_arrays(rows):
    keys = sorted({key for row in rows for key in row})
    return {
        key: np.asarray([row.get(key, np.nan) for row in rows], dtype=np.float32)
        for key in keys
    }


def weight_tag(weight: float) -> str:
    return f"{weight:.0e}".replace("-", "m")


def display_path(path: Path) -> str:
    path = Path(path)
    try:
        return str(path.resolve().relative_to(SCRIPT_DIR))
    except ValueError:
        return str(path)


def save_training_plots(history_df: pd.DataFrame, output_loss_png: Path) -> None:
    plot_df = history_df.copy()
    plot_df["global_step"] = np.arange(len(plot_df))
    loss_columns = [
        column
        for column in [
            "loss",
            "loss_rec",
            "loss_condition_ce",
            "loss_batch_ce",
            "loss_cell_batch_mmd",
            "loss_condition_batch_mmd",
            "loss_condition_entropy",
            "loss_a_cell_condition_overlap",
            "loss_a_cell_batch_overlap",
            "loss_a_condition_batch_overlap",
            "loss_a_condition_within_cell_overlap",
        ]
        if column in plot_df.columns
    ]
    if not loss_columns:
        return
    stage_colors = {
        1: "tab:blue",
        2: "tab:gray",
        3: "tab:orange",
        4: "tab:green",
        5: "tab:red",
    }
    fig, axes = plt.subplots(
        len(loss_columns),
        1,
        figsize=(10, max(3, 2.4 * len(loss_columns))),
        sharex=True,
    )
    if len(loss_columns) == 1:
        axes = [axes]
    for ax, column in zip(axes, loss_columns):
        for stage, stage_df in plot_df.groupby("stage"):
            stage_df = stage_df.dropna(subset=[column])
            if stage_df.empty:
                continue
            stage_int = int(stage)
            ax.plot(
                stage_df["global_step"],
                stage_df[column],
                label=f"stage {stage_int}",
                color=stage_colors.get(stage_int),
            )
        ax.set_ylabel(column)
        ax.grid(alpha=0.3)
        ax.legend(loc="best")
    axes[-1].set_xlabel("Recorded history step")
    fig.suptitle("MultiFact training losses", y=1.01)
    plt.tight_layout()
    fig.savefig(output_loss_png, dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_result(
    *,
    model: MultiFactModel,
    adata_train,
    history,
    config: MultiFactConfig,
    condition_categories,
    batch_categories,
    condition_codes,
    batch_codes,
    open_condition_factor_indices,
    output_prefix: Path,
) -> None:
    history_df = pd.DataFrame(history)
    history_csv = output_prefix.with_name(output_prefix.name + "_history.csv")
    loss_png = output_prefix.with_name(output_prefix.name + "_losses.png")
    h5ad_path = output_prefix.with_suffix(".h5ad")
    checkpoint_path = output_prefix.with_name(output_prefix.name + "_checkpoint.pt")

    history_df.to_csv(history_csv, index=False)
    save_training_plots(history_df, loss_png)

    outputs = model.export_embeddings()
    U_all = outputs["U_all"].numpy()
    U_cell = outputs["U_cell"].numpy()
    U_condition_tensor = outputs["U_condition"].numpy()
    U_batch_tensor = outputs["U_batch"].numpy()
    U_condition = U_condition_tensor.reshape(adata_train.n_obs, -1)
    U_batch = U_batch_tensor.reshape(adata_train.n_obs, -1)
    A_all = outputs["A_all"].numpy()
    cell_factor_ids = outputs["cell_factor_ids"].numpy()
    hard_mask = F.one_hot(
        outputs["cell_factor_ids"], num_classes=model.k_cell
    ).numpy().astype(np.float32)

    A_cell = torch.softmax(model.A_cell.detach(), dim=0).cpu().numpy()
    A_condition = torch.softmax(
        model.A_condition.detach().reshape(adata_train.n_vars, -1), dim=0
    ).cpu().numpy()
    A_batch = torch.softmax(
        model.A_batch.detach().reshape(adata_train.n_vars, -1), dim=0
    ).cpu().numpy()

    adata_out = adata_train.copy()
    adata_out.obsm["multifact_embedding"] = U_all
    adata_out.obsm["multifact_U_cell"] = U_cell
    adata_out.obsm["multifact_U_condition"] = U_condition
    adata_out.obsm["multifact_U_batch"] = U_batch
    adata_out.obsm["multifact_cell_factor_mask"] = hard_mask
    adata_out.obs["multifact_cell_factor"] = pd.Categorical(cell_factor_ids.astype(str))
    adata_out.varm["multifact_A"] = A_all
    adata_out.varm["multifact_A_cell"] = A_cell
    adata_out.varm["multifact_A_condition"] = A_condition
    adata_out.varm["multifact_A_batch"] = A_batch
    adata_out.uns["multifact"] = {
        "config": config.to_dict(),
        "training_history": history_to_arrays(history),
        "batch_factor_scores": np.asarray(
            model.pruning_report["scores"], dtype=np.float32
        ),
        "kept_original_factor_ids": np.asarray(
            model.pruning_report["kept_original_factor_ids"], dtype=np.int64
        ),
        "removed_original_factor_ids": np.asarray(
            model.pruning_report["removed_original_factor_ids"], dtype=np.int64
        ),
        "initial_k_cell": int(model.initial_k_cell),
        "retained_k_cell": int(model.k_cell),
        "condition_key": CONDITION_KEY,
        "batch_key": BATCH_KEY,
        "celltype_key": CELLTYPE_KEY if CELLTYPE_KEY in adata_out.obs else "None",
        "condition_categories": np.asarray(condition_categories, dtype=str),
        "open_condition_factor_labels": np.asarray(
            [condition_categories[index] for index in open_condition_factor_indices],
            dtype=str,
        ),
        "open_condition_factor_indices": np.asarray(
            open_condition_factor_indices,
            dtype=np.int64,
        ),
        "shared_condition_factor_indices": np.asarray([], dtype=str),
        "shared_condition_factor_labels": np.asarray([], dtype=str),
        "batch_categories": np.asarray(batch_categories, dtype=str),
        "embedding_layout": "[U_cell, flatten(U_condition), flatten(U_batch)]",
        "train_gene_names": np.asarray(adata_train.var_names.astype(str), dtype=str),
    }
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "config": config.to_dict(),
        "n_cells": model.n_cells,
        "n_genes": model.n_genes,
        "n_conditions": model.n_conditions,
        "n_batches": model.n_batches,
        "initial_k_cell": model.initial_k_cell,
        "retained_k_cell": model.k_cell,
        "pruning_report": model.pruning_report,
        "condition_categories": condition_categories,
        "open_condition_factor_labels": [
            condition_categories[index] for index in open_condition_factor_indices
        ],
        "open_condition_factor_indices": open_condition_factor_indices,
        "shared_condition_factor_indices": [],
        "shared_condition_factor_labels": [],
        "batch_categories": batch_categories,
        "condition_codes": condition_codes,
        "batch_codes": batch_codes,
        "train_gene_names": adata_train.var_names.astype(str).tolist(),
    }
    adata_out.write_h5ad(h5ad_path)
    torch.save(checkpoint, checkpoint_path)
    print("Saved:", display_path(h5ad_path))
    print("Saved:", display_path(history_csv))
    print("Saved:", display_path(loss_png))
    print("Saved:", display_path(checkpoint_path))


def main() -> None:
    print("Project root:", display_path(PROJECT_ROOT))
    print("Input:", display_path(INPUT_H5AD))
    print("Output directory:", display_path(OUTPUT_DIR))
    print("Train genes:", "all" if N_TRAIN_GENES is None else N_TRAIN_GENES)
    print("Output name suffix:", OUTPUT_NAME_SUFFIX or "(none)")
    print("Stage iterations:", STAGE_ITERS)
    print("Entropy weights:", ENTROPY_WEIGHTS)
    print("PyTorch:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())
    if not INPUT_H5AD.exists():
        raise FileNotFoundError(f"Input h5ad does not exist: {display_path(INPUT_H5AD)}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    adata = sc.read_h5ad(INPUT_H5AD)
    required_obs = [CONDITION_KEY, BATCH_KEY]
    missing_obs = [key for key in required_obs if key not in adata.obs]
    if missing_obs:
        raise KeyError(f"Missing required adata.obs columns: {missing_obs}")
    if adata.obs[required_obs].isna().any().any():
        raise ValueError("Condition and batch labels must not contain missing values.")

    if N_TRAIN_GENES is None:
        adata_train = adata.copy()
    else:
        adata_train = adata[:, : min(int(N_TRAIN_GENES), adata.n_vars)].copy()

    condition_cat = adata_train.obs[CONDITION_KEY].astype("category")
    batch_cat = adata_train.obs[BATCH_KEY].astype("category")
    condition_codes = condition_cat.cat.codes.to_numpy(dtype=np.int64)
    batch_codes = batch_cat.cat.codes.to_numpy(dtype=np.int64)
    condition_categories = condition_cat.cat.categories.astype(str).tolist()
    batch_categories = batch_cat.cat.categories.astype(str).tolist()
    open_condition_factor_indices = resolve_condition_factor_indices(
        OPEN_CONDITION_FACTOR_LABELS,
        OPEN_CONDITION_FACTOR_INDICES,
        condition_categories,
    )

    X = adata_train.X
    if hasattr(X, "toarray"):
        X = X.toarray()
    X = torch.as_tensor(np.asarray(X), dtype=torch.float32)
    if torch.any(X < 0):
        raise ValueError("The expression matrix must be non-negative.")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    base_config = MultiFactConfig(
        k_cell=K_CELL,
        stage1_iters=STAGE_ITERS[0],
        stage3_iters=STAGE_ITERS[1],
        stage4_iters=STAGE_ITERS[2],
        stage5_iters=0,
        lr=LEARNING_RATE,
        batch_score_threshold=BATCH_SCORE_THRESHOLD,
        min_cell_factors=MIN_CELL_FACTORS,
        lambda_condition_ce=LAMBDA_CONDITION_CE,
        lambda_batch_ce=LAMBDA_BATCH_CE,
        lambda_cell_batch_mmd=LAMBDA_CELL_BATCH_MMD,
        lambda_condition_batch_mmd=LAMBDA_CONDITION_BATCH_MMD,
        lambda_stage5_condition_batch_mmd=LAMBDA_STAGE5_CONDITION_BATCH_MMD,
        lambda_stage5_condition_entropy=0.0,
        stage5_freeze_u_cell=STAGE5_FREEZE_U_CELL,
        stage5_freeze_u_batch=STAGE5_FREEZE_U_BATCH,
        stage5_use_cell_mmd=STAGE5_USE_CELL_MMD,
        stage5_use_condition_batch_mmd=STAGE5_USE_CONDITION_BATCH_MMD,
        cell_mmd_mode=CELL_MMD_MODE,
        mmd_cells_per_batch=MMD_CELLS_PER_BATCH,
        lambda_a_cell_condition_overlap=LAMBDA_A_CELL_CONDITION_OVERLAP,
        lambda_a_cell_batch_overlap=LAMBDA_A_CELL_BATCH_OVERLAP,
        lambda_a_condition_batch_overlap=LAMBDA_A_CONDITION_BATCH_OVERLAP,
        lambda_a_condition_within_cell_overlap=LAMBDA_A_CONDITION_WITHIN_CELL_OVERLAP,
        open_condition_factor_indices=open_condition_factor_indices,
        shared_condition_factor_indices=SHARED_CONDITION_FACTOR_INDICES,
        use_amp=True,
        verbose=True,
        history_interval=1 if QUICK_RUN else 10,
    )

    base_model = MultiFactModel(
        n_cells=adata_train.n_obs,
        n_genes=adata_train.n_vars,
        condition_labels=torch.as_tensor(condition_codes),
        batch_labels=torch.as_tensor(batch_codes),
        n_conditions=len(condition_categories),
        n_batches=len(batch_categories),
        config=base_config,
        device=device,
    )
    print("Device:", device)
    print("Conditions:", condition_categories)
    print("Open condition factor indices:", open_condition_factor_indices)
    print("Batches:", batch_categories)
    print("Training base model through stage 4.")
    base_history = fit_multifact(base_model, X)
    P = prepare_target(X, device=base_model.device)

    base_prefix = OUTPUT_DIR / f"{OUTPUT_BASENAME}_stage4_base{OUTPUT_NAME_SUFFIX}"
    save_result(
        model=base_model,
        adata_train=adata_train,
        history=base_history,
        config=base_config,
        condition_categories=condition_categories,
        batch_categories=batch_categories,
        condition_codes=condition_codes,
        batch_codes=batch_codes,
        open_condition_factor_indices=open_condition_factor_indices,
        output_prefix=base_prefix,
    )

    if SKIP_STAGE5_SWEEP:
        print("\nSkipping stage 5 sweep because MULTIFACT_SKIP_STAGE5_SWEEP is enabled.")
        return

    summary_rows = []
    for entropy_weight in ENTROPY_WEIGHTS:
        print(f"\nRunning stage 5 with entropy weight {entropy_weight:g}.")
        model = deepcopy(base_model)
        model.config.stage5_iters = STAGE_ITERS[3]
        model.config.lambda_stage5_condition_entropy = float(entropy_weight)
        model.config.stage5_freeze_u_cell = STAGE5_FREEZE_U_CELL
        model.config.stage5_freeze_u_batch = STAGE5_FREEZE_U_BATCH
        model.config.stage5_use_cell_mmd = STAGE5_USE_CELL_MMD
        model.config.stage5_use_condition_batch_mmd = STAGE5_USE_CONDITION_BATCH_MMD
        model.freeze_factor_matrices()
        stage5_history = _train_stage(
            model,
            P,
            stage=5,
            max_iters=model.config.stage5_iters,
        )
        history = [dict(row) for row in base_history] + stage5_history
        model.history = history

        tag = weight_tag(entropy_weight)
        prefix = OUTPUT_DIR / f"{OUTPUT_BASENAME}_stage5_entropy_{tag}{OUTPUT_NAME_SUFFIX}"
        save_result(
            model=model,
            adata_train=adata_train,
            history=history,
            config=model.config,
            condition_categories=condition_categories,
            batch_categories=batch_categories,
            condition_codes=condition_codes,
            batch_codes=batch_codes,
            open_condition_factor_indices=open_condition_factor_indices,
            output_prefix=prefix,
        )
        final_row = history[-1]
        summary_rows.append(
            {
                "entropy_weight": float(entropy_weight),
                "loss": final_row.get("loss", np.nan),
                "loss_rec": final_row.get("loss_rec", np.nan),
                "loss_condition_entropy": final_row.get(
                    "loss_condition_entropy", np.nan
                ),
                "loss_a_condition_within_cell_overlap": final_row.get(
                    "loss_a_condition_within_cell_overlap", np.nan
                ),
            }
        )

    summary = pd.DataFrame(summary_rows)
    summary_path = OUTPUT_DIR / f"stage5_entropy_sweep_summary{OUTPUT_NAME_SUFFIX}.csv"
    summary.to_csv(summary_path, index=False)
    print("\nSaved sweep summary:", display_path(summary_path))
    print(summary)


if __name__ == "__main__":
    main()
