# MultiFact

This repository contains a minimal implementation and runnable example for
MultiFact, a staged factorization model for separating cell, condition, and
batch factors in single-cell expression data.

## Contents

- `MultiFact/`: Python package with the model, losses, and training loop.
- `train.py`: runnable example script using the included simulated dataset.
- `sim.h5ad`: simulated single-cell dataset used by the example script.

## Installation

Create a Python environment and install the dependencies:

```bash
pip install -r requirements.txt
```

The code uses PyTorch. Install a CUDA-enabled PyTorch build separately if your
environment needs a specific CUDA version.

## Quick Smoke Test

Run a minimal end-to-end check:

```bash
MULTIFACT_FULL_STAGE_ITERS=1,1,1,1 \
MULTIFACT_SKIP_STAGE5_SWEEP=1 \
MULTIFACT_OUTPUT_DIR=/tmp/multifact_smoketest \
MULTIFACT_OUTPUT_SUFFIX=_smoketest \
python train.py
```

This reads `sim.h5ad`, runs stages 1, 3, and 4 for one iteration each, and writes
outputs to `/tmp/multifact_smoketest`.

## Full Example Run

```bash
python train.py
```

By default, the full example uses substantially more iterations. You can adjust
the run without editing code:

```bash
MULTIFACT_FULL_STAGE_ITERS=100,100,100,100 python train.py
```

Useful environment variables:

- `MULTIFACT_INPUT_H5AD`: input `.h5ad` file path.
- `MULTIFACT_OUTPUT_DIR`: output directory.
- `MULTIFACT_N_TRAIN_GENES`: number of genes to use, or `all`.
- `MULTIFACT_FULL_STAGE_ITERS`: four comma-separated iteration counts.
- `MULTIFACT_SKIP_STAGE5_SWEEP`: set to `1` to skip the stage-5 sweep.
- `MULTIFACT_OUTPUT_SUFFIX`: suffix appended to output names.

## Package Usage

See `MultiFact/README.md` for direct Python API usage.
