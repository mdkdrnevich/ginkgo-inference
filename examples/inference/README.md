# 1D Inference Example Guide (Ginkgo + Trellis)

This example demonstrates an end-to-end workflow:

1. Generate an observed jet dataset with Ginkgo.
2. Evaluate Trellis likelihoods on a 1D grid of QCD splitting-rate values $\lambda$.
3. Analyze the likelihood surface and infer the generating parameter in `inference.ipynb`.

The commands below assume you are running from this directory:

```bash
cd examples/inference
```

## Directory Contents

- `make_dataset.sh`: convenience wrapper to generate observed data.
- `run_trellis_grid_1D.sh`: launches a local parallel Trellis grid scan.
- `run_trellis_grid_1D_MLE.sh`: optional second scan using an MLE-generated reference dataset.
- `run_grid_1D_local.py`: local multiprocess runner for 1D grid jobs.
- `trellis_1D.config`: extra CLI arguments passed to every Trellis job.
- `trellis_1D_MLE.config`: config for the optional MLE reference pass.
- `inference.ipynb`: notebook that loads outputs and performs inference/plots.

## Prerequisites

- Python environment with project dependencies installed (including `ginkgo` and `ClusterTrellis`).
- Ability to run scripts from the repository root layout (this example calls scripts under `../../scripts`).
- Write access to output directories configured in `trellis_1D.config` and `trellis_1D_MLE.config`.

## Workflow

### Step 1: Generate observed data

Run:

```bash
bash make_dataset.sh
```

This script runs:

```bash
python ../../scripts/make_ginkgo_dataset.py --output-dir ./data
```

With defaults, it generates a dataset equivalent to:

- `nsamples = 10000`
- `qcd-rate = 2.4` (encoded as `lambda_24` in filename)
- `pt-min = 30`
- `jet-p = 400`

Expected output file (pickle):

```text
./data/ginkgo_10000_jets_no_cuts_lambda_24_pt_min_30_jetp_400_with_perm.pkl
```

If you want a different observed dataset, run `make_ginkgo_dataset.py` directly with custom flags (for example `--qcd-rate`, `--nsamples`, `--output-name`).

### Step 2: Run Trellis over a 1D $\lambda$ grid

Run:

```bash
bash run_trellis_grid_1D.sh
```

This launches:

```bash
python run_grid_1D_local.py \
	--framework trellis \
	--script ../../scripts/run_trellis_grid_1D.py \
	--log-dir logs_trellis \
	--config-file trellis_1D.config
```

#### What happens under the hood

- `run_grid_1D_local.py` launches one task per grid index, default `0..149` (150 points).
- Each task runs `../../scripts/run_trellis_grid_1D.py <job_num> ...extra_args_from_config...`.
- Per-task logs are written to `logs_trellis/task_XXX.log`.
- By default, worker count is `CPU count - 1`.

#### Configure input dataset and output location

`trellis_1D.config` is read as whitespace-separated command-line args and appended to every task command.

Current config:

```text
./data/ginkgo_10000_jets_no_cuts_lambda_24_pt_min_30_jetp_400_with_perm
--outdir /mnt/g/ginkgo/trellis
```

Notes:

- The dataset path is passed **without** `.pkl` (the script appends it internally).
- `--outdir` should match where you want Trellis result files written.
- You can add more options in the config, e.g.:
	- `--n-lambda 150`
	- `--lambda-min 1.9`
	- `--lambda-max 3.05`
	- `--max-njets 10000`
	- `--nleaves-min ...`, `--nleaves-max ...`

Each task writes one result file named like:

```text
trellis_10000_jets_1D_lambda_<lambda_x1000>_ptcut_30_<job_num>_with_perm.pkl
```

in your configured `--outdir`.

#### Optional: run subset/debug jobs first

For a quick check before full scan:

```bash
python run_grid_1D_local.py \
	--framework trellis \
	--script ../../scripts/run_trellis_grid_1D.py \
	--log-dir logs_trellis \
	--config-file trellis_1D.config \
	--start 0 --end 4 --workers 2
```

### Step 3: Run the notebook and infer $\lambda$

Open and run all cells in:

```text
inference.ipynb
```

The notebook:

- Loads the observed dataset from `./data/...`.
- Loads Trellis grid outputs from `datadir` (default in notebook is `/mnt/g/ginkgo/trellis`).
- Builds likelihood surfaces over $\lambda$ and identifies the MLE bin/value.
- Optionally generates an MLE reference dataset and computes partition-ratio corrections.

If your Trellis output directory differs, update the notebook variable `datadir` accordingly.

## Optional MLE Reference Pass

After identifying an MLE point, you can run a second grid scan against an MLE-generated dataset:

```bash
bash run_trellis_grid_1D_MLE.sh
```

This uses `trellis_1D_MLE.config`, which should point to:

- the MLE-generated dataset path (without `.pkl`)
- an MLE output directory (for example `/mnt/g/ginkgo/trellis/MLE`)

The notebook includes sections that consume these files for partition-ratio and corrected-likelihood studies.

## Quick Validation Checklist

Before running the notebook end-to-end, verify:

1. Dataset file exists in `./data`.
2. `logs_trellis/` exists and contains per-task logs.
3. Trellis output directory contains a full (or expected) set of `trellis_..._with_perm.pkl` files.
4. Notebook `datadir` points to your actual Trellis output directory.

## Common Issues

- `FileNotFoundError` for dataset:
	- Confirm `make_dataset.sh` completed.
	- Confirm dataset base path in `trellis_1D.config` matches generated filename.
- Missing Trellis outputs for some bins:
	- Check corresponding `logs_trellis/task_XXX.log` files.
	- Re-run failed task range with `--start/--end`.
- Notebook cannot load results:
	- Update notebook `datadir` to your configured `--outdir`.
	- Ensure output files follow expected naming pattern from `run_trellis_grid_1D.py`.