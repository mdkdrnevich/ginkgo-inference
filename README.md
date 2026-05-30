# Ginkgo Inference

Physics-informed probabilistic inference pipeline for jet clustering, combining synthetic event generation (Ginkgo) with exact/approximate hierarchical likelihood evaluation (ClusterTrellis).

This project explores how to recover latent physics parameters from simulated collider jets by scanning model likelihoods over a parameter grid and comparing the inferred optimum to known truth settings.

<!-- ## Why This Project Is Valuable

This repository demonstrates the ability to:

- Build reproducible scientific ML workflows from data generation to statistical inference.
- Integrate external research codebases into a unified pipeline.
- Scale the same experiment design from local multiprocessing to SLURM array jobs.
- Engineer parameter-sweep infrastructure and post-hoc analysis for model validation.

In short, it is an end-to-end example of applied research engineering: numerical modeling, automation, parallel execution, and analysis communication. -->

## Project Scope

The workflow implemented in this repository:

1. Generate jet-tree datasets with a configurable physics-inspired simulator.
2. Evaluate Trellis likelihoods over a scan of splitting-rate parameter values.
3. Aggregate likelihood surfaces and estimate best-fit parameters.
4. Compare inferred and truth configurations, including optional MLE-based reweighting analysis.

<!-- ## Technical Highlights

- Language and stack: Python, NumPy, PyTorch, Matplotlib, pickle-based scientific data flows.
- Core methods: likelihood scans, partition function evaluation, MAP hierarchy energy extraction.
- Parallelization patterns:
	- local: multiprocessing launcher for 1D parameter scans,
	- cluster: SLURM scripts for array jobs and batched minibatch experiments.
- Research tooling: notebook-driven interpretation and visualization tied to scripted experiment outputs. -->

## Repository Structure

```text
batch_jobs/                 # SLURM launchers for dataset generation and grid scans
examples/inference/         # End-to-end example (scripts, configs, notebook, sample logs/data)
scripts/                    # Core experiment scripts (dataset generation + trellis/ginkgo scans)
data/                       # Default location for generated datasets (can be overridden)
```

## Core Scripts

### Data generation

- `scripts/make_ginkgo_dataset.py`
	- Generates truth jet datasets with configurable sample count, leaf constraints, rates, and output naming.
	- Main useful arguments:
		- `--nsamples`, `--min-leaves`, `--max-leaves`, `--max-ntry`
		- `--pt-min`, `--qcd-rate`, `--qcd-mass`, `--jet-p`
		- `--output-dir`, `--output-name`

### Trellis inference

- `scripts/run_trellis_grid_1D.py`
	- Runs 1D scan over lambda for a fixed `pt_cut`.
	- Each job processes one lambda index and writes one result file.
	- Supports configurable scan bounds and output directory.

- `scripts/run_trellis.py`
	- Legacy 2D-style grid script (lambda and pt-cut mesh) used in earlier workflows.

- `scripts/run_trellis_minibatches.py`
	- Minibatch-based Trellis runs for repeated sampling/robustness studies.

### Ginkgo scan baselines

- `scripts/run_ginkgo_grid_1D.py`
	- Generates 1D baseline jet samples across lambda values.

- `scripts/run_grid.py`
	- Legacy 2D baseline generation script.

## Quick Start

### 1. Clone this repository

```bash
git clone https://github.com/mdkdrnevich/ginkgo-inference.git
cd ginkgo-inference
```

### 2. Install dependencies

This project depends on both this codebase and external research libraries:

- ClusterTrellis: https://github.com/SebastianMacaluso/ClusterTrellis
- Ginkgo: https://github.com/SebastianMacaluso/ginkgo

In practice, create the project environment from `environment.yml` (for example, `conda env create -f environment.yml`) and follow the external repository installation instructions above for ClusterTrellis and Ginkgo.

### 3. Run the end-to-end example

```bash
cd examples/inference
bash make_dataset.sh
bash run_trellis_grid_1D.sh
```

Then open `inference.ipynb` and execute all cells to:

- load truth jets and Trellis grid outputs,
- filter invalid likelihood entries,
- estimate MLE in the scanned lambda range,
- compare and visualize agreement metrics.

## Local Execution Pattern

The local runner supports parallel task execution and reproducible job slicing:

```bash
cd examples/inference
python run_grid_1D_local.py \
	--framework trellis \
	--script ../../scripts/run_trellis_grid_1D.py \
	--log-dir logs_trellis \
	--config-file trellis_1D.config \
	--start 0 --end 149
```

For quick validation/debug:

```bash
python run_grid_1D_local.py \
	--framework trellis \
	--script ../../scripts/run_trellis_grid_1D.py \
	--log-dir logs_trellis \
	--config-file trellis_1D.config \
	--start 0 --end 4 --workers 2
```

## Cluster Execution Pattern (SLURM)

The `batch_jobs/` directory includes templates for HPC runs using Singularity and SLURM arrays:

- `run-trellis-grid-1D.s`
- `run-ginkgo-grid-1D.s`
- `run-trellis-grid.s`
- `run-ginkgo-grid.s`
- `run-trellis-minibatches-grid.s`
- `run-make-ginkgo-dataset.s`

These scripts provide practical examples of resource requests, array indexing, containerized environment setup, and result/log routing.

## Output Artifacts

Typical outputs include:

- generated jet datasets (`.pkl`),
- per-parameter Trellis result files (`.pkl`) containing likelihood arrays and metadata,
- histogram arrays (`.npy`) for leaf-distribution diagnostics,
- task logs for each local or SLURM job.

## Notes on Reproducibility

- Many scripts assume absolute output paths (for example `/mnt/g/ginkgo/...` or `/scratch/...`).
- Use CLI flags and config files to redirect outputs to your environment.
- The example configs in `examples/inference/` are the recommended starting point.

## Research Context

- Paper: https://ml4physicalsciences.github.io/2022/files/NeurIPS_ML4PS_2022_32.pdf
- Presentation: https://indico.cern.ch/event/980214/contributions/4413534/