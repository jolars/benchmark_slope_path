# Benchmark Repository for the SLOPE Path

[![Build Status](https://github.com/jolars/benchmark_slope_path/workflows/Tests/badge.svg)](https://github.com/jolars/benchmark_slope_path/actions)

This repository is dedicated to algorithms that fit the full Sorted L-One Penalized Estimation (SLOPE) path, which consists in solving the following program:

$$\text{minimize}_{\beta_0 \in \mathbb{R}, \beta \in \mathbb{R}^p} \bigg\{ \tfrac{1}{2n} \Vert y - \beta_0 - X\beta \Vert^2_2 + \alpha J(\beta, \lambda)\bigg\}$$

for a sequence of values of $\alpha$, where

$$J(\beta, \lambda) = \sum_{j=1}^p \lambda_j \| \beta_{(j)}\|$$

is the sorted $\ell_1$ norm, with $\lambda_1 \geq \lambda_2 \geq ... \geq \lambda_p$ and $\|\beta_{(1)}\| \geq \|\beta_{(2)}\| \geq ... \geq \|\beta_{(p)}\|$.

We note $n$ (or n_samples) the number of samples and $p$ (or n_features) the number of features.
We also have that $X\in \mathbb{R}^{n\times p}$ and $y\in \mathbb{R}^n$.

## Installing Benchopt

This benchmark relies on benchopt, a generic framework for running numerical benchmarks.
The recommended way to use benchopt is within a conda environment. So, begin by creating and activating
a new conda environment and install benchopt in it:

```bash
conda create -n benchopt python
conda activate benchopt
pip install -U benchopt
```

## Installing the Benchmark

To install the benchmark, clone this repository and move to its folder:

```bash
git clone https://github.com/jolars/benchmark_slope_path
cd benchmark_slope_path/
```

To install the dependencies for the solvers and datasets for the benchmark,
first make sure that you have activated the conda environment where benchopt is
installed. Then, you can either install all the dependencies with:

```bash
benchopt install .
```

Or you can install only a subset of solvers by specifying them with the `-s` option.

```bash
benchopt install . -s solver_name -d dataset_name
```

## Running the Benchmark

To run the benchmark, simply use the `benchopt run` command:

```bash
benchopt run .
```

By default, all solvers and datasets are run. You can restrict the benchmark to some solvers or datasets, e.g.:

```bash
benchopt run -s solver_name -d simulated --max-runs 10 --n-repetitions 5
```

You can also specify a YAML configuration file to set the parameters of the benchmark.
An example config is provided in `example_config.yml`.

```bash
benchopt run --config example_config.yml .
```

Use `benchopt run -h` for more details about these options, or visit <https://benchopt.github.io/api.html>.
