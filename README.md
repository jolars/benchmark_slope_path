# Benchmark Repository for the SLOPE Path

[![Build Status](https://github.com/jolars/benchmark_slope_path/workflows/Tests/badge.svg)](https://github.com/jolars/benchmark_slope_path/actions)

This repository is dedicated algorithms that fit the
full Sorted L-One Penalized
Estimation (SLOPE) path, which consists in solving the following program:

$$\text{minimize}_{\beta_0, \beta \in \mathbb{R}^p} \left\( \tfrac{1}{2n} \Vert y - \beta_0 - X\beta \Vert^2_2 + \alpha J(\beta, \lambda)\right\)$$

for a sequence of values of $\alpha$, where

$$J(\beta, \lambda) = \sum_{j=1}^p \lambda_j \| \beta_{(j)}\|$$

with $\lambda_1 \geq \lambda_2 \geq ... \geq \lambda_p$ and $\|\beta_{(1)}\| \geq \|\beta_{(2)}\| \geq ... \geq \|\beta_{(p)}\|$
is the sorted $\ell_1$ norm.

## Install

This benchmark can be installed and run using the following commands:

```bash
pip install -U benchopt
git clone https://github.com/jolars/benchmark_slope_path
benchopt install ./benchmark_slope_path
benchopt run ./benchmark_slope_path  --config example_config
```

Apart from the problem, options can be passed to `benchopt run`, to restrict the benchmarks to some solvers or datasets, e.g.:

```bash
benchopt run ./benchmark_slope -s sortedl1 -d simulated --max-runs 10 --n-repetitions 5
```

Use `benchopt run -h` for more details about these options, or visit <https://benchopt.github.io/cli.html>.
