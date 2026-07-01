# `perf/` — performance & density-stability benchmark

A reproducible benchmark across the package's inference methods, with two jobs:

1. **Track performance** (wall-clock time and allocations) of each MCMC/VI method as the code evolves.
2. **Validate that density estimates don't drift** when algorithmic changes are made — the exception
   being bug fixes, which are *expected* to change the output.

## What it runs

- **Densities:** the first 10 Marron–Wand (1992) normal-mixture test densities (`src/marron_wand.jl`).
- **Sample sizes:** **400 and 2000** synthetic draws per density in the default `reduced` run. (The
  heavier `full` mode uses 250, 1000, 5000; `smoke` uses 200.)
- **Methods** (`src/methods.jl`): MCMC for RandomBernsteinPoly, BSplineMixture, HistSmoother,
  FiniteGaussianMixture, RandomFiniteGaussianMixture, PitmanYorMixture; VI for all of those except
  RandomBernsteinPoly.
- **Per cell:** `reduced` runs **5 chains**, `full` runs 10 — each of 10 000 non-burn-in draws + 1 000
  burn-in, except `RandomBernsteinPoly` and `HistSmoother-MCMC`, the slow samplers, which are capped at
  5 000 non-burn-in (per-method `max_samples` in `src/methods.jl`). VI: one fit + N Monte-Carlo sample
  sets of `n_mc_vi` draws (N = number of chains).
- **RNG:** all synthetic data *and* all chains are driven by `StableRNGs`, keyed purely by
  (density, n, chain), so runs are bit-reproducible across machines and Julia versions.

### Modes

| mode | densities | sample sizes | chains | non-burn-in | use |
|------|-----------|--------------|--------|-------------|-----|
| `reduced` (default) | all 10 | 400, 2000 | 5 | 10 000 (5 000 for RBP / HistSmoother) | the standard run / CI |
| `full` | all 10 | 250, 1000, 5000 | 10 | 10 000 (5 000 capped) | exhaustive; multi-hour |
| `smoke` | 2 | 200 | 3 | 250 | ~1 min end-to-end sanity check |

## Usage

```bash
julia perf/setup.jl                                      # one-time: build the perf environment
julia -t auto --project=perf perf/benchmark.jl smoke     # ~1 min sanity run (2 densities, tiny settings)
julia -t auto --project=perf perf/benchmark.jl reduced   # DEFAULT: 10 densities, n ∈ {400,2000}, 5 chains
julia -t auto --project=perf perf/benchmark.jl full      # exhaustive (n up to 5000, 10 chains; multi-hour)
```

`reduced` is the default (running `benchmark.jl` with no mode argument uses it) and is what the GitHub
Actions workflow runs. Results are written incrementally to `perf/results/<mode>/`:
- `timings.csv` — `method,kind,regression,density,n,min_time_s,alloc_MiB,n_samples,n_burnin,n_chains`
  (`n_samples` = the effective non-burn-in draw count, which is 5 000 for the two capped methods)
- `densities.csv` — `method,kind,regression,density,n,nchains,k,x,pooled_mean,chain_std`

`-t auto` enables threading; chains within a cell run in parallel (the model/variational objects are
treated as read-only during sampling).

## Cross-version density check

Comparison is on the **pooled posterior-mean density** (average over chains) on a fixed per-density
grid, judged against the **Monte-Carlo noise floor** estimated from the chain-to-chain spread:

```
Δ     = max_grid |pooled_new − pooled_baseline|
noise = max_grid sqrt(std_new² + std_baseline²) / sqrt(nchains)
ratio = Δ / noise
```

Equivalent estimates give `ratio` around 1–3 (a max over the grid inflates it slightly); a genuine
change makes `ratio ≫ 1`. Run:

```bash
julia --project=perf perf/compare.jl perf/baselines/densities.csv perf/results/reduced/densities.csv
```

It prints a per-cell table (Δ, noise, ratio, PASS/REVIEW), flags `ratio > threshold` (default 5) for
REVIEW, and exits non-zero if anything needs review — suitable for CI.

### Baselines and the bug-fix exception

- The committed reference lives in `perf/baselines/densities.csv`. Generate/refresh it from a run
  (`cp perf/results/reduced/densities.csv perf/baselines/densities.csv`), or via the workflow's
  `update_baseline` input. Keep the baseline and the comparison run on the **same mode** so the grids
  and sample sizes line up.
- After an intentional **bug fix** (which changes the correct output), the old baseline for the
  affected method is stale. Temporarily mark that method `regression = false` in `src/methods.jl` so
  `compare.jl` reports it as EXCLUDED rather than a spurious REVIEW; once a baseline from the fixed
  code is committed, flip it back to `true`. (RandomBernsteinPoly went through this after its telescope
  sampler was corrected and is now `regression = true` again — so its baseline must be generated from
  the fixed sampler.)
- Optimizations that are meant to preserve behaviour should keep every method `PASS`.