# HistSmoother

Documentation for the histogram smoother of [Wand2022engines](@citet).

This model is available through the `BayesDensityHistSmoother` package.

The `HistSmoother` approach is based on modelling the data-generating density as a log-spline with a fixed and relatively large basis dimension ``K``, with a smoothness-inducing prior distribution on the spline coefficients. More specifically, the model specification is given by the hierarchical scheme

```math
\begin{align*}
x_i\,|\, \boldsymbol{\beta} &\sim \frac{\exp\{\sum_{k=1}^K \beta_k z_k(x_i)\}}{\int \exp\{\sum_{k=1}^K \beta_k z_k(x)\}\,\text{d}x}, &i = 1,\ldots, n,\\
\beta_k &\sim \mathrm{Normal}(0, \sigma_\beta^2), &k = 1, 2,\\
\beta_j \, |\, \sigma^2 &\sim \mathrm{Normal}(0, \sigma^2), &k \geq 3\\
\sigma^2 &\sim \text{HalfCauchy}(0, s_\sigma),
\end{align*}
```
where ``z_k`` are O'Sullivan spline basis functions [Wand2008Semiparametric](@citep) and ``\sigma_\beta, s_\beta > 0`` are fixed hyperparameters.

The estimates are then obtained by smoothing a likelihood based on the Poisson approximation to a histogram.

This module implements the Gibbs sampling approach of [Wand2022engines](@citet) for Markov chain Monte Carlo-based inference. We also provide an implementation of the semiparametric mean-field variational inference algorithm proposed in the same paper.

## Module API

```@docs
HistSmoother
```

### Evaluating the pdf and cdf
```@docs
BayesDensityHistSmoother.pdf(::HistSmoother, ::NamedTuple{Names, Vals}, ::Real) where {Names, Vals<:Tuple}
BayesDensityHistSmoother.cdf(::HistSmoother, ::NamedTuple{Names, Vals}, ::Real) where {Names, Vals<:Tuple}
```

### Utility functions

```@docs
hyperparams(::HistSmoother)
```

### Markov chain Monte Carlo
```@docs
sample(::Random.AbstractRNG, ::HistSmoother, ::Int)
```

### Variational inference
```@docs
HistSmootherVIPosterior
varinf(shs::HistSmoother)
```