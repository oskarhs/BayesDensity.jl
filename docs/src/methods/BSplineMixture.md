# BSplineMixture

Documentation for B-Spline Mixture Models.

This model is available through the `BayesDensityBSplineMixture` package.

`BSplineMixture` models the data-generating density as a mixture of cubic B-spline basis functions of a fixed, high dimension. The regularity of the estimate is controlled via a smoothness-inducing prior on the spline coefficients. Mathematically, the model can be written as

```math
\begin{align*}
    x_i\,|\, \boldsymbol{\beta} &\sim \sum_{k=1}^K \theta_k b_k(x_i), &i = 1,\ldots, n,\\
    \beta_k &\sim \text{Normal}(0, \sigma_\beta^2), &k = 1, 2,\\
    \Delta^2 \big\{\beta_k - \mu_k\big\} \,|\, \tau^2, \boldsymbol{\delta}^2 &\sim \text{Normal}(0, \tau^2\delta_k^2), &k \geq 3\\
    \tau^2 &\sim \text{InverseGamma}(a_\tau, b_\tau),\\
    \delta^2_k &\sim \text{InverseGamma}(a_\delta, b_\delta), &k = 1, \ldots, K-3,\\
\end{align*}
```
where ``b_k(\cdot)`` is a [B-spline](https://en.wikipedia.org/wiki/B-spline) basis function, normalized to have unit integral, ``\boldsymbol{\mu}\in \mathbb{R}^{K-1}, \sigma_\beta, a_\tau, b_\tau, a_\delta, b_\delta > 0`` are fixed hyperparameters, ``\Delta^2 \alpha_k= (\alpha_k - 2\alpha_{k-1} + \alpha_{k-2})`` is the discrete second-order difference operator and ``\boldsymbol{\theta} = \boldsymbol{\theta}(\boldsymbol{\beta})`` is defined by the logistic stickbreaking-map,
```math
\begin{align*}
    \theta_k &= \frac{e^{\beta_k}}{1 + e^{\beta_k}} \prod_{j = 1}^{k-1} \frac{1}{1 + e^{\beta_j}}, &k = 1,\ldots, K-1\\
    \theta_K &= 1 - \sum_{k=1}^{K-1} \theta_k.
\end{align*}
```

## Module API

```@docs
BSplineMixture
```

### Evaluating the pdf and cdf
```@docs
BayesDensityBSplineMixture.pdf(::BSplineMixture, ::NamedTuple{Names, Vals}, ::Real) where {Names, Vals<:Tuple}
BayesDensityBSplineMixture.cdf(::BSplineMixture, ::NamedTuple{Names, Vals}, ::Real) where {Names, Vals<:Tuple}
```

### Utility functions
```@docs
hyperparams(::BSplineMixture)
```

### Markov chain Monte Carlo
```@docs
sample(::Random.AbstractRNG, ::BSplineMixture, ::Int)
```

### Variational inference
```@docs
BSplineMixtureVIPosterior
varinf(::BSplineMixture)
```