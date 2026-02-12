# FiniteGaussianMixture

Documentation for finite Gaussian mixture models, with a fixed number of mixture components.

This model is available through the `BayesDensityFiniteGaussianMixture` package.

`FiniteGaussianMixture` models the data-generating density as a mixture of normal distributions with a known number of mixture components. The prior-model specification can be written as follows:

```math
\begin{align*}
x_i\,|\, \boldsymbol{w}, \boldsymbol{\mu}, \boldsymbol{\sigma}^2 &\sim \sum_{k=1}^K \frac{w_k}{\sigma_k} \phi\Big(\frac{x_i - \mu_k}{\sigma_k}\Big), &i = 1,\ldots, n,\\
w_k &\sim \mathrm{Dirichlet}_K(\alpha, \ldots, \alpha),\\
\mu_k &\sim \mathrm{Normal}(\mu_0, \sigma_0^2), &k = 1,\ldots, K,\\
\sigma_k^2 \,|\, \beta &\sim \mathrm{InverseGamma}(a_\sigma, \beta), &k = 1,\ldots, K,\\
\beta &\sim \mathrm{Gamma}(a_\beta, b_\beta),\\
\end{align*}
```
where ``\phi(\cdot)`` denotes the density of the standard normal distribution, ``\mu_0 \in \mathbb{R}, \alpha, \sigma_0^2, a_\sigma, a_\beta, b_\beta > 0`` are fixed hyperparameters and ``K`` is a fixed positive integer.[^1]

[^1]:
    We use the rate parameterization of the [Gamma distribution](https://en.wikipedia.org/wiki/Gamma_distribution) here. This differs from the scale-parameterization used by `Distributions`.

!!! note
    When using [`FiniteGaussianMixture`](@ref), the number of mixture components ``K`` must be specified by the user. A more flexible version of this model, where ``K`` is further treated as a random variable with its own prior distribution, is available as [`RandomFiniteGaussianMixture`](@ref).

For Markov chain Monte Carlo based inference, this module implements an augmented Gibbs sampling approach. The algorithm used is essentially the Gibbs sampler sweep (excluding the reversible jump-move) of [Richardson1997Mixtures](@citet).
For variational inference, we implement a variant of algorithm 5 in [Ormerod2010explaining](@citet). Note that our version also includes an additional hyperprior on the rate parameters of the mixture scales and that the algorithm has been adjusted to account for this fact.

## Module API

```@docs
FiniteGaussianMixture
```

### Evaluating the pdf and cdf
```@docs
BayesDensityFiniteGaussianMixture.pdf(::FiniteGaussianMixture, ::NamedTuple, ::Real)
BayesDensityFiniteGaussianMixture.cdf(::FiniteGaussianMixture, ::NamedTuple, ::Real)
```

### Utility functions
```@docs
hyperparams(::FiniteGaussianMixture)
```

### Markov chain Monte Carlo
```@docs
sample(::Random.AbstractRNG, ::FiniteGaussianMixture, ::Int)
```

### Variational inference
```@docs
FiniteGaussianMixtureVIPosterior
varinf(::FiniteGaussianMixture)
```