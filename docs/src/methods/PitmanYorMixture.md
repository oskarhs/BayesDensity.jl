# PitmanYorMixture

Documentation for Pitman--Yor mixture models [Ishwaran2001Gibbs](@citet), with a normal kernel and a normal-inverse gamma base measure.

`PitmanYorMixture` is an infinite-dimensional mixture model, where the regularity of the density is primarily governed through the mixture weights. Mathematically, the Pitman--Yor mixture model can be described as follows:

```math
\begin{align*}
x_i \,|\, \mu_i, \sigma_i^2 &\sim \mathrm{Normal}(\mu_i, \sigma_i^2), &i = 1,\ldots, n,\\
\mu_i, \sigma_i^2 \,|\, G &\sim G, &i = 1,\ldots, n,\\
G &\sim \mathrm{PitmanYor}(\alpha, \theta, G_0),
\end{align*}
```
where ``0 \leq \theta < 1``, ``\alpha > -\theta`` and ``G_0`` is the ``\mathrm{NormalInverseGamma}(\mu_0, \lambda, a, b)``-distribution for some fixed hyperparameters ``\mu_0\in \mathbb{R}, \lambda, a, b > 0``.

Alternatively, the Pitman--Yor mixture model can be written down in its stickbreaking form,
```math
\begin{align*}
x_i \,|\, \boldsymbol{v}, \boldsymbol{\mu}, \boldsymbol{\sigma}^2 &\sim \sum_{k=1}^\infty \frac{w_k}{\sigma_k} \phi\Big(\frac{x_i - \mu_k}{\sigma_k}\Big), &i = 1,\ldots, n,\\
v_k &\sim \text{Beta}(1-\theta, \alpha + k\theta), &k \in \mathbb{N},\\
\mu_k, \sigma_k^2 &\sim \text{NormalInverseGamma}(\mu_0, \lambda, a, b), &k \in \mathbb{N}
\end{align*}
```
where ``\phi(\cdot)`` is the density of the standard normal distribution and ``w_k = v_k\prod_{j=1}^{k-1} (1 - v_j)`` for ``k\in \mathbb{N}``.

For Markov chain Monte Carlo based inference, this module implements algorithm 2 by [Neal2000Markov](@citet).
For variational inference, we implement the truncated stickbreaking approach of [Blei2006DirichletVariational](@citet).

!!! note
    Since Dirichlet process mixture models are equivalent to a Pitman-Yor mixture model with discount parameter equal to `0`, this module can also be used to fit the former type of models.

## Module API

```@docs
PitmanYorMixture
```

### Evaluating the pdf and cdf
```@docs
BayesDensityPitmanYorMixture.pdf(::PitmanYorMixture, ::NamedTuple, ::Real)
BayesDensityPitmanYorMixture.cdf(::PitmanYorMixture, ::NamedTuple, ::Real)
```

### Utility functions
```@docs
hyperparams(::PitmanYorMixture)
```

### Markov chain Monte Carlo
```@docs
sample(::Random.AbstractRNG, ::PitmanYorMixture, ::Int)
```

### Variational inference
```@docs
PitmanYorMixtureVIPosterior
varinf(::PitmanYorMixture)
```