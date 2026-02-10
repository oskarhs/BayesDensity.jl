# RandomBernsteinPoly

Documentation for random Bernstein polynomials [Petrone1999bernstein](@citep).

`RandomBernsteinPoly` models the data-generating density as a mixture of fixed ``\text{Beta}`` distributions with an unknown number of mixture components. The smoothness of the density estiates is primarily controlled through the number of basis densities ``K``, which is selected in a data-driven manner. Mathematically, the Bernstein density model and prior distribution takes the following form,

```math
\begin{align*}
x_i \,|\, K, \boldsymbol{w} &\sim \sum_{k=1}^K w_{k}b_{K,k}(x_i), &i = 1,\ldots, n,\\
\boldsymbol{w} \,|\, K &\sim \text{Dirichlet}_K(a/K, \ldots, a/K),\\
K &\sim p(K),
\end{align*}
```
where ``b_{K,k}(\cdot)`` denotes the probability density function of the ``\text{Beta}(k, K-k+1)``-distribution, ``a> 0`` a fixed hyperparameter and ``p(K)`` is a probability mass function supported on a finite subset of the positive integers.

This module implements the Gibbs sampler of [Petrone1999bernstein](@citet) for Markov chain Monte Carlo-based posterior inference.

!!! note
    Keep in mind that this sampler is quite slow even for moderate sample sizes, due to the expensive sweeps involved in the Gibbs sampler.

## Module API

```@docs
RandomBernsteinPoly
```

### Evaluating the pdf and cdf
```@docs
BayesDensityRandomBernsteinPoly.pdf(::RandomBernsteinPoly, ::NamedTuple, ::Real)
BayesDensityRandomBernsteinPoly.cdf(::RandomBernsteinPoly, ::NamedTuple, ::Real)
```

### Utility functions
```@docs
hyperparams(::RandomBernsteinPoly)
```

### Markov chain Monte Carlo
```@docs
sample(::Random.AbstractRNG, ::RandomBernsteinPoly, ::Int)
```