# RandomBernsteinPoly

Documentation for random Bernstein polynomials [Petrone1999bernstein](@citep).

The Bernstein density model and prior distribution takes the following form,

```math
\begin{align*}
x_i \,|\, f &\sim \sum_{k=1}^K w_{K,k}b_{K,k}(x_i),\\
\boldsymbol{w}_{K} \,|\, K &\sim \text{Dirichlet}(a/K),\\
K &\sim p(K),
\end{align*}
```
where ``b_{K,k}(\cdot)`` denotes the probability density function of the ``\text{Beta}(k, K-k+1)``-distribution and ``p(K)`` is a distribution supported on a finite subset of the positive integers.

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