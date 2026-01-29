# RandomFiniteGaussianMixture

Documentation for finite Gaussian mixture models, with a variable (random) number of mixture components.

The variational inference algorithm used to compute the posterior first proceeds by separately fitting mixture models for different values of ``K``, recording the corresponding value of the optimized evidence lower bound, ``\mathrm{ELBO}(K)`` at the end of each optimization.
The posterior over the number of mixture components ``p(K\,|\, \boldsymbol{x})`` is then approximated via
```math
q(K) \propto p(K)\,\exp\big\{\mathrm{ELBO}(K)\big\}.
```
This approximation can be justified in light of the fact that the ELBO is a lower bound on the log-marginal likelihood ``p(\boldsymbol{x}, K)``. The approximate posterior for the number of mixture components together with the optimal variational densities given ``K`` defines a distribution over a space of mixture of variable dimension, which is then used to make inferences about the density of the given sample.

The algorithm used to compute the conditional variational posterior ``q(\boldsymbol{\mu}|k)\,q(\boldsymbol{\sigma}^2|k)\,q(\boldsymbol{w}|k)`` is variant of the algorithm 5 in [Ormerod2010explaining](@citet). Note that our version also includes an additional hyperprior on the rate parameters of the mixture scales.

## Module API

```@docs
RandomFiniteGaussianMixture
```

### Evaluating the pdf and cdf
```@docs
BayesDensityFiniteGaussianMixture.pdf(::RandomFiniteGaussianMixture, ::NamedTuple{Names, Vals}, ::Real) where {Names, Vals<:Tuple}
BayesDensityFiniteGaussianMixture.cdf(::RandomFiniteGaussianMixture, ::NamedTuple{Names, Vals}, ::Real) where {Names, Vals<:Tuple}
```

### Utility functions
```@docs
support(::RandomFiniteGaussianMixture)
hyperparams(::RandomFiniteGaussianMixture)
```

### Variational inference
```@docs
RandomFiniteGaussianMixtureVIPosterior
varinf(::RandomFiniteGaussianMixture)
```