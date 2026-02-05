# FiniteGaussianMixture

Documentation for finite Gaussian mixture models, with a fixed number of mixture components.

This model is available through the `BayesDensityFiniteGaussianMixture` package.

For Markov chain Monte Carlo based inference, this module implements an augmented Gibbs sampling approach. The algorithm used is essentially the Gibbs sampler sweep (excluding the reversible jump-move) of [Richardson1997Mixtures](@citet).
For variational inference, we implement a variant of the algorithm 5 in [Ormerod2010explaining](@citet). Note that our version also includes an additional hyperprior on the rate parameters of the mixture scales and that the algorithm has been adjusted to account for this fact.

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