# RandomFiniteGaussianMixture

Documentation for finite Gaussian mixture models, with a variable (random) number of mixture components.

This model is available through the `BayesDensityFiniteGaussianMixture` package.

The variational inference algorithm used to compute the posterior first proceeds by separately fitting mixture models for different values of ``K``, recording the corresponding value of the optimized evidence lower bound, ``\mathrm{ELBO}(K)`` at the end of each optimization.
The posterior over the number of mixture components ``p(K\,|\, \boldsymbol{x})`` is then approximated via
```math
q(K) \propto p(K)\,\exp\big\{\mathrm{ELBO}(K)\big\}.
```
This approximation can be justified in light of the fact that the ELBO is a lower bound on the log-marginal likelihood ``p(\boldsymbol{x}, K)``. The approximate posterior for the number of mixture components together with the optimal variational densities given ``K`` defines a distribution over a space of mixture of variable dimension, which is then used to make inferences about the density of the given sample.

The algorithm used to compute the conditional variational posterior ``q(\boldsymbol{\mu}|k)\,q(\boldsymbol{\sigma}^2|k)\,q(\boldsymbol{w}|k)`` is a variant of the algorithm 5 in [Ormerod2010explaining](@citet). Note that our version also includes an additional hyperprior on the rate parameters of the mixture scales and that the algorithm has been adjusted to account for this fact.

There are two main ways of proceeding with Bayesian inference for the variational posterior. One possibility is to proceed with the single value ``\hat{K}`` that maximizes the variational probability ``q(K)``, the so-called maximum a posteriori model. Posterior inference then proceeds via the conditional variational posterior ``q\big(\boldsymbol{\mu}, \boldsymbol{\sigma}^2, \boldsymbol{w} | \hat{K}\big)``. This model can be retrieved by utilizing the [`maximum_a_posteriori`](@ref) method on a fitted variational posterior, which can then be used for posterior inference.

Another possibility is to take a fully Bayesian approach, where we do not condition on a single value of ``K``, but treat it as a random variable. To pursure this approach to posterior inference, one can simply use the object returned by calling [`varinf`](@ref) directly (e.g. for plotting or computing other posterior summary statistics).

## Module API

```@docs
RandomFiniteGaussianMixture
```

### Evaluating the pdf and cdf
```@docs
BayesDensityFiniteGaussianMixture.pdf(::RandomFiniteGaussianMixture, ::NamedTuple, ::Real)
BayesDensityFiniteGaussianMixture.cdf(::RandomFiniteGaussianMixture, ::NamedTuple, ::Real)
```

### Utility functions
```@docs
hyperparams(::RandomFiniteGaussianMixture)
```

### Variational inference
```@docs
RandomFiniteGaussianMixtureVIPosterior
varinf(::RandomFiniteGaussianMixture)
posterior_prob_components
maximum_a_posteriori
```