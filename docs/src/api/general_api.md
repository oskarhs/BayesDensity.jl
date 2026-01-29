# General API

```@setup general_api
using BayesDensityBSplineMixture
```

This page explains how to fit the Bayesian density models implemented in `BayesDensity.jl.`
Most of the methods implemented in this package support two modes of posterior inference: simulation consistent inference through Markov chain Monte Carlo (MCMC) and approximate through variational inference (VI).
We also document most of the convenience methods available for computing select posterior quantities of interest, such as the posterior mean or quantiles of ``f(t)`` for some ``t \in \mathbb{R}``.

The plotting API of this package is documented on a [separate page](plotting_api.md)

## Defining models
The first step to estimating a density with this package is to create a model object for which posterior inference is desired. All density models in this package are subtypes of `AbstractBayesDensityModel`:
```@docs
AbstractBayesDensityModel
```

In order to create a model object, we call the corresponding contructor with the data and other positional- and keyword arguments. For example, we can create a [`BSplineMixture`](@ref) object with default hyperparameters as follows:
```@example general_api
bsm = BSplineMixture(randn(1000))
nothing # hide
```
For more detailed information on the arguments supported by each specific Bayesian density model we refer the reader to the [methods documentation](../methods/index.md).


### Evaluating the density and the cumulative distribution function
The density estimators implemented in this package all specify a model ``f(t\,|\, \boldsymbol{\eta})`` for the density of the data, which depends on a parameter ``\boldsymbol{\eta}``.
In order to calculate ``f(\cdot)`` for a given ``\boldsymbol{\eta}``, each Bayesian density model implements the `pdf` method.
```@docs
pdf(::AbstractBayesDensityModel, ::Any, ::Real)
```

For models that only implement the signature `pdf(::AbstractBayesDensityModel, ::Any, ::Real)`, a generic fallback method is provided for vectors of parameters and vector evaluation grids. However, it is recommended that most models provide specialized methods for vectors of parameters and vectors of evaluation points, as it is often possible to implement batch evaluation more efficiently (e.g. by leveraging BLAS calls instead of loops) when the parameters and the evaluation grid are provided in batches.

The cumulative distribution function of a model can be computed in a similar way by using the `cdf` method:
```@docs
cdf(::AbstractBayesDensityModel, ::Any, ::Real)
```
Generic fallback methods for computing the cdf for vectors of parameters and vector evaluation grids are also provided for models that implement the signature `cdf(::AbstractBayesDensityModel, ::Any, ::Real)`.

### Other methods
All of the density models implemented in this package depend on the choice of various hyperparameters, which can be retrieved by utilizing the following method:
```@docs
hyperparams(::AbstractBayesDensityModel)
```
For the exact format of the returned hyperparameters for a specific Bayesian density model type, we refer to the docs of the individual density estimators.

To compute the support of a given model, the `support` method is provided.
```@docs
support(::AbstractBayesDensityModel)
```

The element type of the model object can be determined via the `eltype` method:
```@docs
eltype(::AbstractBayesDensityModel)
```

## Markov chain Monte Carlo
The main workhorse of MCMC-based inference is the `sample` method, which takes a Bayesian density model object as input and generates posterior samples through a specialized MCMC routine.
```@docs
sample(::AbstractBayesDensityModel, ::Int)
```

All of the implemented MCMC methods return an object of type `PosteriorSamples`:
```@docs
PosteriorSamples
```

The following methods can be used to extract useful information about the model object, such as the underlying model object or the element type.
```@docs
model(::PosteriorSamples)
```

By default, `PosteriorSamples` objects also store the burn-in samples from the MCMC routine. These can be discarded via the following method:
```@docs
drop_burnin(::PosteriorSamples)
```

Multiple `PosteriorSamples` objects can also be concatenated to create a single `PosteriorSamples` object.
This is particularly useful when a preliminary MCMC run is deemed to be too short, and one wants to pool the original samples with the samples from a new MCMC run.
```@docs
vcat(::PosteriorSamples...)
```

#### Computing posterior summary statistics
When using Bayesian density estimators, we are often interested in computing various summary statistics of the posterior draws from an MCMC procedure. For instance, we may be interested in providing an estimate of the density ``f`` (e.g. the posterior mean) and to quantify the uncertainty in this estimate (e.g. via credible bands).

To this end, `BayesDensityCore` provides methods for `PosteriorSamples` objects that let us easily compute relevant summary statistics for the density ``f``, as shown in the short example below:

```@example general_api
bsm = BSplineMixture(randn(1000))
posterior = sample(bsm, 2000; n_burnin=400)

# Compute the posterior mean of f(0.5)
mean(posterior, pdf, 0.5)

# Compute the posterior 0.05 and 0.95-quantiles of f(0.5)
# Note that supplying pdf as the second argument is optional here
quantile(posterior, 0.5, [0.05, 0.95]) == quantile(posterior, pdf, 0.5, [0.05, 0.95])
nothing # hide
```

In some cases it may also be of interest to carry out posterior inference for the cumulative distribution function ``F(t) = \int_{-\infty}^t f(s)\, \text{d}s``. Computing posterior summary statistics for the cdf instead of the pdf is easily achieved by replacing the `pdf` in the second argument with `cdf` instead:
```@example general_api
# Compute the posterior mean of F(0.5)
mean(posterior, cdf, 0.5)

# Compute the posterior 0.05 and 0.95-quantiles of F(0.5)
# Note that supplying cdf as the second argument is necessary here
quantile(posterior, cdf, 0.5, [0.05, 0.95])
nothing # hide
```

The posterior summary statistics available through `BayesDensityCore` are the following:
```@docs
mean(::PosteriorSamples)
quantile(::PosteriorSamples)
median(::PosteriorSamples)
var(::PosteriorSamples)
std(::PosteriorSamples)
```

## Variational inference
The `varinf` method can be used to compute a variational approximation to the posterior distribution:
```@docs
varinf(::AbstractBayesDensityModel)
```

Any call to `varinf` will return a subtype of the abstract type `AbstractVIPosterior`:
```@docs
AbstractVIPosterior
```

For most models, `varinf` also returns an object which stores the result of the optimization procedure, see [`VariationalOptimizationResult`](@ref).

The following convenience methods are also part of the public API:
```@docs
model(::AbstractVIPosterior)
BayesDensity.eltype(::AbstractVIPosterior)
```

#### Generating samples from the variational posterior
The `sample` method makes it possible to generate independent samples from the variational posterior. This is particularly useful in cases where inference for multiple posterior quantities (e.g. medians, variances) is desired.
```@docs
sample(::AbstractVIPosterior, ::Int)
```
As shown in the above docstring, using the `sample` method on a `AbstractVIPosterior` object returns an object of type [`PosteriorSamples`](@ref). As such, all of the convenience methods showcased in the previous subsection will also work for the object returned by `sample`.

#### Computing posterior summary statistics
`BayesDensityCore` also provides convenience methods for `AbstractVIPosterior` objects that let us easily compute relevant summary statistics for the density ``f`` and the cdf ``F`` directly from the variational posterior object:

```@example general_api
bsm = BSplineMixture(randn(1000))
viposterior, info = varinf(bsm)

# Compute the (variational) posterior mean of f(0.5)
mean(viposterior, pdf, 0.5)

# Compute the (variational) posterior median of F(0.5)
median(viposterior, cdf, 0.5)

# Compute the (variational) posterior 0.05 and 0.95-quantiles of f(0.5)
# Note that supplying pdf as the second argument is optional here
quantile(viposterior, 0.5, [0.05, 0.95]) ≈ quantile(viposterior, pdf, 0.5, [0.05, 0.95])
nothing # hide
```

The full list of available summary statistics is the same as that for `PosteriorSamples` objects:
```@docs
mean(::AbstractVIPosterior)
quantile(::AbstractVIPosterior)
median(::AbstractVIPosterior)
var(::AbstractVIPosterior)
std(::AbstractVIPosterior)
```

!!! note
    Note that each call to `mean`, `quantile`, `median`, `var` or `std` in most cases will first simulate a random sample from the posterior distribution, and then uses this sample to compute a Monte Carlo approximation of the quantity of interest using these samples.
    If posterior inference for multiple quantities is desired, then it is recommended to first use [`sample`](@ref), and call these functions on this object as only a single batch of posterior samples is generated in this case.

#### Storing info from the variational optimization
In order to provide a simple way of performing convergence diagnostics for variational optimization problems, `BayesDensityCore` exports the [`VariationalOptimizationResult`](@ref) type.
```@docs
VariationalOptimizationResult
elbo
n_iter
converged
tolerance
posterior
```