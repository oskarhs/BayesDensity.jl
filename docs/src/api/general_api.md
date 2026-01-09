# General API

This page explains how to fit the Bayesian density models implemented in BayesDensity.jl.
Most of the methods implemented in this package support two modes of posterior inference: simulation consistent inference through Markov chain Monte Carlo (MCMC) and approximate through variational inference (VI).
We also document most of the convenience methods available for computing select posterior quantities of interest, such as the posterior mean or quantiles of ``f(t)`` for some ``t \in \mathbb{R}``.
The plotting API of this package is documented on a [separate page](plotting_api.md)

## Defining models
The first step to estimating a density with this package is to create a model object for which posterior inference is desired. All density models in this package are subtypes of `AbstractBayesDensityModel`:
```@docs
AbstractBayesDensityModel
```

In order to create a model object, we call the corresponding contructor with the data and other positional- and keyword arguments. For example, we can create a [`BSplineMixture`](@ref) object with default hyperparameters as follows:
```@repl
using BayesDensity
bsm = BSplineMixture(randn(1000))
```
For more detailed information on the arguments supported by each specific Bayesian density model we refer the reader to the METHODS DOCUMENTATION.


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


## Markov chain Monte Carlo
The main workhorse of MCMC-based inference is the `sample` method, which takes a Bayesian density model object as input and generates posterior samples through a specialized MCMC routine.
```@docs
sample(::AbstractBayesDensityModel, ::Int)
```

All of the implemented MCMC methods return an object of type `PosteriorSamples`:
```@docs
PosteriorSamples
```

The following methods can be used to extract useful information about the model object, such as
```@docs
model(::PosteriorSamples)
eltype(::PosteriorSamples)
```

#### Computing posterior quantities of interest:
The following methods can be used to compute different posterior quantities of interest:
```@docs
mean(::PosteriorSamples, ::AbstractVector{<:Real})
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
model(::AbstractVIPosterior)
```

The `sample` method makes it possible to generate independent samples from the variational posterior. This is particularly useful in cases where inference for multiple posterior quantities (e.g. medians, quantiles) is desired.
```@docs
sample(::AbstractVIPosterior, ::Int)
```
As shown in the above docstring, using the `sample` method on a `AbstractVIPosterior` object returns an object of type [`PosteriorSamples`](@ref). As such, all of the convenience methods showcased in the previous subsection will also work for the object returned by `sample`.

#### Computing posterior quantities of interest:
Alternatively, various posterior quantities of interest can be computed directly as follows:
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