# General API

This explains how to fit the Bayesian density models implemented in BayesDensity.jl.
Most of the methods implemented in this package support two modes of posterior inference: asymptotically exact inference through Markov chain Monte Carlo (MCMC) and approximate through variational inference (VI).
We also document most of the convenience methods available for computing select posterior quantities of interest, such as the posterior mean or quantiles of ``f(t)`` for some ``t \in \mathbb{R}``.
The plotting API of this package is documented on a [separate page](plotting_api.md)

## Defining models
The first step to estimating a density with this package is to create a model object for which posterior inference is desired. All density models in this package are subtypes of `AbstractBayesDensityModel`:
```@docs
AbstractBayesDensityModel
```

All of the density models implemented in this package depend on the choice of various hyperparameters, which can be retrieved by utilizing the following method:
```@docs
hyperparams(::AbstractBayesDensityModel)
```
For the exact format of the returned hyperparameters for a specific Bayesian density model type, we refer to the docs of the individual density estimators.

In order to create a model object, we call the corresponding contructor with the data and other positional- and keyword arguments. For example, we can create a [`BSMModel`](@ref) object with default hyperparameters as follows:
```@repl
using BayesDensity
bsm = BSMModel(randn(1000))
```
For more detailed information on the arguments supported by each specific Bayesian density model we refer the reader to the METHODS DOCUMENTATION.


### Evaluating the density
The density estimators implemented in this package all specify a model ``f(t\,|\, \boldsymbol{\eta})`` for the density of the data, which depends on a parameter vector ``\boldsymbol{\eta}``.
In order to calculate ``f(\cdot)`` for a given ``\eta``, each Bayesian density model implements the `pdf` method.
```@docs
pdf(::AbstractBayesDensityModel, ::Any, ::Real)
```

For Models that only implement the signature `pdf(::BayesDensityModel, ::Any, ::Real)`, a generic fallback method is provided when the . However, it is recommended that most models provide specialized methods for vectors of parameters and evaluation points, as it is often possible to implement batch evaluation more efficiently (e.g. by leveraging BLAS calls instead of loops) when the parameters and the evaluation grid are provided in batches.

### Evaluating the cdf

TODO: Write this section once we have provided a generic fallback method...

## Markov chain Monte Carlo
The main workhorse of MCMC-based inference is the `sample` method, which takes a Bayesian density model object as input and generates posterior samples through a specialized MCMC routine.
```@docs
sample(::AbstractBayesDensityModel, ::Int)
```

All of the implemented MCMC methods return an object of type `PosteriorSamples`:
```@docs
PosteriorSamples
model(::PosteriorSamples)
```

#### Computing posterior quantities of interest:
The following methods can be used to compute different posterior quantities of interest:
```@docs
mean(::PosteriorSamples, ::AbstractVector{<:Real})
quantile(::PosteriorSamples, ::Any, ::Real)
median(::PosteriorSamples, ::Union{Real, AbstractVector{<:Real}})
var(::PosteriorSamples, ::AbstractVector{<:Real})
std(::PosteriorSamples, ::AbstractVector{<:Real})
```

## Variational inference
To compute the variational approximation to the posterior, we use the `varinf` method
```@docs
varinf(::AbstractBayesDensityModel)
```

Any call to `varinf` will return a subtype of the abstract type `AbstractVIPosterior`:
```@docs
AbstractVIPosterior
```

The `sample` method makes it possible to generate i.i.d. samples from the variational posterior:
```@docs
sample(::AbstractVIPosterior, ::Int)
```
As shown in the above docstring, using the `sample` method on a `AbstractVIPosterior` object returns an object of type [`PosteriorSamples`](@ref). As such, all of the convenience methods showcased in the previous subsection will also work for the object returned by `sample`.

#### Computing posterior quantities of interest:
Alternatively, various posterior quantities of interest can be computed directly as follows:
```@docs
mean(::AbstractVIPosterior, ::Union{Real, <:AbstractVector{<:Real}}, ::Int)
quantile(::AbstractVIPosterior, ::Union{Real, <:AbstractVector{<:Real}}, ::Union{Real, <:AbstractVector{<:Real}}, ::Int)
median(::AbstractVIPosterior, ::Union{Real, <:AbstractVector{<:Real}}, ::Int)
var(::AbstractVIPosterior, ::Union{Real, <:AbstractVector{<:Real}}, ::Int)
std(::AbstractVIPosterior, ::Union{Real, <:AbstractVector{<:Real}}, ::Int)
```

!!! note
    Note that each call to any of the above functions will in most cases first simulate a random sample from the posterior distribution, and then approximate the quantity of interest using these samples.
    If posterior inference for multiple quantities is desired, then it is recommended to first use [`sample`], and call these functions on this object as only a single batch of posterior samples is generated in this case.