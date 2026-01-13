# BSplineMixture

Documentation for B-Spline Mixture Models.

This model is available through the `BayesDensityBSplineMixture` module.

## Example usage

## Module API

The first step to fitting a B-spline mixture model to a given dataset is to construct a [`BSplineMixture`](@ref) model object:
```@docs
BSplineMixture
```

### Evaluating the pdf and cdf
```@docs
BayesDensityBSplineMixture.pdf(::BSplineMixture, ::NamedTuple{Names, Vals}, ::Real) where {Names, Vals<:Tuple}
BayesDensityBSplineMixture.cdf(::BSplineMixture, ::NamedTuple{Names, Vals}, ::Real) where {Names, Vals<:Tuple}
```

### Utility functions
```@docs
support(::BSplineMixture)
hyperparams(::BSplineMixture)
```

### Markov chain Monte Carlo
```@docs
sample(::Random.AbstractRNG, ::BSplineMixture, ::Int)
```

### Variational inference
```@docs
BSplineMixtureVIPosterior
varinf(::BSplineMixture)
```