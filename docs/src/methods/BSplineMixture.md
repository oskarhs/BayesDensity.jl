# BSplineMixture

Documentation for B-Spline Mixture Models.

This model is available through the `BayesDensityBSplineMixture` module.

## Module API

The first step to fitting a B-spline mixture model to a given dataset is to construct a [`BSplineMixture`](@ref) model object:
```@docs
BSplineMixture
BayesDensityBSplineMixture.pdf(::BSplineMixture, ::NamedTuple{Names, Vals}, ::Real) where {Names, Vals<:Tuple}
```

### Variational inference
```@docs
BSplineMixtureVIPosterior
```

### Utility functions
```@docs
support(::BSplineMixture)
hyperparams(::BSplineMixture)
```