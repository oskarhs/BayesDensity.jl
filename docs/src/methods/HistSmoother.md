# HistSmoother

Documentation for the histogram smoother of [Wand2022engines](@citet).

This model is available through the `BayesDensityHistSmoother` module.

## Module API

The first step to fitting a histogram smoother is to create a `HistSmoother` model object:
```@docs
HistSmoother
BayesDensityHistSmoother.pdf(::HistSmoother, ::NamedTuple{Names, Vals}, ::Real) where {Names, Vals<:Tuple}
```

### Variational inference
TODO

### Utility functions
```@docs
support(::HistSmoother)
hyperparams(::HistSmoother)
```