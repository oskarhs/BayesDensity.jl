# Histogram smoother

Documentation for the histogram smoother of [Wand2022engines](@citet).
```@docs
SHSModel
BayesDensitySHS.pdf(::SHSModel, ::NamedTuple{Names, Vals}, ::Real) where {Names, Vals<:Tuple}
```

### Variational inference
TODO

### Utility functions
```@docs
support(::SHSModel)
hyperparams(::SHSModel)
```