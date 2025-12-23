# B-Spline Mixture Model

Documentation for B-Spline Mixture Models.
```@docs
BSMModel
BayesDensityBSM.pdf(::BSMModel, ::NamedTuple{Names, Vals}, ::Real) where {Names, Vals<:Tuple}
```

### Variational inference
```@docs
BSMVIPosterior
```

### Utility functions
```@docs
support(::BSMModel)
hyperparams(::BSMModel)
```