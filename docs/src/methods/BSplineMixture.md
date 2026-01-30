# BSplineMixture

Documentation for B-Spline Mixture Models.

This model is available through the `BayesDensityBSplineMixture` package.

## Module API

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