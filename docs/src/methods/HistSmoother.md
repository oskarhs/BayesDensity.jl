# HistSmoother

Documentation for the histogram smoother of [Wand2022engines](@citet).

This model is available through the `BayesDensityHistSmoother` module.

## Example usage

## Module API

```@docs
HistSmoother
```

## Evaluating the pdf and cdf
```@docs
BayesDensityHistSmoother.pdf(::HistSmoother, ::NamedTuple{Names, Vals}, ::Real) where {Names, Vals<:Tuple}
BayesDensityHistSmoother.cdf(::HistSmoother, ::NamedTuple{Names, Vals}, ::Real) where {Names, Vals<:Tuple}
```

### Utility functions
```@docs
support(::HistSmoother)
hyperparams(::HistSmoother)
```

### Markov chain Monte Carlo
```@docs
sample(::Random.AbstractRNG, ::HistSmoother, ::Int)
```

### Variational inference
```@docs
HistSmootherVIPosterior
varinf(shs::HistSmoother)
```