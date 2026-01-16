# PitmanYorMixture

Documentation for Pitman-Yor mixture models [Ishwaran2001Gibbs](@citet).

Markov Chain monte Carlo is via algorithm 2 in [Neal2000Markov](@citet).

## Example usage

## Module API

```@docs
PitmanYorMixture
```

### Evaluating the pdf and cdf
```@docs
BayesDensityPitmanYorMixture.pdf(::PitmanYorMixture, ::NamedTuple{Names, Vals}, ::Real) where {Names, Vals<:Tuple}
BayesDensityPitmanYorMixture.cdf(::PitmanYorMixture, ::NamedTuple{Names, Vals}, ::Real) where {Names, Vals<:Tuple}
```

### Utility functions
```@docs
support(::PitmanYorMixture)
hyperparams(::PitmanYorMixture)
```

### Markov chain Monte Carlo
```@docs
sample(::Random.AbstractRNG, ::PitmanYorMixture, ::Int)
```