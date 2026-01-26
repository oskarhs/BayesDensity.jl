# PitmanYorMixture

Documentation for Pitman-Yor mixture models [Ishwaran2001Gibbs](@citet), with a normal kernel and a normal-inverse gamma base measure.

For Markov chain Monte Carlo based inference, this module implements algorithm 2 by [Neal2000Markov](@citet).
For variational inference, we implement the truncated stickbreaking approach of [Blei2006DirichletVariational](@citet).

!!! note
    Since Dirichlet process mixture models are equivalent to a Pitman-Yor mixture model with discount parameter equal to `0`, this module can also be used to fit the former type of models.

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

### Variational inference
```@docs
PitmanYorMixtureVIPosterior
varinf(::PitmanYorMixture)
```