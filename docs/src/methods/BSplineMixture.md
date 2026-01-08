# BSplineMixture

Documentation for B-Spline Mixture Models.

This model is available through the `BayesDensityBSplineMixture` module, which can be installed as a separate Julia package as follows:
```julia
using Pkg
Pkg.add(url="https://github.com/oskarhs/BayesianDensityEstimation.jl/lib/BayesDensityBSplineMixture.jl")
```

To use the functionality of the module, one can import either of the following packages:
```julia
using BayesDensityBSplineMixture
using BayesDensity # Requires a working BayesDensity installation
```

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