# HistSmoother

Documentation for the histogram smoother of [Wand2022engines](@citet).

This model is available through the `BayesDensityHistSmoother` module, which can be installed as a separate Julia package as follows:
```julia
using Pkg
Pkg.add(url="https://github.com/oskarhs/BayesianDensityEstimation.jl/lib/BayesDensityHistSmoother.jl")
```

To use the functionality of the module, one can import either of the following packages:
```julia
using BayesDensityHistSmoother
using BayesDensity # Requires a working BayesDensity installation
```

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