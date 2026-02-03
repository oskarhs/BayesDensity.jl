```@meta
CurrentModule = BayesDensity
```

# BayesDensity.jl

[BayesDensity.jl](https://github.com/oskarhs/BayesianDensityEstimation.jl) is a Julia package for univariate nonparametric Bayesian density estimation. The package provides access to several density estimators from the Bayesian nonparametrics literature. For most of the implemented methods, posterior inference is possible through both Markov chain Monte Carlo (MCMC) methods and variational inference (VI).

## Installation
We note that each of the models implemented in `BayesDensity` can be installed independently from all the by downloading the corresponding module. For instance, if we would like to use the [`HistSmoother`](@ref) model we need to install the `BayesDensityHistSmoother` package:
```julia
using Pkg
Pkg.add(url="https://github.com/oskarhs/BayesianDensityEstimation.jl/lib/BayesDensityHistSmoother.jl")
```

We can now use the model by importing the downloaded package:
```julia
using BayesDensityHistSmoother
```

Alternatively, if one wants to have access to more models, one can install the `BayesDensity` package instead:
```julia
using Pkg
Pkg.add(url="https://github.com/oskarhs/BayesianDensityEstimation.jl/lib/BayesDensity.jl")
```

We can now import all the models implemented in this package by running the following code snippet:
```julia
using BayesDensity
```

## Quick start
To illustarte the basic use of the package, we show one can fit a histogram smoother to a simulated dataset.

```julia
using BayesDensityHistSmoother, Distributions, Random
rng = Random.Xoshiro(1) # for reproducibility

# Simulate some data:
d_true = MixtureModel([Normal(-0.2, 0.25), Normal(0.5, 0.15)], [0.4, 0.6])
x = rand(rng, d_true, 1000)

# Create a HistSmoother model object:
smoother = HistSmoother(x)
```

Having specified a model for the data, we can perform posterior inference through Markov chain Monte Carlo methods or variational inference:

```julia
mcmc_fit = sample(rng, smoother, 2100; n_burnin=100) # MCMC
vi_fit = varinf(smoother)                            # VI
```

The resulting fits can easily be plotted using the [Plots.jl](https://github.com/JuliaPlots/Plots.jl) and [Makie.jl](https://github.com/MakieOrg/Makie.jl) package extensions. For example, the posterior mean and ``95 \%`` pointwise credible bands can be plotted via Makie as follows:
```julia
using CairoMakie
plot(mcmc_fit) # Based on MCMC
plit(vi_fit)   # Based on VI
```