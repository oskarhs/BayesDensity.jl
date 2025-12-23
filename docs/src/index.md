```@meta
CurrentModule = BayesDensity
```

# BayesDensity.jl

[BayesDensity.jl](https://github.com/oskarhs/BayesianDensityEstimation.jl) is a Julia package for univariate nonparametric Bayesian density estimation. The package provides access to several density estimators from the Bayesian nonparametrics literature. For most of the implemented methods, posterior inference is possible through both Markov chain Monte Carlo (MCMC) methods and variational inference (VI).

## Installation

The BayesDensity.jl package is currently not part of any package repository, but can be installed from its GitHub repository as follows:
```julia
using Pkg
Pkg.add(url="https://github.com/oskarhs/BayesianDensityEstimation.jl/lib/BayesDensity.jl")
```
After installation, `using BayesDensity` will load all of the estimators implemented by this package.

Alternatively, it is possible to install each of the Bayesian density estimators implemented in this package separately. For instance, the B-spline mixture model estimator can be downloaded as follows:
```julia
Pkg.add(url="https://github.com/oskarhs/BayesianDensityEstimation.jl/lib/BayesDensityBSM.jl")
```
Each of the density estimators can then be accessed separately via e.g. `using BayesDensityBSM`.

## Quick start
To illustarte the basic use of the package, we show one can fit a B-spline mixture model to a simulated dataset.

```julia
using BayesDensity, Distributions, Random
rng = Random.Xoshiro(1) # for reproducibility

# Simulate some data:
d_true = MixtureModel([Normal(-0.2, 0.25), Normal(0.5, 0.15)], [0.4, 0.6])
x = rand(rng, d_true, 1000)

# Create a B-Spline mixture model object:
bsm = BSMModel(x)
```

Having specified a model for the data, we can perform posterior inference through Markov chain Monte Carlo methods or variational inference:

```julia
mcmc_fit = sample(rng, bsm, 5000; n_burnin=1000) # MCMC
vi_fit = varinf(bsm) # VI
```

The resulting fits can easily be plotted using the [Plots.jl](https://github.com/JuliaPlots/Plots.jl) and [Makie.jl](https://github.com/MakieOrg/Makie.jl) package extensions. For example, the posterior mean and ``95 \%`` pointwise credible bands can be plotted via Make as follows:
```julia
using CairoMakie
plot(mcmc_fit)
```