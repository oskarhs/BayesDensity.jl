# <img src="docs/src/assets/logo.svg" alt="alt text" width="80" height="80" align="center"> BayesianDensityEstimation.jl

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://oskarhs.github.io/BayesianDensityEstimation.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://oskarhs.github.io/BayesianDensityEstimation.jl/dev/)
[![Build Status](https://github.com/oskarhs/BayesianDensityEstimation.jl/actions/workflows/CI.yml/badge.svg?branch=master)](https://github.com/oskarhs/BayesianDensityEstimation.jl/actions/workflows/CI.yml?query=branch%3Amaster)

A Julia package for nonparametric univariate Bayesian density estimation. Provides access to many different from the statistical literature under a uniform API. Supports model fitting through Markov chain Monte Carlo and approximate inference through variational inference algorithms.

## Installation

PUT SOMETHING HERE ABOUT BEING ABLE TO INSTALL DIFFERENT DENSITY ESTIMATORS SEPARATELY.

## Quick start

To get started, we illustrate the basic use of the package by fitting a B-spline mixture model to a two-component mixture of normal densities:

```julia
using BayesianDensityEstimation, Distributions, Random
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
vi_fit = ... # VI
```

The resulting fitted model objects can be used to compute posterior quantities of interest such as the posterior median of $f(t)$ through `median(mcmc_fit, t)`. Additionally, the package also provides convenience plotting functions through its [Makie.jl](https://github.com/MakieOrg/Makie.jl) and [Plots.jl](https://github.com/JuliaPlots/Plots.jl) extensions, making it easy to visualize the density estimates. For instance, one can easily plot the posterior mean, along with a 95% credible interval as follows:

```julia
using CairoMakie
plot(mcmc_fit)
```

For a more thorough introduction to the API and the capabilities of the package, we refer the interested reader to the DOCUMENTATION