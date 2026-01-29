<h1 align="center">
<img src="docs/src/assets/logo.svg" alt="alt text" width="60" height="60" align="center"> BayesDensity.jl
</h1>

<div align="center">

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://oskarhs.github.io/BayesDensity.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://oskarhs.github.io/BayesDensity.jl/dev/)
[![Build Status](https://github.com/oskarhs/BayesDensity.jl/actions/workflows/CI.yml/badge.svg?branch=master)](https://github.com/oskarhs/BayesDensity.jl/actions/workflows/CI.yml?query=branch%3Amaster)
[![Aqua](https://raw.githubusercontent.com/JuliaTesting/Aqua.jl/master/badge.svg)](https://github.com/JuliaTesting/Aqua.jl)
</div>


BayesDensity.jl is a Julia package for nonparametric univariate Bayesian density estimation.
It offers a unified interface to a variety of density estimators developed in the statistical literature, and supports posterior inference via both Markov chain Monte Carlo (MCMC) and variational inference methods.
Moreover, the package is designed to be extensible, allowing new estimators to make use of its built-in methods for posterior inference without requiring additional boilerplate code.

## Installation

The BayesDensity.jl package is currently not part of any package repository, but can be installed from its GitHub repository as follows:
```julia
using Pkg
Pkg.add(url="https://github.com/oskarhs/BayesianDensityEstimation.jl/lib/BayesDensity.jl")
```

Alternatively, it is possible to install each of the Bayesian density estimators implemented in this package separately. For instance, the B-spline mixture model estimator can be downloaded as follows:
```julia
Pkg.add(url="https://github.com/oskarhs/BayesianDensityEstimation.jl/lib/BayesDensityBSplineMixture.jl")
```

## Quick start

To get started, we illustrate the basic use of the package by fitting a B-spline mixture model to a two-component mixture of normal densities:

```julia
using BayesDensity, Distributions, Random
rng = Random.Xoshiro(1) # for reproducibility

# Simulate some data:
d_true = MixtureModel([Normal(-0.2, 0.25), Normal(0.5, 0.15)], [0.4, 0.6])
x = rand(rng, d_true, 1000)

# Create a B-Spline mixture model object:
bsm = BSplineMixture(x)
```

Having specified a model for the data, we can perform posterior inference through Markov chain Monte Carlo methods or variational inference:

```julia
mcmc_fit = sample(rng, bsm, 5000; n_burnin=1000) # MCMC
vi_fit = varinf(bsm) # VI
```

The resulting fitted model objects can be used to compute posterior quantities of interest such as the posterior median of $f(t)$ through `median(mcmc_fit, t)`. Additionally, the package also provides convenience plotting functions through its [Makie.jl](https://github.com/MakieOrg/Makie.jl) and [Plots.jl](https://github.com/JuliaPlots/Plots.jl) extensions, making it easy to visualize the density estimates. For instance, one can easily plot the posterior mean, along with a 95% credible interval with Makie as follows:

```julia
using CairoMakie
plot(mcmc_fit)
```

For a more thorough introduction to the API and the capabilities of the package, we refer the interested reader to the DOCUMENTATION

## Development phase
The package is currently under a period of heavy development, and new features will as such be added in rapid succession.
To be able to use the latest features of the package, make sure that you currently have the latest version of the package installed.

In particular, here is a non-exhaustive list of planned features (roguhly in order of priority)

- Switch to using ProductDistribution where this is possible.
- Implement an MCMC algorithm for RandomFixedGaussianMixture
- Finish writing the general documentation, including a primer on Bayesian nonparametric density estimation.
- Implement Bernstein polynomial model (variable K)