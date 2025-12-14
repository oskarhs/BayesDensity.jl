# <img src="docs/src/assets/logo.svg" alt="alt text" width="80" height="80" align="center"> BayesianDensityEstimation.jl

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://oskarhs.github.io/BayesianDensityEstimation.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://oskarhs.github.io/BayesianDensityEstimation.jl/dev/)
[![Build Status](https://github.com/oskarhs/BayesianDensityEstimation.jl/actions/workflows/CI.yml/badge.svg?branch=master)](https://github.com/oskarhs/BayesianDensityEstimation.jl/actions/workflows/CI.yml?query=branch%3Amaster)

A Julia package for nonparametric univariate Bayesian density estimation.

## Introduction

The main goal of this package is to provide easy access to various nonparametric Bayesian density estimators under a common API. BayesianDensityEstimation.jl is fully integrated with the Julia plotting ecosystem, allowing the user to easily visualize Bayesian density estimates, along with pointwise credible bands through the [Makie.jl](https://github.com/MakieOrg/Makie.jl) and [Plots.jl](https://github.com/JuliaPlots/Plots.jl) packages.

## Installation

PUT SOMETHING HERE ABOUT BEING ABLE TO INSTALL DIFFERENT DENSITY ESTIMATORS SEPARATELY.

## Quick start

To get started, we illustrate the basic use of the package by fitting a B-spline mixture model to a two-component mixture of normal densities:

```julia
using BayesianDensityEstimation, Distributions, Random
rng = Random.Xoshiro(1) # for reproducibility

# Simulate some data:
d_true = MixtureModel([Normal(-0.2, 0.25), Normal(0.5, 0.15)], [0.4, 0.6])
x = rand(rng, d_true, 1000) # 1000 random samples

# Create a B-Spline mixture model object:
bsm = BSMModel(x)

# Generate samples from the posterior distribution using Markov chain Monte Carlo methods:
posterior_samples = sample(rng, bsm, 5000; n_burnin=1000)
```

The `posterior_samples` object can be used to compute posterior quantities of interest such as the posterior mean of $f(t)$ through `mean(posterior_samples, t)`. Additionally, convenient plotting functions. For instance, one can easily plot the posterior median, along with a 

```julia
using CairoMakie
plot(posterior_samples, estimate=:median)
```

For a more thorough introduction to the API and the capabilities of the package, we refer the interested reader to the DOCUMENTATION