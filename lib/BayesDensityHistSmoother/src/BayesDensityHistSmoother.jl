module BayesDensityHistSmoother

using Reexport

@reexport using BayesDensityCore

import BayesDensityCore: linear_binning

using Distributions, BSplineKit, LinearAlgebra, SparseArrays, StatsBase, Random, MixedModels, DataFrames, Logging

import Distributions: support

include("HistSmoother.jl")

export HistSmoother

include("spline_utils.jl")

include("mcmc.jl")

include("variational.jl")

export HistSmootherVIPosterior

end # module BayesDensitySHS
