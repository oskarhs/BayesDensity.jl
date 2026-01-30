module BayesDensityHistSmoother

using Reexport

@reexport using BayesDensityCore
import BayesDensityCore: linear_binning, support

using BSplineKit
using DataFrames
using Distributions
using LinearAlgebra
using Logging
using MixedModels
using Random
using SparseArrays
using SpecialFunctions
using StatsBase

include("HistSmoother.jl")
export HistSmoother

include("spline_utils.jl")

include("mcmc.jl")

include("variational.jl")
export HistSmootherVIPosterior

end # module BayesDensitySHS
