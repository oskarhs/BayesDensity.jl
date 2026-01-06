module BayesDensitySHS

using Reexport

@reexport using BayesDensityCore

using Distributions, BSplineKit, LinearAlgebra, SparseArrays, StatsBase, Random, MixedModels

import Distributions: support

include("SHSModel.jl")

export SHSModel

include("spline_utils.jl")

include("mcmc.jl")

end # module BayesDensitySHS
