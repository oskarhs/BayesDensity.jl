module BayesDensityBSplineMixture

using Reexport

@reexport using BayesDensityCore
import BayesDensityCore: support
using BandedMatrices
using BSplineKit
using Distributions
using LinearAlgebra
using PolyaGammaHybridSamplers
using Random
using SelectedInversion
using SparseArrays
using SpecialFunctions
using StatsBase


include("spline_utils.jl")
include("utils.jl")

include("BSplineMixture.jl")
export BSplineMixture, order, knots, basis

include("mcmc.jl")

include("variational.jl")
export varinf, BSplineMixtureVIPosterior

end # module
