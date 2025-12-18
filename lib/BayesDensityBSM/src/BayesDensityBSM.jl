module BayesDensityBSM

using Reexport

@reexport using BayesDensityCore

import BayesDensityCore: logit, softmax, logistic_stickbreaking, bin_regular, unitvector

using BSplineKit

using Random, Distributions, Base.Threads, StatsBase, BandedMatrices, PolyaGammaHybridSamplers, LinearAlgebra, SparseArrays, SelectedInversion

include("spline_utils.jl")
include("BSMModel.jl")
include("mcmc.jl")

export BSMModel, sample, hyperparams, pdf, order, knots, basis

include("variational.jl")

export varinf, BSMVIPosterior


end # module
