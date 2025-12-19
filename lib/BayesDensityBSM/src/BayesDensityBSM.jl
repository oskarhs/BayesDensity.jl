module BayesDensityBSM

using Reexport

@reexport using BayesDensityCore

import BayesDensityCore: logit, softmax, logistic_stickbreaking, bin_regular, unitvector

using BSplineKit

using Random, Distributions, Base.Threads, StatsBase, BandedMatrices, PolyaGammaHybridSamplers, LinearAlgebra, SparseArrays, SelectedInversion

import Distributions: support # Not exported by Distributions for some reason.

include("spline_utils.jl")
include("BSMModel.jl")
include("mcmc.jl")

export BSMModel, sample, hyperparams, order, knots, basis, support

include("variational.jl")

export varinf, BSMVIPosterior


end # module
