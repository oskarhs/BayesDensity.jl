module BayesDensityBSplineMixture

using Reexport

@reexport using BayesDensityCore

import BayesDensityCore: logit, softmax, logistic_stickbreaking, bin_regular, unitvector

using BSplineKit

using Random, Distributions, StatsBase, BandedMatrices, PolyaGammaHybridSamplers, LinearAlgebra, SparseArrays, SelectedInversion

import Distributions: support

include("spline_utils.jl")
include("BSplineMixture.jl")
include("mcmc.jl")

export BSplineMixture, sample, hyperparams, order, knots, basis

include("variational.jl")

export varinf, BSplineMixtureVIPosterior


end # module
