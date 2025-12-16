module BayesianDensityEstimationBSM

using Reexport

@reexport using BayesianDensityEstimationCore

import BayesianDensityEstimationCore: logit, softmax, logistic_stickbreaking, bin_regular, unitvector

using BSplineKit

using Random, Distributions, Base.Threads, StatsBase, BandedMatrices, PolyaGammaHybridSamplers, LinearAlgebra, SparseArrays

include("spline_utils.jl")
include("BSMModel.jl")
include("mcmc.jl")
include("variational.jl")

export BSMModel, sample, hyperparams, pdf, basis, order, knots
public support

end # module BayesianDensityEstimationBSM
