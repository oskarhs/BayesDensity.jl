module BayesianDensityEstimationBSM

using BayesianDensityEstimationCore

export BayesianDensityChain, model

import BayesianDensityEstimationCore: logit, softmax, logistic_stickbreaking, bin_regular, unitvector

using BSplineKit

using Random, Distributions, Base.Threads, StatsBase, BandedMatrices, PolyaGammaHybridSamplers, LinearAlgebra, SparseArrays

include("spline_utils.jl")
include("BSMModel.jl")
include("mcmc.jl")

export BSMModel, sample, hyperparams, pdf

end # module BayesianDensityEstimationBSM
