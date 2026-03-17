module BayesDensityLogisticGaussianProcess

using BayesDensityCore
using BSplineKit
using Distributions
using Krylov
using LinearAlgebra
using LinearOperators
using NLSolversBase
using Optim
using Random
using StatsBase

include("utils.jl")

include("LogisticGaussianProcess.jl")
export LogisticGaussianProcess

include("laplace.jl")
export LogisticGaussianProcessLaplacePosterior, laplace_approximation

end
