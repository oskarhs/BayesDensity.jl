module BayesDensityLogisticGaussianProcess

using Reexport

@reexport using BayesDensityCore
import BayesDensityCore: support
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
