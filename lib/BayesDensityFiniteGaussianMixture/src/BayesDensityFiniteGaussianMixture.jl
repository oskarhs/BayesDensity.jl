module BayesDensityFiniteGaussianMixture

using Reexport

@reexport using BayesDensityCore
using Distributions
using Random
using SpecialFunctions
using StatsBase

include("utils.jl")

include("FiniteGaussianMixture/FiniteGaussianMixture.jl")
export FiniteGaussianMixture

include("FiniteGaussianMixture/mcmc.jl")

include("FiniteGaussianMixture/variational.jl")
export FiniteGaussianMixtureVIPosterior

include("RandomFiniteGaussianMixture/RandomFiniteGaussianMixture.jl")
export RandomFiniteGaussianMixture

end # module
