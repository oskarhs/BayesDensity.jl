module BayesDensityRandomBernsteinPoly

using Reexport
@reexport using BayesDensityCore
using Distributions
using Random
using SpecialFunctions
using StatsBase

include("utils.jl")

include("RandomBernsteinPoly.jl")
export RandomBernsteinPoly

include("mcmc.jl")

end # module