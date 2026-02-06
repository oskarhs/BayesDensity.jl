module BayesDensityRandomBernsteinPoly

using Reexport
@reexport using BayesDensityCore
using Distributions
using Random
using StatsBase

include("RandomBernsteinPoly.jl")
export RandomBernsteinPoly

end # module