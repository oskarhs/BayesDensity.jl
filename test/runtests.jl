using BayesDensityCore, BayesDensityBSM
using Test

# Core:
include(joinpath(@__DIR__, "..", "lib", "BayesDensityCore", "test", "runtests.jl"))

# BSM:
include(joinpath(@__DIR__, "..", "lib", "BayesDensityBSM", "test", "runtests.jl"))