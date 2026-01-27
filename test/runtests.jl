push!(LOAD_PATH, joinpath(@__DIR__, "..", "lib"))
using BayesDensityCore
using BayesDensityBSplineMixture
using BayesDensityHistSmoother
using BayesDensityPitmanYorMixture
using Test

# Core:
include(joinpath(@__DIR__, "..", "lib", "BayesDensityCore", "test", "runtests.jl"))

# BSplineMixture
include(joinpath(@__DIR__, "..", "lib", "BayesDensityBSplineMixture", "test", "runtests.jl"))

# HistSmoother
include(joinpath(@__DIR__, "..", "lib", "BayesDensityHistSmoother", "test", "runtests.jl"))

# PitmanYorMixture
include(joinpath(@__DIR__, "..", "lib", "BayesDensityPitmanYorMixture", "test", "runtests.jl"))