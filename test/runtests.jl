push!(LOAD_PATH, joinpath(@__DIR__, "..", "lib"))
using Test

# Core:
include(joinpath(@__DIR__, "..", "lib", "BayesDensityCore", "test", "runtests.jl"))

# BSplineMixture
include(joinpath(@__DIR__, "..", "lib", "BayesDensityBSplineMixture", "test", "runtests.jl"))

# HistSmoother
include(joinpath(@__DIR__, "..", "lib", "BayesDensityHistSmoother", "test", "runtests.jl"))

# PitmanYorMixture
include(joinpath(@__DIR__, "..", "lib", "BayesDensityPitmanYorMixture", "test", "runtests.jl"))

# FiniteGaussianMixture
include(joinpath(@__DIR__, "..", "lib", "BayesDensityFiniteGaussianMixture", "test", "runtests.jl"))

# RandomBernsteinPoly
include(joinpath(@__DIR__, "..", "lib", "BayesDensityRandomBernsteinPoly", "test", "runtests.jl"))
