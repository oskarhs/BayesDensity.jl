using BayesDensityRandomFiniteGaussianMixture
using Test, Aqua

@testset "RandomFiniteGaussianMixture: Aqua.jl" begin
    Aqua.test_all(BayesDensityPitmanYorMixture)
end