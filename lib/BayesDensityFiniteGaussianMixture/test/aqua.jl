using BayesDensityFiniteGaussianMixture
using Test, Aqua

@testset "FiniteGaussianMixture: Aqua.jl" begin
    Aqua.test_all(BayesDensityFiniteGaussianMixture)
end