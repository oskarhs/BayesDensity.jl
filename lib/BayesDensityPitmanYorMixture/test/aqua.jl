using BayesDensityPitmanYorMixture
using Test, Aqua

@testset "PitmanYorMixture: Aqua.jl" begin
    Aqua.test_all(BayesDensityPitmanYorMixture)
end