using BayesDensityCore
using Test, Aqua

@testset "Core: Aqua.jl" begin
    Aqua.test_all(BayesDensityCore)
end