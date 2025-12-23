using BayesDensitySHS
using Test, Aqua

@testset "SHSM: Aqua.jl" begin
    Aqua.test_all(BayesDensitySHS)
end