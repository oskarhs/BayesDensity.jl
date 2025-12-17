using BayesDensityBSM
using Test, Aqua

@testset "BSMM: Aqua.jl" begin
    Aqua.test_all(BayesDensityBSM)
end