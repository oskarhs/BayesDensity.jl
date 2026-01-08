using BayesDensityBSplineMixture
using Test, Aqua

@testset "BSplineMixture: Aqua.jl" begin
    Aqua.test_all(BayesDensityBSplineMixture)
end