using BayesDensityHistSmoother
using Test, Aqua

@testset "HistSmoother: Aqua.jl" begin
    Aqua.test_all(BayesDensityHistSmoother)
end