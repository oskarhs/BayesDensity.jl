using BayesDensityRandomBernsteinPoly
using Test, Aqua

@testset "RandomBernsteinPoly: Aqua.jl" begin
    Aqua.test_all(BayesDensityRandomBernsteinPoly)
end