using Test
using BayesDensityRandomBernsteinPoly
using Distributions, Random

const rng = Random.Xoshiro(1)

# include("aqua.jl")

@testset "RandomBernsteinPoly: Constructor and model object" begin
    x = [1.0, -1.0]

    rbp = RandomBernsteinPoly(x)

    # Test equality
    @test rbp == RandomBernsteinPoly(x)

    # Test that we have subtyped AbstractBayesDensityModel
    @test rbp isa AbstractBayesDensityModel{Float64}

    # Check that the hyperparams method can be used without any issues
    @test hyperparams(rbp) isa NamedTuple

    # Check that the support is returned correctly
    @test Distributions.support(rbp) == (-1.1, 1.1)

    # Check that out of bounds hyperparameter values throw errors
    for hyp in (:prior_strength,)
        @eval @test_throws ArgumentError $RandomBernsteinPoly($x, $hyp = -1)
    end

    # Test show method
    io = IOBuffer() # just checks that we can call the show method
    show(io, rbp)
    output = String(take!(io))
    @test output isa String
end
