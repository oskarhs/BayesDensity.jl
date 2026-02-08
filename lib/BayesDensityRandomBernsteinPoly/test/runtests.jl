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

@testset "RandomBernsteinPoly: pdf and cdf" begin
    x = [0.5, 0.7]
    
    rbp = RandomBernsteinPoly(x; prior_strength = 1, support=(0.0, 1.0))
    w = [0.2, 0.8]
    dist_ref = MixtureModel([Beta(1, 2), Beta(2, 1)], w)

    n_rep = 10
    parameters = (w = copy(w),)
    parameters_vec = fill(parameters, n_rep)
    # Evaluate at single point:
    @test isapprox(pdf(rbp, parameters, 0.0), pdf(dist_ref, 0.0))
    @test isapprox(pdf(rbp, parameters_vec, 0.0), fill(pdf(dist_ref, 0.0), (1, n_rep)))
    @test isapprox(cdf(rbp, parameters, 0.0), cdf(dist_ref, 0.0))
    @test isapprox(cdf(rbp, parameters_vec, 0.0), fill(cdf(dist_ref, 0.0), (1, n_rep)))

    # Evaluate at multiple points:
    t = LinRange(-5, 5, 11)
    @test isapprox(pdf(rbp, parameters, t), pdf(dist_ref, t))
    @test isapprox(pdf(rbp, parameters_vec, t), mapreduce(x->pdf(dist_ref, t), hcat, fill(0, n_rep)))
    @test isapprox(cdf(rbp, parameters, t), cdf(dist_ref, t))
    @test isapprox(cdf(rbp, parameters_vec, t), mapreduce(x->cdf(dist_ref, t), hcat, fill(0, n_rep)))
end