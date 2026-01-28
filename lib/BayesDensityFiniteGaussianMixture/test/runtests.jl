using Test
using BayesDensityFiniteGaussianMixture
using Distributions, Random

const rng = Random.Xoshiro(1)

#include("aqua.jl")

@testset "FiniteGaussianMixture: Constructor and model object" begin
    x = [1.0, -1.0]

    gm = FiniteGaussianMixture(x, 2)

    # Test that we have subtyped AbstractBayesDensityModel
    @test typeof(gm) <: AbstractBayesDensityModel{Float64}

    # Check that the hyperparams method can be used without any issues
    @test typeof(hyperparams(gm)) <: NamedTuple

    # Check that the support is returned correctly
    @test Distributions.support(gm) == (-Inf, Inf)

    # Check that out of bounds hyperparameter values throw errors
    for hyp in (:prior_strength, :prior_variance, :prior_shape, :hyperprior_rate, :hyperprior_shape)
        @eval @test_throws ArgumentError $FiniteGaussianMixture($x, 2; $hyp = -1)
    end

    # Test show method
    io = IOBuffer() # just checks that we can call the show method
    show(io, gm)
    output = String(take!(io))
    @test typeof(output) == String
end

@testset "RandomFiniteGaussianMixture: Constructor and model object" begin
    x = [1.0, -1.0]

    gm = RandomFiniteGaussianMixture(x)

    # Test that we have subtyped AbstractBayesDensityModel
    @test typeof(gm) <: AbstractBayesDensityModel{Float64}

    # Check that the hyperparams method can be used without any issues
    @test typeof(hyperparams(gm)) <: NamedTuple

    # Check that the support is returned correctly
    @test Distributions.support(gm) == (-Inf, Inf)

    # Check that out of bounds hyperparameter values throw errors
    for hyp in (:prior_strength, :prior_variance, :prior_shape, :hyperprior_rate, :hyperprior_shape)
        @eval @test_throws ArgumentError $RandomFiniteGaussianMixture($x; $hyp = -1)
    end

    # Test show method
    io = IOBuffer() # just checks that we can call the show method
    show(io, gm)
    output = String(take!(io))
    @test typeof(output) == String
end


@testset "FiniteGaussianMixture: pdf and cdf" begin
    x = [1.0, -1.0]

    for gm in (FiniteGaussianMixture(x, 2; prior_strength = 1), RandomFiniteGaussianMixture(x; prior_strength = 1))

        n_rep = 10
        parameters = (μ = [0.0], σ2 = [1.0], w = [1.0])
        parameters_vec = fill(parameters, n_rep)

        # Evaluate at single point:
        @test isapprox(pdf(gm, parameters, 0.0), pdf(Normal(0, 1), 0.0))
        @test isapprox(pdf(gm, parameters_vec, 0.0), fill(pdf(Normal(0, 1), 0.0), (1, n_rep)))
        @test isapprox(cdf(gm, parameters, 0.0), cdf(Normal(0, 1), 0.0))
        @test isapprox(cdf(gm, parameters_vec, 0.0), fill(cdf(Normal(0, 1), 0.0), (1, n_rep)))

        # Evaluate at multiple points:
        t = LinRange(-5, 5, 11)
        @test isapprox(pdf(gm, parameters, t), pdf(Normal(0, 1), t))
        @test isapprox(pdf(gm, parameters_vec, t), mapreduce(x->pdf(Normal(0, 1), t), hcat, fill(0, n_rep)))
        @test isapprox(cdf(gm, parameters, t), cdf(Normal(0, 1), t))
        @test isapprox(cdf(gm, parameters_vec, t), mapreduce(x->cdf(Normal(0, 1), t), hcat, fill(0, n_rep)))
    end
end

@testset "FiniteGaussianMixture: MCMC sample" begin
    K = 2
    x = collect(-5:0.1:5)

    fgm = FiniteGaussianMixture(x, 2)
    @test typeof(sample(rng, fgm, 20)) <: PosteriorSamples{Float64}
end