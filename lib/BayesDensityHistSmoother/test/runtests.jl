using BayesDensityHistSmoother
using Test
using Random, BSplineKit, Distributions, LinearAlgebra

const rng = Random.Xoshiro(1)

#include("aqua.jl")

@testset "HistSmoother: Constructor and model object" begin
    x = randn(rng, 10)

    shs = HistSmoother(x)
    @test typeof(shs) <: HistSmoother{Float64, <:AbstractBSplineBasis, <:NamedTuple}

    # Check that we can retrieve hyerparameter defaults
    @test hyperparams(shs) == (σ_β = 1e3, s_σ = 1e3)

    # Test equality method
    @test HistSmoother(x) == shs

    for model in (HistSmoother(x),)
        io = IOBuffer() # just checks that we can call the show method
        show(io, model)
        output = String(take!(io))
        @test typeof(output) == String
    end
end

@testset "HistSmoother: Constructor throws error" begin
    x = collect(-5:0.1:5)

    @test_throws ArgumentError HistSmoother(x; bounds=(-1, 1))
    @test_throws ArgumentError HistSmoother(x; bounds=(1, -1))

    for hyp in [:σ_β, :s_σ]
        @eval @test_throws ArgumentError $HistSmoother($x; $hyp = -1)
    end

    @test_throws ArgumentError HistSmoother(x; n_bins=-10)
end

@testset "HistSmoother: pdf, cdf, support" begin
    # Set all betas to 0. Then the pdf should equal the uniform density 
    x = collect(LinRange(0, 1, 51))

    K = 20
    shs = HistSmoother(x; K = K, bounds = (0, 1))
    bs_min, bs_max = BayesDensityHistSmoother.support(shs)
    t = LinRange(bs_min, bs_max, 11)

    # Test evaluation for single parameter sample, vector of evaluation points
    @test isapprox(pdf(shs, (β = zeros(K),), t), fill(1/1.10, length(t)))
    @test isapprox(cdf(shs, (β = zeros(K),), t), collect(0:0.1:1))

    # Vector of parameter samples, vector of evaluation points
    n_fill = 20
    @test isapprox(pdf(shs, fill((β = zeros(K),), n_fill), t), fill(1/1.10, (length(t), n_fill)))
    @test isapprox(cdf(shs, fill((β = zeros(K),), n_fill), t), reduce(hcat, [collect(0:0.1:1) for _ in 1:n_fill]))

    # Single parameter, single evaluation point
    @test isapprox(pdf(shs, (β = zeros(K),), 0.5), 1/1.1)
    @test isapprox(cdf(shs, (β = zeros(K),), 0.5), 0.5)

    # Vector of parameter samples, single evaluation point
    @test isapprox(pdf(shs, fill((β = zeros(K),), n_fill), 0.5), fill(1/1.10, (1, n_fill)))
    @test isapprox(cdf(shs, fill((β = zeros(K),), n_fill), 0.5), fill(0.5, (1, n_fill)))

    # Now test values not in the support of the model:
    @test pdf(shs, (β = zeros(K),), bs_min - 1e-10) == 0.0
    @test cdf(shs, (β = zeros(K),), bs_min - 1e-10) == 0.0

    @test isapprox(pdf(shs, (β = zeros(K),), bs_min + 1e-10), 1/1.1)
    @test isapprox(cdf(shs, (β = zeros(K),), bs_min + 1e-10), 0.0; atol=1e-5)

    @test pdf(shs, (β = zeros(K),), bs_max + 1e-4) == 0.0
    @test isapprox(cdf(shs, (β = zeros(K),), bs_max + 1e-10), 1.0)

    @test isapprox(pdf(shs, (β = zeros(K),), bs_max - 1e-10), 1/1.1)
    @test isapprox(cdf(shs, (β = zeros(K),), bs_max - 1e-10), 1.0)
end

@testset "HistSmoother: MC: sample" begin
    K = 10
    x = collect(-5:0.1:5)

    shs = HistSmoother(x; K = K, n_bins = 20)
    @test typeof(sample(rng, shs, 100)) <: PosteriorSamples{Float64}
end

@testset "HistSmoother: varinf" begin
    K = 10
    x = collect(-5:0.1:5)

    shs = HistSmoother(x; K = K, n_bins = 20)
    vip = varinf(shs)
    @test typeof(vip) <: AbstractVIPosterior{Float64}
    @test typeof(sample(rng, vip, 100)) <: PosteriorSamples{Float64}
end

@testset "HistSmoother: HistSmootherVIPosterior" begin
    # Create a dummy VI posterior object where the posterior for β
    # is close to a point mass at 0. Then the posterior mean should be uniform...
    K = 10
    x = collect(-5:0.1:5)
    shs = HistSmoother(x; K = K, n_bins = 20)
    vip = HistSmootherVIPosterior{Float64}(zeros(K), 1e-12 * Diagonal(ones(K)), 1.0, 1.0, shs)

    # Now verify that the posterior looks reasonable
    bs_min, bs_max = BayesDensityHistSmoother.support(shs)
    t = LinRange(bs_min, bs_max, 11)
    @test isapprox(mean(vip, t), fill(1/(bs_max - bs_min), length(t)); rtol=1e-5)
end