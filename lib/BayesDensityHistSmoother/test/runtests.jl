using BayesDensityHistSmoother
using Test
using Random, BSplineKit

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

@testset "HistSmoother: pdf, support" begin
    # Set all betas to 0. Then the pdf should equal the uniform density 
    x = collect(LinRange(0, 1, 51))
    t = LinRange(0, 1, 11)

    K = 20
    shs = HistSmoother(x; K = K, bounds = (0, 1))

    # Test evaluation for single parameter sample, vector of evaluation points
    @test isapprox(pdf(shs, (β = zeros(K),), t), fill(1/1.10, length(t)))

    # Vector of parameter samples, vector of evaluation points
    n_fill = 20
    @test isapprox(pdf(shs, fill((β = zeros(K),), n_fill), t), fill(1/1.10, (length(t), n_fill)))

    # Single parameter, single evaluation point
    @test isapprox(pdf(shs, (β = zeros(K),), t[div(length(t), 2)]), 1/1.1)

    # Vector of parameter samples, single evaluation point
    @test isapprox(pdf(shs, fill((β = zeros(K),), n_fill), t[div(length(t), 2)]), fill(1/1.10, (1, n_fill)))

    # Now test values not in the support of the model:
    smin, smax = BayesDensityHistSmoother.support(shs)
    @test pdf(shs, (β = zeros(K),), smin - 1e-4) == 0.0
    @test isapprox(pdf(shs, (β = zeros(K),), smin + 1e-4), 1/1.1)
    @test pdf(shs, (β = zeros(K),), smax + 1e-4) == 0.0
    @test isapprox(pdf(shs, (β = zeros(K),), smax - 1e-4), 1/1.1)
end