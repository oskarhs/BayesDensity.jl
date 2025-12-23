using BayesDensitySHS
using Test
using Random, BSplineKit

const rng = Random.Xoshiro(1)

#include("aqua.jl")

@testset "SHSM: Constructor and model object" begin
    x = randn(rng, 10)

    shs = SHSModel(x)
    @test typeof(shs) <: SHSModel{Float64, <:AbstractBSplineBasis, <:NamedTuple}

    # Check that we can retrieve hyerparameter defaults
    @test hyperparams(shs) == (σ_β = 1e3, s_σ = 1e3)

#=     for model in (SHSModel(x), SHSModel(x))
        io = IOBuffer() # just checks that we can call the show method
        show(io, BSMModel(x))
        output = String(take!(io))
        @test typeof(output) == String
    end =#
end

@testset "SHSM: Constructor throws error" begin
    K = 20
    x = collect(-5:0.1:5)

    @test_throws ArgumentError SHSModel(x; bounds=(-1, 1))
    @test_throws ArgumentError SHSModel(x; bounds=(1, -1))

    for hyp in [:σ_β, :s_σ]
        @eval @test_throws ArgumentError $SHSModel($x; $hyp = -1)
    end

    @test_throws ArgumentError SHSModel(x; n_bins=-10)
end

@testset "SHSM: pdf, support" begin
    # Set all betas to 0. Then the pdf should equal the uniform density 
    x = collect(LinRange(0, 1, 51))
    t = LinRange(0, 1, 11)

    K = 20
    shs = SHSModel(x, K; bounds = (0, 1))

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
    smin, smax = BayesDensitySHS.support(shs)
    @test pdf(shs, (β = zeros(K),), smin - 1e-4) == 0.0
    @test isapprox(pdf(shs, (β = zeros(K),), smin + 1e-4), 1/1.1)
    @test pdf(shs, (β = zeros(K),), smax + 1e-4) == 0.0
    @test isapprox(pdf(shs, (β = zeros(K),), smax - 1e-4), 1/1.1)
end