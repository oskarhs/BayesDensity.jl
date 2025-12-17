using BayesDensityBSM
using Test
using Random, Distributions

const rng = Random.Xoshiro(1)

include("aqua.jl")

@testset "BSMM: Constructor and model object" begin
    K = 20
    x = randn(rng, 10)

    @test length(basis(BSMModel(x, K))) == 20

    @test order(BSMModel(x)) == 4

    @test typeof(hyperparams(BSMModel(x))) <: NTuple{4, <:Real}

    @test Distributions.support(BSMModel([0.0], (-1.0, 1.0))) == (-1.0, 1.0)

    for model in (BSMModel(x), BSMModel(x, n_bins=nothing))
        io = IOBuffer() # just checks that we can call the show method
        show(io, BSMModel(x))
        output = String(take!(io))
        @test typeof(output) == String
    end
end

@testset "BSMM: Constructor throws error" begin
    K = 20
    x = collect(-5:0.1:5)

    @test_throws ArgumentError BSMModel(x, (-1, 1))
    @test_throws ArgumentError BSMModel(x, (1, -1))

    for hyp in [:a_τ, :b_τ, :a_δ, :b_δ]
        @eval @test_throws ArgumentError $BSMModel($x; $hyp = -1)
    end

    @test_throws ArgumentError BSMModel(x; n_bins=-10)
end

@testset "BSMM: MC Sample" begin
    K = 20
    x = collect(-5:0.1:5)

    bsm1 = BSMModel(x; n_bins = 20)
    @test typeof(sample(rng, bsm1, 100)) <: PosteriorSamples

    bsm2 = BSMModel(x; n_bins = nothing)
    @test typeof(sample(rng, bsm2, 100)) <: PosteriorSamples
end

@testset "BSMM: pdf and mean" begin
    K = 20
    x = collect(0:0.1:1)
    t = LinRange(0, 1, 11)

    bsm = BSMModel(x, K, (0,1))

    samples1 = [(spline_coefs = ones(K),) for _ in 1:10]
    ps1 = PosteriorSamples(samples1, bsm, 100, 0)
    @test isapprox(pdf(bsm, samples1, t), ones((length(t), length(samples1))))
    @test isapprox(mean(ps1, t), ones(length(t)))

    samples2 = [(β = BayesDensityBSM.compute_μ(basis(bsm)),) for _ in 1:10]
    ps2 = PosteriorSamples(samples2, bsm, 100, 0)
    @test isapprox(pdf(bsm, samples2, t), ones((length(t), length(samples2))))
    @test isapprox(mean(ps2, t), ones(length(t)))
end