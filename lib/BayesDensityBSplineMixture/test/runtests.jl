using BayesDensityBSplineMixture
using Test
using Random, Distributions, LinearAlgebra

const rng = Random.Xoshiro(1)

include("aqua.jl")

@testset "BSplineMixture: Constructor and model object" begin
    K = 20
    x = randn(rng, 10)

    @test length(basis(BSplineMixture(x; K=K))) == 20

    @test order(BSplineMixture(x)) == 4

    @test typeof(hyperparams(BSplineMixture(x))) <: NamedTuple

    @test Distributions.support(BSplineMixture(LinRange(-0.5, 0.5, 11); bounds = (-1.0, 1.0))) == (-1.0, 1.0)

    @test BSplineMixture(x) == BSplineMixture(x)

    for model in (BSplineMixture(x), BSplineMixture(x, n_bins=nothing))
        io = IOBuffer() # just checks that we can call the show method
        show(io, model)
        output = String(take!(io))
        @test typeof(output) == String
    end
end

@testset "BSplineMixture: Constructor throws error" begin
    K = 20
    x = collect(-5:0.1:5)

    @test_throws ArgumentError BSplineMixture(x; bounds=(-1, 1))
    @test_throws ArgumentError BSplineMixture(x; bounds=(1, -1))

    for hyp in [:a_τ, :b_τ, :a_δ, :b_δ]
        @eval @test_throws ArgumentError $BSplineMixture($x; $hyp = -1)
    end

    @test_throws ArgumentError BSplineMixture(x; n_bins=-10)
end

@testset "BSplineMixture: MC: sample" begin
    K = 20
    x = collect(-5:0.1:5)

    bsm1 = BSplineMixture(x; n_bins = 20)
    @test typeof(sample(rng, bsm1, 100)) <: PosteriorSamples{Float64}

    bsm2 = BSplineMixture(x; n_bins = nothing)
    @test typeof(sample(rng, bsm2, 100)) <: PosteriorSamples{Float64}
end

@testset "BSplineMixture: MC: pdf and mean" begin
    K = 20
    x = collect(0:0.1:1)
    t = LinRange(0, 1, 11)

    bsm = BSplineMixture(x; K=K, bounds=(0,1))

    samples1 = [(spline_coefs = ones(K),) for _ in 1:10]
    ps1 = PosteriorSamples{Float64}(samples1, bsm, 100, 0)
    @test isapprox(pdf(bsm, samples1, t), ones((length(t), length(samples1))))
    @test isapprox(mean(ps1, t), ones(length(t)))

    samples2 = [(β = BayesDensityBSplineMixture.compute_μ(basis(bsm)),) for _ in 1:10]
    ps2 = PosteriorSamples{Float64}(samples2, bsm, 100, 0)
    @test isapprox(pdf(bsm, samples2, t), ones((length(t), length(samples2))))
    @test isapprox(mean(ps2, t), ones(length(t)))
end

@testset "BSplineMixture: VI: varinf, sample, print" begin
    K = 20
    x = collect(0:0.1:1)
    t = LinRange(0, 1, 11)

    bsm1 = BSplineMixture(x; K=K, bounds=(0,1))
    vip1 = varinf(bsm1; max_iter = 10)
    
    @test typeof(vip1) <: AbstractVIPosterior
    @test typeof(sample(vip1, 10)) <: PosteriorSamples{Float64}

    bsm2 = BSplineMixture(x; K=K, bounds=(0,1), n_bins=nothing)
    vip2 = varinf(bsm2; max_iter = 10)
    @test typeof(vip1) <: AbstractVIPosterior
    @test typeof(sample(vip1, 10)) <: PosteriorSamples{Float64}

    io = IOBuffer() # just checks that we can call the show method
    show(io, vip1)
    output = String(take!(io))
    @test typeof(output) == String
end

@testset "BSplineMixture: VI: mean" begin
    K = 20
    x = collect(0:0.1:1)
    t = LinRange(0, 1, 11)
    bsm = BSplineMixture(x; K=K, bounds=(0,1))

    # Create a dummy variational posterior with q_β = MvNormalCanon(inv_Σ * μ, inv_Σ), where inv_Σ is a diagonal matrix with very large entried (then q_β is almost a point mass at μ)
    # Then the VIP mean should be approximately uniform...
    inv_Σ = Diagonal(fill(1e10, K-1))
    μ = BayesDensityBSplineMixture.compute_μ(basis(bsm))
    vip = BSplineMixtureVIPosterior{Float64}(μ, inv_Σ, 1.0, 1.0, fill(1.0, K-3), fill(1.0, K-3), bsm)
    @test isapprox(mean(rng, vip, t), fill(1.0, length(t)), rtol=1e-3)
end