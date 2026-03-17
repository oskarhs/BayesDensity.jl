using BayesDensityCore
using BayesDensityLogisticGaussianProcess
using Distributions
using LinearAlgebra
using Random
using StatsBase
using Test

const rng = Random.Xoshiro(1)

include("aqua.jl")

@testset "LogisticGaussianProcess: Constructor and model object" begin
    x = rand(rng, 10)

    @test LogisticGaussianProcess(x) == LogisticGaussianProcess(x; n_bins = 400)

    @test BayesDensityLogisticGaussianProcess.hyperparams(LogisticGaussianProcess(x)) isa NamedTuple

    @test BayesDensityCore.support(LogisticGaussianProcess(x; bounds=(0,1))) == (0.0, 1.0)

    # Check that we can call the show method:
    io = IOBuffer() # just checks that we can call the show method
    show(io, LogisticGaussianProcess(x))
    output = String(take!(io))
    @test output isa String
end

@testset "LogisticGaussianProcess: Constructor throws error" begin
    K = 20
    x = collect(-5:0.1:5)

    @test_throws ArgumentError LogisticGaussianProcess(x; n_bins=-10)

    # Test invalid bounds
    @test_throws ArgumentError LogisticGaussianProcess(x; bounds=(-1, 1))
    @test_throws ArgumentError LogisticGaussianProcess(x; bounds=(1, -1))

    # Test negative hyperparameters
    hyperparams = [:prior_variance_scale, :prior_length_scale]

    for hyp in hyperparams
        kwargs = Dict(hyp => -1)
        @test_throws ArgumentError LogisticGaussianProcess(x; kwargs...)
    end
end

@testset "LogisticGaussianProcess: pdf and cdf" begin
    n_bins = 20
    x = collect(0:0.1:1)
    L = 11
    t = LinRange(0, 1, L)

    lgp = LogisticGaussianProcess(x; n_bins = n_bins, bounds=(0,1))

    samples1 = [(β = zeros(n_bins),) for _ in 1:10]

    @test isapprox(pdf(lgp, samples1, t), ones((length(t), length(samples1))))
    @test isapprox(pdf(lgp, samples1, 0.5), fill(1.0, (1, 10)))
    @test isapprox(pdf(lgp, (β = zeros(n_bins),), 0.5), 1.0)

    @test isapprox(cdf(lgp, samples1, t), [j/(L-1) for j in 0:(L-1), i in eachindex(samples1)])
    @test isapprox(cdf(lgp, samples1, 0.5), fill(0.5, (1, 10)))
    @test isapprox(cdf(lgp, (β = zeros(n_bins),), 0.5), 0.5)
end

@testset "LogisticGaussianProcess: LogisticGaussianProcessLaplacePosterior" begin
    # Create a dummy Laplace approximation that is essentially a point mass at 0. Then the posterior of f(x) is approximately uniform.
    n_bins = 20
    x = collect(0:0.1:1)
    L = 11
    t = LinRange(0, 1, L)

    lgp = LogisticGaussianProcess(x; n_bins = n_bins, bounds=(0,1))
    lap = LogisticGaussianProcessLaplacePosterior{Float64}(zeros(n_bins), 1e-12 * I(n_bins), 1.0, 1.0, lgp)

    ps1 = sample(rng, lap, 1)

    # Test pdf evaluation when val_pdf/val_cdf is present
    @test isapprox(pdf(model(ps1), samples(ps1)[1], t), fill(1.0, length(t)); rtol=1e-5)
    @test isapprox(cdf(model(ps1), samples(ps1)[1], t), collect(t); rtol=1e-5)

    ps2 = sample(rng, lap, 10)

    @test isapprox(pdf(lgp, samples(ps2), t), ones((length(t), length(samples(ps2)))); rtol=1e-4)
    @test isapprox(cdf(lgp, samples(ps2), t), [j/(L-1) for j in 0:(L-1), i in eachindex(samples(ps2))]; rtol=1e-4)
end
