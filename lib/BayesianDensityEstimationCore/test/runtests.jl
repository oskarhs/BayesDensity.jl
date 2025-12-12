using BayesianDensityEstimationCore
using Distributions, Random, StatsBase
using Test

const rng = Random.Xoshiro(1)

# Random Histogram model on [0, 1] with K-dimensional Dir(fill(a, K))-prior.
struct RandomHistogramModel{T<:Real, NT<:NamedTuple} <: AbstractBayesianDensityModel
    data::NT
    K::Int
    a::T
    function RandomHistogramModel{T}(x::AbstractVector{<:Real}, K::Int; a::Real=1.0) where {T<:Real}
        T_x = T.(x)
        xmin, xmax = T(0), T(1)
        binedges = LinRange(xmin, xmax, K+1)
        bincounts = BayesianDensityEstimationCore.bin_regular(T_x, xmin, xmax, K, true)
        data = (x = x, binedges=binedges, bincounts=bincounts)
        new{T, typeof(data)}(data, K, a)
    end
end

function Distributions.pdf(rhm::RandomHistogramModel, params::NT, t::Real) where {NT<:NamedTuple}
    (; data, K, a) = rhm
    (; θ) = params
    breaks = data.binedges

    val = 0.0
    if (breaks[1] ≤ t ≤ breaks[end])
        idx = max(1, searchsortedfirst(breaks, t) - 1)
        val = K*θ[idx]
    end
    return val
end

function StatsBase.sample(rng::Random.AbstractRNG, rhm::RandomHistogramModel, n_samples::Int; n_burnin = min(1000, div(n_samples, 5)))
    (; data, K, a) = rhm
    samples = Vector{NamedTuple{(:θ,), Tuple{Vector{typeof(a)}}}}(undef, n_samples)
    for i in 1:n_samples
        θ = rand(rng, Dirichlet(fill(a, K) + data.bincounts))
        samples[i] = (θ = θ,)
    end
    return BayesianDensitySamples{typeof(a)}(samples, rhm, n_samples, n_burnin)
end

@testset "pdf fallback methods" begin
    K = 15
    a = 1.0
    x = vcat(fill(0.11, 100), fill(0.51, 100), fill(0.91, 100))
    rhm = RandomHistogramModel{Float64}(x, K; a=a)

    L = 1001
    n_rep = 10

    params = (θ = fill(1/K, K),) # Uniform parameter
    params_vec = fill(params, n_rep)

    #@test pdf(rhm, params, t) == 1/K

    # Test evaluation for single params, a collection of t's
    @test pdf(rhm, params, LinRange(0, 1, L)) == fill(1.0, L)

    # Test evaluation for vector of params, single t
    @test pdf(rhm, params_vec, 0.2) == fill(1.0, (1, length(params_vec)))

    # Test evaluation for vector of params, vector of t's
    @test pdf(rhm, params_vec, LinRange(0, 1, L)) == fill(1.0, (L, length(params_vec)))
end

@testset "sample" begin
    K = 15
    a = 1.0
    x = vcat(fill(0.11, 100), fill(0.51, 100), fill(0.91, 100))
    rhm = RandomHistogramModel{Float64}(x, K; a=a)

    n_samples = 2000
    model_fit = sample(rng, rhm, n_samples)

    @test typeof(model_fit) <: BayesianDensitySamples
    @test length(model_fit.samples) == n_samples
end

@testset "mean and quantile fallback methods" begin
    K = 15
    a = 1.0
    x = vcat(fill(0.11, 100), fill(0.51, 100), fill(0.91, 100))
    rhm = RandomHistogramModel{Float64}(x, K; a=a)

    # True posterior mean:
    θ_mean = (fill(a, K) + rhm.data.bincounts) / (a + length(x))

    # Create dummy BayesianDensitySamples object where all samples are equal to θ_mean:
    n_samples = 2000
    params_vec = fill((θ = θ_mean,), n_samples)
    posterior = BayesianDensitySamples{Float64}(params_vec, rhm, n_samples, 0)

    t = LinRange(0, 1, 1001)
    posterior_mean = mean(posterior, t)
    posterior_median = median(posterior, t)

    # Test vector version (mean)
    @test isapprox(posterior_mean, pdf(rhm, params_vec[1], t))

    # Test scalar version (mean)
    @test isapprox(posterior_mean[1], mean(posterior, t[1]))

    # Test vector version (median)
    @test isapprox(posterior_median, pdf(rhm, params_vec[1], t))

    # Test scalar version (median)
    @test isapprox(posterior_median[1], median(posterior, t[1]))
end