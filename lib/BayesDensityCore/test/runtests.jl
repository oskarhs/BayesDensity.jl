using BayesDensityCore
using Distributions, Random, StatsBase
using Test

const rng = Random.Xoshiro(1)

# include("aqua.jl")

# Random Histogram model on [0, 1] with K-dimensional Dir(fill(a, K))-prior.
struct RandomHistogram{T<:Real, NT<:NamedTuple} <: AbstractBayesDensityModel{T}
    data::NT
    K::Int
    a::T
    function RandomHistogram{T}(x::AbstractVector{<:Real}, K::Int; a::Real=1.0) where {T<:Real}
        T_x = T.(x)
        xmin, xmax = T(0), T(1)
        binedges = LinRange(xmin, xmax, K+1)
        bincounts = BayesDensityCore.bin_regular(T_x, xmin, xmax, K, true)
        data = (x = x, binedges=binedges, bincounts=bincounts)
        new{T, typeof(data)}(data, K, a)
    end
end

function Base.:(==)(rh1::RandomHistogram, rh2::RandomHistogram)
    return rh1.data == rh2.data && rh1.a == rh2.a && rh1.K == rh2.K
end

# Define pdf method for Random Histogram:
function Distributions.pdf(rhm::RandomHistogram, params::NT, t::Real) where {NT<:NamedTuple}
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

# Define cdf method for Random Histogram:
function Distributions.cdf(rhm::RandomHistogram, params::NT, t::Real) where {NT<:NamedTuple}
    (; data, K, a) = rhm
    (; θ) = params
    breaks = data.binedges

    val = 0.0
    if (breaks[1] ≤ t ≤ breaks[end])
        idx = max(1, searchsortedfirst(breaks, t) - 1)
        val = sum(θ[1:idx-1]) + θ[idx] * (t-breaks[idx]) * K
    end
    return val
end

function StatsBase.sample(rng::Random.AbstractRNG, rhm::RandomHistogram{T, NT}, n_samples::Int; n_burnin = min(1000, div(n_samples, 5))) where {T, NT}
    (; data, K, a) = rhm
    samples = Vector{NamedTuple{(:θ,), Tuple{Vector{T}}}}(undef, n_samples)
    for i in 1:n_samples
        θ = rand(rng, Dirichlet(fill(a, K) + data.bincounts))
        samples[i] = (θ = θ,)
    end
    return PosteriorSamples{T}(samples, rhm, n_samples, n_burnin)
end

# This is actually the true posterior in this case
struct RHPosterior{T<:Real, S<:RandomHistogram} <: AbstractVIPosterior{T}
    α::Vector{T}
    model::S
    function RHPosterior{T}(rhm::RandomHistogram) where {T<:Real}
        (; data, K, a) = rhm
        α = a .+ data.bincounts
        return new{T, typeof(rhm)}(α, rhm)
    end
end

function StatsBase.sample(rng::Random.AbstractRNG, rhp::RHPosterior{T, S}, n_samples::Int) where {T, S}
    (; α, model) = rhp
    samples = Vector{NamedTuple{(:θ,), Tuple{Vector{Float64}}}}(undef, n_samples)
    for i in 1:n_samples
        θ = rand(rng, Dirichlet(α))
        samples[i] = (θ = θ,)
    end
    return PosteriorSamples{T}(samples, model, n_samples, 0)
end

function BayesDensityCore.varinf(rhm::RandomHistogram{T, NT}) where {T, NT}
    posterior = RHPosterior{T}(rhm)
    info = VariationalOptimizationResult{T}([0.0], true, 1, 0.0, posterior)
end

@testset "Core: pdf and cdf fallback methods" begin
    K = 15
    a = 1.0
    x = vcat(fill(0.11, 100), fill(0.51, 100), fill(0.91, 100))
    rhm = RandomHistogram{Float64}(x, K; a=a)

    L = 1001
    n_rep = 10

    params = (θ = fill(1/K, K),) # Uniform parameter
    params_vec = fill(params, n_rep)

    # Test evaluation for single params, a collection of t's
    @test pdf(rhm, params, LinRange(0, 1, L)) == fill(1.0, L)

    # Test evaluation for vector of params, single t
    @test pdf(rhm, params_vec, 0.2) == fill(1.0, (1, length(params_vec)))

    # Test evaluation for vector of params, vector of t's
    @test pdf(rhm, params_vec, LinRange(0, 1, L)) == fill(1.0, (L, length(params_vec)))

    # Test evaluation for single params, a collection of t's
    @test cdf(rhm, params, LinRange(0, 1, L)) ≈ [j/(L-1) for j in 0:(L-1)]

    # Test evaluation for vector of params, single t
    @test cdf(rhm, params_vec, 0.2) ≈ fill(0.2, (1, length(params_vec)))

    # Test evaluation for vector of params, vector of t's
    @test cdf(rhm, params_vec, LinRange(0, 1, L)) ≈ [j/(L-1) for j in 0:(L-1), i in eachindex(params_vec)]
end

@testset "Core: MC sample" begin
    K = 15
    a = 1.0
    x = vcat(fill(0.11, 100), fill(0.51, 100), fill(0.91, 100))
    rhm = RandomHistogram{Float64}(x, K; a=a)

    n_samples = 2000
    model_fit = sample(rng, rhm, n_samples)

    @test typeof(model_fit) <: PosteriorSamples{Float64}
    @test length(model_fit.samples) == n_samples
end

@testset "Core: PosteriorSamples: drop_burnin, vcat" begin
    K = 15
    a = 1.0
    x = vcat(fill(0.11, 100), fill(0.51, 100), fill(0.91, 100))
    rhm = RandomHistogram{Float64}(x, K; a=a)

    n_samples = 2000
    model_fit1 = sample(rng, rhm, n_samples)
    model_fit2 = sample(rng, rhm, n_samples)
    model_fit3 = sample(rng, RandomHistogram{Float64}(LinRange(0, 1, 11), K), n_samples)

    @test n_burnin(drop_burnin(model_fit1)) == 0

    @test typeof(model_fit1) <: PosteriorSamples{Float64}
    @test length(model_fit1.samples) == n_samples

    @test_throws ArgumentError vcat(model_fit1, model_fit3)
    @test typeof(vcat(model_fit1, model_fit2)) <: PosteriorSamples{Float64}
end

@testset "Core: MC mean, var, std and quantile fallback methods" begin
    K = 15
    a = 1.0
    x = vcat(fill(0.11, 100), fill(0.51, 100), fill(0.91, 100))
    rhm = RandomHistogram{Float64}(x, K; a=a)

    # True posterior mean:
    θ_mean = (fill(a, K) + rhm.data.bincounts) / (a + length(x))

    # Create dummy PosteriorSamples object where all samples are equal to θ_mean:
    n_samples = 100
    params_vec = fill((θ = θ_mean,), n_samples)
    posterior = PosteriorSamples{Float64}(params_vec, rhm, n_samples, 0)

    t = LinRange(0, 1, 1001)
    posterior_mean_pdf = mean(posterior, t)
    posterior_mean_cdf = mean(posterior, cdf, t)
    posterior_median_pdf = median(posterior, t)
    posterior_median_cdf = median(posterior, cdf, t)
    posterior_std_pdf = std(posterior, t)
    posterior_std_cdf = std(posterior, cdf, t)

    # Test vector version (mean)
    @test isapprox(posterior_mean_pdf, pdf(rhm, params_vec[1], t))
    @test isapprox(posterior_mean_cdf, cdf(rhm, params_vec[1], t))

    # Test scalar version (mean)
    @test isapprox(posterior_mean_pdf[1], mean(posterior, t[1]))
    @test isapprox(posterior_mean_cdf[1], mean(posterior, cdf, t[1]))

    # Test vector version (std)
    @test all(posterior_std_pdf .== std(posterior, t[1]))
    @test posterior_std_cdf[1] == std(posterior, cdf, t[1])

    # Test vector version (median)
    @test isapprox(posterior_median_pdf, pdf(rhm, params_vec[1], t))
    @test isapprox(posterior_median_cdf, cdf(rhm, params_vec[1], t))

    # Test scalar version (median)
    @test isapprox(posterior_median_pdf[1], median(posterior, t[1]))
    @test isapprox(posterior_median_cdf[1], median(posterior, cdf, t[1]))
end

@testset "Core: VI: varinf, sample" begin
    K = 15
    a = 1.0
    x = vcat(fill(0.11, 100), fill(0.51, 100), fill(0.91, 100))
    rhm = RandomHistogram{Float64}(x, K; a=a)
    rhp, info = varinf(rhm)

    @test converged(info)
    @test elbo(info) == [0.0]
    @test n_iter(info) == 1
    @test tolerance(info) isa Real
    
    n_samples = 100

    @test typeof(rhp) <: AbstractVIPosterior

    ps = sample(rng, rhp, n_samples)

    @test typeof(ps) <: PosteriorSamples{Float64}
    @test length(ps.samples) == n_samples
end

@testset "Core: VI mean, var, std and quantile fallback methods" begin
    # Make a RandomHistogram with the parameter `a` set to a very high value (e.g. a = 1e20).
    # Then the posterior draws of `f(t)` should be very close to `1` no matter the data
    # If our implementation is correct, we should have e.g. `mean(vip, pdf, t) ≈ 1` and
    # `std(vip, pdf, t) ≈, 0` 
    K = 15
    a = 1e20
    x = vcat(fill(0.11, 100), fill(0.51, 100), fill(0.91, 100))
    rhm = RandomHistogram{Float64}(x, K; a=a)
    rhp = RHPosterior{Float64}(rhm)

    t = LinRange(0, 1, 11)
    qs = [0.2, 0.8]

    @test isapprox(quantile(rng, rhp, t, 0.2), ones(length(t)); atol=1e-3)
    @test isapprox(quantile(rng, rhp, cdf, t, 0.2), collect(t); atol=1e-3)

    @test isapprox(quantile(rng, rhp, t, qs), ones((length(t), length(qs))); atol=1e-3)
    @test isapprox(quantile(rng, rhp, cdf, t, qs), reduce(hcat, [collect(t) for _ in eachindex(qs)]); atol=1e-3)

    # Evaluate for scalar and vector inputs:
    for statistic in (:mean, :median)
        @eval begin
            @test isapprox($statistic($rng, $rhp, $t), ones(length($t)); atol=1e-3)
            @test isapprox($statistic($rng, $rhp, cdf, $t), collect($t); atol=1e-3)

            @test isapprox($statistic($rng, $rhp, $t[1]), 1.0; atol=1e-3)
            @test isapprox($statistic($rng, $rhp, cdf, $t[1]), $t[1]; atol=1e-3)
        end
    end

    # Evaluate for scalar and vector inputs:
    for statistic in (:var, :std)
        @eval begin
            @test isapprox($statistic($rng, $rhp, $t), zeros(length($t)); atol=1e-3)
            @test isapprox($statistic($rng, $rhp, cdf, $t), zeros(length($t)); atol=1e-3)

            @test isapprox($statistic($rng, $rhp, $t[1]), 0.0; atol=1e-3)
            @test isapprox($statistic($rng, $rhp, cdf, $t[1]), 0.0; atol=1e-3)
        end
    end
end