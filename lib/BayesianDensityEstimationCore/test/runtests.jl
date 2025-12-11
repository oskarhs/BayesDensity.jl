using BayesianDensityEstimationCore
using Distributions
using Test

# Random Histogram model on [0, 1] with K-dimensional Dir(a)-prior.
struct RandomHistogramModel{T<:Real, NT<:NamedTuple} <: AbstractBayesianDensityModel
    data::NT
    K::Int
    a::T
    function RandomHistogramModel{T}(x::AbstractVector{<:Real}, K::Int; a::Real=1.0) where {T<:Real}
        T_x = T.(x)
        xmin, xmax = T(0), T(1)
        binedges = LinRange(xmin, xmax, K+1)
        bincounts = bin_regular(T_x, xmin, xmax, K, true)
        data = (x = x, binedges=binedges, bincounts=bincounts)
        new{T, typeof(data)}(data, K, a)
    end
end

function Distributions.pdf(rhm::RandomHistogramModel, params::NT, t::Real) where {NT}
    (; data, K, a) = rhm
    (; θ) = params
    breaks = data.binedges

    val = 0.0
    if (breaks[1] ≤ t ≤ breaks[end])
        idx = max(1, searchsortedfirst(hbreaks, x) - 1)
        val = K*θ[idx]
    end
    return val
end

@testset "pdf fallback methods" begin
    K = 15
    x = vcat(fill(0.11, 100), fill(0.51, 100), fill(0.91, 100))
    rhm = RandomHistogramModel{Float64}(x, K; a=1.0)

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
    @test pdf(rhm, params_vec, LinRange(0, 1, L)) == fill(1.0, (L, legnth(params_vec)))
end

@testset "sample" begin
    
end

@testset "mean fallback methods" begin
    
end

@testset "quantile fallback methods" begin
    
end