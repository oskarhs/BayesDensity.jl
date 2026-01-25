struct TDistLocationScale{T<:Real} <: ContinuousUnivariateDistribution
    df::T
    location::T
    scale::T
end

function Distributions.logpdf(tdist::TDistLocationScale{T}, t::Union{S, AbstractVector{S}}) where {T<:Real, S<:Real}
    R = promote_type(T, S)
    (; df, location, scale) = tdist
    return @. loggamma((df+1)/2) - loggamma(df/2) - log(df*R(pi)*scale) - (df + 1) / 2 * log(1 + (t - location)^2/(df*scale))
end

function Distributions.rand(rng::AbstractRNG, tdist::TDistLocationScale)
    (; df, location, scale) = tdist
    standard = rand(rng, TDist(df))
    return location + standard * scale
end

function Distributions.cdf(tdist::TDistLocationScale, t::Union{Real, AbstractVector{<:Real}})
    (; df, location, scale) = tdist
    return cdf(TDist(df), @.((t-location)/scale))
end

struct NormalInverseGamma{T<:Real} <: ContinuousUnivariateDistribution
    location::T
    inv_scale_fac::T
    shape::T
    rate::T
end

function Distributions.rand(rng::AbstractRNG, dist::NormalInverseGamma)
    (; location, inv_scale_fac, shape, rate) = dist
    σ2 = rand(rng, InverseGamma(shape, rate))
    μ = rand(rng, Normal(location, sqrt(σ2 / inv_scale_fac)))
    return μ, σ2
end