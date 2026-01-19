struct TDistLocationScale{T<:Real} <: ContinuousUnivariateDistribution
    df::T
    location::T
    scale::T
end
TDistLocationScale(args...) = TDistLocationScale{Float64}(args...)

function Distributions.logpdf(tdist::TDistLocationScale{T}, t::Union{S, AbstractVector{S}}) where {T<:Real, S<:Real}
    R = promote_type(T, S)
    (; df, loc, scale) = tdist
    return @. loggamma((df+1)/2) - loggamma(df/2) - log(df*R(pi)*scale) - (df + 1) / 2 * log(1 + (t - loc)^2/(df*scale))
end

function Distributions.rand(rng::AbstractRNG, tdist::TDistLocationScale)
    (; df, loc, scale) = tdist
    standard = rand(rng, TDist(df))
    return @.((standard - loc) / scale)
end

function Distributions.cdf(tdist::TDistLocationScale, t::Union{S, AbstractVector{S}})
    (; df, loc, scale) = tdist
    return cdf(TDist(df), @.((t-loc)/scale))
end

function _tdist_logpdf(df::T, loc::T, scale::T, t::Union{S, AbstractVector{S}}) where {T<:Real, S<:Real}
    R = promote_type(T, S)
    return @. loggamma((df+1)/2) - loggamma(df/2) - log(df*R(pi)*scale) - (df + 1) / 2 * log(1 + (t - loc)^2/(df*scale))
end