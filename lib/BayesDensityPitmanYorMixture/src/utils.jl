xlogx(x::Real) = ifelse(x > 0.0, x*log(x), 0.0)

# Numerically stable softmax
function softmax(x::AbstractVector{T}) where {T<:Real}
    xmax = maximum(x)
    num = @. exp(x - xmax)
    return num /sum(num)
end

# Map an unconstrained K-1 dimensional vector to the K-simplex through logistic stickbreaking, defined as the composition of the logistic and stickbreaking maps.
# To ensure numerical stability, the calculation is performed in log-space.
function logistic_stickbreaking(β::AbstractVector{T}) where {T<:Real}
    K = length(β) + 1
    log_π = Vector{T}(undef, K)
    softplus_sum = zero(T)
    for k in 1:K-1
        softplus_sum += softplus(β[k])
        log_π[k] = β[k] - softplus_sum
    end
    log_π[K] = -softplus_sum
    return softmax(log_π)
end

# Stickbreaking map. Takes a vector in [0, 1]^(K-1) and outputs a simplex-vector
# NB! It is implicit that V[K] = 1 in the input here.
function truncated_stickbreaking(v::AbstractVector{T}) where {T<:Real}
    K = length(v)+1
    w = Vector{T}(undef, K)
    w[1] = v[1]
    cum_w = w[1]
    for k in 2:K-1
        w[k] = v[k] * (1 - cum_w)
        cum_w += w[k]
    end
    w[K] = 1 - cum_w
    return w
end

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

struct NormalInverseGamma{T<:Real} <: ContinuousMultivariateDistribution
    location::T
    inv_scale_fac::T
    shape::T
    rate::T
end

Base.length(::NormalInverseGamma) = 2

function Distributions._rand!(rng::AbstractRNG, dist::NormalInverseGamma{<:Real}, θ::AbstractArray{<:Real})
    (; location, inv_scale_fac, shape, rate) = dist
    θ[2] = rand(rng, InverseGamma(shape, rate))
    θ[1] = rand(rng, Normal(location, sqrt(θ[2] / inv_scale_fac)))
    return θ
end