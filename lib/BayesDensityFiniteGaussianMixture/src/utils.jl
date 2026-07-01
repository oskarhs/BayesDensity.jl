xlogx(x::Real) = ifelse(x > 0.0, x*log(x), 0.0)

# Numerically stable softmax
function softmax(x::AbstractVector{T}) where {T<:Real}
    xmax = maximum(x)
    num = @. exp(x - xmax)
    return num /sum(num)
end

# In-place numerically stable softmax over the first `L` entries: dst[1:L] .= softmax(src[1:L]).
# `dst` and `src` may alias. Allocation-free.
function _softmax!(dst::AbstractVector{T}, src::AbstractVector{T}, L::Int = length(src)) where {T<:Real}
    m = src[1]
    @inbounds for k in 2:L
        src[k] > m && (m = src[k])
    end
    s = zero(T)
    @inbounds for k in 1:L
        e = exp(src[k] - m)
        dst[k] = e
        s += e
    end
    @inbounds for k in 1:L
        dst[k] /= s
    end
    return dst
end

function _get_default_location(x::AbstractVector{<:Real})
    xmin, xmax = extrema(x)
    return (xmax + xmin) / 2
end

function _get_default_variance(x::AbstractVector{<:Real})
    xmin, xmax = extrema(x)
    R = xmax - xmin
    return R^2
end

function _get_default_hyperprior_rate(x::AbstractVector{<:Real})
    xmin, xmax = extrema(x)
    R = xmax - xmin
    return 10/R^2
end

function _check_finitegaussianmixturekwargs(prior_strength::Real, prior_variance::Real, prior_shape::Real, hyperprior_shape::Real, hyperprior_rate::Real)
    (prior_strength > 0) || throw(ArgumentError("Prior strength `prior_strength` must be positive."))
    (prior_variance > 0) || throw(ArgumentError("Prior standard deviation `prior_variance` must be positive."))
    (prior_shape > 0) || throw(ArgumentError("Prior shape parameter `prior_shape` must be positive."))
    (hyperprior_shape > 0) || throw(ArgumentError("Hyperprior shape parameter `hyperprior_shape` must be positive."))
    (hyperprior_rate > 0) || throw(ArgumentError("Hyperprior rate parameter `hyperprior_rate` must be positive."))
end

function _get_suffstats_binned(x::AbstractVector{T}, breaks::AbstractVector{T}) where {T}
    K = length(breaks)-1
    bin_counts = zeros(Int, K)
    bin_sums = zeros(T, K)
    bin_sumsqs = zeros(T, K)
    for val in x
        idval = min(max(1, searchsortedfirst(breaks, val) - 1), K)
        bin_counts[idval] +=1
        bin_sums[idval] += val
        bin_sumsqs[idval] += val^2
    end
    return bin_counts, bin_sums, bin_sumsqs
end