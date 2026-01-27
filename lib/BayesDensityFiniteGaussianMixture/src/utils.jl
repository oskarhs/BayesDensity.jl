function _get_default_location(x::AbstractVector{<:Real})
    xmin, xmax = extrema(x)
    return (xmax + xmin) / 2
end

function _get_default_variance(x::AbstractVector{<:Real})
    xmin, xmax = extrema(x)
    R = xmax - xmin
    return R
end

function _get_default_rate(x::AbstractVector{<:Real})
    xmin, xmax = extrema(x)
    R = xmax - xmin
    return 0.2*R^2
end

function _check_finitegaussianmixturekwargs(prior_strength::Real, prior_variance::Real, prior_shape::Real, prior_rate::Real)
    (prior_strength > 0) || throw(ArgumentError("Prior strength `prior_strength` must be positive."))
    (prior_variance > 0) || throw(ArgumentError("Prior standard deviation `prior_variance` must be positive."))
    (prior_shape > 0) || throw(ArgumentError("Prior shape parameter `prior_shape` must be positive."))
    (prior_rate > 0) || throw(ArgumentError("Prior rate parameter `prior_rate` must be positive."))
end