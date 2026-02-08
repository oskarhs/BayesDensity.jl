"""
    RandomBernsteinPoly{T<:Real} <: AbstractBayesDensityModel{T}
    
Struct representing a random Bernstein polyomial.

# Constructors
    RandomBernsteinPoly(x::AbstractVector{<:Real}; kwargs...)
    RandomBernsteinPoly{T}(x::AbstractVector{<:Real}; kwargs...)

# Arguments
* `x`: The data vector.

# Keyword arguments
* `prior_components`: A [`Distributions.DiscreteNonParametric`](@extref Distributions.DiscreteNonParametric) distribution instance specifying the prior on the number of components `K`. Defaults to `DiscreteNonParametric(1:50, fill(T(1/50), 50))`, corresponding to a uniform prior on the set {1, …, 50\\}.
* `prior_strength`: Strength parameter of the symmetric Dirichlet prior on the mixture weights. E.g. the prior is Dirichlet(strength, ..., strength). Defaults to `1.0`.
* `bounds`: A tuple giving the support of the B-spline mixture model.

# Returns
* `rbp`: A random Bernstein polynomial model object.

# Examples

# Extended help

"""
struct RandomBernsteinPoly{T<:Real, NT<:NamedTuple, W<:DiscreteNonParametric{Int, T}} <: AbstractBayesDensityModel{T}
    data::NT
    prior_components::W
    prior_strength::T
    bounds::NTuple{2, T}
    function RandomBernsteinPoly{T}(
        x::AbstractVector{<:Real};
        prior_components::DiscreteNonParametric=DiscreteNonParametric(1:100, fill(T(1/100), 100)),
        prior_strength::Real=1.0,
        bounds::Tuple{<:Real,<:Real} = _get_default_bounds(x)
        ) where {T<:Real}
        _check_rbpkwargs(prior_components, prior_strength, bounds, x)
        T_bounds = (T(bounds[1]), T(bounds[2]))
        T_x = T.(x)
        x_trans = @. (T_x - T_bounds[1])/(T_bounds[2] - T_bounds[1])
        data = (x = T_x, n = length(x), x_trans = x_trans)
        return new{T, typeof(data), typeof(prior_components)}(data, prior_components, T(prior_strength), T_bounds)
    end
end
RandomBernsteinPoly(args...; kwargs...) = RandomBernsteinPoly{Float64}(args...; kwargs...)

Base.:(==)(rbp1::RandomBernsteinPoly, rbp2::RandomBernsteinPoly) = (rbp1.data == rbp2.data) && (hyperparams(rbp1) == hyperparams(rbp2))


BayesDensityCore.support(rbp::RandomBernsteinPoly) = rbp.bounds

"""
    hyperparams(
        rbp::RandomBernsteinPoly{T}
    ) where {T} -> @NamedTuple{prior_components::DiscreteNonParametric{Int, T}, prior_strength::T}

Returns the hyperparameters of the random Bernstein polynomial `rbp` as a `NamedTuple`.
"""
BayesDensityCore.hyperparams(rbp::RandomBernsteinPoly) = (
    prior_components = rbp.prior_components,
    prior_strength = rbp.prior_strength
)

function Base.show(io::IO, ::MIME"text/plain", rbp::RandomBernsteinPoly{T}) where {T}
    println(io, nameof(typeof(rbp)), '{', T, "} with ", length(probs(rbp.prior_components)), " values for the number mixture components.")
    println(io, "Using ", rbp.data.n, " observations.")
    let io = IOContext(io, :compact => true, :limit => true)
        println(io, "Hyperparameters:")
        print(io, " prior_strength = " , rbp.prior_strength)
    end
    nothing
end

Base.show(io::IO, rbp::RandomBernsteinPoly) = show(io, MIME("text/plain"), rbp)

"""
    pdf(
        bsm::RandomBernsteinPoly,
        params::NamedTuple,
        t::Union{Real, AbstractVector{<:Real}}
    ) -> Union{Real, Vector{<:Real}}

    pdf(
        bsm::RandomBernsteinPoly,
        params::AbstractVector{NamedTuple},
        t::Union{Real, AbstractVector{<:Real}}
    ) -> Matrix{<:Real}

Evaluate ``f(t\\, |\\, \\boldsymbol{\\eta})`` for a given `RandomBernsteinPoly` when the model parameters of the NamedTuple `params` are given by ``\\boldsymbol{\\eta}``.

The named tuple should contain fields named `:w`.
"""
Distributions.pdf(rbp::RandomBernsteinPoly, params::NamedTuple, t::Real) = _pdf(rbp, params, t)
Distributions.pdf(rbp::RandomBernsteinPoly, params::NamedTuple, t::AbstractVector{<:Real}) = _pdf(rbp, params, t)
Distributions.pdf(rbp::RandomBernsteinPoly, params::AbstractVector{<:NamedTuple}, t::AbstractVector{<:Real}) = _pdf(rbp, params, t)
Distributions.pdf(rbp::RandomBernsteinPoly, params::AbstractVector{<:NamedTuple}, t::Real) = _pdf(rbp, params, [t])

"""
    cdf(
        bsm::RandomBernsteinPoly,
        params::NamedTuple,
        t::Union{Real, AbstractVector{<:Real}}
    ) -> Union{Real, Vector{<:Real}}

    cdf(
        bsm::RandomBernsteinPoly,
        params::AbstractVector{NamedTuple},
        t::Union{Real, AbstractVector{<:Real}}
    ) -> Matrix{<:Real}

Evaluate ``F(t\\, |\\, \\boldsymbol{\\eta})`` for a given `RandomBernsteinPoly` when the model parameters of the NamedTuple `params` are given by ``\\boldsymbol{\\eta}``.

The named tuple should contain fields named `:w`.
"""
Distributions.cdf(rbp::RandomBernsteinPoly, params::NamedTuple, t::Real) = _cdf(rbp, params, t)
Distributions.cdf(rbp::RandomBernsteinPoly, params::NamedTuple, t::AbstractVector{<:Real}) = _cdf(rbp, params, t)
Distributions.cdf(rbp::RandomBernsteinPoly, params::AbstractVector{<:NamedTuple}, t::AbstractVector{<:Real}) = _cdf(rbp, params, t)
Distributions.cdf(rbp::RandomBernsteinPoly, params::AbstractVector{<:NamedTuple}, t::Real) = _cdf(rbp, params, [t])

for funcs in ((:_pdf, :pdf, :(xmax - xmin)), (:_cdf, :cdf, :(1)))
    @eval begin
        function $(funcs[1])(
            rbp::RandomBernsteinPoly{T},
            params::NamedTuple{(:w,), V},
            t::S
        ) where {T<:Real, S<:Real, V<:Tuple}
            xmin, xmax = support(rbp)
            R = xmax - xmin
            t_trans = @. (t - xmin) / R
            w = params.w
            K = length(w)
            val = zero(promote_type(T, S))
            for k in eachindex(w)
                val += w[k] * $(funcs[2])(Beta(k, K - k + 1), t_trans)
            end
            return val / $(funcs[3])
        end
        function $(funcs[1])(
            rbp::RandomBernsteinPoly{T},
            params::NamedTuple{(:w,), V},
            t::AbstractVector{S}
        ) where {T<:Real, S<:Real, V<:Tuple}
            xmin, xmax = support(rbp)
            R = xmax - xmin
            t_trans = @. (t - xmin) / R
            w = params.w
            K = length(w)
            val = zeros(promote_type(T, S), length(t))
            for k in eachindex(w)
                val += w[k] * $(funcs[2])(Beta(k, K - k + 1), t_trans)
            end
            return val / $(funcs[3])
        end
        function $(funcs[1])(
            rbp::RandomBernsteinPoly{T},
            params::AbstractVector{NamedTuple{(:w,), V}},
            t::AbstractVector{S}
        ) where {T<:Real, S<:Real, V<:Tuple}
            xmin, xmax = support(rbp)
            R = xmax - xmin
            t_trans = @. (t - xmin) / R
            val = zeros(promote_type(T, S), (length(t), length(params)))
            for m in eachindex(params)
                w = params[m].w
                K = length(w)
                for k in eachindex(w)
                    val[:, m] .+= w[k] * $(funcs[2])(Beta(k, K - k + 1), t_trans)
                end
            end
            return val / $(funcs[3])
        end
    end
end

function _get_default_bounds(x::AbstractVector{<:Real})
    xmin, xmax = extrema(x)
    R = xmax - xmin
    return xmin - 0.05*R, xmax + 0.05*R
end

function _check_rbpkwargs(
    prior_components::DiscreteNonParametric,
    prior_strength::Real,
    bounds::Tuple{<:Real, <:Real},
    x::AbstractVector{<:Real}
)
    all(support(prior_components) .>= 1) || throw(ArgumentError("Prior on the number of mixture components must have strictly positive support."))
    (prior_strength > 0) || throw(ArgumentError("Prior strength `prior_strength` must be positive."))
        xmin, xmax = extrema(x)
    (bounds[1] < bounds[2]) || throw(ArgumentError("Supplied upper bound must be strictly greater than the lower bound."))
    (bounds[1] ≤ xmin ≤ xmax ≤ bounds[2]) || throw(ArgumentError("Data is not contained within supplied bounds."))
end