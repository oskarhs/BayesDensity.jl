"""
    RandomFiniteGaussianMixture{T<:Real} <: AbstractBayesDensityModel{T}

Struct representing a finite Gaussian mixture model with a variable (random) number of components.

# Constructors
    RandomFiniteGaussianMixture(x::AbstractVector{<:Real}; kwargs...)
    RandomFiniteGaussianMixture{T}(x::AbstractVector{<:Real}; kwargs...)

# Arguments
* `x`: The data vector.

# Keyword arguments
* `prior_components`: A vector containing the prior probabilities on the number of mixture components. Defaults to `fill(1, 30)`, corresponding to a uniform prior on the set {1, …, 30}
* `prior_strength`: Strength parameter of the symmetric Dirichlet prior on the mixture weights. E.g. the prior is Dirichlet(strength, ..., strength). Defaults to `1.0`.
* `prior_location`: Prior mean of the location parameters `μ[k]`. Defaults to the midpoint of the minimum and maximum values in the sample.
* `prior_variance`: The prior variance of the location parameter `μ[k]`. Defaults to the sample range.
* `prior_shape`: Prior shape parameter of the squared scale parameters `σ2[k]`: Defaults to `2.0`.
* `prior_rate`: Prior rate parameter of the squared scale parameters `σ2[k]`. Defaults to `0.2*R^2`, where `R` is the sample range.
"""
struct RandomFiniteGaussianMixture{T<:Real, NT<:NamedTuple, W<:StatsBase.AbstractWeights} <: AbstractBayesDensityModel{T}
    data::NT
    prior_components::W
    prior_strength::T
    prior_location::T
    prior_variance::T
    prior_shape::T
    prior_rate::T
    function RandomFiniteGaussianMixture{T}(
        x::AbstractVector{<:Real};
        prior_components::AbstractVector{<:Real}=fill(1, 30),
        prior_strength::Real=1.0,
        prior_location::Real=_get_default_location(x),
        prior_variance::Real=_get_default_variance(x),
        prior_shape::Real=2.0,
        prior_rate::Real=_get_default_rate(x)
    ) where {T<:Real}
        _check_finitegaussianmixturekwargs(prior_strength, prior_variance, prior_shape, prior_rate)
        prior_components = pweights(T.(prior_components))
        data = (x = T.(x), n  = length(x))
        return new{T, typeof(data), typeof(prior_components)}(data, prior_components, T(prior_strength), T(prior_location), T(prior_variance), T(prior_shape), T(prior_rate))
    end
end
RandomFiniteGaussianMixture(args...; kwargs...) =  RandomFiniteGaussianMixture{Float64}(args...; kwargs...)

Base.:(==)(gm1::RandomFiniteGaussianMixture, gm2::RandomFiniteGaussianMixture) = (gm1.data == gm2.data) && (hyperparams(gm1) == hyperparams(gm2))

"""
    support(gm::RandomFiniteGaussianMixture{T}) where {T} -> NTuple{2, T}

Get the support of the finite Gaussian mixture model `gm`.
"""
BayesDensityCore.support(::RandomFiniteGaussianMixture{T}) where {T} = (-T(Inf), T(Inf))

"""
    hyperparams(
        gm::RandomFiniteGaussianMixture{T}
    ) where {T} -> @NamedTuple{prior_strength::T, prior_location::T, prior_variance::T, prior_shape::T, prior_rate::T}

Returns the hyperparameters of the finite Gaussian mixture model `gm` as a `NamedTuple`.
"""
BayesDensityCore.hyperparams(gm::RandomFiniteGaussianMixture) = (prior_components = gm.prior_components, prior_strength = gm.prior_strength, prior_location=gm.prior_location, prior_variance=gm.prior_variance, prior_shape=gm.prior_shape, prior_rate=gm.prior_rate)

# Print method for unbinned data
function Base.show(io::IO, ::MIME"text/plain", gm::RandomFiniteGaussianMixture{T}) where {T}
    println(io, nameof(typeof(gm)), '{', T, "} with maximum number of components ", length(gm.prior_components), ":")
    println(io, "Using ", gm.data.n, " observations.")
    let io = IOContext(io, :compact => true, :limit => true)
        println(io, "Hyperparameters:")
        println(io, " prior_location = " , gm.prior_location, ", prior_variance = ", gm.prior_variance)
        println(io, " prior_shape = ", gm.prior_shape, ", prior_rate = ", gm.prior_rate)
        print(io, " prior_strength =", gm.prior_strength)
    end
    nothing
end

Base.show(io::IO, gm::RandomFiniteGaussianMixture) = show(io, MIME("text/plain"), gm)

"""
    pdf(
        bsm::RandomFiniteGaussianMixture,
        params::NamedTuple,
        t::Union{Real, AbstractVector{<:Real}}
    ) -> Union{Real, Vector{<:Real}}

    pdf(
        bsm::RandomFiniteGaussianMixture,
        params::AbstractVector{NamedTuple},
        t::Union{Real, AbstractVector{<:Real}}
    ) -> Matrix{<:Real}

Evaluate ``f(t | \\boldsymbol{\\eta})`` for a given `RandomFiniteGaussianMixture` when the model parameters of the NamedTuple `params` are given by ``\\boldsymbol{\\eta}``.

The named tuple should contain fields named `:μ`, `:σ2` and `:w`.
"""
Distributions.pdf(pym::RandomFiniteGaussianMixture, params::NamedTuple, t::Real) = _pdf(pym, params, t)
Distributions.pdf(pym::RandomFiniteGaussianMixture, params::NamedTuple, t::AbstractVector{<:Real}) = _pdf(pym, params, t)
Distributions.pdf(pym::RandomFiniteGaussianMixture, params::AbstractVector{NamedTuple}, t::AbstractVector{<:Real}) = _pdf(pym, params, t)
Distributions.pdf(pym::RandomFiniteGaussianMixture, params::AbstractVector{NamedTuple}, t::Real) = _pdf(pym, params, t)

"""
    cdf(
        bsm::RandomFiniteGaussianMixture,
        params::NamedTuple,
        t::Union{Real, AbstractVector{<:Real}}
    ) -> Union{Real, Vector{<:Real}}

    cdf(
        bsm::RandomFiniteGaussianMixture,
        params::AbstractVector{NamedTuple},
        t::Union{Real, AbstractVector{<:Real}}
    ) -> Matrix{<:Real}

Evaluate ``F(t | \\boldsymbol{\\eta})`` for a given `RandomFiniteGaussianMixture` when the model parameters of the NamedTuple `params` are given by ``\\boldsymbol{\\eta}``.

The named tuple should contain fields named `:μ`, `:σ2` and `:w`.
"""
Distributions.cdf(pym::RandomFiniteGaussianMixture, params::NamedTuple, t::Real) = _cdf(pym, params, t)
Distributions.cdf(pym::RandomFiniteGaussianMixture, params::NamedTuple, t::AbstractVector{<:Real}) = _cdf(pym, params, t)
Distributions.cdf(pym::RandomFiniteGaussianMixture, params::AbstractVector{NamedTuple}, t::AbstractVector{<:Real}) = _cdf(pym, params, t)
Distributions.cdf(pym::RandomFiniteGaussianMixture, params::AbstractVector{NamedTuple}, t::Real) = _cdf(pym, params, t)

for funcs in ((:_pdf, :pdf), (:_cdf, :cdf))
    @eval begin
        function $(funcs[1])(
            ::RandomFiniteGaussianMixture{T},
            params::NamedTuple{(:μ, :σ2, :w), V},
            t::S
        ) where {T<:Real, S<:Real, V<:Tuple}
            (; μ, σ2, w) = params
            val = zero(promote_type(T, S))
            for k in eachindex(μ)
                val += w[k] * $(funcs[2])(Normal(μ[k], sqrt(σ2[k])), t)
            end
            return val
        end
        function $(funcs[1])(
            ::RandomFiniteGaussianMixture{T},
            params::NamedTuple{(:μ, :σ2, :w), V},
            t::AbstractVector{S}
        ) where {T<:Real, S<:Real, V<:Tuple}
            (; μ, σ2, w) = params
            val = zeros(promote_type(T, S), length(t))
            for k in eachindex(μ)
                val .+= w[k] * $(funcs[2])(Normal(μ[k], sqrt(σ2[k])), t)
            end
            return val
        end
        function $(funcs[1])(
            ::RandomFiniteGaussianMixture{T},
            params::AbstractVector{NamedTuple{(:μ, :σ2, :w), V}},
            t::AbstractVector{S}
        ) where {T<:Real, S<:Real, V<:Tuple}
            val = zeros(promote_type(T, S), (length(t), length(params)))
            for m in eachindex(params)
                (; μ, σ2, w) = params[m]
                for k in eachindex(μ)
                    val[:, k] .+= w[k] * $(funcs[2])(Normal(μ[k], sqrt(σ2[k])), t)
                end
            end
            return val
        end
    end
end