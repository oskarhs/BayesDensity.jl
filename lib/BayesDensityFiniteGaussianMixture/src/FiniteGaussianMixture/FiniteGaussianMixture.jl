"""
    FiniteGaussianMixture{T<:Real} <: AbstractBayesDensityModel{T}

Struct representing a finite Gaussian mixture model with a fixed number of components.

# Constructors
    FiniteGaussianMixture(x::AbstractVector{<:Real}, K::Int; kwargs...)
    FiniteGaussianMixture{T}(x::AbstractVector{<:Real}, K::Int; kwargs...)

# Arguments
* `x`: The data vector.

# Keyword arguments
* `prior_strength`: Strength parameter of the symmetric Dirichlet prior on the mixture weights. E.g. the prior is Dirichlet(strength, ..., strength). Defaults to `1.0`.
* `prior_location`: Prior mean of the location parameters `μ[k]`. Defaults to the midpoint of the minimum and maximum values in the sample.
* `prior_variance`: The prior variance of the location parameter `μ[k]`. Defaults to the sample range.
* `prior_shape`: Prior shape parameter of the squared scale parameters `σ2[k]`: Defaults to `2.0`.
* `hyperprior_shape`: Prior shape parameter of the hyperprior on the rate parameter of `σ2[k]`. Defaults to `0.2`.
* `hyperprior_rate`: Prior rate parameter of the hyperprior on the rate parameter of `σ2[k]`. Defaults to `0.2*R^2`, where `R` is the sample range.
"""
struct FiniteGaussianMixture{T<:Real, NT<:NamedTuple} <: AbstractBayesDensityModel{T}
    data::NT
    K::Int
    prior_strength::T
    prior_location::T
    prior_variance::T
    prior_shape::T
    hyperprior_rate::T
    hyperprior_shape::T
    function FiniteGaussianMixture{T}(
        x::AbstractVector{<:Real},
        K::Int;
        prior_strength::Real=1.0,
        prior_location::Real=_get_default_location(x),
        prior_variance::Real=_get_default_variance(x),
        prior_shape::Real=2.0,
        hyperprior_shape::Real=0.2,
        hyperprior_rate::Real=_get_default_hyperprior_rate(x)
    ) where {T<:Real}
        _check_finitegaussianmixturekwargs(prior_strength, prior_variance, prior_shape, hyperprior_rate, hyperprior_shape)
        data = (x = T.(x), n  = length(x))
        return new{T, typeof(data)}(data, K, T(prior_strength), T(prior_location), T(prior_variance), T(prior_shape), T(hyperprior_rate), T(hyperprior_shape))
    end
end
FiniteGaussianMixture(args...; kwargs...) = FiniteGaussianMixture{Float64}(args...; kwargs...)

Base.:(==)(gm1::FiniteGaussianMixture, gm2::FiniteGaussianMixture) = (gm1.data == gm2.data) && (hyperparams(gm1) == hyperparams(gm2))

"""
    support(gm::FiniteGaussianMixture{T}) where {T} -> NTuple{2, T}

Get the support of the finite Gaussian mixture model `gm`.
"""
BayesDensityCore.support(::FiniteGaussianMixture{T}) where {T} = (-T(Inf), T(Inf))

"""
    hyperparams(
        gm::FiniteGaussianMixture{T}
    ) where {T} -> @NamedTuple{prior_strength::T, prior_location::T, prior_variance::T, prior_shape::T, prior_rate::T}

Returns the hyperparameters of the finite Gaussian mixture model `gm` as a `NamedTuple`.
"""
BayesDensityCore.hyperparams(gm::FiniteGaussianMixture) = (
    prior_strength = gm.prior_strength,
    prior_location = gm.prior_location,
    prior_variance = gm.prior_variance,
    prior_shape = gm.prior_shape,
    hyperprior_shape = gm.hyperprior_shape,
    hyperprior_rate = gm.hyperprior_rate
)

# Print method for unbinned data
function Base.show(io::IO, ::MIME"text/plain", gm::FiniteGaussianMixture{T}) where {T}
    println(io, nameof(typeof(gm)), '{', T, "} with ", gm.K, " components:")
    println(io, "Using ", gm.data.n, " observations.")
    let io = IOContext(io, :compact => true, :limit => true)
        println(io, "Hyperparameters:")
        println(io, " prior_location = " , gm.prior_location, ", prior_variance = ", gm.prior_variance)
        println(io, " prior_shape = ", gm.prior_shape, ", hyperprior_shape = ", gm.hyperprior_shape, ", hyperprior_rate = ", gm.hyperprior_rate)
        print(io, " prior_strength = ", gm.prior_strength)
    end
    nothing
end

Base.show(io::IO, gm::FiniteGaussianMixture) = show(io, MIME("text/plain"), gm)

"""
    pdf(
        bsm::FiniteGaussianMixture,
        params::NamedTuple,
        t::Union{Real, AbstractVector{<:Real}}
    ) -> Union{Real, Vector{<:Real}}

    pdf(
        bsm::FiniteGaussianMixture,
        params::AbstractVector{NamedTuple},
        t::Union{Real, AbstractVector{<:Real}}
    ) -> Matrix{<:Real}

Evaluate ``f(t | \\boldsymbol{\\eta})`` for a given `FiniteGaussianMixture` when the model parameters of the NamedTuple `params` are given by ``\\boldsymbol{\\eta}``.

The NamedTuple `params` should contain fields named `:μ`, `:σ2` and `:w`.
"""
Distributions.pdf(pym::FiniteGaussianMixture, params::NamedTuple, t::Real) = _pdf(pym, params, t)
Distributions.pdf(pym::FiniteGaussianMixture, params::NamedTuple, t::AbstractVector{<:Real}) = _pdf(pym, params, t)
Distributions.pdf(pym::FiniteGaussianMixture, params::AbstractVector{NamedTuple}, t::AbstractVector{<:Real}) = _pdf(pym, params, t)
Distributions.pdf(pym::FiniteGaussianMixture, params::AbstractVector{NamedTuple}, t::Real) = _pdf(pym, params, t)

"""
    cdf(
        bsm::FiniteGaussianMixture,
        params::NamedTuple,
        t::Union{Real, AbstractVector{<:Real}}
    ) -> Union{Real, Vector{<:Real}}

    cdf(
        bsm::FiniteGaussianMixture,
        params::AbstractVector{NamedTuple},
        t::Union{Real, AbstractVector{<:Real}}
    ) -> Matrix{<:Real}

Evaluate ``F(t | \\boldsymbol{\\eta})`` for a given `FiniteGaussianMixture` when the model parameters of the NamedTuple `params` are given by ``\\boldsymbol{\\eta}``.

The NamedTuple `params` should contain fields named `:μ`, `:σ2` and `:w`.
"""
Distributions.cdf(pym::FiniteGaussianMixture, params::NamedTuple, t::Real) = _cdf(pym, params, t)
Distributions.cdf(pym::FiniteGaussianMixture, params::NamedTuple, t::AbstractVector{<:Real}) = _cdf(pym, params, t)
Distributions.cdf(pym::FiniteGaussianMixture, params::AbstractVector{NamedTuple}, t::AbstractVector{<:Real}) = _cdf(pym, params, t)
Distributions.cdf(pym::FiniteGaussianMixture, params::AbstractVector{NamedTuple}, t::Real) = _cdf(pym, params, t)

for funcs in ((:_pdf, :pdf), (:_cdf, :cdf))
    @eval begin
        function $(funcs[1])(
            ::FiniteGaussianMixture{T},
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
            ::FiniteGaussianMixture{T},
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
            ::FiniteGaussianMixture{T},
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