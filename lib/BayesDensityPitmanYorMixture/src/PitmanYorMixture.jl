"""
    PitmanYorMixture{T<:Real} <: AbstractBayesDensityModel{T}
    
Struct representing a Pitman-Yor mixture model with a normal kernel and a conjugate Normal-InverseGamma base measure.

# Constructors
    PitmanYorMixture(x::AbstractVector{<:Real}; kwargs...)
    PitmanYorMixture{T}(x::AbstractVector{<:Real}; kwargs...)

# Arguments
* `x`: The data vector.

# Keyword arguments
* `discount`: Discount parameter of the Pitman-Yor process. Defaults to `0.0`, corresponding to a Dirichlet Process.
* `strength`: Strength parameter of the Pitman-Yor process. Defaults to `1.0`.
* `prior_location`: Prior mean of the location parameter `μ`. Defaults to `mean(x)`.
* `prior_inv_scale_fac`: Factor by which the conditional prior variance `σ2` of `μ` is scaled. Defaults to `1`.
* `prior_shape`: Prior shape parameter of the squared scale parameter `σ2`: Defaults to `2.0`.
* `prior_rate`: Prior rate parameter of the squared scale parameter `σ2`. Defaults to `var(x)`.

# Examples
```julia-repl
julia> x = (1.0 .- (1.0 .- LinRange(0.0, 1.0, 5000)) .^(1/3)).^(1/3);

julia> pym = PitmanYorMixture(x)
PitmanYorMixture{Float64}:
Using 5000 observations.
Hyperparameters:
 discount = 0.0, strength = 1.0
 prior_location = 0.578555, prior_inv_scale_fac = 1.0
 prior_shape = 2.0, prior_rate = 0.0334916

julia> pym = PitmanYorMixture(x; strength = 2, discount = 0.5);
```

# Extended help
"""
struct PitmanYorMixture{T<:Real, NT<:NamedTuple} <: AbstractBayesDensityModel{T}
    data::NT
    discount::T
    strength::T
    prior_location::T
    prior_inv_scale_fac::T
    prior_shape::T
    prior_rate::T
    function PitmanYorMixture{T}(x::AbstractVector{<:Real}; discount::Real=0.0, strength::Real=1.0, prior_location::Real=mean(x), prior_inv_scale_fac::Real=1.0, prior_shape::Real=2.0, prior_rate::Real=var(x)) where {T<:Real}
        _check_pitmanyorkwargs(discount, strength, prior_inv_scale_fac, prior_shape, prior_rate)
        data = (x = T.(x), n = length(x))

        return new{T,typeof(data)}(data, T(discount), T(strength), T(prior_location), T(prior_inv_scale_fac), T(prior_shape), T(prior_rate))
    end
end
PitmanYorMixture(args...; kwargs...) = PitmanYorMixture{Float64}(args...; kwargs...)

Base.:(==)(pym1::PitmanYorMixture, pym2::PitmanYorMixture) = (pym1.data == pym2.data) && (hyperparams(pym1) == hyperparams(pym2))

BayesDensityCore.support(::PitmanYorMixture{T}) where {T} = (-T(Inf), T(Inf))

"""
    hyperparams(
        pym::PitmanYorMixture{T}
    ) where {T} -> @NamedTuple{discount::T, strength::T, prior_location::T, prior_inv_scale_fac::T, prior_shape::T, prior_rate::T}

Returns the hyperparameters of the Pitman-Yor mixture model `pym` as a `NamedTuple`.
"""
BayesDensityCore.hyperparams(pym::PitmanYorMixture) = (discount=pym.discount, strength=pym.strength, prior_location=pym.prior_location, prior_inv_scale_fac=pym.prior_inv_scale_fac, prior_shape=pym.prior_shape, prior_rate=pym.prior_rate)

# Print method for unbinned data
function Base.show(io::IO, ::MIME"text/plain", pym::PitmanYorMixture{T}) where {T}
    println(io, nameof(typeof(pym)), '{', T, "}:")
    println(io, "Using ", pym.data.n, " observations.")
    let io = IOContext(io, :compact => true, :limit => true)
        println(io, "Hyperparameters:")
        println(io, " discount = " , pym.discount, ", strength = ", pym.strength)
        println(io, " prior_location = " , pym.prior_location, ", prior_inv_scale_fac = ", pym.prior_inv_scale_fac)
        print(io, " prior_shape = ", pym.prior_shape, ", prior_rate = ", pym.prior_rate)
    end
    nothing
end

Base.show(io::IO, pym::PitmanYorMixture) = show(io, MIME("text/plain"), pym)

"""
    pdf(
        bsm::PitmanYorMixture,
        params::NamedTuple,
        t::Union{Real, AbstractVector{<:Real}}
    ) -> Union{Real, Vector{<:Real}}

    pdf(
        bsm::PitmanYorMixture,
        params::AbstractVector{NamedTuple},
        t::Union{Real, AbstractVector{<:Real}}
    ) -> Matrix{<:Real}

Evaluate ``f(t\\, |\\, \\boldsymbol{\\eta})`` for a given `PitmanYorMixture` when the model parameters of the NamedTuple `params` are given by ``\\boldsymbol{\\eta}``.

The named tuple should contain fields named `:μ`, `:σ2` and a third field named `:cluster_counts` or `:w`, depending on whether the marginal or stickbreaking parameterization is used.
"""
Distributions.pdf(pym::PitmanYorMixture, params::NamedTuple, t::Real) = _pdf(pym, params, t)
Distributions.pdf(pym::PitmanYorMixture, params::NamedTuple, t::AbstractVector{<:Real}) = _pdf(pym, params, t)
Distributions.pdf(pym::PitmanYorMixture, params::AbstractVector{<:NamedTuple}, t::AbstractVector{<:Real}) = _pdf(pym, params, t)
Distributions.pdf(pym::PitmanYorMixture, params::AbstractVector{<:NamedTuple}, t::Real) = _pdf(pym, params, t)

"""
    cdf(
        bsm::PitmanYorMixture,
        params::NamedTuple,
        t::Union{Real, AbstractVector{<:Real}}
    ) -> Union{Real, Vector{<:Real}}

    cdf(
        bsm::PitmanYorMixture,
        params::AbstractVector{NamedTuple},
        t::Union{Real, AbstractVector{<:Real}}
    ) -> Matrix{<:Real}

Evaluate ``F(t\\, |\\, \\boldsymbol{\\eta})`` for a given `PitmanYorMixture` when the model parameters of the NamedTuple `params` are given by ``\\boldsymbol{\\eta}``.

The named tuple should contain fields named `:μ`, `:σ2` and a third field named `:cluster_counts` or `:w`, depending on whether the marginal or stickbreaking parameterization is used.
"""
Distributions.cdf(pym::PitmanYorMixture, params::NamedTuple, t::Real) = _cdf(pym, params, t)
Distributions.cdf(pym::PitmanYorMixture, params::NamedTuple, t::AbstractVector{<:Real}) = _cdf(pym, params, t)
Distributions.cdf(pym::PitmanYorMixture, params::AbstractVector{<:NamedTuple}, t::AbstractVector{<:Real}) = _cdf(pym, params, t)
Distributions.cdf(pym::PitmanYorMixture, params::AbstractVector{<:NamedTuple}, t::Real) = _cdf(pym, params, t)


# pdf/cdf evalutation for samplers using the marginal parameterization
for funcs in ((:_pdf, :pdf), (:_cdf, :cdf))
    @eval begin
        function $(funcs[1])(
            pym::PitmanYorMixture{T},
            params::NamedTuple{(:μ, :σ2, :cluster_counts), V},
            t::S
        ) where {T<:Real, S<:Real, V<:Tuple}
            R = promote_type(T, S)
            (; μ, σ2, cluster_counts) = params
            n = pym.data.n
            (; discount, strength, prior_location, prior_inv_scale_fac, prior_shape, prior_rate) = hyperparams(pym)
            K = length(cluster_counts) # Number of existing clusters

            vals = zero(R)
            
            # Compute relative weight of (μ, σ²) belonging to a new cluster
            for k in eachindex(cluster_counts)
                vals += (cluster_counts[k] - discount) / (strength + n) * $(funcs[2])(Normal(μ[k], sqrt(σ2[k])), t) # Contribution from event that (μ, σ²) belong to existing clusters
            end
            marginal_scale = sqrt(prior_rate*(1 + 1/prior_inv_scale_fac)/prior_shape)
            vals += (strength + K * discount) / (strength + n) * $(funcs[2])(TDistLocationScale(2*prior_shape, prior_location, marginal_scale), t) # Contribution from event that (μ, σ²) forms a new cluster
            return vals
        end
        function $(funcs[1])(
            pym::PitmanYorMixture{T},
            params::NamedTuple{(:μ, :σ2, :cluster_counts), V},
            t::AbstractVector{S}
        ) where {T<:Real, S<:Real, V<:Tuple}
            R = promote_type(T, S)
            (; μ, σ2, cluster_counts) = params
            n = pym.data.n
            (; discount, strength, prior_location, prior_inv_scale_fac, prior_shape, prior_rate) = hyperparams(pym)
            K = length(cluster_counts) # Number of existing clusters

            vals = zeros(R, length(t))
            
            # Compute relative weight of (μ, σ²) belonging to a new cluster
            for k in eachindex(cluster_counts)
                vals .+= (cluster_counts[k] - discount) / (strength + n) .* $(funcs[2])(Normal(μ[k], sqrt(σ2[k])), t) # Contribution from event that (μ, σ²) belong to existing clusters
            end
            marginal_scale = sqrt(prior_rate*(1 + 1/prior_inv_scale_fac)/prior_shape)
            vals .+= (strength + K * discount) / (strength + n) .* $(funcs[2])(TDistLocationScale(2*prior_shape, prior_location, marginal_scale), t) # Contribution from event that (μ, σ²) forms a new cluster
            return vals
        end
        function $(funcs[1])(
            pym::PitmanYorMixture{T},
            params::AbstractVector{NamedTuple{(:μ, :σ2, :cluster_counts), V}},
            t::AbstractVector{S}
        ) where {T<:Real, S<:Real, V<:Tuple}
            R = promote_type(T, S)
            n = pym.data.n
            (; discount, strength, prior_location, prior_inv_scale_fac, prior_shape, prior_rate) = hyperparams(pym)
            marginal_scale = sqrt(prior_rate*(1 + 1/prior_inv_scale_fac)/prior_shape)

            vals = zeros(R, (length(t), length(params)))
            
            # Compute relative weight of (μ, σ²) belonging to a new cluster
            for m in eachindex(params)
                (; μ, σ2, cluster_counts) = params[m]
                for k in eachindex(cluster_counts)
                    vals[:, m] .+= (cluster_counts[k] - discount) / (strength + n) .* $(funcs[2])(Normal(μ[k], sqrt(σ2[k])), t) # Contribution from event that (μ, σ²) belong to existing clusters
                end
                vals[:, m] .+= (strength + length(cluster_counts) * discount) / (strength + n) .* $(funcs[2])(TDistLocationScale(2*prior_shape, prior_location, marginal_scale), t) # Contribution from event that (μ, σ²) forms a new cluster
            end
            return vals
        end
    end
end


# pdf/cdf evaluation for samplers/VI algorithms based on the stickbreaking parameterization
for funcs in ((:_pdf, :pdf), (:_cdf, :cdf))
    @eval begin
        function $(funcs[1])(
            ::PitmanYorMixture{T},
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
            ::PitmanYorMixture{T},
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
            ::PitmanYorMixture{T},
            params::AbstractVector{NamedTuple{(:μ, :σ2, :w), V}},
            t::AbstractVector{S}
        ) where {T<:Real, S<:Real, V<:Tuple}
            val = zeros(promote_type(T, S), (length(t), length(params)))
            for m in eachindex(params)
                (; μ, σ2, w) = params[m]
                for k in eachindex(μ)
                    val[:, m] .+= w[k] * $(funcs[2])(Normal(μ[k], sqrt(σ2[k])), t)
                end
            end
            return val
        end
    end
end

_pdf(pym::PitmanYorMixture, params::AbstractVector{<:NamedTuple}, t::Real) = _pdf(pym, params, [t])
_cdf(pym::PitmanYorMixture, params::AbstractVector{<:NamedTuple}, t::Real) = _cdf(pym, params, [t])


function _check_pitmanyorkwargs(discount::Real, strength::Real, prior_inv_scale_fac::Real, prior_shape::Real, prior_rate::Real)
    (0 ≤ discount < 1) || throw(ArgumentError("Discount parameter `discount` must lie in the interval [0,1)."))
    (strength > -discount) || throw(ArgumentError("Strength parameter `strength` must be greater than -discount."))
    (prior_inv_scale_fac > 0) || throw(ArgumentError("Prior inverse scale `prior_inv_scale_fac` must be positive."))
    (prior_shape > 0) || throw(ArgumentError("Prior shape parameter `prior_shape` must be positive."))
    (prior_rate > 0) || throw(ArgumentError("Prior rate parameter `prior_rate` must be positive."))
end