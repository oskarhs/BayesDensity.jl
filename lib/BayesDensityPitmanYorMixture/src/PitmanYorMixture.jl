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
* `location`: Prior mean of the location parameter `μ`. Defaults to `mean(x)`.
* `scale_fac`: Factor by which the conditional prior variance `σ2` of `μ` is scaled. Defaults to `1`.
* `shape`: Prior shape parameter of the squared scale parameter `σ2`: Defaults to `2.0`.
* `rate`: Prior rate parameter of the squared scale parameter `σ2`. Defaults to `var(x)`.

# Returns
* `pym`: A Pitman-Yor mixture model object.

# Examples

# Extended help
"""
struct PitmanYorMixture{T<:Real, NT<:NamedTuple} <: AbstractBayesDensityModel{T}
    data::NT
    discount::T
    strength::T
    location::T
    scale_fac::T
    shape::T
    rate::T
    function PitmanYorMixture{T}(x::AbstractVector{<:Real}; discount::Real=0.0, strength::Real=1.0, location::Real=mean(x), scale_fac::Real=1.0, shape::Real=2.0, rate::Real=var(x)) where {T<:Real}
        _check_pitmanyorkwargs(discount, strength, scale_fac, shape, rate)
        data = (x = T.(x), n = length(x))

        return new{T,typeof(data)}(data, T(discount), T(strength), T(location), T(scale_fac), T(shape), T(rate))
    end
end
PitmanYorMixture(args...; kwargs...) = PitmanYorMixture{Float64}(args...; kwargs...)

Base.:(==)(pym1::PitmanYorMixture, pym2::PitmanYorMixture) = (pym1.data == pym2.data) && (hyperparams(pym1) == hyperparams(pym2))

"""
    support(pym::PitmanYorMixture{T}) where {T} -> NTuple{2, T}

Get the support of the Pitman-Yor mixture model `pym`.
"""
BayesDensityCore.support(::PitmanYorMixture{T}) where {T} = (-T(Inf), T(Inf))

"""
    hyperparams(
        pym::PitmanYorMixture{T}
    ) where {T} -> @NamedTuple{discount::T, strength::T, location::T, scale_fac::T, shape::T, rate::T}

Returns the hyperparameters of the Pitman-Yor mixture model `pym` as a `NamedTuple`.
"""
BayesDensityCore.hyperparams(pym::PitmanYorMixture) = (discount=pym.discount, strength=pym.strength, location=pym.location, scale_fac=pym.scale_fac, shape=pym.shape, rate=pym.rate)

# Print method for unbinned data
function Base.show(io::IO, ::MIME"text/plain", pym::PitmanYorMixture{T}) where {T}
    println(io, nameof(typeof(pym)), '{', T, "}:")
    println(io, "Using ", pym.data.n, "  observations.")
    let io = IOContext(io, :compact => true, :limit => true)
        println(io, "Hyperparameters: ")
        println(io, " discount = " , pym.discount, ", strength = ", pym.strength)
        println(io, " location = " , pym.location, ", scale_fac = ", pym.scale_fac)
        print(io, "shape = ", pym.shape, ", rate = ", pym.rate)
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

Evaluate ``f(t | \\boldsymbol{\\eta})`` for a given `PitmanYorMixture` when the model parameters of the NamedTuple `params` are given by ``\\boldsymbol{\\eta}``.

The named tuple should contain fields named `:μ`, `:σ2` and `:cluster_counts`.
"""
Distributions.pdf(pym::PitmanYorMixture, params::NamedTuple, t::Real) = _pdf(pym, params, t)
Distributions.pdf(pym::PitmanYorMixture, params::NamedTuple, t::AbstractVector{<:Real}) = _pdf(pym, params, t)


function _pdf(pym::PitmanYorMixture{T}, params::NamedTuple, t::S) where {T<:Real, S<:Real}
    R = promote_type(T, S)
    (; μ, σ2, cluster_counts) = params
    n = pym.data.n
    (; discount, strength, location, scale_fac, shape, rate) = hyperparams(pym)
    K = length(cluster_counts) # Number of existing clusters

    vals = zero(R)
    
    # Compute relative weight of (μ, σ²) belonging to a new cluster
    for k in eachindex(cluster_counts)
        vals += (cluster_counts[k] - discount) / (strength + n) * pdf(Normal(μ[k], sqrt(σ2[k])), t) # Contribution from event that (μ, σ²) belong to existing clusters
    end
    scale_new = sqrt(rate*(1 + scale_fac)/shape)
    vals += (strength + K * discount) / (strength + n) * pdf(TDistLocationScale(2*shape, location, scale_new), t) # Contribution from event that (μ, σ²) forms a new cluster
    return vals
end

function _pdf(pym::PitmanYorMixture{T}, params::NamedTuple, t::AbstractVector{S}) where {T<:Real, S<:Real}
    R = promote_type(T, S)
    (; μ, σ2, cluster_counts) = params
    n = pym.data.n
    (; discount, strength, location, scale_fac, shape, rate) = hyperparams(pym)
    K = length(cluster_counts) # Number of existing clusters

    vals = zeros(R, length(t))
    
    # Compute relative weight of (μ, σ²) belonging to a new cluster
    for k in eachindex(cluster_counts)
        vals .+= (cluster_counts[k] - discount) / (strength + n) .* pdf(Normal(μ[k], sqrt(σ2[k])), t) # Contribution from event that (μ, σ²) belong to existing clusters
    end
    scale_new = sqrt(rate*(1 + scale_fac)/shape)
    vals .+= (strength + K * discount) / (strength + n) .* pdf(TDistLocationScale(2*shape, location, scale_new), t) # Contribution from event that (μ, σ²) forms a new cluster
    return vals
end

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

Evaluate ``F(t | \\boldsymbol{\\eta})`` for a given `PitmanYorMixture` when the model parameters of the NamedTuple `params` are given by ``\\boldsymbol{\\eta}``.

The named tuple should contain fields named `:μ`, `:σ2` and `:cluster_counts`.
"""
Distributions.cdf(pym::PitmanYorMixture, params::NamedTuple, t::Real) = _cdf(pym, params, t)
Distributions.cdf(pym::PitmanYorMixture, params::NamedTuple, t::AbstractVector{<:Real}) = _cdf(pym, params, t)

function _cdf(pym::PitmanYorMixture{T}, params::NamedTuple, t::S) where {T<:Real, S<:Real}
    R = promote_type(T, S)
    (; μ, σ2, cluster_counts) = params
    n = pym.data.n
    (; discount, strength, location, scale_fac, shape, rate) = hyperparams(pym)
    K = length(cluster_counts) # Number of existing clusters
    
    vals = zero(R)

    # Compute relative weight of (μ, σ²) belonging to a new cluster
    for k in eachindex(cluster_counts)
        vals += (cluster_counts[k] - discount) / (strength + n) * cdf(Normal(μ[k], sqrt(σ2[k])), t) # Contribution from event that (μ, σ²) belong to existing clusters
    end

    scale_new = sqrt(rate*(1 + scale_fac)/shape)
    vals += (strength + K * discount) / (strength + n) * cdf(TDistLocationScale(2*shape, location, scale_new), t) # Contribution from event that (μ, σ²) forms a new cluster
    return vals
end

function _cdf(pym::PitmanYorMixture{T}, params::NamedTuple, t::AbstractVector{S}) where {T<:Real, S<:Real}
    R = promote_type(T, S)
    (; μ, σ2, cluster_counts) = params
    n = pym.data.n
    (; discount, strength, location, scale_fac, shape, rate) = hyperparams(pym)
    K = length(cluster_counts) # Number of existing clusters
    
    vals = zeros(R, length(t))

    # Compute relative weight of (μ, σ²) belonging to a new cluster
    for k in eachindex(cluster_counts)
        vals .+= (cluster_counts[k] - discount) / (strength + n) .* cdf(Normal(μ[k], sqrt(σ2[k])), t) # Contribution from event that (μ, σ²) belong to existing clusters
    end
    scale_new = sqrt(rate*(1 + scale_fac)/shape)
    vals .+= (strength + K * discount) / (strength + n) .* cdf(TDistLocationScale(2*shape, location, scale_new), t) # Contribution from event that (μ, σ²) forms a new cluster
    return vals
end

function _check_pitmanyorkwargs(discount::Real, strength::Real, scale_fac::Real, shape::Real, rate::Real)
    (0 ≤ discount < 1) || throw(ArgumentError("Discount parameter `discount` must lie in the interval [0,1)."))
    (strength > -discount) || throw(ArgumentError("Strength parameter `strength` must be greater than -discount."))
    (scale_fac > 0) || throw(ArgumentError("Prior standard deviation `scale_fac` must be positive."))
    (shape > 0) || throw(ArgumentError("Prior shape parameter `shape` must be positive."))
    (rate > 0) || throw(ArgumentError("Prior rate parameter `rate` must be positive."))
end