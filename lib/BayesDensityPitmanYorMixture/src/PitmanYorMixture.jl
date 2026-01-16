"""
    PitmanYorMixture{T<:Real} <: AbstractBayesDensityModel{T}
    
Struct representing a Pitman-Yor mixture model with a normal kernel.

# Constructors
    
    PitmanYorMixture(x::AbstractVector{<:Real}; kwargs...)
    PitmanYorMixture{T}(x::AbstractVector{<:Real}; kwargs...)

# Arguments
* `x`: The data vector.

# Keyword arguments
* `d`: Discount parameter of the Pitman-Yor process. Defaults to `0.0`, corresponding to a Dirichlet Process.
* `α`: Strength parameter of the Pitman-Yor process. Defaults to `1.0`.
* `μ0`: Prior mean of `μ`. Defaults to the midrange of the data vector.
* `σ0`: Prior standard deviation of `μ`. Defaults to `sqrt(R)`, where `R` is the sample range.
* `γ`: Prior shape parameter of `σ2`: Defaults to `2.0`.
* `δ`: Prior rate parameter of `σ2`. Defaults to `0.2*R^2` where `R` is the sample range.

# Returns
* `pym`: A Pitman-Yor mixture model object.

# Examples

# Extended help
"""
struct PitmanYorMixture{T<:Real, NT<:NamedTuple} <: AbstractBayesDensityModel{T}
    data::NT
    d::T
    α::T
    μ0::T
    σ0::T
    γ::T
    δ::T
    function PitmanYorMixture{T}(x::AbstractVector{<:Real}; d::Real=0.0, α::Real=1.0, μ0::Real=_get_default_μ0(x), σ0::Real=_get_default_σ0(x), γ::Real=2.0, δ::Real=_get_default_δ(x)) where {T<:Real}
        _check_pitmanyorkwargs(d, α, σ0, γ, δ)
        data = (x = T.(x), n = length(x))

        return new{T,typeof(data)}(data, T(d), T(α), T(μ0), T(σ0), T(γ), T(δ))
    end
end
PitmanYorMixture(args...; kwargs...) = PitmanYorMixture{Float64}(args...; kwargs...)

Base.:(==)(pym1::PitmanYorMixture, pym2::PitmanYorMixture) = (pym1.data == pym2.data) && (hyperparams(pym1) == hyperparams(pym2))

"""
    support(pym::PitmanYorMixture{T}) where {T} -> NTuple{2, T}

Get the support of the Pitman-Yor mixture model `pym`.
"""
BayesDensityCore.support(pym::PitmanYorMixture{T}) where {T} = (-T(Inf), T(Inf))

"""
    hyperparams(
        pym::PitmanYorMixture{T}
    ) where {T} -> @NamedTuple{d::T, α::T, μ0::T, σ0::T, γ::T, δ::T}

Returns the hyperparameters of the Pitman-Yor mixture model `pym` as a `NamedTuple`.
"""
BayesDensityCore.hyperparams(pym::PitmanYorMixture) = (d=pym.d, α=pym.α, μ0=pym.μ0, σ0=pym.σ0, γ=pym.γ, δ=pym.δ)

# Print method for unbinned data
function Base.show(io::IO, ::MIME"text/plain", pym::PitmanYorMixture{T}) where {T}
    println(io, nameof(typeof(pym)), '{', T, "}:")
    println(io, "Using ", pym.data.n, "  observations.")
    let io = IOContext(io, :compact => true, :limit => true)
        println(io, "Hyperparameters: ")
        println(io, " d = " , pym.d, ", α = ", pym.α)
        println(io, " μ0 = " , pym.μ0, ", σ0 = ", pym.σ0)
        print(io, "γ = ", pym.γ, ", δ = ", pym.δ)
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
    (; d, α, μ0, σ0, γ, δ) = hyperparams(pym)
    K = length(cluster_counts) # Number of existing clusters

    vals = zero(R)
    
    # Compute relative weight of (μ, σ²) belonging to a new cluster
    for k in eachindex(cluster_counts)
        vals += (cluster_counts[k] - d) / (α + n) * pdf(Normal(μ[k], sqrt(σ2[k])), t) # Contribution from event that (μ, σ²) belong to existing clusters
    end
    vals += (α + K * d) / (α + n) * exp(_tdist_logpdf(2*γ, μ0, sqrt(σ0^2 + δ/γ), t)) # Contribution from event that (μ, σ²) forms a new cluster
    return vals
end

function _pdf(pym::PitmanYorMixture{T}, params::NamedTuple, t::AbstractVector{S}) where {T<:Real, S<:Real}
    R = promote_type(T, S)
    (; μ, σ2, cluster_counts) = params
    n = pym.data.n
    (; d, α, μ0, σ0, γ, δ) = hyperparams(pym)
    K = length(cluster_counts) # Number of existing clusters

    vals = zeros(R, length(t))
    
    # Compute relative weight of (μ, σ²) belonging to a new cluster
    for k in eachindex(cluster_counts)
        vals .+= (cluster_counts[k] - d) / (α + n) .* pdf(Normal(μ[k], sqrt(σ2[k])), t) # Contribution from event that (μ, σ²) belong to existing clusters
    end
    vals .+= (α + K * d) / (α + n) .* exp.(_tdist_logpdf(2*γ, μ0, sqrt(σ0^2 + δ/γ), t)) # Contribution from event that (μ, σ²) forms a new cluster
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
    (; d, α, μ0, σ0, γ, δ) = hyperparams(pym)
    K = length(cluster_counts) # Number of existing clusters
    
    vals = zero(R)

    # Compute relative weight of (μ, σ²) belonging to a new cluster
    for k in eachindex(cluster_counts)
        vals += (cluster_counts[k] - d) / (α + n) * cdf(Normal(μ[k], sqrt(σ2[k])), t) # Contribution from event that (μ, σ²) belong to existing clusters
    end

    scale_new = sqrt(σ0^2 + δ/γ)
    vals += (α + K * d) / (α + n) * cdf(TDist(2*γ), (t - μ0)/scale_new) # Contribution from event that (μ, σ²) forms a new cluster
    return vals
end

function _cdf(pym::PitmanYorMixture{T}, params::NamedTuple, t::AbstractVector{S}) where {T<:Real, S<:Real}
    R = promote_type(T, S)
    (; μ, σ2, cluster_counts) = params
    n = pym.data.n
    (; d, α, μ0, σ0, γ, δ) = hyperparams(pym)
    K = length(cluster_counts) # Number of existing clusters
    
    vals = zeros(R, length(t))

    # Compute relative weight of (μ, σ²) belonging to a new cluster
    for k in eachindex(cluster_counts)
        vals .+= (cluster_counts[k] - d) / (α + n) .* cdf(Normal(μ[k], sqrt(σ2[k])), t) # Contribution from event that (μ, σ²) belong to existing clusters
    end

    scale_new = sqrt(σ0^2 + δ/γ)
    vals .+= (α + K * d) / (α + n) .* cdf(TDist(2*γ), (t .- μ0)/scale_new) # Contribution from event that (μ, σ²) forms a new cluster
    return vals
end

function _check_pitmanyorkwargs(d::Real, α::Real, σ0::Real, γ::Real, δ::Real)
    (0 ≤ d < 1) || throw(ArgumentError("Discount parameter `d` must lie in the interval [0,1)."))
    (α > -d) || throw(ArgumentError("Strength parameter `α` must be greater than -d."))
    (σ0 > 0) || throw(ArgumentError("Prior standard deviation `σ0` must be positive."))
    (γ > 0) || throw(ArgumentError("Prior shape parameter `γ` must be positive."))
    (δ > 0) || throw(ArgumentError("Prior rate parameter `δ` must be positive."))
end

function _get_default_μ0(x::AbstractVector{<:Real})
    xmin, xmax = extrema(x)
    return (xmax + xmin) / 2
end

function _get_default_σ0(x::AbstractVector{<:Real})
    xmin, xmax = extrema(x)
    return sqrt(xmax - xmin)
end

function _get_default_δ(x::AbstractVector{<:Real})
    xmin, xmax = extrema(x)
    return 0.2*(xmax - xmin)^2
end

# Logpdf of location-shape t-distribution
function _tdist_logpdf(df::T, loc::T, scale::T, t::Union{S, AbstractVector{S}}) where {T<:Real, S<:Real}
    R = promote_type(T, S)
    return @. loggamma((df+1)/2) - loggamma(df/2) - log(df*R(pi)*scale) - (df + 1) / 2 * log(1 + (t - loc)^2/(df*scale))
end