"""
    RandomFiniteGaussianMixtureVIPosterior{T<:Real} <: AbstractVIPosterior{T}

Struct representing the variational posterior distribution of a [`RandomFiniteGaussianMixture`](@ref).

# Fields
* `probs_components`: AbstractWeights object representing the posterior over K, given by q(K) ∝ p(k) ELBO(k).
* `mixture_fits`: Vector of [`FiniteGaussianMixtureVIPosterior`](@ref), containing the fitted variational posterior distributions for differing values of mixture components.
* `rgfm`: The `RandomFiniteGaussianMixture` to which the variational posterior was fit.
"""
struct RandomFiniteGaussianMixtureVIPosterior{T<:Real, A<:StatsBase.AbstractWeights, M<:AbstractVector{<:FiniteGaussianMixtureVIPosterior}, R<:RandomFiniteGaussianMixture} <: AbstractVIPosterior{T}
    posterior_components::A
    mixture_fits::M
    rfgm::R
    function RandomFiniteGaussianMixtureVIPosterior{T}(
        posterior_components::StasBase.AbstractWeights{T},
        mixture_fits::AbstractVector{<:FiniteGaussianMixtureVIPosterior},
        rfgm::RandomFiniteGaussianMixture
    ) where {T<:Real}
        return new{T, typeof(mixture_fits), typeof(rfgm)}(posterior_components, mixture_fits, rfgm)
    end
end

BayesDensityCore.model(vip::RandomFiniteGaussianMixtureVIPosterior) = vip.rfgm

function Base.show(io::IO, ::MIME"text/plain", vip::RandomFiniteGaussianMixtureVIPosterior{T, A, B, M}) where {T, A, B, M}
    K = length(vip.q_θ)
    println(io, nameof(typeof(vip)), "{", T, "} vith truncation level ", K, " and variational densities:")
    println(io, " q_v::", A, ",")
    println(io, " q_θ::", B, ",")
    println(io, "Model:")
    print(io, model(vip))
    nothing
end
Base.show(io::IO, vip::RandomFiniteGaussianMixtureVIPosterior) = show(io, MIME("text/plain"), vip)

function StatsBase.sample(rng::AbstractRNG, vip::RandomFiniteGaussianMixtureVIPosterior{T, A, B, M}, n_samples::Int) where {T<:Real, A, B, M}
    (; posterior_components, mixture_fits, rfgm) = vip
    samples = Vector{NamedTuple{(:μ, :σ2, :w), Tuple{Vector{T}, Vector{T}, Vector{T}}}}(undef, n_samples)

    for m in 1:n_samples
        # Draw K ∼ q(k)
        K = sample(rng, posterior_components)
        # Sample from q(⋯|K)
        samples[m] = sample(rng, mixture_fits[K], 1)
    end
    return PosteriorSamples{T}(samples, fgm, n_samples, 0)
end