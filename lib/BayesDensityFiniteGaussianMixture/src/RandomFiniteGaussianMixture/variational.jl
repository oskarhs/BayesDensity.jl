"""
    RandomFiniteGaussianMixtureVIPosterior{T<:Real} <: AbstractVIPosterior{T}

Struct representing the variational posterior distribution of a [`RandomFiniteGaussianMixture`](@ref).

# Fields
* `probs_components`: AbstractWeights object representing the posterior over K, given by q(K) ∝ p(K) ELBO(K).
* `mixture_fits`: Vector of [`FiniteGaussianMixtureVIPosterior`](@ref), containing the fitted variational posterior distributions for differing values of mixture components.
* `rgfm`: The `RandomFiniteGaussianMixture` to which the variational posterior was fit.
"""
struct RandomFiniteGaussianMixtureVIPosterior{T<:Real, A<:StatsBase.AbstractWeights, M<:AbstractVector{<:FiniteGaussianMixtureVIPosterior{T}}, R<:RandomFiniteGaussianMixture} <: AbstractVIPosterior{T}
    posterior_prob_components::A
    mixture_fits::M
    rfgm::R
    function RandomFiniteGaussianMixtureVIPosterior{T}(
        posterior_prob_components::StasBase.AbstractWeights{T},
        mixture_fits::AbstractVector{<:FiniteGaussianMixtureVIPosterior{T}},
        rfgm::RandomFiniteGaussianMixture
    ) where {T<:Real}
        return new{T, typeof(mixture_fits), typeof(rfgm)}(posterior_prob_components, mixture_fits, rfgm)
    end
end

BayesDensityCore.model(vip::RandomFiniteGaussianMixtureVIPosterior) = vip.rfgm

"""
    posterior_prob_components(vip::RandomFiniteGaussianMixtureVIPosterior{T}) where {T} -> Vector{T}'

Get the variational posterior over the number of mixture components.
"""
posterior_prob_components(vip::RandomFiniteGaussianMixtureVIPosterior) = vip.probs_components

"""
    maximum_a_posteriori(
        vip::RandomFiniteGaussianMixtureVIPosterior{T}
    ) where {T} -> FiniteGaussianMixtureVIPosterior{T}

Get the variational posterior distribution that maximizes the approximate posterior probability on the number of components q(K).
"""
maximum_a_posteriori(vip::RandomFiniteGaussianMixtureVIPosterior) = vip.mixture_fits[argmax(posterior_prob_components(vip))]

function Base.show(io::IO, ::MIME"text/plain", vip::RandomFiniteGaussianMixtureVIPosterior{T, A, M, R}) where {T, A, M, R}
    println(io, nameof(typeof(vip)), "{", T, "}.")
    println(io, "Model:")
    print(io, model(vip))
    nothing
end
Base.show(io::IO, vip::RandomFiniteGaussianMixtureVIPosterior) = show(io, MIME("text/plain"), vip)

function StatsBase.sample(
    rng::AbstractRNG,
    vip::RandomFiniteGaussianMixtureVIPosterior{T},
    n_samples::Int
) where {T<:Real}
    (; posterior_prob_components, mixture_fits, rfgm) = vip
    samples = Vector{NamedTuple{(:μ, :σ2, :w), Tuple{Vector{T}, Vector{T}, Vector{T}}}}(undef, n_samples)

    for m in 1:n_samples
        # Draw K ~ q(K)
        K = sample(rng, posterior_prob_components)
        # Sample (μ, σ2, w) from q(⋯|K)
        samples[m] = sample(rng, mixture_fits[K], 1)
    end
    return PosteriorSamples{T}(samples, rfgm, n_samples, 0)
end

"""
    varinf(
        rfgm::RandomFiniteGaussianMixture{T};
        max_iter::Int              = 2000
        rtol::Real                 = 1e-6
    ) where {T} -> PitmanYorMixtureVIPosterior{T}

Find a variational approximation to the posterior distribution of a [`RandomFiniteGaussianMixture`](@ref) using mean-field variational inference.

# Arguments
* `rfgm`: The `RandomFiniteGaussianMixture` whose posterior we want to approximate.

# Keyword arguments
* `max_iter`: Maximal number of VI iterations. Defaults to `2000`.
* `rtol`: Relative tolerance used to determine convergence. Defaults to `1e-6`.

# Returns
* `vip`: A [`RandomFiniteGaussianMixtureVIPosterior`](@ref) object representing the variational posterior.
* `info`: A [`VariationalOptimizationResult`](@ref) describing the result of the optimization.

!!! note
    To perform the optimization for a fixed number of iterations irrespective of the convergence criterion, one can set `rtol = 0.0`, and `max_iter` equal to the desired total iteration count.
    Note that setting `rtol` to a strictly negative value will issue a warning.

# Examples
```julia-repl
julia> using Random

julia> x = (1.0 .- (1.0 .- LinRange(0.0, 1.0, 5000)) .^(1/3)).^(1/3);

julia> rfgm = RandomFiniteGaussianMixture(x);

julia> vip, info = varinf(rfgm);

julia> vip, info = varinf(rfgm; rtol=1e-7, max_iter=3000);
```
"""
function BayesDensityCore.varinf(
    rfgm::RandomFiniteGaussianMixture;
    max_iter::Int = 2000,
    rtol::Real    = 1e-6
)
    (max_iter >= 1) || throw(ArgumentError("Maximum number of iterations must be positive."))
    (rtol ≥ 0.0) || @warn "Relative tolerance is negative."
    return _variational_inference(rfgm, initial_params, max_iter, rtol)
end

function _variational_inference(
    rfgm::FiniteGaussianMixture{T},
    initial_params::NamedTuple,
    max_iter::Int,
    rtol::Real
) where {T}
    
end