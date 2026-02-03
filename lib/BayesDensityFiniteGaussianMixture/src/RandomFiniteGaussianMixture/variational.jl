"""
    RandomFiniteGaussianMixtureVIPosterior{T<:Real} <: AbstractVIPosterior{T}

Struct representing the variational posterior distribution of a [`RandomFiniteGaussianMixture`](@ref).

# Fields
* `posterior_components`: The posterior probabilities on the number of components. Note that `support(posterior_components)[K]` corresponds to the posterior probability of model `K`.
* `mixture_fits`: Vector of [`FiniteGaussianMixtureVIPosterior`](@ref) objects, containing the fitted variational posterior distributions for differing values of mixture components.
* `rgfm`: The `RandomFiniteGaussianMixture` to which the variational posterior was fit.
"""
struct RandomFiniteGaussianMixtureVIPosterior{T<:Real, D<:DiscreteNonParametric, M<:AbstractVector, R<:RandomFiniteGaussianMixture{T}} <: AbstractVIPosterior{T}
    posterior_components::D
    mixture_fits::M
    rfgm::R
    function RandomFiniteGaussianMixtureVIPosterior{T}(
        posterior_components::DiscreteNonParametric,
        mixture_fits::AbstractVector,
        rfgm::RandomFiniteGaussianMixture
    ) where {T<:Real}
        return new{T, typeof(posterior_components), typeof(mixture_fits), typeof(rfgm)}(posterior_components, mixture_fits, rfgm)
    end
end

BayesDensityCore.model(vip::RandomFiniteGaussianMixtureVIPosterior) = vip.rfgm

"""
    posterior_components(vip::RandomFiniteGaussianMixtureVIPosterior{T}) where {T} -> DiscreteNonParametric{Int, T}

Get the variational posterior probability mass function of the number of mixture components as a `DiscreteNonParametric` instance.
"""
posterior_components(vip::RandomFiniteGaussianMixtureVIPosterior{T}) where {T} = vip.posterior_components

"""
    maximum_a_posteriori(
        vip::RandomFiniteGaussianMixtureVIPosterior{T}
    ) where {T} -> FiniteGaussianMixtureVIPosterior{T}

Get the variational posterior distribution that maximizes the approximate posterior probability on the number of components q(K).
"""
maximum_a_posteriori(vip::RandomFiniteGaussianMixtureVIPosterior) = vip.mixture_fits[argmax(probs(posterior_components(vip)))]

function Base.show(io::IO, ::MIME"text/plain", vip::RandomFiniteGaussianMixtureVIPosterior{T}) where {T}
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
    (; posterior_components, mixture_fits, rfgm) = vip
    samples = Vector{NamedTuple{(:μ, :σ2, :w, :β), Tuple{Vector{T}, Vector{T}, Vector{T}, T}}}(undef, n_samples)

    K_support = support(posterior_components)
    # Extract posterior model probabilities
    posterior_probs_components = probs(posterior_components)
    # Draw K ~ q(K) n_samples times, record the counts
    K_samples = rand(rng, Multinomial(n_samples, posterior_probs_components))

    # Generate samples conditional on the drawn K's
    i0 = 1
    for i in eachindex(K_samples)
        i1 = i0 + (K_samples[i]-1)
        # Sample (μ, σ2, w, β) from q(⋯|K)
        if K_samples[i] >= 1
            samples[i0:i1] .= sample(rng, mixture_fits[i], K_samples[i]).samples
        end
        i0 = i1 + 1
    end
    return PosteriorSamples{T}(samples, rfgm, n_samples, 0)
end

"""
    varinf(
        rfgm::RandomFiniteGaussianMixture{T};
        max_iter::Int = 2000
        rtol::Real    = 1e-6
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

julia> vip = varinf(rfgm);

julia> vip = varinf(rfgm; rtol=1e-7, max_iter=3000);
```
"""
function BayesDensityCore.varinf(
    rfgm::RandomFiniteGaussianMixture;
    max_iter::Int = 2000,
    rtol::Real    = 1e-6
)
    (max_iter >= 1) || throw(ArgumentError("Maximum number of iterations must be positive."))
    (rtol ≥ 0.0) || @warn "Relative tolerance is negative."
    return _variational_inference(rfgm, max_iter, rtol)
end

function _variational_inference(
    rfgm::RandomFiniteGaussianMixture{T},
    max_iter::Int,
    rtol::Real
) where {T}
    (; data, prior_components, prior_strength, prior_location, prior_variance, prior_shape, hyperprior_shape, hyperprior_rate) = rfgm
    (; x, n) = data

    probs_prior_components = probs(prior_components)
    K_support = support(prior_components)

    # Fit models and store the value of the posterior probability on the logscale
    mixture_fits_any = Vector{Any}(undef, length(probs(prior_components)))
    logprobs = Vector{T}(undef, length(probs(prior_components))) # Store logprobabilities in a separate vector for normalization purposes
    vi_type = Any
    for i in eachindex(probs_prior_components)
        fgm = FiniteGaussianMixture(
            x,
            K_support[i];
            prior_strength = prior_strength,
            prior_location = prior_location,
            prior_variance = prior_variance,
            prior_shape = prior_shape,
            hyperprior_shape = hyperprior_shape,
            hyperprior_rate = hyperprior_rate
        )
        vip, info = varinf(fgm; max_iter = max_iter, rtol = rtol)
        logprobs[i] = log(probs_prior_components[i]) + last(elbo(info))
        mixture_fits_any[i] = vip
        vi_type = typeof(vip)
    end

    # Get quantities for numerically stable normalization of probabilities
    probs_posterior_components = softmax(logprobs)
    posterior_components = DiscreteNonParametric(K_support, probs_posterior_components)

    # Create a new dictionary containing the normalized posterior probabilities and the corresponding model.
    mixture_fits = Vector{vi_type}(undef, length(probs(prior_components)))
    for i in eachindex(probs(prior_components))
        mixture_fits[i] = mixture_fits_any[i]
    end
    return RandomFiniteGaussianMixtureVIPosterior{T}(
        posterior_components,
        mixture_fits,
        rfgm
    )
end