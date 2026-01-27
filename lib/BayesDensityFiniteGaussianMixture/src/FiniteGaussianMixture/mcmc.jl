"""
    sample(
        [rng::Random.AbstractRNG],
        bsm::FiniteGaussianMixture{T},
        n_samples::Int;
        n_burnin::Int              = min(100, div(n_samples, 5)),
        initial_params::NamedTuple = _get_default_initparams_mcmc(hs)
    ) where {T} -> PosteriorSamples{T}

Generate `n_samples` posterior samples from a `FiniteGaussianMixture` using an augmented Gibbs sampler.

# Arguments
* `rng`: Optional random seed used for random variate generation.
* ``: The `FiniteGaussianMixture` object for which posterior samples are generated.
* `n_samples`: The total number of samples (including burn-in).

# Keyword arguments
* `n_burnin`: Number of burn-in samples.
* `initial_params`: Initial values used in the MCMC algorithm. Should be supplied as a `NamedTuple` with fields `:β` and `:σ2`, where `:β` is a `K`-dimensional vector and `σ2` is a positive scalar.

# Returns
* `ps`: A [`PosteriorSamples`](@ref) object holding the posterior samples and the original model object.

# Examples

"""
function StatsBase.sample(rng::AbstractRNG, finite_mixture::FiniteGaussianMixture, n_samples::Int; n_burnin::Int = min(div(n_samples, 5), 100), initial_params::NamedTuple=get_default_initparams_mcmc(shs))
    (1 ≤ n_samples ≤ Inf) || throw(ArgumentError("Number of samples must be a positive integer."))
    (0 ≤ n_burnin ≤ Inf) || throw(ArgumentError("Number of burn-in samples must be a nonnegative integer."))
    n_samples ≥ n_burnin || @warn "Number of total samples is smaller than the number of burn-in samples."
    _check_initparams_mcmc(initial_params, finit_emixture)
    return _sample_posterior(rng, finite_mixture, initial_params, n_samples, n_burnin)
end

function _check_initialparams_mcmc(initial_params::NamedTuple{N, T}, finite_mixture::FiniteGaussianMixture) where {N, T}
    (:μ in N && :σ2 in N && :w in N) || throw(ArgumentError("Expected a NamedTuple with fields μ, σ2 and w"))
    (; μ, σ2, w) = initial_params
    (length(μ) == length(σ2) == length(w) == finite_mixture.K) || throw(ArgumentError("Initial μ, σ2 or w dimensions are incompatible with the mixture model dimension."))
end

function _sample_posterior(rng::AbstractRNG, finite_mixture::FiniteGaussianMixture{T}, initial_params::NamedTuple, n_samples::Int, n_burnin::Int) where {T<:Real}
    # Unpack parameters, data
    (; data, K, prior_strength, prior_location, prior_variance, prior_shape, prior_rate) = finite_mixture
    (; x, n) = data
    (; μ, σ2, w) = initial_params
    
    cluster_alloc = Vector{Int}(undef, n)
    logprobs = Vector{K}(undef, T)
    samples = Vector{NamedTuple{(:μ, :σ2, :w), Tuple{Vector{T}, Vector{T}, Vector{T}}}}(undef, n_samples)

    for m in 1:n_samples
        # Sample from p(cluster_alloc|⋯)
        for i in eachindex(x)
            for k in 1:K
                logprobs[k] = logpdf(w[k]) + logpdf(Normal(μ[k], sqrt(σ2[k]), x[i]))
            end
            probs = softmax(logprobs)
            cluster_alloc[i] = wsample(rng, 1:K, probs)
        end

        # Sample from p(w|⋯)
        N = StatsBase.counts(cluster_alloc)
        w = rand(rng, Dirichlet(prior_strength .+ N))

        # Sample from p(μ|⋯)
        for k in 1:K
            ind_k = (cluster_alloc == k)
        end

        # Sample from p(σ2|⋯)

        # Store the samples
        samples[m] = (μ = μ, σ2 = σ2, w = w)
    end
    return PosteriorSamples{T}(samples, finite_mixture, n_samples, n_burnin)
end