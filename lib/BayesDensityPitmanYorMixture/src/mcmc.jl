"""
    sample(
        [rng::Random.AbstractRNG],
        pym::PitmanYorMixture{T},
        n_samples::Int;
        n_burnin::Int = min(1000, div(n_samples, 5)),
        initial_params::NamedTuple = _get_default_initparams_mcmc(pym)
    ) where {T} -> PosteriorSamples{T}

Generate `n_samples` posterior samples from a `PitmanYorMixture` using an augmented marginal Gibbs sampler.

# Arguments
* `rng`: Optional random seed used for random variate generation.
* `pym`: The `PitmanYorMixture` object for which posterior samples are generated.
* `n_samples`: The total number of samples (including burn-in).

# Keyword arguments
* `n_burnin`: Number of burn-in samples.
* `initial_params`: Initial values used in the MCMC algorithm. Should be supplied as a `NamedTuple` with fields `:μ` and `:σ2`, where both are vectors of the same dimension.

# Returns
* `ps`: A [`PosteriorSamples`](@ref) object holding the posterior samples and the original model object.
"""
function StatsBase.sample(
    rng::AbstractRNG,
    pym::PitmanYorMixture,
    n_samples::Int;
    n_burnin::Int = min(1000, div(n_samples, 5)),
    initial_params::NamedTuple=_get_default_initparams_mcmc(pym)
)
    (1 ≤ n_samples ≤ Inf) || throw(ArgumentError("Number of samples must be a positive integer."))
    (0 ≤ n_burnin ≤ Inf) || throw(ArgumentError("Number of burn-in samples must be a nonnegative integer."))
    (n_samples ≥ n_burnin) || @warn "The total number of samples is smaller than the number of burn-in samples."
    _check_initparams(bsm, initial_params)
    return _sample_posterior(rng, pym, initial_params, n_samples, n_burnin)
end

function _sample_posterior(rng::AbstractRNG, pym::PitmanYorMixture{T, D}, initial_params::NamedTuple, n_samples::Int, n_burnin::Int) where {T, D}
    # Unpack hyperparameters and data
    (; data, d, α, μ0, σ0, γ, δ) = pym
    (; x, n) = data

    # Initialize μ, σ2
    (; μ, σ2) = initial_params

    # Cluster allocation vector for all variables
    cluster_alloc = Vector{Int}(undef, n)
    # DO SOME INITIALIZATION HERE
    cluster_counts = StatsBase.counts(cluster_alloc)
    K = length(cluster_counts)

    samples = Vector{NamedTuple{(:μ, :σ2, :cluster_alloc), (Vector{T}, Vector{T}, Vector{T})}}(undef, n_samples)

    for m in 1:n_samples

        for i in 1:n
            logprobs = Vector{T}(undef, K+1)
            for k in 1:K
                cluster_counts_k = sum(cluster_alloc .== k)
                logprobs[k] = logpdf(Normal(μ[k], sqrt(σ2[k])), x[i]) + log(cluster_counts_k - d)
            end
            logprobs[K+1] = _tdist_logpdf(2*γ, μ0, sqrt(σ0^2 + δ/γ), x[i]) + log(α + K * d)
            probs = BayesDensityCore.softmax(logprobs)
            new_alloc = wsample(rng, 1:K+1, probs)
            cluster_counts[new_alloc] += 1
            
            if new_alloc == K+1
                
            end
        end

        samples[m] = (μ = μ, σ2 = σ2, cluster_counts = cluster_counts)
    end

    return PosteriorSamples{T}(samples, pym, n_samples, n_burnin)
end