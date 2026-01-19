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
    #_check_initparams(bsm, initial_params)
    return _sample_posterior(rng, pym, initial_params, n_samples, n_burnin)
end

function _sample_posterior(rng::AbstractRNG, pym::PitmanYorMixture{T, D}, initial_params::NamedTuple, n_samples::Int, n_burnin::Int) where {T, D}
    # Unpack hyperparameters and data
    (; data, discount, strength, location, scale_fac, rate, shape) = pym
    (; x, n) = data

    # Initialize μ, σ2
    (; μ, σ2) = initial_params

    # Cluster allocation vector for all variables
    cluster_alloc = Vector{Int}(undef, n)
    # DO SOME INITIALIZATION HERE
    cluster_counts = StatsBase.counts(cluster_alloc)
    K = length(cluster_counts)

    marginal_scale = sqrt(rate*(1 + scale_fac)/shape)

    samples = Vector{NamedTuple{(:μ, :σ2, :cluster_counts), (Vector{T}, Vector{T}, Vector{Int})}}(undef, n_samples)

    for m in 1:n_samples
        # Update clusters
        for i in 1:n
            if cluster_counts[cluster_alloc[i]] == 1
                # Remove the cluster (swap with last cluster and resize vectors)
                old = cluster_alloc[i]
                μ[cluster_alloc[i]] = μ[K]
                σ2[cluster_alloc[i]] = σ2[K]
                cluster_counts[cluster_alloc[i]] = cluster_counts[K]
                pop!(μ)
                pop!(σ2)
                pop!(cluster_counts)
                K = K - 1
                cluster_alloc[cluster_alloc .== K] .= old
            else
                cluster_counts[cluster_alloc[i]] -= 1
            end

            # Assign obervation x[i] to a new cluster:
            logprobs = Vector{T}(undef, K+1)
            for k in 1:K
                logprobs[k] = logpdf(Normal(μ[k], sqrt(σ2[k])), x[i]) + log(cluster_counts[k] - discount)
            end
            logprobs[K+1] = logpdf(TDistLocationScale(2.0*shape, location, marginal_scale), x[i]) + log(strength + K * discount)
            probs = BayesDensityCore.softmax(logprobs)
            new_alloc = wsample(rng, 1:K+1, probs)
            cluster_alloc[i] = new_alloc
            
            if new_alloc == K+1
                # Sample μ, σ2 from posterior
                shape_new = shape + 1/2
                scale_fac_new = scale_fac + 1
                rate_new = rate + scale_fac * abs2(x[i] - location) / (2*scale_fac_new)
                location_new = (x[i] + scale_fac * location) / (1 + scale_fac)
                push!(σ2, rand(rng, InverseGamma(shape_new, rate_new)))
                push!(μ, Normal(location_new, sqrt(σ2[end] * scale_fac_new)))
                # Update cluster counts
                push!(cluster_counts, 1)
            end
        end
        # Sample parameters given clusters.
        for k in 1:K
            # Compute updated cluster-specific hyperparameters
            clust_k_ind = (cluster_alloc .== k)
            clust_mean = mean(view(x, clust_k_ind))
            scale_fac_post = scale_fac + cluster_counts[k]
            shape_post = shape + cluster_counts[k]/2
            rate_post = rate + (sum(abs2, view(x, clust_k_ind) .- clust_mean) + cluster_counts[k] * scale_fac / scale_fac_post * sum(abs2, clust_mean - location)) / 2
            location_post = (scale_fac * location + cluster_counts[k] * clust_mean) / scale_fac_post

            # Sample cluster parameters
            σ2[k] = rand(rng, InverseGamma(shape_post, rate_post))
            μ[k] = rand(rng, Normal(location_post, sqrt(scale_fac_post * σ2[k])))
        end

        samples[m] = (μ = μ, σ2 = σ2, cluster_counts = cluster_counts)
    end

    return PosteriorSamples{T}(samples, pym, n_samples, n_burnin)
end