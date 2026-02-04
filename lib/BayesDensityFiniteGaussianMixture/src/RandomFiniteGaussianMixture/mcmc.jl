"""
    sample(
        [rng::Random.AbstractRNG],
        rfgm::RandomFiniteGaussianMixture{T},
        n_samples::Int;
        n_burnin::Int              = min(1000, div(n_samples, 5)),
        initial_params::NamedTuple = _get_default_initparams_mcmc(hs)
    ) where {T} -> PosteriorSamples{T}

Generate `n_samples` posterior samples from a `RandomFiniteGaussianMixture` using an augmented Gibbs sampler.

# Arguments
* `rng`: Optional random seed used for random variate generation.
* `rfgm`: The `RandomFiniteGaussianMixture` object for which posterior samples are generated.
* `n_samples`: The total number of samples (including burn-in).

# Keyword arguments
* `n_burnin`: Number of burn-in samples.
* `initial_params`: Initial values used in the MCMC algorithm. Should be supplied as a `NamedTuple` with fields `:μ`, `:σ2` and `:w`, where all are `K`-dimensional vectors.
The following constraints must also be satisfied: `σ2[k]>0` for all `k` and `w[k]≥0` for all `k` and `sum(w) ≈ 1`

# Returns
* `ps`: A [`PosteriorSamples`](@ref) object holding the posterior samples and the original model object.

# Examples
```julia-repl
julia> using Random

julia> x = (1.0 .- (1.0 .- LinRange(0.0, 1.0, 5000)) .^(1/3)).^(1/3);

julia> rfgm = RandomFiniteGaussianMixture(x, 2)

julia> ps1 = sample(rfgm, 5_000);

julia> ps2 = sample(rfgm, 5_000; n_burnin=2_000, initial_params = (μ = [0.2, 0.8], σ2 = [1.0, 2.0], w = [0.7, 0.3]));
```
"""
function StatsBase.sample(
    rng::AbstractRNG,
    rfgm::RandomFiniteGaussianMixture,
    n_samples::Int;
    n_burnin::Int = min(div(n_samples, 5), 1000),
    initial_params::NamedTuple=_get_default_initial_params_mcmc(rfgm)
)
    (1 ≤ n_samples ≤ Inf) || throw(ArgumentError("Number of samples must be a positive integer."))
    (0 ≤ n_burnin ≤ Inf) || throw(ArgumentError("Number of burn-in samples must be a nonnegative integer."))
    n_samples ≥ n_burnin || @warn "Number of total samples is smaller than the number of burn-in samples."
    _check_initial_params_mcmc(initial_params, rfgm)
    return _sample_posterior(rng, rfgm, initial_params, n_samples, n_burnin)
end

function _get_default_initial_params_mcmc(rfgm::RandomFiniteGaussianMixture{T}) where {T}
    (; x, n) = rfgm.data
    # Initialize by the most a prior probable K.
    K_support = support(rfgm.prior_components)
    K = K_support[argmax[probs(rfgm.prior_components)]]

    breaks = quantile(x, LinRange{T}(0, 1, K+1))
    bin_counts, bin_sums, bin_sumsqs = _get_suffstats_binned(x, breaks)
    nonzero_ind = (bin_counts .!= 0)

    μ = fill(rfgm.prior_location, K)
    σ2 = fill(rfgm.prior_variance, K)

    μ[nonzero_ind] = bin_sums[nonzero_ind] ./ bin_counts[nonzero_ind]
    σ2[nonzero_ind] = (bin_sumsqs[nonzero_ind] - 2*μ[nonzero_ind] .* bin_sums[nonzero_ind] + bin_counts[nonzero_ind] .* μ[nonzero_ind].^2) ./ bin_counts[nonzero_ind] # Inflate the variances a bit (all obs should not necessarily belong to the nearest cluster)
    w = max.(bin_counts, T(1e-6)*n) # Prevent degenerate 0-weight clusters in initialization
    w = w / sum(w)
    return (μ = copy(μ), σ2 = copy(σ2), w = copy(w))
end

function _check_initial_params_mcmc(initial_params::NamedTuple{N, T}, rfgm::RandomFiniteGaussianMixture) where {N, T}
    (:μ in N && :σ2 in N && :w in N) || throw(ArgumentError("Expected a NamedTuple with fields μ, σ2 and w"))
    (; μ, σ2, w) = initial_params
    all(σ2 .> 0) || throw(ArgumentError("Initial σ2 vector contains negative values."))
    (all(w .≥ 0) && isapprox(sum(w), 1)) || throw(ArgumentError("Initial w vector does not belong to the K-simplex."))
    (length(μ) == length(σ2) == length(w)) || throw(ArgumentError("Initial μ, σ2 or w dimensions are incompatible."))
    (pdf(rfgm, length(w)) > 0) || throw(ArgumentError("Initial number of mixture components has probability 0.")) 
end

function _sample_posterior(rng::AbstractRNG, rfgm::RandomFiniteGaussianMixture{T}, initial_params::NamedTuple, n_samples::Int, n_burnin::Int) where {T<:Real}
    # Unpack parameters, data
    (; data, prior_components, prior_strength, prior_location, prior_variance, prior_shape, hyperprior_rate, hyperprior_shape) = rfgm
    (; x, n) = data
    (; μ, σ2, w) = initial_params

    K_support = support(prior_components)
    
    cluster_alloc = Vector{Int}(undef, n)
    cluster_alloc_new = Vector{Int}(undef, n)
    cluster_sum = Vector{T}(undef, K)
    cluster_sumsq = Vector{T}(undef, K)
    samples = Vector{NamedTuple{(:μ, :σ2, :w, :β), Tuple{Vector{T}, Vector{T}, Vector{T}, T}}}(undef, n_samples)

    for m in 1:n_samples
        # Sample from p(cluster_alloc|⋯)
        log_w = log.(w)
        for i in eachindex(x)
            logprobs = Vector{T}(undef, K)
            for k in 1:K
                logprobs[k] = log_w[k] + logpdf(Normal(μ[k], sqrt(σ2[k])), x[i])
            end
            probs = softmax(logprobs)
            cluster_alloc[i] = wsample(rng, 1:K, probs)
        end
        # Relabel clusters so that the K_plus first clusters are filled:
        cluster_counts = StatsBase.counts(cluster_alloc, K)
        cluster_rearranged_ind = vcat(findall(cluster_counts .> 0), findall(cluster_counts .== 0)) # length K
        K_plus = sum(cluster_rearranged_ind) # count number of indices for which the allocation counts are positive
        
        # Update μ, σ2 according to new labelling
        μ = μ[cluster_rearranged_ind]
        σ2 = σ2[cluster_rearranged_ind]
        # Update cluster allocations according to new labelling
        for k in eachindex(cluster_rearranged_ind)
            cluster_alloc_new[cluster_alloc .== k] .= findall(cluster_rearranged_ind .== k)
        end
        copy!(cluster_alloc, cluster_alloc_new) # Update allocation
        
        # Update the cluster counts to use the new labels
        cluster_counts = StatsBase.counts(cluster_alloc, K)

        # Sample from p(β|⋯)
        β = rand(rng, Gamma(hyperprior_shape + K*prior_shape, inv(hyperprior_rate + sum(inv, σ2))))

        # Sample from p(K|⋯)
        K_support_given_K_plus = setdiff(K_support, 1:K_plus-1) # possible values are those in the support of the prior that are ≥ the number of nonempty components
        logprobs_K = Vector{T}(undef, length(K_support_given_K_plus))
        for j in eachindex(K_support_given_K_plus)
            logprobs_K[j] = logpdf(prior_components, K_support_given_K_plus[j]) + loggamma(K + 1) - loggamma(K - K_plus + 1) + sum(loggamma(cluster_counts .+ prior_strength)) - K_plus * loggamma(1 + prior_strength)
        end
        K = wsample(rng, K_support_given_K_plus, softmax(logprobs_K))
        # Add empty clusters if K > K_plus
        μ_new = Vector{T}(undef, K)
        σ2_new = Vector{T}(undef, K)
        μ_new[1:K_plus] = μ_new[1:K_plus]
        σ2_new[1:K_plus] = σ2_new[1:K_plus]
        if K > K_plus
            # Sample new empty components from the prior conditional on the current β
            μ[K_plus+1:K] .= rand(rng, Normal(prior_location, sqrt(prior_variance)), K-K_plus)
            σ2[K_plus+1:K] .= rand(rng, InverseGamma(prior_shape, β), K-K_plus)
        end

        # Sample from p(w|⋯)
        cluster_counts = StatsBase.counts(cluster_alloc, K)
        w = rand(rng, Dirichlet(prior_strength .+ cluster_counts))

        # Sample from p(μ|⋯)
        cluster_sum = Vector{T}(undef, K_plus)   # sufficient statistics
        cluster_sumsq = Vector{T}(undef, K_plus)
        for k in 1:K_plus
            ind_k = (cluster_alloc .== k) # This is O(n), total complexity O(k*n). Better to have a separate loop that computes suffstats, as this would be O(n) instead
            cluster_sum[k] = sum(x[ind_k])
            cluster_sumsq[k] = sum(abs2, x[ind_k])
            variance_k = inv(1/prior_variance + cluster_counts[k]/σ2[k])
            mean_k = variance_k * (prior_location / prior_variance + cluster_sum[k]/σ2[k])
            μ[k] = rand(rng, Normal(mean_k, sqrt(variance_k)))
        end

        # Sample from p(σ2|⋯)
        for k in 1:K_plus
            rss_k = (cluster_sumsq[k] - 2*μ[k]*cluster_sum[k] + cluster_counts[k] * μ[k]^2)
            σ2[k] = rand(rng, InverseGamma(prior_shape + T(0.5)*cluster_counts[k], β + rss_k/2))
        end

        # Store the samples
        samples[m] = (μ = copy(μ), σ2 = copy(σ2), w = w, β = β)
    end
    return PosteriorSamples{T}(samples, rfgm, n_samples, n_burnin)
end