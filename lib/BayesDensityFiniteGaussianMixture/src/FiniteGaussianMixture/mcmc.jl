"""
    sample(
        [rng::Random.AbstractRNG],
        fgm::FiniteGaussianMixture{T},
        n_samples::Int;
        n_burnin::Int              = min(1000, div(n_samples, 5)),
        initial_params::NamedTuple = _get_default_initparams_mcmc(rfgm)
    ) where {T} -> PosteriorSamples{T}

Generate `n_samples` posterior samples from a `FiniteGaussianMixture` using an augmented Gibbs sampler.

# Arguments
* `rng`: Optional random seed used for random variate generation.
* `fgm`: The `FiniteGaussianMixture` object for which posterior samples are generated.
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

julia> fgm = FiniteGaussianMixture(x, 4);

julia> ps1 = sample(fgm, 5_000);

julia> ps2 = sample(fgm, 5_000; n_burnin=2_000, initial_params = (μ = [0.2, 0.8], σ2 = [1.0, 2.0], w = [0.7, 0.3]));
```
"""
function StatsBase.sample(
    rng::AbstractRNG,
    fgm::FiniteGaussianMixture,
    n_samples::Int;
    n_burnin::Int = min(div(n_samples, 5), 1000),
    initial_params::NamedTuple=_get_default_initial_params_mcmc(fgm)
)
    (1 ≤ n_samples ≤ Inf) || throw(ArgumentError("Number of samples must be a positive integer."))
    (0 ≤ n_burnin ≤ Inf) || throw(ArgumentError("Number of burn-in samples must be a nonnegative integer."))
    n_samples ≥ n_burnin || @warn "Number of total samples is smaller than the number of burn-in samples."
    _check_initial_params_mcmc(initial_params, fgm)
    return _sample_posterior(rng, fgm, initial_params, n_samples, n_burnin)
end

function _get_default_initial_params_mcmc(fgm::FiniteGaussianMixture{T}) where {T}
    (; x, n) = fgm.data
    K = fgm.K
    breaks = quantile(x, LinRange{T}(0, 1, K+1))
    bin_counts, bin_sums, bin_sumsqs = _get_suffstats_binned(x, breaks)
    nonzero_ind = (bin_counts .!= 0)

    μ = fill(fgm.prior_location, K)
    σ2 = fill(fgm.prior_variance, K)

    μ[nonzero_ind] = bin_sums[nonzero_ind] ./ bin_counts[nonzero_ind]
    σ2[nonzero_ind] = (bin_sumsqs[nonzero_ind] - 2*μ[nonzero_ind] .* bin_sums[nonzero_ind] + bin_counts[nonzero_ind] .* μ[nonzero_ind].^2) ./ bin_counts[nonzero_ind] # Inflate the variances a bit (all obs should not necessarily belong to the nearest cluster)
    w = max.(bin_counts, T(1e-6)*n) # Prevent degenerate 0-weight clusters in initialization
    w = w / sum(w)
    return (μ = copy(μ), σ2 = copy(σ2), w = copy(w))
end

function _check_initial_params_mcmc(initial_params::NamedTuple{N, T}, fgm::FiniteGaussianMixture) where {N, T}
    (:μ in N && :σ2 in N && :w in N) || throw(ArgumentError("Expected a NamedTuple with fields μ, σ2 and w"))
    (; μ, σ2, w) = initial_params
    all(σ2 .> 0) || throw(ArgumentError("Initial σ2 vector contains negative values."))
    (all(w .≥ 0) && isapprox(sum(w), 1)) || throw(ArgumentError("Initial w vector does not belong to the K-simplex."))
    (length(μ) == length(σ2) == length(w) == fgm.K) || throw(ArgumentError("Initial μ, σ2 or w dimensions are incompatible with the mixture model dimension."))
end

function _sample_posterior(rng::AbstractRNG, fgm::FiniteGaussianMixture{T}, initial_params::NamedTuple, n_samples::Int, n_burnin::Int) where {T<:Real}
    # Unpack parameters, data
    (; data, K, prior_strength, prior_location, prior_variance, prior_shape, hyperprior_rate, hyperprior_shape) = fgm
    (; x, n) = data
    (; μ, σ2, w) = initial_params
    
    cluster_alloc = Vector{Int}(undef, n)
    logprobs = Vector{T}(undef, K)
    samples = Vector{NamedTuple{(:μ, :σ2, :w, :β), Tuple{Vector{T}, Vector{T}, Vector{T}, T}}}(undef, n_samples)

    for m in 1:n_samples
        # Sample from p(cluster_alloc|⋯)
        log_w = log.(w)
        for i in eachindex(x)
            for k in 1:K
                logprobs[k] = log_w[k] + logpdf(Normal(μ[k], sqrt(σ2[k])), x[i])
            end
            probs = softmax(logprobs)
            cluster_alloc[i] = wsample(rng, 1:K, probs)
        end

        # Sample from p(β|⋯)
        β = rand(rng, Gamma(hyperprior_shape + K*prior_shape, inv(hyperprior_rate + sum(inv, σ2))))

        # Sample from p(w|⋯)
        cluster_counts = StatsBase.counts(cluster_alloc, K)
        w = rand(rng, Dirichlet(prior_strength .+ cluster_counts))
        
        # Compute sufficient statistics for non-empty components
        cluster_sum = zeros(T, K)
        cluster_sumsq = zeros(T, K)
        for i in eachindex(x)
            k_ind = cluster_alloc[i]
            cluster_sum[k_ind] += x[i]
            cluster_sumsq[k_ind] += x[i]^2
        end        

        # Sample from p(μ|⋯)
        for k in 1:K
            variance_k = inv(1/prior_variance + cluster_counts[k]/σ2[k])
            mean_k = variance_k * (prior_location / prior_variance + cluster_sum[k]/σ2[k])
            μ[k] = rand(rng, Normal(mean_k, sqrt(variance_k)))
        end

        # Sample from p(σ2|⋯)
        for k in 1:K
            rss_k = (cluster_sumsq[k] - 2*μ[k]*cluster_sum[k] + cluster_counts[k] * μ[k]^2)
            σ2[k] = rand(rng, InverseGamma(prior_shape + T(0.5)*cluster_counts[k], β + rss_k/2))
        end

        # Store the samples
        samples[m] = (μ = copy(μ), σ2 = copy(σ2), w = w, β = β)
    end
    return PosteriorSamples{T}(samples, fgm, n_samples, n_burnin)
end