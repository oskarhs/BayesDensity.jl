"""
    sample(
        [rng::Random.AbstractRNG],
        rbp::RandomBernsteinPoly{T},
        n_samples::Int;
        n_burnin::Int              = min(1000, div(n_samples, 5)),
        initial_params::NamedTuple = _get_default_initparams_mcmc(rbp)
    ) where {T} -> PosteriorSamples{T}

Generate `n_samples` posterior samples from a `RandomBernsteinPoly` using the telescope sampler.

# Arguments
* `rng`: Optional random seed used for random variate generation.
* `rbp`: The `RandomBernsteinPoly` object for which posterior samples are generated.
* `n_samples`: The total number of samples (including burn-in).

# Keyword arguments
* `n_burnin`: Number of burn-in samples.
* `initial_params`: Initial values used in the MCMC algorithm. Should be supplied as a `NamedTuple` with a single field `K`, where `K` is a positive integer. Defaults to

# Returns
* `ps`: A [`PosteriorSamples`](@ref) object holding the posterior samples and the original model object.

# Examples
```julia-repl
julia> using Random

julia> x = (1.0 .- (1.0 .- LinRange(0.0, 1.0, 5000)) .^(1/3)).^(1/3);

julia> rbp = RandomBernsteinPoly(x);

julia> ps1 = sample(rbp, 5_000);

julia> ps2 = sample(rbp, 5_000; n_burnin=2_000, initial_params = (K = 5,));
```
"""
function StatsBase.sample(
    rng::AbstractRNG,
    rbp::RandomBernsteinPoly,
    n_samples::Int;
    n_burnin::Int = min(div(n_samples, 5), 1000),
    initial_params::NamedTuple=_get_default_initial_params_mcmc(rbp)
)
    (1 ≤ n_samples ≤ Inf) || throw(ArgumentError("Number of samples must be a positive integer."))
    (0 ≤ n_burnin ≤ Inf) || throw(ArgumentError("Number of burn-in samples must be a nonnegative integer."))
    n_samples ≥ n_burnin || @warn "Number of total samples is smaller than the number of burn-in samples."
    _check_initial_params_mcmc(initial_params, rbp)
    return _sample_posterior(rng, rbp, initial_params, n_samples, n_burnin)
end

function _get_default_initial_params_mcmc(rbp::RandomBernsteinPoly{T}) where {T}
    (; x, n) = rbp.data
    # Initialize by choosing the value of K in the support closest to 0.5√n
    K_support = support(rbp.prior_components)
    K = K_support[argmin(abs.(K_support .- 0.5*sqrt(n)))]
    return (K = K,)
end

function _check_initial_params_mcmc(initial_params::NamedTuple{N, T}, rbp::RandomBernsteinPoly) where {N, T}
    (:K in N) || throw(ArgumentError("Expected a NamedTuple with fields `:K`"))
    (; K) = initial_params
    (K .≥ 0) || throw(ArgumentError("Initial number of basis functions `K` must be positive."))
    (pdf(rbp.prior_components, K) > 0) || throw(ArgumentError("Initial number of mixture components has probability 0.")) 
end

function _sample_posterior(
    rng::AbstractRNG,
    rbp::RandomBernsteinPoly{T},
    initial_params::NamedTuple,
    n_samples::Int,
    n_burnin::Int,
) where {T<:Real}
    # Unpack model, data
    (; data, prior_components, prior_strength, bounds) = rbp
    (; x, n, x_trans) = data

    # Initial parameters
    (; K) = initial_params
    K_support = support(prior_components)

    # Initial number of components
    logprobs_K = Vector{T}(undef, length(K_support))
    probs_y = Vector{T}(undef, n)
    y = copy(x_trans)
    cluster_alloc = bin_regular_ind(y, zero(T), one(T), K)

    # mcmc
    samples = Vector{NamedTuple{(:w,), Tuple{Vector{T}}}}(undef, n_samples)
    for m in 1:n_samples
        # Sample from p(y|K)
        for i in eachindex(y)
            probs_y[i] = zero(T)
            for k in 1:K
                probs_y[i] += pdf(Beta(k, K - k + 1), x_trans[i])
            end
            probs_y[i] *= prior_strength / K
            for j in eachindex(setdiff(1:n, i))
                probs_y[j] = pdf(Beta(cluster_alloc[j], K-cluster_alloc[j]+1), x_trans[i])
            end
            j = wsample(rng, 1:n, probs_y / sum(probs_y))
            if j == i # sample new component
                new_bin_prob = Vector{T}(undef, K)
                for k in 1:K
                    new_bin_prob[k] = pdf(Beta(k, K-k+1), x_trans[i])
                end
                k = wsample(rng, 1:K, new_bin_prob / sum(new_bin_prob))
                y[i] = rand(rng, Uniform((k-1)/K, k/K))
                cluster_alloc[i] = k
            else # tie
                y[i] = y[j]
                cluster_alloc[i] = cluster_alloc[j]
            end
        end

        # Sample from p(K|y)
        for j in eachindex(K_support)
            K = K_support[j]
            cluster_alloc = bin_regular_ind(y, zero(T), one(T), K)
            logprobs_K[j] = logpdf(prior_components, K)
            for i in eachindex(cluster_alloc)
                logprobs_K[j] += logpdf(Beta(cluster_alloc[i], K - cluster_alloc[i]+1), x_trans[i])
            end
        end
        K = wsample(rng, K_support, softmax(logprobs_K))
        cluster_alloc = bin_regular_ind(y, zero(T), one(T), K)

        # Sample from p(w|K, y)
        cluster_counts = StatsBase.counts(cluster_alloc, K)
        w = rand(rng, Dirichlet(prior_strength .+ cluster_counts))

        # Store the samples
        samples[m] = (w = w,)
    end
    return PosteriorSamples{T}(samples, rbp, n_samples, n_burnin)
end