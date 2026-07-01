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
* `initial_params`: Initial values used in the MCMC algorithm. Should be supplied as a `NamedTuple` with a single field `K`, where `K` is a positive integer. Defaults to the integer nearest to `0.5*sqrt(n)` which has positive prior probability, where `n` is the sample size.

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

# Precompute the Bernstein basis functions evaluated at the (fixed) data points for every
# candidate number of components `K`. `basis_pdf[K]` and `basis_logpdf[K]` are `K × n` matrices
# with entry `[k, i]` equal to `pdf`/`logpdf` of `Beta(k, K - k + 1)` at `x_trans[i]`.
# Since `x_trans` never changes during sampling, these values are constant and are otherwise
# recomputed (via fresh `Beta` constructions) in the innermost MCMC loops.
#
# The tables are returned as `Vector`s indexed directly by `K` (length `maximum(K_support)`);
# entries for `K` outside the support are left undefined and never accessed. This avoids the
# per-access hashing of a `Dict` in the hot `p(K|y)` loop.
function _precompute_basis(x_trans::AbstractVector{T}, K_support) where {T<:Real}
    n = length(x_trans)
    Kmax = maximum(K_support)
    basis_pdf = Vector{Matrix{T}}(undef, Kmax)
    basis_logpdf = Vector{Matrix{T}}(undef, Kmax)
    for K in K_support
        Bp = Matrix{T}(undef, K, n)
        Bl = Matrix{T}(undef, K, n)
        for k in 1:K
            d = Beta(k, K - k + 1)
            for i in 1:n
                Bp[k, i] = pdf(d, x_trans[i])
                Bl[k, i] = logpdf(d, x_trans[i])
            end
        end
        basis_pdf[K] = Bp
        basis_logpdf[K] = Bl
    end
    return basis_pdf, basis_logpdf
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

    # Precompute the (log-)basis functions at the data points for each candidate K. These are
    # independent of the sampler state, so they are computed once instead of inside the loops.
    basis_pdf, basis_logpdf = _precompute_basis(x_trans, K_support)

    # Initial number of components
    Kmax = maximum(K_support)
    logprobs_K = Vector{T}(undef, length(K_support))
    choice_w = Vector{T}(undef, Kmax + 1)    # p(y_i|…) weights: tie-to-bin 1:K, new component at K+1
    members = [Int[] for _ in 1:Kmax]        # members[ℓ] = observation indices currently in bin ℓ
    pos = Vector{Int}(undef, n)              # pos[j] = index of j within members[cluster_alloc[j]]
    cluster_scratch = Vector{Int}(undef, n)  # scratch buffer for the p(K|y) step
    y = copy(x_trans)
    cluster_alloc = bin_regular_ind(y, zero(T), one(T), K)

    # mcmc
    samples = Vector{NamedTuple{(:w,), Tuple{Vector{T}}}}(undef, n_samples)
    for m in 1:n_samples
        # Sample from p(y|K). The tie weight of point i to observation j depends on j only through
        # its bin, so observations are grouped by bin: the categorical over {tie to any member of
        # bin ℓ} ∪ {open a new component} has K+1 outcomes instead of n, turning the O(n²) sweep into
        # O(nK). Per-bin membership lists are maintained incrementally (swap-remove), so drawing a
        # uniform member of a bin is O(1).
        Bpdf = basis_pdf[K]
        for ℓ in 1:K
            empty!(members[ℓ])
        end
        for j in 1:n
            push!(members[cluster_alloc[j]], j)
            pos[j] = length(members[cluster_alloc[j]])
        end
        for i in 1:n
            # Column i holds the basis β(x_i; k, K-k+1) for k = 1:K (contiguous, cache-resident).
            col = @view Bpdf[:, i]
            ci = cluster_alloc[i]
            @inbounds for ℓ in 1:K
                cnt = length(members[ℓ]) - (ℓ == ci)   # members of bin ℓ excluding i itself
                choice_w[ℓ] = cnt * col[ℓ]
            end
            choice_w[K+1] = sum(col) * prior_strength / K
            c = wsample(rng, 1:(K+1), @view choice_w[1:(K+1)])
            if c == K + 1 # open a new component
                b = wsample(rng, 1:K, col)
                y[i] = rand(rng, Uniform((b-1)/K, b/K))
            else # tie to a uniformly chosen member of bin c (excluding i)
                Lc = length(members[c])
                if c == ci
                    r = rand(rng, 1:(Lc - 1))
                    jsel = r == pos[i] ? members[c][Lc] : members[c][r]
                else
                    jsel = members[c][rand(rng, 1:Lc)]
                end
                y[i] = y[jsel]
                b = c
            end
            if b != ci # i changed bins: swap-remove from bin ci, append to bin b
                v = members[ci]; p = pos[i]; lastj = v[end]
                v[p] = lastj; pos[lastj] = p; pop!(v)
                push!(members[b], i); pos[i] = length(members[b])
                cluster_alloc[i] = b
            end
        end

        # Sample from p(K|y)
        for j in eachindex(K_support)
            Kc = K_support[j]
            bin_regular_ind!(cluster_scratch, y, zero(T), one(T), Kc)
            Blog = basis_logpdf[Kc]
            acc = logpdf(prior_components, Kc)
            for i in eachindex(cluster_scratch)
                acc += Blog[cluster_scratch[i], i]
            end
            logprobs_K[j] = acc
        end
        K = wsample(rng, K_support, softmax(logprobs_K))
        bin_regular_ind!(cluster_alloc, y, zero(T), one(T), K)

        # Sample from p(w|K, y)
        cluster_counts = StatsBase.counts(cluster_alloc, K)
        w = rand(rng, Dirichlet(prior_strength/K .+ cluster_counts))

        # Store the samples
        samples[m] = (w = w,)
    end
    return PosteriorSamples{T}(samples, rbp, n_samples, n_burnin)
end