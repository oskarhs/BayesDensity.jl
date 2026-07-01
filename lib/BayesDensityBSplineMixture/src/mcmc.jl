"""
    sample(
        [rng::Random.AbstractRNG],
        bsm::BSplineMixture{T},
        n_samples::Int;
        n_burnin::Int              = min(1000, div(n_samples, 5)),
        initial_params::NamedTuple = get_default_initparams_mcmc(bsm)
    ) where {T} -> PosteriorSamples{T}

Generate `n_samples` posterior samples from a `BSplineMixture` using an augmented Gibbs sampler.

# Arguments
* `rng`: Optional random seed used for random variate generation.
* `bsm`: The `BSplineMixture` object for which posterior samples are generated.
* `n_samples`: The total number of samples (including burn-in).

# Keyword arguments
* `n_burnin`: Number of burn-in samples.
* `initial_params`: Initial values used in the MCMC algorithm. Should be supplied as a `NamedTuple` with fields `:β` and `:τ2`, where `:β` is a `K-1`-dimensional vector and `τ2` is a positive scalar.

# Returns
* `ps`: A [`PosteriorSamples`](@ref) object holding the posterior samples and the original model object.

# Examples
```julia-repl
julia> using Random

julia> x = (1.0 .- (1.0 .- LinRange(0.0, 1.0, 5000)) .^(1/3)).^(1/3);

julia> bsm = BSplineMixture(x);

julia> ps = sample(Xoshiro(1), bsm, 5000);
```
"""
function StatsBase.sample(
    rng::AbstractRNG,
    bsm::BSplineMixture,
    n_samples::Int;
    n_burnin::Int = min(1000, div(n_samples, 5)),
    initial_params::NamedTuple=_get_default_initparams_mcmc(bsm)
)
    (1 ≤ n_samples ≤ Inf) || throw(ArgumentError("Number of samples must be a positive integer."))
    (0 ≤ n_burnin ≤ Inf) || throw(ArgumentError("Number of burn-in samples must be a nonnegative integer."))
    (n_samples ≥ n_burnin) || @warn "The total number of samples is smaller than the number of burn-in samples."
    _check_initparams(bsm, initial_params)
    return _sample_posterior(rng, bsm, initial_params, n_samples, n_burnin)
end

function _check_initparams(bsm::BSplineMixture, initial_params::NamedTuple{N, V}) where {N, V}
    (:β in N && :τ2 in N) || throw(ArgumentError("Expected a NamedTuple with fields β and τ2"))
    K = length(BSplineKit.basis(bsm))
    (; β, τ2) = initial_params

    (β isa AbstractVector && length(β) == K-1) || throw(ArgumentError("Dimension of supplied initial β does not match that of the spline basis."))
    (τ2 isa Real && τ2 > 0) || throw(ArgumentError("Supplied value of τ2 must be positive."))
end

# Lazy initialization
function _get_default_initparams_mcmc(bsm::BSplineMixture{T}) where {T}
    β = copy(bsm.data.μ)
    τ2 = one(T)                # Global smoothing parameter
    return (β = β, τ2 = τ2)
end

# --- Performance helpers for the augmented Gibbs sampler ---

# In-place numerically stable softmax over the first `L` entries: dst[1:L] .= softmax(src[1:L]).
function _softmax!(dst::AbstractVector{T}, src::AbstractVector{T}, L::Int) where {T<:Real}
    m = src[1]
    @inbounds for l in 2:L
        src[l] > m && (m = src[l])
    end
    s = zero(T)
    @inbounds for l in 1:L
        e = exp(src[l] - m)
        dst[l] = e
        s += e
    end
    @inbounds for l in 1:L
        dst[l] /= s
    end
    return dst
end

# Draw counts ~ Multinomial(m, probs[1:L]) into counts[1:L] by direct categorical counting.
# Allocation-free and avoids constructing a `Multinomial`; efficient for the small L (≤5) here.
function _multinomial_counts!(rng::AbstractRNG, counts::AbstractVector{<:Integer}, m::Integer, probs::AbstractVector{T}, L::Int) where {T<:Real}
    @inbounds for l in 1:L
        counts[l] = 0
    end
    @inbounds for _ in 1:m
        u = rand(rng, T)
        c = 1
        acc = probs[1]
        while c < L && u > acc
            c += 1
            acc += probs[c]
        end
        counts[c] += 1
    end
    return counts
end

# Sample β ~ N(J⁻¹h, J⁻¹) from the canonical (precision) parametrization, exploiting the banded
# (pentadiagonal) structure of the precision `J` via a banded Cholesky.
# Removes some overhead from the generic `rand(MvNormalCanon(...))` approach.
function _sample_canon_banded(rng::AbstractRNG, h::AbstractVector{T}, J::AbstractMatrix{T}) where {T<:Real}
    C = cholesky(Symmetric(J))
    z = randn(rng, length(h))
    return (C \ h) .+ (C.U \ z)   # mean J⁻¹h, and Cov = (UᵀU)⁻¹ = J⁻¹
end

# To do: make a multithreaded version (also one for unbinned data)
function _sample_posterior(rng::AbstractRNG, bsm::BSplineMixture{T, A, NamedTuple{(:x, :log_B, :b_ind, :bincounts, :μ, :P, :n), Vals}}, initial_params::NamedTuple, n_samples::Int, n_burnin::Int) where {T, A, Vals}
    basis = BSplineKit.basis(bsm)
    K = length(basis)
    (; log_B, b_ind, bincounts, μ, P, n) = bsm.data
    n_bins = length(bincounts)

    # Prior Hyperparameters
    (; prior_global_shape, prior_global_rate, prior_local_shape, prior_local_rate, prior_stdev) = hyperparams(bsm)
    Q0 = Diagonal(vcat(fill(1/prior_stdev^2, 2), zeros(T, K-3)))

    # Initial parameters
    (; β, τ2) = initial_params

    # Initialize other params
    δ2 = Vector{T}(undef, K-3)
    ω = Vector{T}(undef, K-1)

    n_overlap = size(log_B, 2)

    # Preallocated buffers reused across sweeps
    logprobs = Vector{T}(undef, n_overlap)
    probs    = Vector{T}(undef, n_overlap)
    counts   = Vector{Int}(undef, n_overlap)
    N        = Vector{Int}(undef, K)
    S        = Vector{Int}(undef, K-1)

    #θ = Vector{T}(undef, K) # Mixture probabilities
    θ = max.(eps(), logistic_stickbreaking(β))
    θ = θ / sum(θ)
    log_θ = log.(θ)

    # Get normalization factor
    norm_fac = compute_norm_fac(basis, T)

    # Initialize vector of samples
    samples = Vector{NamedTuple{(:spline_coefs, :β, :τ2, :δ2), Tuple{Vector{T}, Vector{T}, T, Vector{T}}}}(undef, n_samples)

    for m in 1:n_samples
        # Second differences of (β - μ): Δ[k] = (P(β-μ))[k]; shared by the δ² and τ² updates.
        Δ = P * (β .- μ)

        # Update δ2
        for k in 1:K-3
            a_δ_k_new = prior_local_shape + T(0.5)
            b_δ_k_new = prior_local_rate + T(0.5) * abs2(Δ[k]) / τ2
            δ2[k] = rand(rng, InverseGamma(a_δ_k_new, b_δ_k_new))
        end

        # Update τ2
        a_τ_new = prior_global_shape + T(0.5) * (K - 1)
        b_τ_new = prior_global_rate + sum(abs2, view(β, 1:2) - view(μ, 1:2)) / (2*prior_stdev^2)
        for k in 1:K-3
            b_τ_new += T(0.5) * abs2(Δ[k]) / δ2[k]
        end
        τ2 = rand(rng, InverseGamma(a_τ_new, b_τ_new))

        # Update z (accumulate class-label counts N)
        fill!(N, 0)
        for i in 1:n_bins
            k0 = b_ind[i]
            for l in 1:n_overlap
                logprobs[l] = log_B[i,l] + log_θ[k0 + l - 1]
            end
            _softmax!(probs, logprobs, n_overlap)
            _multinomial_counts!(rng, counts, bincounts[i], probs, n_overlap)
            @inbounds for l in 1:n_overlap
                N[k0 + l - 1] += counts[l]
            end
        end

        # Update ω. S[k] = n - Σ_{j<k} N[j] (no clamp, matching the original binned-x variant).
        acc = 0
        for k in 1:K-1
            S[k] = n - acc
            acc += N[k]
            ω[k] = rand(rng, PolyaGammaHybridSampler(S[k], β[k]))
        end

        # Update β (canonical Gaussian; pentadiagonal precision → banded Cholesky draw)
        D = Diagonal(1 ./ (τ2 .* δ2))
        Q = transpose(P) * D * P + (Q0 / τ2)
        inv_Σ_new = Diagonal(ω) + Q                       # Q + Ω retains banded structure
        canon_mean_new = Q * μ .+ (view(N, 1:K-1) .- S ./ 2)
        β = _sample_canon_banded(rng, canon_mean_new, inv_Σ_new)

        # Record θ
        θ = max.(eps(), logistic_stickbreaking(β))
        θ = θ / sum(θ)
        log_θ = log.(θ)

        # Compute coefficients in terms of unnormalized B-spline basis
        spline_coefs = θ .* norm_fac
        samples[m] = (spline_coefs = spline_coefs, β = β, τ2 = τ2, δ2 = δ2)
    end
    return PosteriorSamples{T}(samples, bsm, n_samples, n_burnin)
end

function _sample_posterior(rng::AbstractRNG, bsm::BSplineMixture{T, A, NamedTuple{(:hist, :log_B, :b_ind, :bincounts, :μ, :P, :n), Vals}}, initial_params::NamedTuple, n_samples::Int, n_burnin::Int) where {T, A, Vals}
    basis = BSplineKit.basis(bsm)
    K = length(basis)
    (; log_B, b_ind, bincounts, μ, P, n) = bsm.data
    n_bins = length(bincounts)

    # Prior Hyperparameters
    (; prior_global_shape, prior_global_rate, prior_local_shape, prior_local_rate, prior_stdev) = hyperparams(bsm)
    Q0 = Diagonal(vcat(fill(1/prior_stdev^2, 2), zeros(T, K-3)))

    # Initial parameters
    (; β, τ2) = initial_params

    # Initialize other params
    δ2 = Vector{T}(undef, K-3)
    ω = Vector{T}(undef, K-1)

    n_overlap = size(log_B, 2)

    # Preallocated buffers reused across sweeps
    logprobs = Vector{T}(undef, n_overlap)
    probs    = Vector{T}(undef, n_overlap)
    counts   = Vector{Int}(undef, n_overlap)
    N        = Vector{Int}(undef, K)
    S        = Vector{Int}(undef, K-1)

    #θ = Vector{T}(undef, K) # Mixture probabilities
    θ = max.(eps(), logistic_stickbreaking(β))
    θ = θ / sum(θ)
    log_θ = log.(θ)

    # Get normalization factor
    norm_fac = compute_norm_fac(basis, T)

    # Initialize vector of samples
    samples = Vector{NamedTuple{(:spline_coefs, :β, :τ2, :δ2), Tuple{Vector{T}, Vector{T}, T, Vector{T}}}}(undef, n_samples)

    for m in 1:n_samples
        # Second differences of (β - μ): Δ[k] = (P(β-μ))[k]; shared by the δ² and τ² updates.
        Δ = P * (β .- μ)

        # Update δ2
        for k in 1:K-3
            a_δ_k_new = prior_local_shape + T(0.5)
            b_δ_k_new = prior_local_rate + T(0.5) * abs2(Δ[k]) / τ2
            δ2[k] = rand(rng, InverseGamma(a_δ_k_new, b_δ_k_new))
        end

        # Update τ2
        a_τ_new = prior_global_shape + T(0.5) * (K - 1)
        b_τ_new = prior_global_rate + sum(abs2, view(β, 1:2) - view(μ, 1:2)) / (2*prior_stdev^2)
        for k in 1:K-3
            b_τ_new += T(0.5) * abs2(Δ[k]) / δ2[k]
        end
        τ2 = rand(rng, InverseGamma(a_τ_new, b_τ_new))

        # Update z (accumulate class-label counts N)
        fill!(N, 0)
        for i in 1:n_bins
            ks = b_ind[i]
            w = ks[2] - ks[1] + 1
            for l in 1:w
                logprobs[l] = log_B[i,l] + log_θ[ks[1] + l - 1]
            end
            _softmax!(probs, logprobs, w)
            _multinomial_counts!(rng, counts, bincounts[i], probs, w)
            @inbounds for l in 1:w
                N[ks[1] + l - 1] += counts[l]
            end
        end

        # Update ω. S[k] = max(n - Σ_{j<k} N[j], 0).
        acc = 0
        for k in 1:K-1
            S[k] = max(n - acc, 0)
            acc += N[k]
            ω[k] = rand(rng, PolyaGammaHybridSampler(S[k], β[k]))
        end

        # Update β (canonical Gaussian; pentadiagonal precision → banded Cholesky draw)
        D = Diagonal(1 ./ (τ2 .* δ2))
        Q = transpose(P) * D * P + (Q0 / τ2)
        inv_Σ_new = Diagonal(ω) + Q                       # Q + Ω retains banded structure
        canon_mean_new = Q * μ .+ (view(N, 1:K-1) .- S ./ 2)
        β = _sample_canon_banded(rng, canon_mean_new, inv_Σ_new)

        # Record θ
        θ = max.(eps(), logistic_stickbreaking(β))
        θ = θ / sum(θ)
        log_θ = log.(θ)

        # Compute coefficients in terms of unnormalized B-spline basis
        spline_coefs = θ .* norm_fac
        samples[m] = (spline_coefs = spline_coefs, β = β, τ2 = τ2, δ2 = δ2)
    end
    return PosteriorSamples{T}(samples, bsm, n_samples, n_burnin)
end

function _sample_posterior(rng::AbstractRNG, bsm::BSplineMixture{T, A, NamedTuple{(:x, :log_B, :b_ind, :μ, :P, :n), Vals}}, initial_params::NamedTuple, n_samples::Int, n_burnin::Int) where {T, A, Vals}
    basis = BSplineKit.basis(bsm)
    K = length(basis)
    (; log_B, b_ind, μ, P, n) = bsm.data

    # Prior Hyperparameters
    (; prior_global_shape, prior_global_rate, prior_local_shape, prior_local_rate, prior_stdev) = hyperparams(bsm)
    Q0 = Diagonal(vcat(fill(1/prior_stdev^2, 2), zeros(T, K-3)))
    
    # Initial parameters
    (; β, τ2) = initial_params

    # Initialize other params
    δ2 = Vector{T}(undef, K-3)
    ω = Vector{T}(undef, K-1)

    n_overlap = size(log_B, 2)

    # Preallocated buffers reused across sweeps
    logprobs = Vector{T}(undef, n_overlap)
    probs    = Vector{T}(undef, n_overlap)
    counts   = Vector{Int}(undef, n_overlap)
    N        = Vector{Int}(undef, K)
    S        = Vector{Int}(undef, K-1)

    #θ = Vector{T}(undef, K) # Mixture probabilities
    θ = max.(eps(), logistic_stickbreaking(β))
    θ = θ / sum(θ)
    log_θ = log.(θ)

    # Get normalization factor
    norm_fac = compute_norm_fac(basis, T)

    # Initialize vector of samples
    samples = Vector{NamedTuple{(:spline_coefs, :β, :τ2, :δ2), Tuple{Vector{T}, Vector{T}, T, Vector{T}}}}(undef, n_samples)
    spline_coefs = theta_to_coef(θ, basis)
    samples[1] = (spline_coefs = spline_coefs, β = β, τ2 = τ2, δ2 = δ2)

    for m in 2:n_samples
        # Second differences of (β - μ): Δ[k] = (P(β-μ))[k]; shared by the δ² and τ² updates.
        Δ = P * (β .- μ)

        # Update δ2
        for k in 1:K-3
            a_δ_k_new = prior_local_shape + T(0.5)
            b_δ_k_new = prior_local_rate + T(0.5) * abs2(Δ[k]) / τ2
            δ2[k] = rand(rng, InverseGamma(a_δ_k_new, b_δ_k_new))
        end

        # Update τ2
        a_τ_new = prior_global_shape + T(0.5) * (K - 1)
        b_τ_new = prior_global_rate + sum(abs2, view(β, 1:2) - view(μ, 1:2)) / (2*prior_stdev^2)
        for k in 1:K-3
            b_τ_new += T(0.5) * abs2(Δ[k]) / δ2[k]
        end
        τ2 = rand(rng, InverseGamma(a_τ_new, b_τ_new))

        # Update z (accumulate class-label counts N)
        fill!(N, 0)
        for i in 1:n
            k0 = b_ind[i]
            for l in 1:n_overlap
                logprobs[l] = log_B[i,l] + log_θ[k0 + l - 1]
            end
            _softmax!(probs, logprobs, n_overlap)
            _multinomial_counts!(rng, counts, 1, probs, n_overlap)
            @inbounds for l in 1:n_overlap
                N[k0 + l - 1] += counts[l]
            end
        end

        # Update ω. S[k] = max(n - Σ_{j<k} N[j], 0).
        acc = 0
        for k in 1:K-1
            S[k] = max(n - acc, 0)
            acc += N[k]
            ω[k] = rand(rng, PolyaGammaHybridSampler(S[k], β[k]))
        end

        # Update β (canonical Gaussian; pentadiagonal precision → banded Cholesky draw)
        D = Diagonal(1 ./ (τ2 .* δ2))
        Q = transpose(P) * D * P + (Q0 / τ2)
        inv_Σ_new = Diagonal(ω) + Q                       # Q + Ω retains banded structure
        canon_mean_new = Q * μ .+ (view(N, 1:K-1) .- S ./ 2)
        β = _sample_canon_banded(rng, canon_mean_new, inv_Σ_new)

        # Record θ
        θ = max.(eps(), logistic_stickbreaking(β))
        θ = θ / sum(θ)
        log_θ = log.(θ)

        # Compute coefficients in terms of unnormalized B-spline basis
        spline_coefs = θ .* norm_fac
        samples[m] = (spline_coefs = spline_coefs, β = β, τ2 = τ2, δ2 = δ2)
    end
    return PosteriorSamples{T}(samples, bsm, n_samples, n_burnin)
end