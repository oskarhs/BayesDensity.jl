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
    initial_params::NamedTuple=get_default_initparams_mcmc(bsm)
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
function get_default_initparams_mcmc(bsm::BSplineMixture{T}) where {T}
    β = copy(bsm.data.μ)
    τ2 = one(T)                # Global smoothing parameter
    return (β = β, τ2 = τ2)
end

# To do: make a multithreaded version (also one for unbinned data)
function _sample_posterior(rng::AbstractRNG, bsm::BSplineMixture{T, A, NamedTuple{(:x, :log_B, :b_ind, :bincounts, :μ, :P, :n), Vals}}, initial_params::NamedTuple, n_samples::Int, n_burnin::Int) where {T, A, Vals}
    basis = BSplineKit.basis(bsm)
    K = length(basis)
    (; x, log_B, b_ind, bincounts, μ, P, n) = bsm.data
    n_bins = length(bincounts)

    # Prior Hyperparameters
    (; prior_global_shape, prior_global_rate, prior_local_shape, prior_local_rate, prior_stdev) = hyperparams(bsm)
    Q0 = Diagonal(vcat([1/prior_stdev^2, 1/prior_stdev^2], zeros(T, K-3)))

    # Initial parameters
    (; β, τ2) = initial_params

    # Initialize other params
    δ2 = Vector{T}(undef, K-3)
    ω = Vector{T}(undef, K-1)
    
    logprobs = Vector{T}(undef, 4)  # class label probabilities

    #θ = Vector{T}(undef, K) # Mixture probabilities
    θ = max.(eps(), logistic_stickbreaking(β))
    θ = θ / sum(θ)
    log_θ = log.(θ)

    # Get normalization factor
    norm_fac = compute_norm_fac(basis, T)
    
    # Initialize vector of samples
    samples = Vector{NamedTuple{(:spline_coefs, :β, :τ2, :δ2), Tuple{Vector{T}, Vector{T}, T, Vector{T}}}}(undef, n_samples)

    for m in 1:n_samples

        # Update δ2: (some inefficiencies here, but okay for now)
        for k in 1:K-3
            a_δ_k_new = prior_local_shape + T(0.5)
            b_δ_k_new = prior_local_rate + T(0.5) * abs2( β[k+2] -  μ[k+2] - ( 2*(β[k+1] - μ[k+1]) - (β[k] - μ[k]) )) / τ2
            δ2[k] = rand(rng, InverseGamma(a_δ_k_new, b_δ_k_new))
        end

        # Update τ2
        a_τ_new = prior_global_shape + T(0.5) * (K - 3)
        b_τ_new = prior_global_rate
        for k in 1:K-3
            b_τ_new += T(0.5) * abs2( β[k+2] -  μ[k+2] - ( 2*(β[k+1] - μ[k+1]) - (β[k] - μ[k]) )) / δ2[k]
        end
        τ2 = rand(rng, InverseGamma(a_τ_new, b_τ_new))

        # Update z (N and S)
        N = zeros(Int, K)               # class label counts (of z[i]'s)
        for i in 1:n_bins
            # Compute the four nonzero probabilities:
            k0 = b_ind[i]
            for l in 1:4
                k = k0 + l - 1
                logprobs[l] = log_B[i,l] + log_θ[k] 
            end
            probs = softmax(logprobs)
            counts = rand(rng, Multinomial(bincounts[i], probs))
            N[k0:k0+3] .+= counts
        end
        # Update ω
        # Compute N and S
        S = n .- cumsum(vcat(0, N[1:K-1]))
        for k in 1:K-1
            c_k_new = S[k]
            d_k_new = β[k]
            ω[k] = rand(rng, PolyaGammaHybridSampler(c_k_new, d_k_new))
        end

        # Update β
        # Compute the Q matrix
        D = Diagonal(1 ./(τ2*δ2))
        Q = transpose(P) * D * P + Q0
        # Compute the Ω matrix (Note: Q + D retains the banded structure!)
        Ω = Diagonal(ω)
        inv_Σ_new = Ω + Q
        # Compute inv(Σ_new) * μ_new
        canon_mean_new = Q * μ + (N[1:K-1] - S[1:K-1]/2)
        # Sample β
        β = rand(rng, MvNormalCanon(canon_mean_new, inv_Σ_new))

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
    (; x, log_B, b_ind, μ, P, n) = bsm.data

    # Prior Hyperparameters
    (; prior_global_shape, prior_global_rate, prior_local_shape, prior_local_rate, prior_stdev) = hyperparams(bsm)
    Q0 = Diagonal(vcat([1/prior_stdev^2, 1/prior_stdev^2], zeros(T, K-3)))
    
    # Initial parameters
    (; β, τ2) = initial_params

    # Initialize other params
    δ2 = Vector{T}(undef, K-3)
    ω = Vector{T}(undef, K-1)

    logprobs = Vector{T}(undef, 4)  # class label probabilities

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

        # Update δ2: (some inefficiencies here, but okay for now)
        for k in 1:K-3
            a_δ_k_new = prior_local_shape + T(0.5)
            b_δ_k_new = prior_local_rate + T(0.5) * abs2( β[k+2] -  μ[k+2] - ( 2*(β[k+1] - μ[k+1]) - (β[k] - μ[k]) )) / τ2
            δ2[k] = rand(rng, InverseGamma(a_δ_k_new, b_δ_k_new))
        end

        # Update τ2
        a_τ_new = prior_global_shape + T(0.5) * (K - 3)
        b_τ_new = prior_global_rate
        for k in 1:K-3
            b_τ_new += T(0.5) * abs2( β[k+2] -  μ[k+2] - ( 2*(β[k+1] - μ[k+1]) - (β[k] - μ[k]) )) / δ2[k]
        end
        τ2 = rand(rng, InverseGamma(a_τ_new, b_τ_new))

        # Update z (N and S)
        N = zeros(Int, K)               # class label counts (of z[i]'s)
        for i in 1:n
            # Compute the four nonzero probabilities:
            k0 = b_ind[i]
            for l in 1:4
                k = k0 + l - 1
                logprobs[l] = log_B[i,l] + log_θ[k] 
            end
            probs = softmax(logprobs)
            counts = rand(rng, Multinomial(1, probs))
            N[k0:k0+3] .+= counts
        end
        # Update ω
        # Compute N and S
        S = n .- cumsum(vcat(0, N[1:K-1]))
        for k in 1:K-1
            c_k_new = S[k]
            d_k_new = β[k]
            ω[k] = rand(rng, PolyaGammaHybridSampler(c_k_new, d_k_new))
        end

        # Update β
        # Compute the Q matrix
        D = Diagonal(1 ./(τ2*δ2))
        Q = transpose(P) * D * P + Q0
        # Compute the Ω matrix (Note: Q + Ω retains the banded structure!)
        Ω = Diagonal(ω)
        inv_Σ_new = Ω + Q
        # Compute inv(Σ_new) * μ_new
        canon_mean_new = Q * μ + (N[1:K-1] - S[1:K-1]/2)
        # Sample β
        β = rand(rng, MvNormalCanon(canon_mean_new, inv_Σ_new))

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