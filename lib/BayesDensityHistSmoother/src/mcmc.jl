"""
    sample(
        [rng::Random.AbstractRNG],
        hs::HistSmoother{T},
        n_samples::Int;
        n_burnin::Int              = min(100, div(n_samples, 5)),
        initial_params::NamedTuple = get_default_initparams_mcmc(hs)
    ) where {T} -> PosteriorSamples{T}

Generate `n_samples` posterior samples from a `HistSmoother` using an augmented Gibbs sampler.

# Arguments
* `rng`: Optional random seed used for random variate generation.
* `hs`: The `HistSmoother` object for which posterior samples are generated.
* `n_samples`: The total number of samples (including burn-in).

# Keyword arguments
* `n_burnin`: Number of burn-in samples.
* `initial_params`: Initial values used in the MCMC algorithm. Should be supplied as a `NamedTuple` with fields `:β` and `:σ2`, where `:β` is a `K`-dimensional vector and `σ2` is a positive scalar.

# Returns
* `ps`: A [`PosteriorSamples`](@ref) object holding the posterior samples and the original model object.

# Examples
```julia-repl
julia> using Random

julia> x = (1.0 .- (1.0 .- LinRange(0.0, 1.0, 5000)) .^(1/3)).^(1/3);

julia> hs = HistSmoother(x);

julia> ps = sample(Xoshiro(1), hs, 1100);
```
"""
function StatsBase.sample(
    rng::AbstractRNG,
    shs::HistSmoother,
    n_samples::Int;
    n_burnin::Int = min(div(n_samples, 5), 100),
    initial_params::NamedTuple=get_default_initparams_mcmc(shs)
)
    (1 ≤ n_samples ≤ Inf) || throw(ArgumentError("Number of samples must be a positive integer."))
    (0 ≤ n_burnin ≤ Inf) || throw(ArgumentError("Number of burn-in samples must be a nonnegative integer."))
    n_samples ≥ n_burnin || @warn "Number of total samples is smaller than the number of burn-in samples."
    _check_initparams(shs, initial_params)
    return _sample_posterior(rng, shs, initial_params, n_samples, n_burnin)
end

function _check_initparams(shs::HistSmoother, initial_params::NamedTuple{N, V}) where {N, V}
    (:β in N && :σ2 in N) || throw(ArgumentError("Expected a NamedTuple with fields β and τ2"))
    K = length(shs.bs)
    (; β, σ2) = initial_params

    (β isa AbstractVector && length(β) == K) || throw(ArgumentError("Dimension of supplied initial β does not match that of the spline basis."))
    (σ2 isa Real && σ2 > 0) || throw(ArgumentError("Supplied value of τ2 must be positive."))
end

# Lazy initialization
function get_default_initparams_mcmc(shs::HistSmoother{T}) where {T}
    K = length(shs.bs)
    β = zeros(T, K)
    σ2 = shs.prior_scale_fixed^2
    return (β = β, σ2 = σ2)
end

function _sample_posterior(rng::AbstractRNG, shs::HistSmoother{T}, initial_params::NamedTuple, n_samples::Int, n_burnin::Int) where {T<:Real}
    # Unpack model:
    (; data, bs, prior_scale_fixed, prior_scale_random) = shs
    (; x, n, x_grid, N, C, LZ, bounds) = data
    prior_scaled_fixed2 = prior_scale_fixed^2

    n_bins = length(N)
    K = length(bs)

    # Initialize parameters:
    (; β, σ2) = initial_params

    CTN = transpose(C) * N
    Cβ_k = C * β # Equal to Cbeta

    samples_temp = Vector{NamedTuple{(:β, :σ2), Tuple{Vector{T}, T}}}(undef, n_samples)

    for m in 1:n_samples
        v = vcat([prior_scaled_fixed2, prior_scaled_fixed2], fill(σ2, K-2))
        
        # Sample beta (slice sampler) (consider trying ARS instead at a later point in time)
        for k in 1:K
            Cβ_k .-= β[k] * C[:, k]
            logdensity = β_k -> CTN[k] * β_k - β_k^2 / (2*v[k]) - sum(exp, β_k*C[:,k] + Cβ_k)
            β[k] = slice_sampling_univariate(rng, 1.0, logdensity, β[k])
            Cβ_k .+= β[k] * C[:, k] 
        end

        # Sample ξ
        a_ξ = T(1)
        b_ξ = 1/σ2 + 1/prior_scale_random
        ξ = rand(rng, InverseGamma(a_ξ, b_ξ))

        # Sample σ2
        a_σ = (K-1) / 2
        b_σ = sum(abs2, view(β, 3:K)) / 2 + 1 / ξ
        σ2 = rand(rng, InverseGamma(a_σ, b_σ))

        samples_temp[m] = (β = copy(β), σ2 = σ2) # We compute the normalization constants after completing the MCMC loop
    end

    # Compute normalization constants:
    eval_grid, val_cdf, l1_norm_vec = compute_norm_constants_cdf_grid(shs, samples_temp)
    samples = Vector{NamedTuple{(:β, :σ2, :norm, :eval_grid, :val_cdf), Tuple{Vector{T}, T, T, LinRange{T}, Vector{T}}}}(undef, n_samples)
    for m in 1:n_samples
        samples[m] = (β = samples_temp[m].β, σ2 = samples_temp[m].σ2,
                      norm = l1_norm_vec[m], eval_grid = eval_grid,
                      val_cdf = val_cdf[m])
    end
    return PosteriorSamples{T}(samples, shs, n_samples, n_burnin)
end

# Adopted from SliceSampling.jl under the MIT license (accessed on 23. Dec. 2025): https://github.com/TuringLang/SliceSampling.jl/blob/main/src/univariate/univariate.jl
function slice_sampling_univariate(
    rng::Random.AbstractRNG, w, logdensity, θ::F
) where {F<:Real}
    #w, max_prop = alg.window, alg.max_proposals
    max_prop = 10_000
    ℓπ          = logdensity(θ)
    ℓy          = ℓπ - Random.randexp(rng, F)
    L, R, props = find_interval(rng, logdensity, w, ℓy, θ)

    for _ in 1:max_prop
        U     = rand(rng, F)
        θ′    = L + U * (R - L)
        ℓπ′   = logdensity(θ′)
        props += 1
        if (ℓy < ℓπ′)
            return θ′
        end

        if θ′ < θ
            L = θ′
        else
            R = θ′
        end
    end
    return throw(ErrorException("Slice sampler exceeded maximum number of proposals"))
end

# Adopted from SliceSampling.jl under the MIT license (accessed on 23. Dec. 2025): https://github.com/TuringLang/SliceSampling.jl/blob/main/src/univariate/steppingout.jl
function find_interval(
    rng::Random.AbstractRNG, logdensity, w::Real, ℓy::Real, θ₀::F
) where {F<:Real}
    #m      = alg.max_stepping_out
    m      = 32
    u      = rand(rng, F)
    L      = θ₀ - w * u
    R      = L + w
    V      = rand(rng, F)
    J      = floor(Int, m * V)
    K      = (m - 1) - J
    n_eval = 0

    while J > 0 && ℓy < logdensity(L)
        L = L - w
        J = J - 1
        n_eval += 1
    end
    while K > 0 && ℓy < logdensity(R)
        R = R + w
        K = K - 1
        n_eval += 1
    end
    return L, R, n_eval
end