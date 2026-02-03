"""
    FiniteGaussianMixtureVIPosterior{T<:Real} <: AbstractVIPosterior{T}

Struct representing the variational posterior distribution of a [`FiniteGaussianMixture`](@ref).

# Fields
* `q_w`: Distribution representing the optimal variational densities of the component weights q*(w|K).
* `q_μ`: Product distribution representing the optimal variational densitiy of the component means q*(μ|K).
* `q_σ2`: Product distribution representing the optimal variational densitiy of the component variances q*(σ2|K).
* `q_β`: The optimal variational density q*(β|k) of the rate hyperparameter of the component variance σ2[k].
* `fgm`: The `FiniteGaussianMixture` to which the variational posterior was fit.
"""
struct FiniteGaussianMixtureVIPosterior{T<:Real, A<:Dirichlet, B<:ContinuousDistribution, C<:ContinuousDistribution, D<:InverseGamma{T}, F<:FiniteGaussianMixture} <: AbstractVIPosterior{T}
    q_w::A
    q_μ::B
    q_σ2::C
    q_β::D
    fgm::F
    function FiniteGaussianMixtureVIPosterior{T}(
        dirichlet_params::AbstractVector{T},
        location_params::AbstractVector{T},
        variance_params::AbstractVector{T},
        shape_params::AbstractVector{T},
        rate_params::AbstractVector{T},
        hyper_shape_param::T,
        hyper_rate_param::T,
        fgm::FiniteGaussianMixture
    ) where {T<:Real}
        q_w = Dirichlet(dirichlet_params)
        q_μ = product_distribution([Normal(location_params[i], variance_params[i]) for i in eachindex(location_params)])
        q_σ2 = product_distribution([InverseGamma(shape_params[i], rate_params[i]) for i in eachindex(shape_params)])
        q_β = InverseGamma(hyper_shape_param, hyper_rate_param)
        return new{T,typeof(q_w), typeof(q_μ), typeof(q_σ2), typeof(q_β), typeof(fgm)}(q_w, q_μ, q_σ2, q_β, fgm)
    end
end

BayesDensityCore.model(vip::FiniteGaussianMixtureVIPosterior) = vip.fgm

function Base.show(io::IO, ::MIME"text/plain", vip::FiniteGaussianMixtureVIPosterior{T, A, B, C, D}) where {T, A, B, C, D}
    K = length(vip.q_w)
    println(io, nameof(typeof(vip)), "{", T, "} vith variational densities:")
    println(io, " q_w::", A, ",")
    println(io, " q_μ::", B, ",")
    println(io, " q_σ2::", C, ",")
    println(io, " q_β::", D, ",")
    println(io, "Model:")
    print(io, model(vip))
    nothing
end
Base.show(io::IO, vip::FiniteGaussianMixtureVIPosterior) = show(io, MIME("text/plain"), vip)

function StatsBase.sample(rng::AbstractRNG, vip::FiniteGaussianMixtureVIPosterior{T}, n_samples::Int) where {T<:Real}
    (; q_w, q_μ, q_σ2, q_β, fgm) = vip
    samples = Vector{NamedTuple{(:μ, :σ2, :w, :β), Tuple{Vector{T}, Vector{T}, Vector{T}, T}}}(undef, n_samples)
    
    for m in 1:n_samples
        samples[m] = (
            μ = rand(rng, q_μ),
            σ2 = rand(rng, q_σ2),
            w = rand(rng, q_w),
            β = rand(rng, q_β)
        )
    end
    return PosteriorSamples{T}(samples, fgm, n_samples, 0)
end

"""
    varinf(
        fgm::FiniteGaussianMixture{T};
        initial_params::NamedTuple = _get_default_initparams(x),
        max_iter::Int              = 2000
        rtol::Real                 = 1e-6
    ) where {T} -> PitmanYorMixtureVIPosterior{T}

Find a variational approximation to the posterior distribution of a [`FiniteGaussianMixture`](@ref) using mean-field variational inference.
# Arguments
* `fgm`: The `FiniteGaussianMixture` whose posterior we want to approximate.

# Keyword arguments
* `initial_params`: Initial values of the VI parameters `dirichlet_params` `location_params`, `variance_params`, `shape_params` and `rate_params`, supplied as a NamedTuple.
* `max_iter`: Maximal number of VI iterations. Defaults to `1000`.
* `rtol`: Relative tolerance used to determine convergence. Defaults to `1e-6`.

# Returns
* `vip`: A [`FiniteGaussianMixtureVIPosterior`](@ref) object representing the variational posterior.
* `info`: A [`VariationalOptimizationResult`](@ref) describing the result of the optimization.

!!! note
    To perform the optimization for a fixed number of iterations irrespective of the convergence criterion, one can set `rtol = 0.0`, and `max_iter` equal to the desired total iteration count.
    Note that setting `rtol` to a strictly negative value will issue a warning.

# Examples
```julia-repl
julia> using Random

julia> x = (1.0 .- (1.0 .- LinRange(0.0, 1.0, 5000)) .^(1/3)).^(1/3);

julia> fgm = FiniteGaussianMixture(x, 10);

julia> vip, info = varinf(fgm);

julia> vip, info = varinf(fgm; rtol=1e-7, max_iter=3000);
```

# Extended help
## Convergence
The criterion used to determine convergence is that the relative change in the ELBO falls below the given `rtol`.
"""
function BayesDensityCore.varinf(
    fgm::FiniteGaussianMixture;
    initial_params::NamedTuple = _get_default_initparams_varinf(fgm),
    max_iter::Int              = 2000,
    rtol::Real                 = 1e-6
)
    (max_iter >= 1) || throw(ArgumentError("Maximum number of iterations must be positive."))
    (rtol ≥ 0.0) || @warn "Relative tolerance is negative."
    _check_initialparams_varinf(initial_params, fgm)
    return _variational_inference(fgm, initial_params, max_iter, rtol)
end

# Simple initialization where quantiles are used to initialize component means
function _get_default_initparams_varinf(fgm::FiniteGaussianMixture{T}) where {T}
    (; x, n) = fgm.data
    K = fgm.K
    breaks = quantile(x, LinRange{T}(0, 1, K+1))
    bin_counts, bin_sums, bin_sumsqs = _get_suffstats_binned(x, breaks)
    nonzero_ind = (bin_counts .!= 0)

    μ = fill(fgm.prior_location, K)
    σ2 = fill(fgm.prior_variance, K)

    μ[nonzero_ind] = bin_sums[nonzero_ind] ./ bin_counts[nonzero_ind]
    σ2[nonzero_ind] = (bin_sumsqs[nonzero_ind] - 2*μ[nonzero_ind] .* bin_sums[nonzero_ind] + bin_counts[nonzero_ind] .* μ[nonzero_ind].^2) ./ bin_counts[nonzero_ind]
    (; prior_strength, prior_location, prior_variance, prior_shape, hyperprior_shape, hyperprior_rate) = hyperparams(fgm)
    return (
        dirichlet_params = prior_strength .+ bin_counts,
        location_params = copy(μ),
        variance_params = fill(ifelse(K ≥ 2, var(μ), var(x)), K),
        shape_params = fill(prior_shape, K),
        rate_params = fill(hyperprior_shape / hyperprior_rate, K)
    )
end

function _variational_inference(
    fgm::FiniteGaussianMixture{T},
    initial_params::NamedTuple,
    max_iter::Int,
    rtol::Real
) where {T}
    # Unpack model
    (; data, K, prior_strength, prior_location, prior_variance, prior_shape, hyperprior_rate, hyperprior_shape) = fgm
    (; x, n) = data

    # Unpack parameters
    (; dirichlet_params, location_params, variance_params, shape_params, rate_params) = initial_params

    # Initialize relevant model quantitites
    shape_hyperparam = zero(T)
    rate_hyperparam = zero(T)
    prior_dirichlet_params = fill(prior_strength, K)
    q_mat = Matrix{T}(undef, (K, n))
    E_inv_σ2 = shape_params ./ rate_params 
    kernel_terms = Matrix{T}(undef, (K, n))
    digamma_sum_dirichlet_params = digamma(sum(dirichlet_params))
    non_kernel_terms = @. digamma(dirichlet_params) - digamma_sum_dirichlet_params
    kernel_term0 = @. -log(2*T(pi))/2 - (log(rate_params) - digamma(shape_params)) / 2
    for i in eachindex(x)
        kernel_terms[:, i] = kernel_term0 - @. E_inv_σ2 * ((x[i] - location_params)^2 + variance_params) / 2
    end

    # Optimization loop
    converged = false
    ELBO = Vector{T}(undef, max_iter)
    # Arbitrary init of ELBO
    # (we perform at least 2 iterations, due to not all variables being initialized properly)
    ELBO_last = 1.0
    iter = 1
    while !converged && iter ≤ max_iter
        # Update q(z|k) 
        E_N = zeros(T, K)
        for i in eachindex(x)
            logprobs = kernel_terms[:, i] + non_kernel_terms
            probs = softmax(logprobs)
            q_mat[:, i] = probs
            E_N .+= probs
        end
        weighted_sum = q_mat * x
        # ELBO contribution from -E[log q(z)]
        ELBO_q_term = -sum(xlogx, q_mat)

        # Update q(β|k)
        shape_hyperparam = hyperprior_shape + K*prior_shape
        rate_hyperparam = hyperprior_rate + sum(E_inv_σ2)
        E_β = shape_hyperparam / rate_hyperparam
        E_log_β = - log(rate_hyperparam) + digamma(shape_hyperparam)
        # KL between q(β|k) and p(β|k)
        KL_β = shape_hyperparam * log(rate_hyperparam) - loggamma(shape_hyperparam) - hyperprior_shape * log(hyperprior_rate) + loggamma(hyperprior_shape)
        KL_β += (shape_hyperparam - hyperprior_shape) * E_log_β + (hyperprior_rate - rate_hyperparam) * E_β

        # Update q(w|k)
        dirichlet_params = prior_strength .+ E_N
        digamma_sum_dirichlet_params = digamma(sum(dirichlet_params))
        non_kernel_terms = @. digamma(dirichlet_params) - digamma_sum_dirichlet_params
        # KL between q(w|k) and p(w|k)
        KL_w = sum(
            @. loggamma(prior_dirichlet_params) - loggamma(dirichlet_params) + (dirichlet_params - prior_dirichlet_params) * non_kernel_terms
        ) + loggamma(sum(dirichlet_params)) - loggamma(sum(prior_dirichlet_params))
        # ELBO contributions from terms in the likelihood depending on w
        ELBO_w_term = sum(@. E_N * non_kernel_terms)

        # Update q(μ|k)
        variance_params = @. inv(1/prior_variance + E_inv_σ2 * E_N)
        location_params = @. variance_params * (prior_location / prior_variance + E_inv_σ2 * weighted_sum)
        weighted_rss = zeros(T, K)
        for i in eachindex(x)
            weighted_rss += @. q_mat[:,i] * (x[i] - location_params)^2
        end
        weighted_rss += E_N .* variance_params
        # KL between q(μ|k) and p(μ|k)
        KL_μ = sum(
            @. (location_params - prior_location)^2 / (2*prior_variance) + (variance_params / prior_variance - 1 - log(variance_params / prior_variance)) / 2
        )

        # Update q(σ2|k)
        shape_params = prior_shape .+ E_N / 2
        rate_params = E_β .+ weighted_rss/2
        E_inv_σ2 = shape_params ./ rate_params
        E_log_σ2 = @. log(rate_params) - digamma(shape_params)
        # KL between q(σ2|k) and p(σ2|k, E_β)
        KL_σ2 = sum(
            @. (prior_shape - shape_params) * E_log_σ2 + (E_β-rate_params) * E_inv_σ2
        )
        KL_σ2 += sum(
            @. shape_params * log(rate_params) - hyperprior_shape * E_log_β + loggamma(hyperprior_shape) - loggamma(shape_params)
        )

        kernel_term0 = @. -log(2*T(pi))/2 - E_log_σ2 / 2
        for i in eachindex(x)
            kernel_terms[:, i] = kernel_term0 - @. E_inv_σ2 * ((x[i] - location_params)^2 + variance_params) / 2
        end
        # ELBO contributions from terms in the likelihood depending on μ, σ2
        ELBO_θ_term = sum(q_mat .* kernel_terms)

        # Check convergence
        ELBO[iter] =  ELBO_q_term + ELBO_w_term + ELBO_θ_term - KL_w - KL_β - KL_μ - KL_σ2
        converged = (abs(ELBO[iter]-ELBO_last)/abs(ELBO[iter]) < rtol) && iter ≥ 2
        ELBO_last = ELBO[iter]
        iter += 1
    end

    # Return posterior, diagnostics
    converged || @warn "Failed to meet convergence criterion in $(iter-1) iterations."
    variational_posterior = FiniteGaussianMixtureVIPosterior{T}(
        dirichlet_params,
        location_params,
        variance_params,
        shape_params,
        rate_params,
        shape_hyperparam,
        rate_hyperparam,
        fgm
    )
    info = VariationalOptimizationResult{T}(ELBO[1:iter-1], converged, iter-1, rtol, variational_posterior)
    return variational_posterior, info
end


function _check_initialparams_varinf(initial_params::NamedTuple{N, T}, ::FiniteGaussianMixture) where {N, T}
    (:dirichlet_params in N &&
    :location_params  in N &&
    :variance_params  in N &&
    :shape_params     in N &&
    :rate_params      in N) || throw(ArgumentError("Expected a NamedTuple with fields dirichlet_params, location_params, variance_params, shape_params and rate_params"))
    (; dirichlet_params, location_params, variance_params, shape_params, rate_params) = initial_params
    (length(dirichlet_params) == length(location_params) == length(variance_params) == length(shape_params) == length(rate_params)) || throw(ArgumentError("Initial dirichlet_params, location_params, variance_params, shape_params and rate_params dimensions are incompatible."))
end