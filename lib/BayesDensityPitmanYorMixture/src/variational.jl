"""
    PitmanYorMixtureVIPosterior{T<:Real} <: AbstractVIPosterior{T}

Struct representing the variational posterior distribution of a [`PitmanYorMixture`](@ref).

# Fields
* `q_v`: Vector of distributions representing the optimal variational densities q*(vₖ), i.e. the density of the stick-breaking weights.
* `q_θ`: Vector of distributions representing the optimal variational densities q*(θₖ), i.e. the joint density of the mixture component means and variances.
* `pym`: The `PitmanYorMixture` to which the variational posterior was fit.
"""
struct PitmanYorMixtureVIPosterior{T<:Real, A<:AbstractVector{<:Beta{T}}, B<:AbstractVector{<:NormalInverseGamma{T}}, M<:PitmanYorMixture} <: AbstractVIPosterior{T}
    q_v::A
    q_θ::B
    pym::M
    function PitmanYorMixtureVIPosterior{T}(
        a_v::AbstractVector{T},
        b_v::AbstractVector{T},
        locations::AbstractVector{T},
        inv_scale_facs::AbstractVector{T},
        shapes::AbstractVector{T},
        rates::AbstractVector{T},
        pym::M
    ) where {T<:Real, M<:PitmanYorMixture}
        K = length(locations)
        q_v = Vector{Beta{T}}(undef, K-1)
        q_θ = Vector{NormalInverseGamma{T}}(undef, K)
        for k in 1:K-1
            q_v[k] = Beta{T}(a_v[k], b_v[k])
            q_θ[k] = NormalInverseGamma{T}(locations[k], inv_scale_facs[k], shapes[k], rates[k])
        end
        q_θ[K] = NormalInverseGamma{T}(locations[K], inv_scale_facs[K], shapes[K], rates[K])
        return new{T,Vector{Beta{T}},Vector{NormalInverseGamma{T}},M}(q_v, q_θ, pym)
    end
end

BayesDensityCore.model(vip::PitmanYorMixtureVIPosterior) = vip.pym

function Base.show(io::IO, ::MIME"text/plain", vip::PitmanYorMixtureVIPosterior{T, A, B, M}) where {T, A, B, M}
    K = length(vip.q_θ)
    println(io, nameof(typeof(vip)), "{", T, "} vith truncation level ", K, " and variational densities:")
    println(io, " q_v::", A, ",")
    println(io, " q_θ::", B, ",")
    println(io, "Model:")
    print(io, model(vip))
    nothing
end
Base.show(io::IO, vip::PitmanYorMixtureVIPosterior) = show(io, MIME("text/plain"), vip)

function StatsBase.sample(rng::AbstractRNG, vip::PitmanYorMixtureVIPosterior{T, A, B, M}, n_samples::Int) where {T<:Real, A, B, M}
    (; q_v, q_θ, pym) = vip
    K = length(q_θ)
    samples = Vector{NamedTuple{(:μ, :σ2, :w), Tuple{Vector{T}, Vector{T}, Vector{T}}}}(undef, n_samples)
    
    # Initialize vectors holding samples
    v = Vector{T}(undef, K-1)
    μ = Vector{T}(undef, K)
    σ2 = Vector{T}(undef, K)

    for m in 1:n_samples
        for k in 1:K-1
            v[k] = rand(rng, q_v[k])
            μ[k], σ2[k] = rand(rng, q_θ[k])
        end
        μ[K], σ2[K] = rand(rng, q_θ[K])
        w = truncated_stickbreaking(v)
        samples[m] = (μ = copy(μ), σ2 = copy(σ2), w = w)
    end
    return PosteriorSamples{T}(samples, pym, n_samples, 0)
end

"""
    varinf(
        pym::PitmanYorMixture{T};
        truncation_level::Int=30,
        initial_params::NamedTuple=nothing,
        max_iter::Int=3000
        rtol::Real=1e-6
    ) where {T} -> PitmanYorMixtureVIPosterior{T}

Find a variational approximation to the posterior distribution of a [`PitmanYorMixture`](@ref) using mean-field variational inference based on a truncated stickbreaking-approach.

# Arguments
* `pym`: The `PitmanYorMixture` whose posterior we want to approximate.

# Keyword arguments
* `truncation level`: Positive integer specifying the truncation level of the variational approximation. Defaults to `30`. This parameter is ignored `initial_params` is set to another value than nothing. 
* `initial_params`: Initial values of the VI parameters `a_v` `b_v`, `locations` and `inv_scale_facs`, `shapes` and `rates`, supplied as a NamedTuple.
* `max_iter`: Maximal number of VI iterations. Defaults to `3000`.
* `rtol`: Relative tolerance used to determine convergence. Defaults to `1e-6`.

# Returns
* `vip`: A [`PitmanYorMixtureVIPosterior`](@ref) object representing the variational posterior.
* `info`: A [`VariationalOptimizationResult`](@ref) describing the result of the optimization.

!!! note
    To perform the optimization for a fixed number of iterations irrespective of the convergence criterion, one can set `rtol = 0.0`, and `max_iter` equal to the desired total iteration count.
    Note that setting `rtol` to a strictly negative value will issue a warning.

# Extended help
## Convergence
The criterion used to determine convergence is that the relative change in the ELBO falls below the given `rtol`.

## Truncation
The truncation level determines the maximal number of components used in the variational approximation.
Generally, setting the truncation level to a higher value leads to an approximating class with a greater representational capacity, at the cost of increased computation.
"""
function BayesDensityCore.varinf(
    pym::PitmanYorMixture;
    truncation_level::Int=30,
    initial_params::NamedTuple=_get_default_initparams(pym, truncation_level),
    max_iter::Int=3000,
    rtol::Real=1e-6
)
    (truncation_level >= 1) || throw(ArgumentError("Truncation level must be positive."))
    (max_iter >= 1) || throw(ArgumentError("Maximum number of iterations must be positive."))
    (rtol ≥ 0.0) || @warn "Relative tolerance is negative."
    return _variational_inference(pym, initial_params, max_iter, rtol)
end

# Simple initialization where quantiles are used to initialize component means
function _get_default_initparams(pym::PitmanYorMixture{T, NT}, truncation_level::Int) where {T, NT}
    x = pym.data.x
    (; discount, strength, location, inv_scale_fac, shape, rate) = hyperparams(pym)
    locations = quantile(x, LinRange(1/(truncation_level+1), (truncation_level)/(truncation_level+1), truncation_level))
    return (
        a_v = fill(1, truncation_level-1),
        b_v = fill(1, truncation_level-1),
        locations = locations,
        inv_scale_facs = fill(inv_scale_fac, truncation_level),
        shapes = fill(shape, truncation_level),
        rates = fill(rate, truncation_level)
        )
end

function _variational_inference(
    pym::PitmanYorMixture{T, NT},
    initial_params::NamedTuple,
    max_iter::Int,
    rtol::Real
) where {T, NT}
    # Unpack parameters
    (; data, discount, strength, location, inv_scale_fac, shape, rate) = pym
    (; x, n) = data

    # Get initial parameter values
    (; a_v, b_v, locations, inv_scale_facs, shapes, rates) = initial_params
    K = length(locations)
    
    # Matrix of q(zᵢ = k)
    q_mat = Matrix{T}(undef, (K, n))

    # Precompute some quantities that are used for the ELBO
    E_log_v = @. digamma(a_v) - digamma(a_v + b_v)
    E_log_cv = @. digamma(b_v) - digamma(a_v + b_v)
    kernel_terms = Matrix{T}(undef, (K, n))
    kernel_term0 = @. -1/2 * (log(rates) - digamma(shapes)) - 1/(2*inv_scale_facs) - log(2*T(pi)) / 2
    for i in eachindex(x)
        kernel_terms[:, i] = @. kernel_term0 - shapes * (x[i] - locations)^2 / (2*rates) # Parts of this can be precomputed before the loop!
    end

    # Optimize:
    ELBO = Vector{T}(undef, max_iter)
    ELBO_old = 1.0
    converged = false
    iter = 1
    while !converged && iter ≤ max_iter
        # Update q(z)
        # Compute contribution to logprobabilities from non-kernel terms
        non_kernel_term = vcat(E_log_v, zero(T)) + cumsum(vcat(zero(T), E_log_cv))
        for i in eachindex(x)
            # Contributions from kernel
            logprobs = kernel_terms[:, i] + non_kernel_term
            probs = softmax(logprobs)
            q_mat[:, i] = probs
        end
        ELBO_q_term = -sum(xlogx, q_mat)
        # Vector of number of observations with label greater than k
        E_N = vec(sum(q_mat; dims=2))
        E_S = n .- cumsum(E_N)
        wmeans = (q_mat * x) ./ (E_N)
        wsumsq = Vector{T}(undef, K)
        for k in 1:K
            wsumsq[k] = sum(q_mat[k, :] .* (x .- wmeans[k]).^2)
        end

        # Update q(v)
        a_v = @. 1 - discount + E_N[1:K-1]
        b_v = @. E_S[1:K-1] + strength + discount * (1:K-1)
        # Compute ELBO contribution from the v part of the likelihood
        E_log_v = @. digamma(a_v) - digamma(a_v + b_v)
        E_log_cv = @. digamma(b_v) - digamma(a_v + b_v)
        ELBO_v_term = sum(@. (E_N[1:K-1]) * E_log_v[1:K-1] + (E_S[1:K-1]) * E_log_cv[1:K-1])

        # Update q(θ)
        inv_scale_facs = inv_scale_fac .+ E_N
        locations = (inv_scale_fac * location .+ E_N .* wmeans) ./ inv_scale_facs
        shapes = shape .+ E_N / 2
        rates = rate .+ (wsumsq + inv_scale_fac * E_N ./ inv_scale_facs .* (wmeans .- location).^2) / 2
        # ELBO contribution
        kernel_term0 = @. -1/2 * (log(rates) - digamma(shapes)) - 1/(2*inv_scale_facs) - 1/2 * log(2*T(pi))
        for i in eachindex(x)
            kernel_terms[:, i] = @. kernel_term0 - shapes * (x[i] - locations)^2 / (2*rates) # Parts of this can be precomputed before the loop!
        end
        ELBO_θ_term = sum(q_mat .* kernel_terms) # Here we just esentially get what we compute when considering q_z

        # Compute KL-divergences between q and priors
        KL_v = sum(@. logbeta(1-discount, strength + discount*(1:K-1)) - logbeta(a_v, b_v) + (a_v-(1-discount))*digamma(a_v) + (b_v - (strength + discount*(1:K-1)))*digamma(b_v) + (1 - discount + strength + discount*(1:K-1) - b_v - a_v) * digamma(a_v + b_v))
        KL_norm_const = @. (log(inv_scale_fac) - log(inv_scale_facs))/ 2 + shapes * log(rates) - shape * log(rate) - loggamma(shapes) + loggamma(shape)
        KL_IG = @. (shape - shapes) * (log(rates) - digamma(shapes)) + (rate - rates) * shapes / rates
        KL_N = @. 1 - inv_scale_fac/inv_scale_facs - shapes * (location - locations)^2 / rates
        KL_θ = sum(KL_norm_const + KL_IG + KL_N)
        # Check convergence
        ELBO[iter] = ELBO_v_term + ELBO_θ_term + ELBO_q_term - KL_v - KL_θ
        converged = (abs(ELBO[iter] - ELBO_old)/abs(ELBO_old) ≤ rtol && iter ≥ 2)
        ELBO_old = ELBO[iter]
        iter += 1
    end

    converged || @warn "Failed to meet convergence criterion in $(iter-1) iterations."
    variational_posterior = PitmanYorMixtureVIPosterior{T}(
        a_v,
        b_v,
        locations,
        inv_scale_facs,
        shapes,
        rates,
        pym
    )
    info = VariationalOptimizationResult{T}(ELBO[1:iter-1], converged, iter-1, rtol, variational_posterior)
    return variational_posterior, info
end