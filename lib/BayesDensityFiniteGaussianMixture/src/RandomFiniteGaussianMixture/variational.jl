"""
    RandomFiniteGaussianMixtureVIPosterior{T<:Real} <: AbstractVIPosterior{T}

Struct representing the variational posterior distribution of a [`RandomFiniteGaussianMixture`](@ref).

# Fields
* `mixture_fits`: Dictionary consisting of 2-tuples, where the values are the posterior probability for a given `K` and the corresponding [`FiniteGaussianMixtureVIPosterior`](@ref), containing the fitted variational posterior distributions for differing values of mixture components.
* `rgfm`: The `RandomFiniteGaussianMixture` to which the variational posterior was fit.
"""
struct RandomFiniteGaussianMixtureVIPosterior{T<:Real, M<:AbstractDict, R<:RandomFiniteGaussianMixture{T}} <: AbstractVIPosterior{T}
    mixture_fits::M
    rfgm::R
    function RandomFiniteGaussianMixtureVIPosterior{T}(
        mixture_fits::AbstractDict,
        rfgm::RandomFiniteGaussianMixture
    ) where {T<:Real}
        return new{T, typeof(mixture_fits), typeof(rfgm)}(mixture_fits, rfgm)
    end
end

BayesDensityCore.model(vip::RandomFiniteGaussianMixtureVIPosterior) = vip.rfgm

"""
    posterior_prob_components(vip::RandomFiniteGaussianMixtureVIPosterior{T}) where {T} -> Dict{Int, T}

Get the variational posterior probability mass function of the number of mixture components as a dictionary.
"""
posterior_prob_components(vip::RandomFiniteGaussianMixtureVIPosterior{T}) where {T} = Dict{Int, T}(key => val[1] for (key, val) in vip.mixture_fits)

"""
    maximum_a_posteriori(
        vip::RandomFiniteGaussianMixtureVIPosterior{T}
    ) where {T} -> FiniteGaussianMixtureVIPosterior{T}

Get the variational posterior distribution that maximizes the approximate posterior probability on the number of components q(K).
"""
function maximum_a_posteriori(vip::RandomFiniteGaussianMixtureVIPosterior{T}) where {T}
    highest_prob = -T(Inf)
    best_model = nothing
    for (key, val) in vip.mixture_fits
        if val[1] > highest_prob
            highest_prob = val[1]
            best_model = val[2]
        end
    end
    return best_model
end

function Base.show(io::IO, ::MIME"text/plain", vip::RandomFiniteGaussianMixtureVIPosterior{T}) where {T}
    println(io, nameof(typeof(vip)), "{", T, "}.")
    println(io, "Model:")
    print(io, model(vip))
    nothing
end
Base.show(io::IO, vip::RandomFiniteGaussianMixtureVIPosterior) = show(io, MIME("text/plain"), vip)

function StatsBase.sample(
    rng::AbstractRNG,
    vip::RandomFiniteGaussianMixtureVIPosterior{T},
    n_samples::Int
) where {T<:Real}
    (; mixture_fits, rfgm) = vip
    samples = Vector{NamedTuple{(:μ, :σ2, :w, :β), Tuple{Vector{T}, Vector{T}, Vector{T}, T}}}(undef, n_samples)

    # Extract posterior model probabilities
    q_K_probs = Vector{T}(undef, length(mixture_fits))
    multinomial_index = Dict{Int, Int}()
    iter = 0
    for (K, val) in mixture_fits
        iter += 1
        q_K_probs[iter] = val[1]
        multinomial_index[iter] = K
    end
    # Draw K ~ q(K) n_samples times, record the counts
    K_samples = rand(rng, Multinomial(n_samples, q_K_probs))

    # Map the computed counts to the correpsonding index and prune K's with 0 draws
    K_counts = Dict{Int, Int}(multinomial_index[i] => K_samples[i] for i in 1:length(mixture_fits) if K_samples[i] > 0)

    # Generate samples conditional on the drawn K's
    i0 = 1
    for (K, K_num_samples) in K_counts
        i1 = i0 + (K_num_samples-1)
        # Sample (μ, σ2, w, β) from q(⋯|K)
        samples[i0:i1] .= sample(rng, mixture_fits[K][2], K_num_samples).samples
        i0 = i1 + 1
    end
    return PosteriorSamples{T}(samples, rfgm, n_samples, 0)
end

"""
    varinf(
        rfgm::RandomFiniteGaussianMixture{T};
        max_iter::Int = 2000
        rtol::Real    = 1e-6
    ) where {T} -> PitmanYorMixtureVIPosterior{T}

Find a variational approximation to the posterior distribution of a [`RandomFiniteGaussianMixture`](@ref) using mean-field variational inference.

# Arguments
* `rfgm`: The `RandomFiniteGaussianMixture` whose posterior we want to approximate.

# Keyword arguments
* `max_iter`: Maximal number of VI iterations. Defaults to `2000`.
* `rtol`: Relative tolerance used to determine convergence. Defaults to `1e-6`.

# Returns
* `vip`: A [`RandomFiniteGaussianMixtureVIPosterior`](@ref) object representing the variational posterior.
* `info`: A [`VariationalOptimizationResult`](@ref) describing the result of the optimization.

!!! note
    To perform the optimization for a fixed number of iterations irrespective of the convergence criterion, one can set `rtol = 0.0`, and `max_iter` equal to the desired total iteration count.
    Note that setting `rtol` to a strictly negative value will issue a warning.

# Examples
```julia-repl
julia> using Random

julia> x = (1.0 .- (1.0 .- LinRange(0.0, 1.0, 5000)) .^(1/3)).^(1/3);

julia> rfgm = RandomFiniteGaussianMixture(x);

julia> vip = varinf(rfgm);

julia> vip = varinf(rfgm; rtol=1e-7, max_iter=3000);
```
"""
function BayesDensityCore.varinf(
    rfgm::RandomFiniteGaussianMixture;
    max_iter::Int = 2000,
    rtol::Real    = 1e-6
)
    (max_iter >= 1) || throw(ArgumentError("Maximum number of iterations must be positive."))
    (rtol ≥ 0.0) || @warn "Relative tolerance is negative."
    return _variational_inference(rfgm, max_iter, rtol)
end

function _variational_inference(
    rfgm::RandomFiniteGaussianMixture{T},
    max_iter::Int,
    rtol::Real
) where {T}
    (; data, prior_components, prior_strength, prior_location, prior_variance, prior_shape, hyperprior_shape, hyperprior_rate) = rfgm
    (; x, n) = data

    # Fit models and store the value of the posterior probability on the logscale
    mixture_fits_logprobs = Dict{Int, Tuple{T, Any}}()
    logprobs = Vector{T}(undef, length(prior_components)) # Store logprobabilities in a separate vector for normalization purposes
    vi_type = Any
    iter = 0
    for (K, val) in prior_components
        iter += 1
        fgm = FiniteGaussianMixture(
            x,
            K;
            prior_strength = prior_strength,
            prior_location = prior_location,
            prior_variance = prior_variance,
            prior_shape = prior_shape,
            hyperprior_shape = hyperprior_shape,
            hyperprior_rate = hyperprior_rate
        )
        vip, info = varinf(fgm; max_iter = max_iter, rtol = rtol)
        logprobs[iter] = log(val) + last(elbo(info))
        mixture_fits_logprobs[K] = (logprobs[iter], vip)
        vi_type = typeof(vip)
    end

    # Get quantities for numerically stable normalization of probabilities
    max_logprobs = maximum(logprobs)
    sum_probs_stable = sum(exp, logprobs .- max_logprobs)

    # Create a new dictionary containing the normalized posterior probabilities and the corresponding model.
    mixture_fits = Dict{Int, Tuple{T, vi_type}}()
    for (K, val) in mixture_fits_logprobs
        logprob, vip = val
        prob = exp(logprob - max_logprobs) / sum_probs_stable
        mixture_fits[K] = (prob, vip)
    end
    return RandomFiniteGaussianMixtureVIPosterior{T}(
        mixture_fits,
        rfgm
    )
end