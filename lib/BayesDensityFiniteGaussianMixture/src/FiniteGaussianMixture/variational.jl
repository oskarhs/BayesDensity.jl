"""
    FiniteGaussianMixtureVIPosterior{T<:Real} <: AbstractVIPosterior{T}

Struct representing the variational posterior distribution of a [`FiniteGaussianMixture`](@ref).

# Fields
* `q_w`: Distribution representing the optimal variational densities of the component weights q*(wₖ|K) for 1 ≤ k ≤ K and 1 ≤ K ≤ K_max.
* `q_μ`: Vector of distributions representing the optimal variational densities of the component means q*(μₖ|K) for 1 ≤ k ≤ K and 1 ≤ K ≤ K_max.
* `q_σ2`: Vector of distributions representing the optimal variational densities of the component variances q*(σ2ₖ|K) for 1 ≤ k ≤ K and 1 ≤ K ≤ K_max.
* `fgm`: The `FiniteGaussianMixture` to which the variational posterior was fit.
"""
struct FiniteGaussianMixtureVIPosterior{T<:Real, A<:Dirichlet, B<:ContinuousDistribution, C<:ContinuousDistribution, F<:FiniteGaussianMixture}
    q_w::A
    q_μ::B
    q_σ2::C
    fgm::F
    function FiniteGaussianMixtureVIPosterior{T}(
        dirichlet_params::AbstractVector{T},
        location_params::AbstractVector{T},
        variance_params::AbstractVector{T},
        shapes_params::AbstractVector{T},
        rate_params::AbstractVector{T},
        fgm::FiniteGaussianMixture
    ) where {T<:Real}
        q_w = Dirichlet(dirichlet_par)
        q_μ = product_distribution([Normal(location_params[i], variance_params[i]) for i in eachindex(location_params)])
        q_σ2 = product_distribution([InverseGamma(shape_params[i], rate_params[i]) for i in eachindex(shape_params)])
        return new{T,typeof(q_w), typeof(q_μ), typeof(q_σ2), typeof(fgm)}(q_w, q_μ, q_σ2, fgm)
    end
end

BayesDensityCore.model(vip::FiniteGaussianMixtureVIPosterior) = vip.fgm

function Base.show(io::IO, ::MIME"text/plain", vip::FiniteGaussianMixtureVIPosterior{T, A, B, C, F}) where {T, A, B, C, F}
    K = length(vip.q_w)
    println(io, "K-dimensional ", nameof(typeof(vip)), "{", T, "} vith variational densities:")
    println(io, " q_w::", A, ",")
    println(io, " q_μ::", B, ",")
    println(io, " q_σ2::", B, ",")
    println(io, "Model:")
    print(io, model(vip))
    nothing
end
Base.show(io::IO, vip::FiniteGaussianMixtureVIPosterior) = show(io, MIME("text/plain"), vip)

function StatsBase.sample(rng::AbstractRNG, vip::FiniteGaussianMixtureVIPosterior{T}, n_samples::Int) where {T<:Real}
    (; q_w, q_μ, q_σ2, fgm) = vip
    samples = Vector{NamedTuple{(:μ, :σ2, :w), Tuple{Vector{T}, Vector{T}, Vector{T}}}}(undef, n_samples)
    
    for m in 1:n_samples
        samples[m] = (
            μ = rand(rng, q_μ),
            σ2 = rand(rng, q_σ2),
            w = rand(rng, q_w)
        )
    end
    return PosteriorSamples{T}(samples, pym, n_samples, 0)
end

"""
    varinf(
        fgm::FiniteGaussianMixture{T};
        truncation_level::Int      = 30,
        initial_params::NamedTuple = _get_default_initparams(x),
        max_iter::Int              = 3000
        rtol::Real                 = 1e-6
    ) where {T} -> PitmanYorMixtureVIPosterior{T}

Find a variational approximation to the posterior distribution of a [`FiniteGaussianMixture`](@ref) using mean-field variational inference based on a truncated stickbreaking-approach.

# Arguments
* `fgm`: The `FiniteGaussianMixture` whose posterior we want to approximate.

# Keyword arguments
* `initial_params`: Initial values of the VI parameters `dirichlet_params` `location_params`, `variance_params`, `shape_params` and `rate_params`, supplied as a NamedTuple.
* `max_iter`: Maximal number of VI iterations. Defaults to `1000`.
* `rtol`: Relative tolerance used to determine convergence. Defaults to `1e-6`.

# Returns
* `vip`: A [`FiniteGaussianMixtureVIPosterior`](@ref) object representing the variational posterior.

!!! note
    To perform the optimization for a fixed number of iterations irrespective of the convergence criterion, one can set `rtol = 0.0`, and `max_iter` equal to the desired total iteration count.
    Note that setting `rtol` to a strictly negative value will issue a warning.

# Extended help
## Convergence
The criterion used to determine convergence is that the relative change in the ELBO falls below the given `rtol`.
"""
function BayesDensityCore.varinf(
    fgm::FiniteGaussianMixture;
    initial_params::NamedTuple=_get_default_initparams(fgm, truncation_level),
    max_iter::Int=3000,
    rtol::Real=1e-6
)
    (max_iter >= 1) || throw(ArgumentError("Maximum number of iterations must be positive."))
    (rtol ≥ 0.0) || @warn "Relative tolerance is negative."
    _check_initialparams(initial_params, fgm)
    return _variational_inference(fgm, initial_params, max_iter, rtol)
end

# Simple initialization where quantiles are used to initialize component means
function _get_default_initparams(fgm::FiniteGaussianMixture{T}) where {T}
    K = fgm.K
    x = fgm.data.x
    (; prior_strength, prior_location, prior_variance, prior_shape, prior_rate) = hyperparams(fgm)
    locations_init = quantile(x, LinRange(1/(K+1), K/(K+1), K))
    return (
        dirichlet_params = fill(1, K),
        location_params = locations_init,
        variance_params = fill(inv_scale_fac, truncation_level),
        shape_params = fill(prior_shape, truncation_level),
        rate_params = fill(prior_rate, truncation_level)
    )
end

function _variational_inference(
    fgm::FiniteGaussianMixture{T, NT},
    initial_params::NamedTuple,
    max_iter::Int,
    rtol::Real
) where {T, NT}
    # Unpack parameters
    (; dirichlet_params, location_params, variance_params, shape_params, rate_params) = initial_params

    # Optimization loop

    # Return posterior, diagnostics
    converged || @warn "Failed to meet convergence criterion in $(iter-1) iterations."
    variational_posterior = FiniteGaussianMixtureVIPosterior{T}(
        a_v,
        b_v,
        locations,
        inv_scale_facs,
        shapes,
        rates,
        fgm
    )
    info = VariationalOptimizationResult{T}(ELBO[1:iter-1], converged, iter-1, rtol, variational_posterior)
    return variational_posterior, info
end


function _check_initialparams(initial_params::NamedTuple{N, T}, ::FiniteGaussianMixture) where {N, T}
    (:dirichlet_params in N &&
    :location_params  in N &&
    :variance_params  in N &&
    :shape_params     in N &&
    :rate_params      in N) || throw(ArgumentError("Expected a NamedTuple with fields dirichlet_params, location_params, variance_params, shape_params and rate_params"))
    (; dirichlet_params, location_params, variance_params, shape_params, rate_params) = initial_params
    (length(dirichlet_params) == length(location_params) == length(variance_params) == length(shape_params) == length(rate_params)) || throw(ArgumentError("Initial dirichlet_params, location_params, variance_params, shape_params and rate_params dimensions are incompatible."))
end