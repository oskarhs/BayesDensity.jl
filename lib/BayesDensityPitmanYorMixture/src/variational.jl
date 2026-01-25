"""
    PitmanYorMixtureVIPosterior{T<:Real} <: AbstractVIPosterior{T}

Struct representing the variational posterior distribution of a [`PitmanYorMixture`](@ref).

# Fields
* `q_v`: Vector of distributions representing the optimal variational densities q*(vₖ), i.e. the density of the stick-breaking weights.
* `q_θ`: Vector of distributions representing the optimal variational densities q*(θₖ), i.e. the joint density of the mixture component means and variances.
* `pym`: The `PitmanYorMixture` to which the variational posterior was fit.
"""
struct PitmanYorMixtureVIPosterior{T<:Real, A<:AbstractVector{<:Beta{T}}, B<:AbstractVector{<:Normal{T}}, M<:PitmanYorMixture} <: AbstractVIPosterior{T}
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
    K = length(model(vip.q_θ))
    println(io, nameof(typeof(vip)), "{", T, "} vith truncation level", K, " and variational densities:")
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
    samples = Vector{NamedTuple{(:w, :μ, :σ2), Tuple{Vector{T}, Vector{T}, Vector{T}}}}(undef, n_samples)

    for m in 1:n_samples
        for k in 1:K-1
            v[k] = rand(rng, q_v[k])
            μ[k], σ2[k] = rand(rng, q_θ[k])
        end
        μ[K], σ2[K] = rand(rng, q_θ[K])
        w = truncated_stickbreaking(v)
        samples[m] = (w = w, μ = μ, σ2 = σ2)
    end
    return PosteriorSamples{T}(samples, pym, n_samples, 0)
end

"""
    varinf(
        bsm::PitmanYorMixture{T};
        truncation_level::Int=30,
        initial_params::Union{Nothing,NamedTuple}=nothing,
        max_iter::Int=3000
        rtol::Real=1e-6
    ) where {T} -> PitmanYorMixtureVIPosterior{T}

Find a variational approximation to the posterior distribution of a [`PitmanYorMixture`](@ref) using mean-field variational inference based on a truncated stickbreaking-approach.

# Arguments
* `bsm`: The `BSplineMixture` whose posterior we want to approximate.

# Keyword arguments
* `truncation level`: Integer specifying the truncation level of the variational approximation. Defaults to `30`. This parameter is ignored `initial_params` is set to another value than nothing. 
* `initial_params`: Initial values of the VI parameters `a_v` `b_v`, `locations` and `inv_scale_facs`, `shapes` and `rates`, supplied as a NamedTuple.
* `max_iter`: Maximal number of VI iterations. Defaults to `3000`.
* `rtol`: Relative tolerance used to determine convergence. Defaults to `1e-6`.

# Returns
* `vip`: A [`PitmanYorMixtureVIPosterior`](@ref) object representing the variational posterior.

!!! note
    To sample for a fixed number of iterations irrespective of the convergence criterion, one can set `rtol = 0.0`, and `max_iter` equal to the desired total iteration count.
    Note that setting `rtol` to a strictly negative value will issue a warning.

# Extended help
## Convergence
The criterion used to determine convergence is that the relative change in the ELBO falls below the given `rtol`.
"""
function BayesDensityCore.varinf(
    pym::PitmanYorMixture;
    truncation_level::Int=30,
    initial_params::Union{Nothing,NamedTuple}=nothing,
    max_iter::Int=3000,
    rtol::Real=1e-6
)
    if isnothing(initial_params)
        initial_params = _get_default_initparams(pym, truncation_level)
    end
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
        b_v = T.(collect(1/(truncation_level-1:1))),
        locations = locations,
        inv_scale_facs = fill(inv_scale_fac, truncation_level),
        shapes = fill(shape, truncation_level),
        rates = fill(rate, truncation_level)
        )
end

function _varinf(
    pym::PitmanYorMixtureVIPosterior{T, NT},
    initial_params::NamedTuple,
    max_iter::Int,
    rtol::Real
) where {T, NT}
    # Unpack parameters
    (; data, discount, strength, location, inv_scale_fac, shape, rate) = pym
    (; x, n) = data

    # Get initial parameter values
    (; a_v, b_v, locations, inv_scale_facs, shapes, rates) = initial_params

    # Optimize:
    ELBO = Vector{T}(undef, max_iter)
    ELBO_old = 1.0
    converged = false
    iter = 1
    while !converged && iter ≤ max_iter
        # Write the CAVI loop here...

        # Check convergence
        ELBO[iter] = ...
        converged = (abs(ELBO[iter] - ELBO_old)/abs(ELBO_old) ≤ rtol && iter ≥ 2)
        ELBO_old = ELBO[iter]
        iter += 1
    end
    vip = PitmanYorMixtureVIPosterior{T}(
        a_v,
        b_v,
        locations,
        inv_scale_facs,
        shapes,
        rates,
        pym
    )
    info = VariationalOptimizationResult{T}(ELBO[1:n_iter], converged, n_iter, tolerance, variational_posterior)
    return vip, info
end