"""
    varinf(
        bdm::AbstractBayesDensityModel,
        args...;
        kwargs...
    ) -> AbstractVIPosterior

Compute a variational approximation to the posterior distribution.

The positional arguments and keyword arguments supported by this function, as well as the type of the returned variational posterior object differs between different subtypes of [`AbstractBayesDensityModel`](@ref).
"""
function varinf(::AbstractBayesDensityModel) end

"""
    AbstractVIPosterior

Abstract super type representing the variational posterior distribution of `AbstractBayesDensityModel`
"""
abstract type AbstractVIPosterior end

"""
    sample(
        [rng::Random.AbstractRNG],
        vip::AbstractVIPosterior,
        n_samples::Int,
    ) -> PosteriorSamples

Generate `n_samples` i.i.d. samples from the variationonal posterior distribution `vip`.

# Examples
```julia
julia> x = (1.0 .- (1.0 .- LinRange(0.0, 1.0, 5000)) .^(1/3)).^(1/3);

julia> vip = varinf(BSMModel(x));

julia> vps = sample(Random.Xoshiro(1812), model, 5000)
PosteriorSamples{Float64} object holding 5000 posterior samples, of which 1000 are burn-in samples.
Model:
200-dimensional BSMModel{Float64}:
Using 5000 binned observations on a regular grid consisting of 1187 bins.
 basis:  200-element BSplineBasis of order 4, domain [-0.05, 1.05]
 order:  4
 knots:  [-0.05, -0.05, -0.05, -0.05, -0.0444162, -0.0388325, -0.0332487, -0.027665, -0.0220812, -0.0164975  …  1.0165, 1.02208, 1.02766, 1.03325, 1.03883, 1.04442, 1.05, 1.05, 1.05, 1.05]
```

"""
function StatsBase.sample(::AbstractRNG, ::AbstractVIPosterior, ::Int) end # NB! Remember to set the number of burn-in samples to zero when implementing this function!
StatsBase.sample(vip::AbstractVIPosterior, n_samples::Int) = sample(Random.default_rng(), vip, n_samples)

"""
    quantile(
        [rng::Random.AbstractRNG],
        vip::AbstractVIPosterior,
        t::Union{Real, AbstractVector{<:Real}},
        q::Union{Real, AbstractVector{<:Real}},
        [n_samples::Int=1000]
    ) -> Union{Real, Vector{<:Real}, Matrix{<:Real}}

Compute the posterior `q`-quantiles of `f(t)` for each element in the collection `t`.

By default this function falls back to `quantile(sample(rng, vip, n_samples), t, q)`

In the case where both `t` and `q` are scalars, the output is a real number.
When `t` is a vector and `q` a scalar, this function returns a vector of the same length as `t`.
If `q` is supplied as a Vector, then this function returns a Matrix of dimension `(length(t), length(q))`, where each column corresponds to a given quantile. This is also the case when `t` is supplied as a scalar.
"""
Distributions.quantile(rng::AbstractRNG, vip::AbstractVIPosterior, t::Union{Real, <:AbstractVector{<:Real}}, q::Union{Real, <:AbstractVector{<:Real}}, n_samples::Int=1000) = quantile(sample(rng, vip, n_samples), t, q)
Distributions.quantile(vip::AbstractVIPosterior, t::Union{Real, <:AbstractVector{<:Real}}, q::Union{Real, <:AbstractVector{<:Real}}, n_samples::Int=1000) = quantile(sample(vip, n_samples), t, q)

"""
    median(
        [rng::Random.AbstractRNG],
        vip::AbstractVIPosterior,
        t::Union{Real, AbstractVector{<:Real}},
        [n_samples::Int=1000]
    ) -> Union{Real, Vector{<:Real}}

Compute the posterior median of ``f(t)`` for each element in the collection `t`.

By default this function falls back to `median(sample(rng, vip, n_samples), t)`

In the case where both `t` and `q` are scalars, the output is a real number.
When `t` is a vector, this function returns a vector of the same length as `t`.
"""
Distributions.median(rng::AbstractRNG, vip::AbstractVIPosterior, t::Union{Real, <:AbstractVector{<:Real}}, n_samples::Int=1000) = quantile(rng, vip, t, 0.5, n_samples)
Distributions.median(vip::AbstractVIPosterior, t::Union{Real, <:AbstractVector{<:Real}}, n_samples::Int=1000) = quantile(vip, t, 0.5, n_samples)

"""
    mean(
        [rng::Random.AbstractRNG],
        vip::AbstractVIPosterior,
        t::Union{Real, AbstractVector{<:Real}},
        [n_samples::Int=1000]
    ) -> Union{Real, Vector{<:Real}}

Compute the approximate posterior mean of ``f(t)`` for every element in the collection `t` using Monte Carlo samples.

If the input `t` is a scalar, a scalar is returned. If `t` is a vector, this function returns a vector the same length as `t`.
"""
Distributions.mean(rng::AbstractRNG, vip::AbstractVIPosterior, t::Union{Real, <:AbstractVector{<:Real}}, n_samples::Int=1000) = mean(sample(rng, vip, n_samples), t)
Distributions.mean(vip::AbstractVIPosterior, t::Union{Real, <:AbstractVector{<:Real}}, n_samples::Int=1000) = mean(sample(vip, n_samples), t)

"""
    var(
        [rng::Random.AbstractRNG],
        vip::AbstractVIPosterior,
        t::Union{Real, AbstractVector{<:Real}},
        [n_samples::Int=1000]
    ) -> Union{Real, Vector{<:Real}}

Compute the approximate posterior variance of ``f(t)`` for every element in the collection `t` using Monte Carlo samples.

If the input `t` is a scalar, a scalar is returned. If `t` is a vector, this function returns a vector the same length as `t`.
"""
Distributions.var(rng::AbstractRNG, vip::AbstractVIPosterior, t::Union{Real, <:AbstractVector{<:Real}}, n_samples::Int=1000) = var(sample(rng, vip, n_samples), t)
Distributions.var(vip::AbstractVIPosterior, t::Union{Real, <:AbstractVector{<:Real}}, n_samples::Int=1000) = var(sample(vip, n_samples), t)

"""
    std(
        [rng::Random.AbstractRNG],
        vip::AbstractVIPosterior,
        t::Union{Real, AbstractVector{<:Real}},
        [n_samples::Int=1000]
    ) -> Union{Real, Vector{<:Real}}

Compute the approximate posterior standard deviation of ``f(t)`` for every element in the collection `t` using Monte Carlo samples.

If the input `t` is a scalar, a scalar is returned. If `t` is a vector, this function returns a vector the same length as `t`.
"""
Distributions.std(rng::AbstractRNG, vip::AbstractVIPosterior, t::Union{Real, <:AbstractVector{<:Real}}, n_samples::Int=1000) = sqrt.(var(rng, vip, t, n_samples))
Distributions.std(vip::AbstractVIPosterior, t::Union{Real, <:AbstractVector{<:Real}}, n_samples::Int=1000) = sqrt.(var(vip, t, n_samples))
