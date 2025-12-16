abstract type AbstractVIPosterior end

"""
    sample(
        [rng::Random.AbstractRNG],
        vip::AbstractVIPosterior,
        n_samples::Int,
    ) -> PosteriorSamples

Generate `n_samples` i.i.d. samples from the variationonal posterior distribution `vip`.

# TODO: Add examples, more explanation here
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
