"""
    varinf(
        bdm::AbstractBayesDensityModel,
        args...;
        kwargs...
    ) -> AbstractVIPosterior

Compute a variational approximation to the posterior distribution.

The positional arguments and keyword arguments supported by this function, as well as the type of the returned variational posterior object differs between different subtypes of [`AbstractBayesDensityModel`](@ref).

# Examples
```julia-repl
julia> x = (1.0 .- (1.0 .- LinRange(0.0, 1.0, 5001)) .^(1/3)).^(1/3);

julia> vip = varinf(BSMModel(x))
BSMVIPosterior{Float64} vith variational densities:
 q_β <: Distributions.MvNormalCanon{Float64},
 q_τ <: Distributions.InverseGamma{Float64},
 q_δ <: Vector{Distributions.InverseGamma{Float64}}.
Model:
200-dimensional BSMModel{Float64}:
Using 5001 binned observations on a regular grid consisting of 1187 bins.
 basis:  200-element BSplineBasis of order 4, domain [-0.05, 1.05]
 order:  4
 knots:  [-0.05, -0.05, -0.05, -0.05, -0.0444162, -0.0388325, -0.0332487, -0.027665, -0.0220812, -0.0164975  …  1.0165, 1.02208, 1.02766, 1.03325, 1.03883, 1.04442, 1.05, 1.05, 1.05, 1.05]
```
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
        n_samples::Int
    ) -> PosteriorSamples

Generate `n_samples` independent samples from the variationonal posterior distribution `vip`.

# Examples
```julia-repl
julia> x = (1.0 .- (1.0 .- LinRange(0, 1, 5001)) .^(1/3)).^(1/3);

julia> vip = varinf(BSMModel(x));

julia> vps = sample(Random.Xoshiro(1812), vip, 5000)
PosteriorSamples{Float64} object holding 5000 posterior samples, of which 0 are burn-in samples.
Model:
200-dimensional BSMModel{Float64}:
Using 5001 binned observations on a regular grid consisting of 1187 bins.
 basis:  200-element BSplineBasis of order 4, domain [-0.05, 1.05]
 order:  4
 knots:  [-0.05, -0.05, -0.05, -0.05, -0.0444162, -0.0388325, -0.0332487, -0.027665, -0.0220812, -0.0164975  …  1.0165, 1.02208, 1.02766, 1.03325, 1.03883, 1.04442, 1.05, 1.05, 1.05, 1.05]
```
"""
StatsBase.sample(vip::AbstractVIPosterior, n_samples::Int) = sample(Random.default_rng(), vip, n_samples)

"""
    model(vip::AbstractVIPosterior)

Get the model object to which the variational posterior `vip` was fitted.
"""
function model(::AbstractVIPosterior) end

"""
    quantile(
        [rng::Random.AbstractRNG],
        vip::AbstractVIPosterior,
        t::Union{Real, AbstractVector{<:Real}},
        q::Union{Real, AbstractVector{<:Real}},
        [n_samples::Int=1000]
    ) -> Union{Real, Vector{<:Real}, Matrix{<:Real}}

Compute the posterior `q`-quantiles of ``f(t)`` for each element in the collection `t`.

By default this function falls back to `quantile(sample(rng, vip, n_samples), t, q)`

In the case where both `t` and `q` are scalars, the output is a real number.
When `t` is a vector and `q` a scalar, this function returns a vector of the same length as `t`.
If `q` is supplied as a Vector, then this function returns a Matrix of dimension `(length(t), length(q))`, where each column corresponds to a given quantile. This is also the case when `t` is supplied as a scalar.

# Examples
```julia-repl
julia> x = (1.0 .- (1.0 .- LinRange(0, 1, 5001)) .^(1/3)).^(1/3);

julia> vip = varinf(BSMModel(x));

julia> quantile(Random.Xoshiro(1), vip, 0.9, 0.5)
0.537450082172813

julia> quantile(Random.Xoshiro(1), vip, [0.2, 0.8], [0.05, 0.95])
2×2 Matrix{Float64}:
 0.34042  0.362478
 1.3039   1.43599
```
"""
Distributions.quantile(vip::AbstractVIPosterior, t::Union{Real, <:AbstractVector{<:Real}}, q::Union{Real, <:AbstractVector{<:Real}}, n_samples::Int=1000) = quantile(sample(vip, n_samples), t, q)
Distributions.quantile(rng::AbstractRNG, vip::AbstractVIPosterior, t::Union{Real, <:AbstractVector{<:Real}}, q::Union{Real, <:AbstractVector{<:Real}}, n_samples::Int=1000) = quantile(sample(rng, vip, n_samples), t, q)

"""
    median(
        [rng::Random.AbstractRNG],
        vip::AbstractVIPosterior,
        t::Union{Real, AbstractVector{<:Real}},
        [n_samples::Int=1000]
    ) -> Union{Real, Vector{<:Real}}

Compute the posterior median of ``f(t)`` for each element in the collection `t`.

Equivalent to `quantile(rng, vip, t, 0.5, n_samples)`.

In the case where both `t` and `q` are scalars, the output is a real number.
When `t` is a vector, this function returns a vector of the same length as `t`.
"""
Distributions.median(vip::AbstractVIPosterior, t::Union{Real, <:AbstractVector{<:Real}}, n_samples::Int=1000) = quantile(vip, t, 0.5, n_samples)
Distributions.median(rng::AbstractRNG, vip::AbstractVIPosterior, t::Union{Real, <:AbstractVector{<:Real}}, n_samples::Int=1000) = quantile(rng, vip, t, 0.5, n_samples)

"""
    mean(
        [rng::Random.AbstractRNG],
        vip::AbstractVIPosterior,
        t::Union{Real, AbstractVector{<:Real}},
        [n_samples::Int=1000]
    ) -> Union{Real, Vector{<:Real}}

Compute the approximate posterior mean of ``f(t)`` for every element in the collection `t` using Monte Carlo samples.

If the input `t` is a scalar, a scalar is returned. If `t` is a vector, this function returns a vector the same length as `t`.

# Examples
```julia-repl
julia> x = (1.0 .- (1.0 .- LinRange(0, 1, 5001)) .^(1/3)).^(1/3);

julia> vip = varinf(BSMModel(x));

julia> mean(Random.Xoshiro(1), vip, 0.1)
0.08615412808594237

julia> mean(Random.Xoshiro(1), vip, [0.1, 0.8])
2-element Vector{Float64}:
 0.08615412808594237
 1.3674886390998342
```
"""
Distributions.mean(vip::AbstractVIPosterior, t::Union{Real, <:AbstractVector{<:Real}}, n_samples::Int=1000) = mean(sample(vip, n_samples), t)
Distributions.mean(rng::AbstractRNG, vip::AbstractVIPosterior, t::Union{Real, <:AbstractVector{<:Real}}, n_samples::Int=1000) = mean(sample(rng, vip, n_samples), t)

"""
    var(
        [rng::Random.AbstractRNG],
        vip::AbstractVIPosterior,
        t::Union{Real, AbstractVector{<:Real}},
        [n_samples::Int=1000]
    ) -> Union{Real, Vector{<:Real}}

Compute the approximate posterior variance of ``f(t)`` for every element in the collection `t` using Monte Carlo samples.

If the input `t` is a scalar, a scalar is returned. If `t` is a vector, this function returns a vector the same length as `t`.

# Examples
```julia-repl
julia> x = (1.0 .- (1.0 .- LinRange(0, 1, 5001)) .^(1/3)).^(1/3);

julia> vip = varinf(BSMModel(x));

julia> var(Random.Xoshiro(1), vip, 0.1)
3.2895012264929507e-6

julia> var(Random.Xoshiro(1), vip, [0.1, 0.8])
2-element Vector{Float64}:
 3.2895012264929507e-6
 0.0014793006688108977
```
"""
Distributions.var(vip::AbstractVIPosterior, t::Union{Real, <:AbstractVector{<:Real}}, n_samples::Int=1000) = var(sample(vip, n_samples), t)
Distributions.var(rng::AbstractRNG, vip::AbstractVIPosterior, t::Union{Real, <:AbstractVector{<:Real}}, n_samples::Int=1000) = var(sample(rng, vip, n_samples), t)

"""
    std(
        [rng::Random.AbstractRNG],
        vip::AbstractVIPosterior,
        t::Union{Real, AbstractVector{<:Real}},
        [n_samples::Int=1000]
    ) -> Union{Real, Vector{<:Real}}

Compute the approximate posterior standard deviation of ``f(t)`` for every element in the collection `t` using Monte Carlo samples.

If the input `t` is a scalar, a scalar is returned. If `t` is a vector, this function returns a vector the same length as `t`.
This method is equivalent to `sqrt.(var(rng, vip, t, n_samples))`.
"""
Distributions.std(vip::AbstractVIPosterior, t::Union{Real, <:AbstractVector{<:Real}}, n_samples::Int=1000) = sqrt.(var(vip, t, n_samples))
Distributions.std(rng::AbstractRNG, vip::AbstractVIPosterior, t::Union{Real, <:AbstractVector{<:Real}}, n_samples::Int=1000) = sqrt.(var(rng, vip, t, n_samples))