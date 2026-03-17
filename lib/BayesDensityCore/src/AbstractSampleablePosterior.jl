"""
    AbstractSampleablePosterior{T}

Abstract super type for posterior approximations from which samples can be drawn.
Subtypes currently include [`AbstractVIPosterior`](@erf) and [`AbstractLaplacePosterior`](@ref).
"""
abstract type AbstractSampleablePosterior{T<:Real} end

"""
    eltype(::AbstractSampleablePosterior{T}) where {T}

Get the element type of a variational posterior object.
"""
Base.eltype(::AbstractSampleablePosterior{T}) where {T} = T

"""
    sample(
        [rng::Random.AbstractRNG],
        app::AbstractSampleablePosterior{T},
        n_samples::Int
    ) where {T} -> PosteriorSamples{T}

Generate `n_samples` independent samples from the variationonal posterior distribution `app`.

# Examples
```julia-repl
julia> x = (1.0 .- (1.0 .- LinRange(0, 1, 5001)) .^(1/3)).^(1/3);

julia> app = varinf(BSplineMixture(x));

julia> vps = sample(Random.Xoshiro(1812), app, 5000);
```
"""
StatsBase.sample(lap::AbstractSampleablePosterior, n_samples::Int) = sample(Random.default_rng(), lap, n_samples)

"""
    model(app::AbstractSampleablePosterior{T}) where {T} -> AbstractBayesDensityModel{T}

Get the model object to which the variational posterior `app` was fitted.
"""
function model(::AbstractSampleablePosterior) end

"""
    quantile(
        [rng::Random.AbstractRNG],
        app::AbstractSampleablePosterior,
        [func = pdf],
        t::Union{Real, AbstractVector{<:Real}},
        q::Union{Real, AbstractVector{<:Real}},
        [n_samples::Int=1000]
    ) -> Union{Real, Vector{<:Real}}

    quantile(
        [rng::Random.AbstractRNG],
        app::AbstractSampleablePosterior,
        [func = pdf],
        t::Union{Real, AbstractVector{<:Real}},
        q::AbstractVector{<:Real},
        [n_samples::Int=1000]
    ) -> Matrix{<:Real}

Compute the posterior `q`-quantile(s) of the pdf ``f(t)`` or the cdf ``F(t)`` for each element in the collection `t`.

By default this function falls back to `quantile(sample(rng, app, n_samples), func, t, q)`

In the case where both `t` and `q` are scalars, the output is a real number.
When `t` is a vector and `q` a scalar, this function returns a vector of the same length as `t`.
If `q` is supplied as a Vector, then this function returns a Matrix of dimension `(length(t), length(q))`, where each column corresponds to a given quantile. This is also the case when `t` is supplied as a scalar.

# Examples
```julia-repl
julia> x = (1.0 .- (1.0 .- LinRange(0, 1, 5001)) .^(1/3)).^(1/3);

julia> app = varinf(BSplineMixture(x));

julia> quantile(Random.Xoshiro(1), app, 0.9, 0.5)
0.537450082172813

julia> quantile(Random.Xoshiro(1), app, [0.2, 0.8], [0.05, 0.95])
2×2 Matrix{Float64}:
 0.34042  0.362478
 1.3039   1.43599
```
"""
Distributions.quantile(app::AbstractSampleablePosterior) = throw(MethodError(quantile, (app)))

# We can just use the methods for PosteriorSamples as a fallback here.
for func in (:pdf, :cdf)
    @eval begin
        Distributions.quantile(app::AbstractSampleablePosterior, ::typeof($func), t::Union{Real, <:AbstractVector{<:Real}}, q::Union{Real, <:AbstractVector{<:Real}}, n_samples::Int=1000) = quantile(sample(app, n_samples), $func, t, q)
        Distributions.quantile(rng::AbstractRNG, app::AbstractSampleablePosterior, ::typeof($func), t::Union{Real, <:AbstractVector{<:Real}}, q::Union{Real, <:AbstractVector{<:Real}}, n_samples::Int=1000) = quantile(sample(rng, app, n_samples), $func, t, q)
    end
end
# Make pdf the default
Distributions.quantile(app::AbstractSampleablePosterior, t::Union{Real, <:AbstractVector{<:Real}}, q::Union{Real, <:AbstractVector{<:Real}}, n_samples::Int=1000) = quantile(sample(app, n_samples), t, q)
Distributions.quantile(rng::AbstractRNG, app::AbstractSampleablePosterior, t::Union{Real, <:AbstractVector{<:Real}}, q::Union{Real, <:AbstractVector{<:Real}}, n_samples::Int=1000) = quantile(sample(rng, app, n_samples), t, q)


"""
    median(
        [rng::Random.AbstractRNG],
        app::AbstractSampleablePosterior,
        [func = pdf],
        t::Union{Real, AbstractVector{<:Real}},
        [n_samples::Int=1000]
    ) -> Union{Real, Vector{<:Real}}

Compute the posterior median of the pdf ``f(t)`` or the cdf ``F(t)`` for each element in the collection `t`.

Equivalent to `quantile(rng, app, t, 0.5, n_samples)`.

In the case where both `t` and `q` are scalars, the output is a real number.
When `t` is a vector, this function returns a vector of the same length as `t`.
"""
Distributions.median(app::AbstractSampleablePosterior) = throw(MethodError(median, (app)))

"""
    mean(
        [rng::Random.AbstractRNG],
        app::AbstractSampleablePosterior,
        [func = pdf],
        t::Union{Real, AbstractVector{<:Real}},
        [n_samples::Int=1000]
    ) -> Union{Real, Vector{<:Real}}

Compute the approximate posterior mean of ``f(t)`` for every element in the collection `t` using Monte Carlo samples.

If the input `t` is a scalar, a scalar is returned. If `t` is a vector, this function returns a vector the same length as `t`.

# Examples
```julia-repl
julia> x = (1.0 .- (1.0 .- LinRange(0, 1, 5001)) .^(1/3)).^(1/3);

julia> app = varinf(BSplineMixture(x));

julia> mean(Random.Xoshiro(1), app, 0.1)
0.08615412808594237

julia> mean(Random.Xoshiro(1), app, [0.1, 0.8])
2-element Vector{Float64}:
 0.08615412808594237
 1.3674886390998342
```
"""
Distributions.mean(app::AbstractSampleablePosterior) = throw(MethodError(mean, (app)))

"""
    var(
        [rng::Random.AbstractRNG],
        app::AbstractSampleablePosterior,
        [func = pdf],
        t::Union{Real, AbstractVector{<:Real}},
        [n_samples::Int=1000]
    ) -> Union{Real, Vector{<:Real}}

Compute the posterior variance of the pdf ``f(t)`` or the cdf ``F(t)`` for each element in the collection `t`.

If the input `t` is a scalar, a scalar is returned. If `t` is a vector, this function returns a vector the same length as `t`.

# Examples
```julia-repl
julia> x = (1.0 .- (1.0 .- LinRange(0, 1, 5001)) .^(1/3)).^(1/3);

julia> app = varinf(BSplineMixture(x));

julia> var(Random.Xoshiro(1), app, 0.1)
3.2895012264929507e-6

julia> var(Random.Xoshiro(1), app, [0.1, 0.8])
2-element Vector{Float64}:
 3.2895012264929507e-6
 0.0014793006688108977
```
"""
Distributions.var(app::AbstractSampleablePosterior) = throw(MethodError(var, (app)))

"""
    std(
        [rng::Random.AbstractRNG],
        app::AbstractSampleablePosterior,
        [func = pdf],
        t::Union{Real, AbstractVector{<:Real}},
        [n_samples::Int=1000]
    ) -> Union{Real, Vector{<:Real}}

Compute the posterior standard deviation of the pdf ``f(t)`` or the cdf ``F(t)`` for each element in the collection `t`.

If the input `t` is a scalar, a scalar is returned. If `t` is a vector, this function returns a vector the same length as `t`.
This method is equivalent to `sqrt.(var(rng, app, t, n_samples))`.
"""
Distributions.std(app::AbstractSampleablePosterior) = throw(MethodError(std, (app)))

# Just reuse the methods defined for PosteriorSamples.
for statistic in (:median, :mean, :var, :std)
    for func in (:pdf, :cdf)
        @eval begin
            Distributions.$statistic(app::AbstractSampleablePosterior, ::typeof($func), t::Union{Real, <:AbstractVector{<:Real}}, n_samples::Int=1000) = $statistic(sample(app, n_samples), $func, t)
            Distributions.$statistic(rng::AbstractRNG, app::AbstractSampleablePosterior, ::typeof($func), t::Union{Real, <:AbstractVector{<:Real}}, n_samples::Int=1000) = $statistic(sample(rng, app, n_samples), $func, t)
        end
    end

    # Make pdf the default
    @eval begin
        Distributions.$statistic(app::AbstractSampleablePosterior, t::Union{Real, <:AbstractVector{<:Real}}, n_samples::Int=1000) = $statistic(sample(app, n_samples), pdf, t)
        Distributions.$statistic(rng::AbstractRNG, app::AbstractSampleablePosterior, t::Union{Real, <:AbstractVector{<:Real}}, n_samples::Int=1000) = $statistic(sample(rng, app, n_samples), pdf, t)
    end
end