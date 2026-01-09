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
    AbstractVIPosterior{T<:Real}

Abstract super type representing the variational posterior distribution of `AbstractBayesDensityModel`
"""
abstract type AbstractVIPosterior{T<:Real} end

"""
    Base.eltype(::AbstractVIPosterior{T}) where {T}

Get the element type of a variational posterior object.
"""
Base.eltype(::AbstractVIPosterior{T}) where {T} = T

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

julia> vip = varinf(BSplineMixture(x));

julia> vps = sample(Random.Xoshiro(1812), vip, 5000);
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
        [func = ::typeof(pdf)],
        t::Union{Real, AbstractVector{<:Real}},
        q::Union{Real, AbstractVector{<:Real}},
        [n_samples::Int=1000]
    ) -> Union{Real, Vector{<:Real}}

    quantile(
        [rng::Random.AbstractRNG],
        vip::AbstractVIPosterior,
        [func::Union{::typeof(pdf), ::typeof(cdf)} = ::typeof(pdf)],
        t::Union{Real, AbstractVector{<:Real}},
        q::AbstractVector{<:Real},
        [n_samples::Int=1000]
    ) -> Matrix{<:Real}

Compute the posterior `q`-quantile(s) of the pdf ``f(t)`` or the cdf ``F(t)`` for each element in the collection `t`.

By default this function falls back to `quantile(sample(rng, vip, n_samples), func, t, q)`

In the case where both `t` and `q` are scalars, the output is a real number.
When `t` is a vector and `q` a scalar, this function returns a vector of the same length as `t`.
If `q` is supplied as a Vector, then this function returns a Matrix of dimension `(length(t), length(q))`, where each column corresponds to a given quantile. This is also the case when `t` is supplied as a scalar.

# Examples
```julia-repl
julia> x = (1.0 .- (1.0 .- LinRange(0, 1, 5001)) .^(1/3)).^(1/3);

julia> vip = varinf(BSplineMixture(x));

julia> quantile(Random.Xoshiro(1), vip, 0.9, 0.5)
0.537450082172813

julia> quantile(Random.Xoshiro(1), vip, [0.2, 0.8], [0.05, 0.95])
2×2 Matrix{Float64}:
 0.34042  0.362478
 1.3039   1.43599
```
"""
Distributions.quantile(vip::AbstractVIPosterior) = throw(MethodError(quantile, (vip)))

# We can just use the methods for PosteriorSamples as a fallback here.
for func in (:pdf, :cdf)
    @eval begin
        Distributions.quantile(vip::AbstractVIPosterior, ::typeof($func), t::Union{Real, <:AbstractVector{<:Real}}, q::Union{Real, <:AbstractVector{<:Real}}, n_samples::Int=1000) = quantile(sample(vip, n_samples), $func, t, q)
        Distributions.quantile(rng::AbstractRNG, vip::AbstractVIPosterior, ::typeof($func), t::Union{Real, <:AbstractVector{<:Real}}, q::Union{Real, <:AbstractVector{<:Real}}, n_samples::Int=1000) = quantile(sample(rng, vip, n_samples), $func, t, q)
    end
end
# Make pdf the default
Distributions.quantile(vip::AbstractVIPosterior, t::Union{Real, <:AbstractVector{<:Real}}, q::Union{Real, <:AbstractVector{<:Real}}, n_samples::Int=1000) = quantile(sample(vip, n_samples), t, q)
Distributions.quantile(rng::AbstractRNG, vip::AbstractVIPosterior, t::Union{Real, <:AbstractVector{<:Real}}, q::Union{Real, <:AbstractVector{<:Real}}, n_samples::Int=1000) = quantile(sample(rng, vip, n_samples), t, q)


"""
    median(
        [rng::Random.AbstractRNG],
        vip::AbstractVIPosterior,
        [func = ::typeof(pdf)],
        t::Union{Real, AbstractVector{<:Real}},
        [n_samples::Int=1000]
    ) -> Union{Real, Vector{<:Real}}

Compute the posterior median of the pdf ``f(t)`` or the cdf ``F(t)`` for each element in the collection `t`.

Equivalent to `quantile(rng, vip, t, 0.5, n_samples)`.

In the case where both `t` and `q` are scalars, the output is a real number.
When `t` is a vector, this function returns a vector of the same length as `t`.
"""
Distributions.median(vip::AbstractVIPosterior) = throw(MethodError(median, (vip)))

"""
    mean(
        [rng::Random.AbstractRNG],
        vip::AbstractVIPosterior,
        [func = ::typeof(pdf)],
        t::Union{Real, AbstractVector{<:Real}},
        [n_samples::Int=1000]
    ) -> Union{Real, Vector{<:Real}}

Compute the approximate posterior mean of ``f(t)`` for every element in the collection `t` using Monte Carlo samples.

If the input `t` is a scalar, a scalar is returned. If `t` is a vector, this function returns a vector the same length as `t`.

# Examples
```julia-repl
julia> x = (1.0 .- (1.0 .- LinRange(0, 1, 5001)) .^(1/3)).^(1/3);

julia> vip = varinf(BSplineMixture(x));

julia> mean(Random.Xoshiro(1), vip, 0.1)
0.08615412808594237

julia> mean(Random.Xoshiro(1), vip, [0.1, 0.8])
2-element Vector{Float64}:
 0.08615412808594237
 1.3674886390998342
```
"""
Distributions.mean(vip::AbstractVIPosterior) = throw(MethodError(mean, (vip)))

"""
    var(
        [rng::Random.AbstractRNG],
        vip::AbstractVIPosterior,
        [func = ::typeof(pdf)],
        t::Union{Real, AbstractVector{<:Real}},
        [n_samples::Int=1000]
    ) -> Union{Real, Vector{<:Real}}

Compute the posterior variance of the pdf ``f(t)`` or the cdf ``F(t)`` for each element in the collection `t`.

If the input `t` is a scalar, a scalar is returned. If `t` is a vector, this function returns a vector the same length as `t`.

# Examples
```julia-repl
julia> x = (1.0 .- (1.0 .- LinRange(0, 1, 5001)) .^(1/3)).^(1/3);

julia> vip = varinf(BSplineMixture(x));

julia> var(Random.Xoshiro(1), vip, 0.1)
3.2895012264929507e-6

julia> var(Random.Xoshiro(1), vip, [0.1, 0.8])
2-element Vector{Float64}:
 3.2895012264929507e-6
 0.0014793006688108977
```
"""
Distributions.var(vip::AbstractVIPosterior) = throw(MethodError(var, (vip)))

"""
    std(
        [rng::Random.AbstractRNG],
        vip::AbstractVIPosterior,
        [func = ::typeof(pdf)],
        t::Union{Real, AbstractVector{<:Real}},
        [n_samples::Int=1000]
    ) -> Union{Real, Vector{<:Real}}

Compute the posterior standard deviation of the pdf ``f(t)`` or the cdf ``F(t)`` for each element in the collection `t`.

If the input `t` is a scalar, a scalar is returned. If `t` is a vector, this function returns a vector the same length as `t`.
This method is equivalent to `sqrt.(var(rng, vip, t, n_samples))`.
"""
Distributions.std(vip::AbstractVIPosterior) = throw(MethodError(std, (vip)))

# Just reuse the methods defined for PosteriorSamples.
for statistic in (:median, :mean, :var, :std)
    for func in (:pdf, :cdf)
        @eval begin
            Distributions.$statistic(vip::AbstractVIPosterior, ::typeof($func), t::Union{Real, <:AbstractVector{<:Real}}, n_samples::Int=1000) = $statistic(sample(vip, n_samples), $func, t)
            Distributions.$statistic(rng::AbstractRNG, vip::AbstractVIPosterior, ::typeof($func), t::Union{Real, <:AbstractVector{<:Real}}, n_samples::Int=1000) = $statistic(sample(rng, vip, n_samples), $func, t)
        end
    end

    # Make pdf the default
    @eval begin
        Distributions.$statistic(vip::AbstractVIPosterior, t::Union{Real, <:AbstractVector{<:Real}}, n_samples::Int=1000) = $statistic(sample(vip, n_samples), pdf, t)
        Distributions.$statistic(rng::AbstractRNG, vip::AbstractVIPosterior, t::Union{Real, <:AbstractVector{<:Real}}, n_samples::Int=1000) = $statistic(sample(rng, vip, n_samples), pdf, t)
    end
end