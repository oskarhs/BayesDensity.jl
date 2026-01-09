"""
    PosteriorSamples{T<:Real}

Struct holding posterior samples of the parameters of a Bayesian density model.

# Fields
* `samples`: Vector holding posterior samples of model parameters.
* `model`: The model object to which samples were fit.
* `n_samples`: Total number of Monte Carlo samples. 
* `n_burnin`: Number of burn-in samples.
"""
struct PosteriorSamples{T<:Real, M<:AbstractBayesDensityModel, V<:AbstractVector}
    samples::V
    model::M
    n_samples::Int
    n_burnin::Int
    function PosteriorSamples{T}(samples::V, model::M, n_samples::Int, n_burnin::Int) where {T, M, V}
        return new{T, M, V}(samples, model, n_samples, n_burnin) 
    end
end

"""
    model(ps::PosteriorSamples) -> AbstractBayesDensityModel

Return the model object of `ps`.
"""
model(ps::PosteriorSamples) = ps.model

"""
    Base.eltype(::PosteriorSamples{T}) where {T}

Get the element type of a `PosteriorSamples` object.
"""
Base.eltype(::PosteriorSamples{T,M,V}) where {T, M, V} = T

function Base.show(io::IO, ::MIME"text/plain", ps::PosteriorSamples{T, M, V}) where {T, M, V}
    println(io, "PosteriorSamples{", T, "} object holding ", ps.n_samples, " posterior samples, of which ", ps.n_burnin, " are burn-in samples.")
    println(io, "Model:")
    print(io, model(ps))
    nothing
end

Base.show(io::IO, bsm::PosteriorSamples) = show(io, MIME("text/plain"), bsm)

"""
    sample(
        [rng::Random.AbstractRNG],
        bdm::AbstractBayesDensityModel,
        n_samples::Int;
        n_burnin::Int=min(1_000, div(n_samples, 5))
    ) -> PosteriorSamples

Generate approximate posterior samples from the density model `bdm` using Markov chain Monte Carlo methods.

This functions returns a [`PosteriorSamples`](@ref) object which can be used to compute posterior quantities of interest such as the posterior mean of ``f(t)`` or posterior quantiles.

# Arguments
* `rng`: Seed used for random variate generation.
* `bdm`: The Bayesian density model object to generate posterior samples from.
* `n_samples`: Number of Monte Carlo samples (including burn-in)

# Keyword arguments
* `n_burnin`: Number of burn-in samples.

# Returns
* `ps`: A [`PosteriorSamples`](@ref) object holding the posterior samples and the original model object.
"""
StatsBase.sample(bdm::AbstractBayesDensityModel, args...; kwargs...) = StatsBase.sample(Random.default_rng(), bdm, args...; kwargs...)

"""
    quantile(
        ps::PosteriorSamples,
        [func = ::typeof(pdf)],
        t::Union{Real, AbstractVector{<:Real}},
        q::RealAbstractVector{<:Real,
    ) -> Union{Real, Vector{<:Real}}

    quantile(
        ps::PosteriorSamples,
        [func = ::typeof(pdf)]
        t::Union{Real, AbstractVector{<:Real}},
        q::AbstractVector{<:Real},
    ) -> Matrix{<:Real}

Compute the approximate posterior quantile(s) of a functional of ``f`` for every element in the collection `t` using Monte Carlo samples.

The target functional can be either be the pdf ``f`` or the cdf ``F``, and is controlled by adjusting the `func` argument.
By default, the posterior quantiles of ``f`` are computed.

In the case where both `t` and `q` are scalars, the output is a real number.
When `t` is a vector and `q` a scalar, this function returns a vector of the same length as `t`.
If `q` is supplied as a Vector, then this method returns a Matrix of dimension `(length(t), length(q))`, where each column corresponds to a given quantile. This is also the case when `t` is supplied as a scalar.

# Examples
```julia-repl
julia> x = (1.0 .- (1.0 .- LinRange(0, 1, 5001)) .^(1/3)).^(1/3);

julia> ps = sample(Random.Xoshiro(1), BSplineMixture(x), 5000);

julia> quantile(ps, 0.1, 0.5); # Get the posterior median of f(0.5)

julia> quantile(ps, cdf, 0.1, 0.5); # Get the posterior median of F(0.5)

julia> quantile(ps, [0.1, 0.8], [0.05, 0.95]); # Get the posterior 0.05, 0.95-quantiles of f(0.1) and f(0.8)
```
"""
Distributions.quantile(ps::PosteriorSamples) = throw(MethodError(quantile, (ps)))

# Get posterior samples of pdf/cdf evaluated on a grid/single point.
# Use the samples to compute the desired quantiles.
for func in (:pdf, :cdf)
    @eval begin
        function Distributions.quantile(ps::PosteriorSamples, ::typeof($func), t, q::Real)
            if !(0 ≤ q ≤ 1)
                throws(DomainError("Requested quantile level is not in [0,1]."))
            end
            func_samp = $func(ps.model, ps.samples[ps.n_burnin+1:end], t)

            return mapslices(x -> quantile(x, q), func_samp; dims=2)[:]
        end
        function Distributions.quantile(ps::PosteriorSamples, ::typeof($func), t, q::AbstractVector{<:Real})
            if !all(0 .≤ q .≤ 1)
                throw(DomainError("All requested quantile levels must lie in the interval [0,1]."))
            end
            func_samp = $func(ps.model, ps.samples[ps.n_burnin+1:end], t)
            
            result = mapslices(x -> quantile(x, q), func_samp; dims=2)
            return result  # shape: (length(t), length(q))
        end
        function Distributions.quantile(ps::PosteriorSamples, ::typeof($func), t::Real, q::Real)
            if !(0 ≤ q ≤ 1)
                throws(DomainError("Requested quantile level is not in [0,1]."))
            end
            func_samp = $func(ps.model, ps.samples[ps.n_burnin+1:end], t)

            return mapslices(x -> quantile(x, q), func_samp; dims=2)[1]
        end
    end
end
# Make pdf default:
Distributions.quantile(ps::PosteriorSamples, t::Union{Real, AbstractVector{<:Real}}, q::Union{Real, AbstractVector{<:Real}}) = quantile(ps, pdf, t, q)

"""
    median(
        ps::PosteriorSamples,
        [func = ::typeof(pdf)]
        t::Union{Real, AbstractVector{<:Real}},
    ) -> Union{Real, Vector{<:Real}}

Compute the approximate posterior median of a functional of ``f`` for every element in the collection `t` using Monte Carlo samples.

The target functional can be either be the pdf ``f`` or the cdf ``F``, and is controlled by adjusting the `func` argument.
By default, the posterior median of ``f`` is computed.

If the input `t` is a scalar, a scalar is returned. If `t` is a vector, this function returns a vector the same length as `t`.
"""
Distributions.median(ps::PosteriorSamples) = throw(MethodError(median, (ps)))

"""
    mean(
        ps::PosteriorSamples,
        [func = ::typeof(pdf)],
        t::Union{Real, AbstractVector{<:Real}}
    ) -> Union{Real, Vector{<:Real}}

Compute the approximate posterior mean of a functional of ``f`` for every element in the collection `t` using Monte Carlo samples.

The target functional can be either be the pdf ``f`` or the cdf ``F``, and is controlled by adjusting the `func` argument.
By default, the posterior mean of ``f`` is computed.

If the input `t` is a scalar, a scalar is returned. If `t` is a vector, this function returns a vector the same length as `t`.

# Examples
```julia-repl
julia> x = (1.0 .- (1.0 .- LinRange(0, 1, 5001)) .^(1/3)).^(1/3);

julia> ps = sample(Random.Xoshiro(1), BSplineMixture(x), 5000);

julia> mean(ps, 0.1)
0.0969450407517681

julia> mean(ps, [0.1, 0.8])
2-element Vector{Float64}:
 0.0969450407517681
 1.3662358915400654
```
"""
Distributions.mean(ps::PosteriorSamples) = throw(MethodError(mean, (ps)))

"""
    var(
        ps::PosteriorSamples,
        [func = ::typeof(pdf)],
        t::Union{Real, AbstractVector{<:Real}}
    ) -> Union{Real, Vector{<:Real}}

Compute the approximate posterior variance of a functional of ``f`` for every element in the collection `t` using Monte Carlo samples.

The target functional can be either be the pdf ``f`` or the cdf ``F``, and is controlled by adjusting the `func` argument.
By default, the posterior variance of ``f`` is computed.

If the input `t` is a scalar, a scalar is returned. If `t` is a vector, this function returns a vector the same length as `t`.

# Examples
```julia-repl
julia> x = (1.0 .- (1.0 .- LinRange(0, 1, 5001)) .^(1/3)).^(1/3);

julia> ps = sample(Random.Xoshiro(1), BSplineMixture(x), 5000);

julia> var(ps, 0.1) # get the posterior variance of f(0.1)
0.00027756364767372627

julia> var(ps, [0.1, 0.8]) # get the posterior variance of f(0.1) and f(0.8)
2-element Vector{Float64}:
 0.00027756364767372627
 0.005977674286240125
```
"""
Distributions.var(ps::PosteriorSamples) = throw(MethodError(var, (ps)))

"""
    std(
        ps::PosteriorSamples,
        [func = ::typeof(pdf)],
        t::Union{Real, AbstractVector{<:Real}}
    ) -> Union{Real, Vector{<:Real}}

Compute the approximate posterior standard deviation of a functional of ``f`` for every element in the collection `t` using Monte Carlo samples.

The target functional can be either be the pdf ``f`` or the cdf ``F``, and is controlled by adjusting the `func` argument.
By default, the posterior standard deviation of ``f`` is computed.

If the input `t` is a scalar, a scalar is returned. If `t` is a vector, this function returns a vector the same length as `t`.
This method is equivalent to `sqrt.(var(rng, vip, t, n_samples))`.
"""
Distributions.std(ps::PosteriorSamples) = throw(MethodError(std, (ps)))

# Use the PosteriorSamples object to evaluate func(t) on a grid/single point.
# Then, use these samples to approximate the desired statistic.
for statistic in (:median, :mean, :var, :std)
    for func in (:pdf, :cdf)
        @eval begin
            function Distributions.$statistic(ps::PosteriorSamples, ::typeof($func), t::AbstractVector{<:Real})
                func_samp = $func(model(ps), ps.samples[ps.n_burnin+1:end], t)
                
                result = mapslices(x -> $statistic(x), func_samp; dims=2)[:]
                return result
            end
            function Distributions.$statistic(ps::PosteriorSamples, ::typeof($func), t::Real)
                func_samp = $func(model(ps), ps.samples[ps.n_burnin+1:end], t)
                
                result = mapslices(x -> $statistic(x), func_samp; dims=2)[1]
                return result
            end
        end
    end
    # Make pdf the default:
    @eval begin 
        Distributions.$statistic(ps::PosteriorSamples, t::AbstractVector{<:Real}) = $statistic(ps, pdf, t)
        Distributions.$statistic(ps::PosteriorSamples, t::Real) = $statistic(ps, pdf, t)
    end
end