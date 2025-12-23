"""
    PosteriorSamples{T}

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

Base.eltype(::PosteriorSamples{T,M,V}) where {T, M, V} = T

function Base.show(io::IO, ::MIME"text/plain", ps::PosteriorSamples{T, M, V}) where {T, M, V}
    println(io, "PosteriorSamples{", T, "} object holding ", ps.n_samples, " posterior samples, of which ", ps.n_burnin, " are burn-in samples.")
    println(io, "Model:")
    println(io, model(ps))
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

# Examples
```julia-repl
julia> x = (1.0 .- (1.0 .- LinRange(0, 1, 5001)) .^(1/3)).^(1/3);

julia> ps = sample(Random.Xoshiro(1), BSMModel(x), 5000; n_burnin = 500)
PosteriorSamples{Float64} object holding 5000 posterior samples, of which 500 are burn-in samples.
Model:
200-dimensional BSMModel{Float64}:
Using 5001 binned observations on a regular grid consisting of 1187 bins.
 basis:  200-element BSplineBasis of order 4, domain [-0.05, 1.05]
 order:  4
 knots:  [-0.05, -0.05, -0.05, -0.05, -0.0444162, -0.0388325, -0.0332487, -0.027665, -0.0220812, -0.0164975  …  1.0165, 1.02208, 1.02766, 1.03325, 1.03883, 1.04442, 1.05, 1.05, 1.05, 1.05]
```
"""
function StatsBase.sample(::AbstractBayesDensityModel, ::Int) end # Perhaps the user should have the option to discard burn-in samples?

"""
    quantile(
        ps::PosteriorSamples,
        t::Union{Real, AbstractVector{<:Real}},
        q::Union{Real, AbstractVector{<:Real}},
    ) -> Union{Real, Vector{<:Real}, Matrix{<:Real}}

Compute the approximate posterior quantiles of ``f(t)`` for every element in the collection `t` using Monte Carlo samples.

In the case where both `t` and `q` are scalars, the output is a real number.
When `t` is a vector and `q` a scalar, this functions returns a vector of the same length as `t`.
If `q` is supplied as a Vector, then this function returns a Matrix of dimension `(length(t), length(q))`, where each column corresponds to a given quantile. This is also the case when `t` is supplied as a scalar.

# Examples
```julia-repl
julia> x = (1.0 .- (1.0 .- LinRange(0, 1, 5001)) .^(1/3)).^(1/3);

julia> ps = sample(Random.Xoshiro(1), BSMModel(x), 5000);

julia> quantile(ps, 0.1, 0.5)
0.08936091272819188

julia> quantile(ps, [0.1, 0.8], [0.05, 0.95])
2×2 Matrix{Float64}:
 0.0661715  0.120343
 1.2408     1.49437
```
"""
function Distributions.quantile(ps::PosteriorSamples, t, q::Real)
    if !(0 ≤ q ≤ 1)
        throws(DomainError("Requested quantile level is not in [0,1]."))
    end
    f_samp = pdf(ps.model, ps.samples[ps.n_burnin+1:end], t)

    return mapslices(x -> quantile(x, q), f_samp; dims=2)[:]
end

function Distributions.quantile(ps::PosteriorSamples, t, q::AbstractVector{<:Real})
    if !all(0 .≤ q .≤ 1)
        throw(DomainError("All requested quantile levels must lie in the interval [0,1]."))
    end
    f_samp = pdf(ps.model, ps.samples[ps.n_burnin+1:end], t)
    
    result = mapslices(x -> quantile(x, q), f_samp; dims=2)
    return result  # shape: (length(t), length(q))
end
function Distributions.quantile(ps::PosteriorSamples, t::Real, q::Real)
    if !(0 ≤ q ≤ 1)
        throws(DomainError("Requested quantile level is not in [0,1]."))
    end
    f_samp = pdf(ps.model, ps.samples[ps.n_burnin+1:end], t)

    return mapslices(x -> quantile(x, q), f_samp; dims=2)[1]
end

"""
    median(
        ps::PosteriorSamples,
        t::Union{Real, AbstractVector{<:Real}},
    ) -> Union{Real, Vector{<:Real}}

Compute the approximate posterior median of ``f(t)`` for every element in the collection `t` using Monte Carlo samples.

If the input `t` is a scalar, a scalar is returned. If `t` is a vector, this function returns a vector the same length as `t`.
"""
Distributions.median(ps::PosteriorSamples, t::Union{Real, AbstractVector{<:Real}}) = quantile(ps, t, 0.5)

"""
    mean(
        ps::PosteriorSamples,
        t::Union{Real, AbstractVector{<:Real}}
    ) -> Union{Real, Vector{<:Real}}

Compute the approximate posterior mean of ``f(t)`` for every element in the collection `t` using Monte Carlo samples.

If the input `t` is a scalar, a scalar is returned. If `t` is a vector, this function returns a vector the same length as `t`.

# Examples
```julia-repl
julia> x = (1.0 .- (1.0 .- LinRange(0, 1, 5001)) .^(1/3)).^(1/3);

julia> ps = sample(Random.Xoshiro(1), BSMModel(x), 5000);

julia> mean(ps, 0.1)
0.0969450407517681

julia> mean(ps, [0.1, 0.8])
2-element Vector{Float64}:
 0.0969450407517681
 1.3662358915400654
```
"""
function Distributions.mean(ps::PosteriorSamples, t::AbstractVector{<:Real})
    f_samp = pdf(model(ps), ps.samples[ps.n_burnin+1:end], t)
    
    result = mapslices(x -> mean(x), f_samp; dims=2)[:]
    return result
end
Base.Broadcast.broadcasted(::typeof(mean), ps::PosteriorSamples, t::AbstractVector{<:Real}) = Distributions.mean(ps, t)
function Distributions.mean(ps::PosteriorSamples, t::Real)
    f_samp = pdf(model(ps), ps.samples[ps.n_burnin+1:end], t)
    
    result = mapslices(x -> mean(x), f_samp; dims=2)[1]
    return result
end

"""
    var(
        ps::PosteriorSamples,
        t::Union{Real, AbstractVector{<:Real}}
    ) -> Union{Real, Vector{<:Real}}

Compute the approximate posterior variance of ``f(t)`` for every element in the collection `t` using Monte Carlo samples.

If the input `t` is a scalar, a scalar is returned. If `t` is a vector, this function returns a vector the same length as `t`.

# Examples
```julia-repl
julia> x = (1.0 .- (1.0 .- LinRange(0, 1, 5001)) .^(1/3)).^(1/3);

julia> ps = sample(Random.Xoshiro(1), BSMModel(x), 5000);

julia> var(ps, 0.1)
0.00027756364767372627

julia> var(ps, [0.1, 0.8])
2-element Vector{Float64}:
 0.00027756364767372627
 0.005977674286240125
```
"""
function Distributions.var(ps::PosteriorSamples, t::AbstractVector{<:Real})
    f_samp = pdf(model(ps), ps.samples[ps.n_burnin+1:end], t)
    
    result = mapslices(x -> var(x), f_samp; dims=2)[:]
    return result
end
Base.Broadcast.broadcasted(::typeof(var), ps::PosteriorSamples, t::AbstractVector{<:Real}) = Distributions.var(ps, t)
function Distributions.var(ps::PosteriorSamples, t::Real)
    f_samp = pdf(model(ps), ps.samples[ps.n_burnin+1:end], t)
    
    result = mapslices(x -> var(x), f_samp; dims=2)[1]
    return result
end

"""
    std(
        ps::PosteriorSamples,
        t::Union{Real, AbstractVector{<:Real}}
    ) -> Union{Real, Vector{<:Real}}

Compute the approximate posterior standard deviation of ``f(t)`` for every element in the collection `t` using Monte Carlo samples.

If the input `t` is a scalar, a scalar is returned. If `t` is a vector, this function returns a vector the same length as `t`.
This method is equivalent to `sqrt.(var(rng, vip, t, n_samples))`.
"""
Distributions.std(ps::PosteriorSamples, t::Union{<:Real, <:AbstractVector{<:Real}}) = sqrt.(var(ps, t))
Base.Broadcast.broadcasted(::typeof(std), ps::PosteriorSamples, t::AbstractVector{<:Real}) = Distributions.std(ps, t)
