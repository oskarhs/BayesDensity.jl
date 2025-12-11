"""
    BayesianDensitySamples{T}

Struct holding posterior samples of the parameters of a Bayesian density model.

# Fields
* `samples`: Vector holding posterior samples of model parameters.
* `model`: The model object to which samples were fit.
* `n_samples`: Total number of Monte Carlo samples. 
* `n_burnin`: Number of burn-in samples.
"""
struct BayesianDensitySamples{T<:Real, M<:AbstractBayesianDensityModel, V<:AbstractVector}
    samples::V
    model::M
    n_samples::Int
    n_burnin::Int
    function BayesianDensitySamples{T}(samples::V, model::M, n_samples::Int, n_burnin::Int) where {T, M, V}
        return new{T, M, V}(samples, model, n_samples, n_burnin) 
    end
end

"""
    model(bdc::BayesianDensitySamples)

Return the model object of `bdc`.
"""
model(bdc::BayesianDensitySamples) = bdc.model

Base.eltype(::BayesianDensitySamples{T,M,V}) where {T, M, V} = T

function Base.show(io::IO, ::MIME"text/plain", bdc::BayesianDensitySamples{T, M, V}) where {T, M, V}
    println(io, "BayesianDensitySamples{", T, "} object holding ", bdc.n_samples, " posterior samples, of which ", bdc.n_burnin, " are burn-in samples.")
    println(io, "Model:")
    println(io, model(bdc))
    nothing
end

Base.show(io::IO, bsm::BayesianDensitySamples) = show(io, MIME("text/plain"), bsm)


"""
    sample(
        [rng::Random.AbstractRNG],
        bdm::AbstractBayesianDensityModel,
        n_samples::Int;
        n_burnin::Int=min(1_000, div(n_samples, 5))
    ) -> AbstractBayesianDensitySamples

Generate approximate posterior samples from the density model `bdm` using Markov chain Monte Carlo methods.

TODO: make the docs here more elaborate, in particular with examples.
"""
function StatsBase.sample(::AbstractRNG, ::AbstractBayesianDensityModel, ::Int; n_burnin::Int=min(1_000, div(n_samples, 5))) end

"""
    quantile(
        bdc::BayesianDensitySamples, t::AbstractVector{<:Real}, q::Real
    ) -> Vector{<:Real}

    quantile(
        bdc::BayesianDensitySamples, t::AbstractVector{<:Real}, q::AbstractVector{<:Real}
    ) -> Matrix{<:Real}

Compute the approximate posterior quantiles of f(t) for every element in the collection `t` using Monte Carlo samples.

In the case where both `t` and `q` are scalars, the output is a real number.
When `t` is a vector and `q` a scalar, this functions returns a vector of the same length as `t`.
If `q` is supplied as a Vector, then this function returns a Matrix of dimension `(length(t), length(q))`, where each column corresponds to a given quantile.
"""
function Distributions.quantile(bdc::BayesianDensitySamples, t, q::Real)
    if !(0 ≤ q ≤ 1)
        throws(DomainError("Requested quantile level is not in [0,1]."))
    end
    f_samp = pdf(bdc.model, bdc.samples[bdc.n_burnin+1:end], t)

    return mapslices(x -> quantile(x, q), f_samp; dims=2)[:]
end

function Distributions.quantile(bdc::BayesianDensitySamples, t, q::AbstractVector{<:Real})
    if !all(0 .≤ q .≤ 1)
        throw(DomainError("All requested quantile levels must lie in the interval [0,1]."))
    end
    f_samp = pdf(bdc.model, bdc.samples[bdc.n_burnin+1:end], t)
    
    result = mapslices(x -> quantile(x, q), f_samp; dims=2)
    return result  # shape: (length(t), length(q))
end
function Distributions.quantile(bdc::BayesianDensitySamples, t::Real, q::Real)
    if !(0 ≤ q ≤ 1)
        throws(DomainError("Requested quantile level is not in [0,1]."))
    end
    f_samp = pdf(bdc.model, bdc.samples[bdc.n_burnin+1:end], t)

    return mapslices(x -> quantile(x, q), f_samp; dims=2)[1]
end

"""
    median(bdc::BayesianDensitySamples, t)

Compute the approximate posterior median of f(t) for every element in the collection `t` using Monte Carlo samples.
"""
Distributions.median(bdc::BayesianDensitySamples, t) = quantile(bdc, t, 0.5)

"""
    mean(bdc::BayesianDensitySamples, t)

Compute the approximate posterior mean of f(t) for every element in the collection `t` using Monte Carlo samples.
"""
function Distributions.mean(bdc::BayesianDensitySamples, t::AbstractVector{<:Real})
    f_samp = pdf(model(bdc), bdc.samples[bdc.n_burnin+1:end], t)
    
    result = mapslices(x -> mean(x), f_samp; dims=2)[:]
    return result
end
Base.Broadcast.broadcasted(::typeof(mean), bdc::BayesianDensitySamples, t::AbstractVector{<:Real}) = Distributions.mean(bdc, t)
function Distributions.mean(bdc::BayesianDensitySamples, t::Real)
    f_samp = pdf(model(bdc), bdc.samples[bdc.n_burnin+1:end], t)
    
    result = mapslices(x -> mean(x), f_samp; dims=2)[1]
    return result
end

"""
    var(bdc::BayesianDensitySamples, t)

Compute the approximate posterior variance of f(t) for every element in the collection `t` using Monte Carlo samples.
"""
function Distributions.var(bdc::BayesianDensitySamples, t::AbstractVector{<:Real})
    f_samp = pdf(model(bdc), bdc.samples[bdc.n_burnin+1:end], t)
    
    result = mapslices(x -> var(x), f_samp; dims=2)[:]
    return result
end
Base.Broadcast.broadcasted(::typeof(var), bdc::BayesianDensitySamples, t::AbstractVector{<:Real}) = Distributions.var(bdc, t)
function Distributions.var(bdc::BayesianDensitySamples, t::Real)
    f_samp = pdf(model(bdc), bdc.samples[bdc.n_burnin+1:end], t)
    
    result = mapslices(x -> var(x), f_samp; dims=2)[1]
    return result
end

"""
    std(bdc::BayesianDensitySamples, t)

Compute the approximate posterior standard deviation of f(t) for every element in the collection `t` using Monte Carlo samples.
"""
Distributions.std(bdc::BayesianDensitySamples, t::Union{<:Real, <:AbstractVector{<:Real}}) = sqrt.(var(bdc, t))
Base.Broadcast.broadcasted(::typeof(std), bdc::BayesianDensitySamples, t::AbstractVector{<:Real}) = Distributions.std(bdc, t)
