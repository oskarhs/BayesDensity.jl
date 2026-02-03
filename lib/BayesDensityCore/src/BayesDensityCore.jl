module BayesDensityCore

using Distributions
using StatsBase
using Random

import Distributions: support

export linebandplot
export linebandplot!

function linebandplot end
function linebandplot! end

"""
    AbstractBayesDensityModel{T<:Real}

Abstract super type for all Bayesian density models implemented in this package.
"""
abstract type AbstractBayesDensityModel{T<:Real} end

export AbstractBayesDensityModel

"""
    eltype(::AbstractBayesDensityModel{T}) where {T}

Return the element type of a Bayesian density model.
"""
Base.eltype(::AbstractBayesDensityModel{T}) where {T} = T

"""
    hyperparams(bdm::AbstractBayesDensityModel) -> @NamedTuple

Return the hyperparameters of the model `bdm` as a `NamedTuple`.
"""
function hyperparams(::AbstractBayesDensityModel) end

export hyperparams

"""
    support(bdm::AbstractBayesDensityModel{T}) where {T} -> NTuple{2, T}

Return the support of the model `bdm` as an 2-dimensional tuple.
"""
function Distributions.support(::AbstractBayesDensityModel{T}) where {T} end

export support

"""
    pdf(
        bdm::AbstractBayesDensityModel,
        parameters::NamedTuple,
        t::Union{Real, AbstractVector{<:Real}}
    ) -> Union{Real, Vector{<:Real}}
    
    pdf(
        bdm::AbstractBayesDensityModel,
        parameters::AbstractVector{<:NamedTuple},
        t::Union{Real, AbstractVector{<:Real}}
    ) -> Matrix{<:Real}

Evaluate ``f(t \\,|\\,\\boldsymbol{\\eta})`` of the Bayesian density model `bdm` for every ``\\boldsymbol{\\eta}`` in `parameters` and every element in the collection `t`.

If a single NamedTuple is passed to the parameters argument, this function outputs either a scalar or a vector depending on the input type of the third argument `t`.

If a vector of NamedTuples is passed to the second positional argument, then this function returns a Matrix of size `(length(t), length(parameters))`.
"""
function Distributions.pdf(::AbstractBayesDensityModel, ::Any, ::Real) end

export pdf

"""
    cdf(
        bdm::AbstractBayesDensityModel,
        parameters::NamedTuple,
        t::Union{Real, AbstractVector{<:Real}}
    ) -> Union{Real, Vector{<:Real}}
    
    cdf(
        bdm::AbstractBayesDensityModel,
        parameters::AbstractVector{<:NamedTuple},
        t::Union{Real, AbstractVector{<:Real}}
    ) -> Matrix{<:Real}

Evaluate the cumulative distribution function ``F(t\\,|\\, \\boldsymbol{\\eta}) = \\int_{-\\infty}^t f(s\\,|\\,\\boldsymbol{\\eta})\\,\\text{d}s`` of the Bayesian density model `bdm` for every ``\\boldsymbol{\\eta}`` in `parameters` and every element in the collection `t`.

If a single NamedTuple is passed to the parameters argument, this function outputs either a scalar or a vector depending on the input type of the third argument `t`.

If a vector of NamedTuples is passed to the second positional argument, then this function returns a Matrix of size `(length(t), length(parameters))`.
"""
function Distributions.cdf(::AbstractBayesDensityModel, ::Any, ::Real) end

export cdf

# Generate fallback methods for pdf and cdf:
for func in (:pdf, :cdf)
    @eval begin
        function Distributions.$func(bdm::AbstractBayesDensityModel{S}, parameters::NamedTuple, t::AbstractVector{T}) where {S<:Real, T<:Real}
            f_samp = Vector{promote_type(T, S)}(undef, length(t))
            for i in eachindex(t)
                f_samp[i] = Distributions.$func(bdm, parameters, t[i])
            end
            return f_samp
        end
    end

    @eval begin
        function Distributions.$func(bdm::AbstractBayesDensityModel{S}, parameters::AbstractVector{<:NamedTuple}, t::T) where {S<:Real, T<:Real}
            f_samp = Matrix{promote_type(T, S)}(undef, (length(t), length(parameters)))
            for j in eachindex(parameters)
                f_samp[:, j] .= Distributions.$func(bdm, parameters[j], t)
            end
            return f_samp
        end
    end

    @eval begin
        function Distributions.$func(bdm::AbstractBayesDensityModel{S}, parameters::AbstractVector{<:NamedTuple}, t::AbstractVector{T}) where {S<:Real, T<:Real}
            f_samp = Matrix{promote_type(T, S)}(undef, (length(t), length(parameters)))
            for j in eachindex(parameters)
                f_samp[:, j] .= Distributions.$func(bdm, parameters[j], t)
            end
            return f_samp
        end
    end
end

include("monte_carlo.jl")
export PosteriorSamples, sample, quantile, mean, median, var, std, model, n_burnin, drop_burnin, samples, n_samples

include("variational.jl")
export AbstractVIPosterior, varinf
export VariationalOptimizationResult, n_iter, elbo, tolerance, converged, posterior

"""
    default_grid_points(bdm::AbstractBayesDensityModel{T}) where {T} -> AbstractVector{T}

Get the default grid used for plotting of density estimates.

Defaults to returning constructing a grid based on the extrema of `bdm.data.x`.
If a given struct does not store a copy of the original data used to construct the model object as `bdm.data.x`, this method should be implemented.
"""
function default_grid_points(bdm::AbstractBayesDensityModel{T}) where {T}
    xmin, xmax = extrema(bdm.data.x)
    R = xmax - xmin
    x = LinRange{T}(xmin - 0.05*R, xmax + 0.05*R, 2001)
    return x
end

export default_grid_points

end # module
