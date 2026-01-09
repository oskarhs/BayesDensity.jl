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

Base.eltype(::AbstractBayesDensityModel{T}) where {T} = T

"""
    hyperparams(bdm::AbstractBayesDensityModel) -> @NamedTuple

Return the hyperparameters of the model `bdm` as a `NamedTuple`.
"""
function hyperparams(::AbstractBayesDensityModel) end

export hyperparams

"""
    support(bdm::AbstractBayesDensityModel) -> NTuple{2, <:Real}

Return the support of the model `bdm` as an 2-dimensional tuple.
"""
function support(::AbstractBayesDensityModel) end

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

Evaluate f(t | η) of the Bayesian density model `bdm` for every η in `parameters` and every element in the collection `t`.

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

Evaluate the cumulative distribution function ``F(t\\,|\\, \\eta) = \\int_{-\\infty}^t f(s\\,|\\,\\eta)\\,\\text{d}s`` of the Bayesian density model `bdm` for every ``\\eta`` in `parameters` and every element in the collection `t`.

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

include("utils.jl")
public softplus, sigmoid, logit, softmax, logistic_stickbreaking, countint, bin_regular, unitvector

include("monte_carlo.jl")
export PosteriorSamples, sample, quantile, mean, median, var, std, model

include("variational.jl")
export AbstractVIPosterior, varinf

end # module
