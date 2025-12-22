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
    hyperparams(bdm::AbstractBayesDensityModel)

Return the hyperparameters of the model `bdm` as a `NamedTuple`.
"""
function hyperparams(::AbstractBayesDensityModel) end

export hyperparams

"""
    hyperparams(bdm::AbstractBayesDensityModel)

Return the support of the model `bdm` as a tuple.
"""
function support(::AbstractBayesDensityModel) end

export support

"""
    pdf(bdm::AbstractBayesDensityModel, parameters, t)
    pdf(bdm::AbstractBayesDensityModel, parameters::AbstractVector, t)

Evaluate f(t | η) of the Bayesian density model `bdm` for every element in the collection `t` when the parameter is equal to η.
"""
function Distributions.pdf(::AbstractBayesDensityModel, ::Any, ::Real) end

function Distributions.pdf(bdm::AbstractBayesDensityModel{S}, parameters::NamedTuple, t::AbstractVector{T}) where {S<:Real, T<:Real}
    f_samp = Vector{promote_type(T, S)}(undef, length(t))
    for i in eachindex(t)
        f_samp[i] = pdf(bdm, parameters, t[i])
    end
    return f_samp
end

function Distributions.pdf(bdm::AbstractBayesDensityModel{S}, parameters::AbstractVector{<:NamedTuple}, t::T) where {S<:Real, T<:Real}
    f_samp = Matrix{promote_type(T, S)}(undef, (length(t), length(parameters)))
    for j in eachindex(parameters)
        f_samp[:, j] .= pdf(bdm, parameters[j], t)
    end
    return f_samp
end

function Distributions.pdf(bdm::AbstractBayesDensityModel{S}, parameters::AbstractVector{<:NamedTuple}, t::AbstractVector{T}) where {S<:Real, T<:Real}
    f_samp = Matrix{promote_type(T, S)}(undef, (length(t), length(parameters)))
    for j in eachindex(parameters)
        f_samp[:, j] .= pdf(bdm, parameters[j], t)
    end
    return f_samp
end

export pdf

include("utils.jl")
public softplus, sigmoid, logit, softmax, logistic_stickbreaking, countint, bin_regular, unitvector

include("monte_carlo.jl")
export PosteriorSamples, sample, quantile, mean, median, var, std, model

include("variational.jl")
export AbstractVIPosterior, varinf

end # module
