module BayesianDensityEstimationCore

using Distributions
using StatsBase
using Random

export linebandplot
export linebandplot!

function linebandplot end
function linebandplot! end


# Abstract super type for model objects
abstract type AbstractBayesianDensityModel end

export AbstractBayesianDensityModel

"""
    pdf(bdm::AbstractBayesianDensityModel, parameters, t)
    pdf(bdm::AbstractBayesianDensityModel, parameters::AbstractVector, t)

Evaluate f(t | η) of the Bayesian density model `bdm` for every element in the collection `t` when η is given by the parameters keyword.
"""
#function Distributions.pdf(::AbstractBayesianDensityModel, ::NT, ::Real) where {NT} end

# Suppose that pdf(bdm, params, t::Real) has been implemented...
# size(f_samp) = (length(t), length(params))
function Distributions.pdf(bdm::AbstractBayesianDensityModel, parameters::NamedTuple{Names, Vals}, t::AbstractVector{T}) where {Names, Vals, T<:Real}
    f_samp = Vector{T}(undef, length(t))
    for i in eachindex(t)
        f_samp[i] = pdf(bdm, parameters, t[i])
    end
    return f_samp
end

function Distributions.pdf(bdm::AbstractBayesianDensityModel, parameters::AbstractVector{NamedTuple{Names, Vals}}, t::Union{T, <:AbstractVector{T}}) where {Names, Vals, T<:Real}
    f_samp = Matrix{T}(undef, (length(t), length(parameters)))
    for j in eachindex(parameters)
        f_samp[:, j] .= pdf(bdm, parameters[j], t)
    end
    return f_samp
end

export pdf

include("utils.jl")
public softplus, sigmoid, logit, softmax, logistic_stickbreaking, countint, bin_regular, unitvector

include("monte_carlo.jl")
export BayesianDensitySamples, sample, quantile, mean, median, model

end # module
