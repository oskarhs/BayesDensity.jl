module BayesianDensityEstimationCore

using Distributions
using StatsBase
using Random


# Abstract super type for model objects
abstract type AbstractBayesianDensityModel end

"""
    pdf(bdm::AbstractBayesianDensityModel, parameters, t)
    pdf(bdm::AbstractBayesianDensityModel, parameters::AbstractVector, t)

Evaluate f(t | η) of the Bayesian density model `bdm` for every element in the collection `t` when η is given by the parameters keyword.


"""
function Distributions.pdf(::AbstractBayesianDensityModel, ::NT, ::Real) where {NT} end

# Suppose that pdf(bdm, params, t::Real) has been implemented...
# size(f_samp) = (length(t), length(params))
function Distributions.pdf(bdm::AbstractBayesianDensityModel, parameters, t::AbstractVector{T}) where {T<:Real}
    f_samp = Vector{T}(undef, length(t))
    for i in eachindex(t)
        f_samp[i] = pdf(bdm, parameters, t[i])
    end
    return f_samp
end

function Distributions.pdf(bdm::AbstractBayesianDensityModel, parameters::AbstractVector, t::Union{T, Vector{T}}) where {T<:Real}
    f_samp = Matrix{T}(undef, (length(t), length(parameters)))
    for j in eachindex(parameters)
        f_samp[:, j] = pdf(bdm, parameters[j], t)
    end
    return f_samp
end

include("monte_carlo.jl")

export BayesianDensityChain
export sample, pdf, quantile, mean, median

end # module
