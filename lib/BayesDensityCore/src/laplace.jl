"""
    laplace_approximation(
        bdm::AbstractBayesDensityModel{T},
        args...;
        kwargs...
    ) where {T<:Real} -> AbstractLaplacePosterior{T}

Compute a Laplace approximation to the posterior distribution.

The positional arguments and keyword arguments supported by this function, as well as the type of the returned variational posterior object differs between different subtypes of [`AbstractBayesDensityModel`](@ref).
"""
function laplace_approximation(::AbstractBayesDensityModel) end

"""
    AbstractVIPosterior{T<:Real} <: AbstractSampleablePosterior{T}

Abstract super type representing the variational posterior distribution of `AbstractBayesDensityModel`
"""
abstract type AbstractLaplacePosterior{T<:Real} <: AbstractSampleablePosterior{T} end

