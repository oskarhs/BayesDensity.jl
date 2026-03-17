"""
    varinf(
        bdm::AbstractBayesDensityModel{T},
        args...;
        kwargs...
    ) where {T<:Real} -> AbstractVIPosterior{T}

Compute a variational approximation to the posterior distribution.

The positional arguments and keyword arguments supported by this function, as well as the type of the returned variational posterior object differs between different subtypes of [`AbstractBayesDensityModel`](@ref).
"""
function varinf(::AbstractBayesDensityModel) end

"""
    AbstractVIPosterior{T<:Real} <: AbstractSampleablePosterior{T}

Abstract super type representing the variational posterior distribution of `AbstractBayesDensityModel`
"""
abstract type AbstractVIPosterior{T<:Real} <: AbstractSampleablePosterior{T} end

"""
    VariationalOptimizationResult{T<:Real}

Struct holding the result of a variational inference procedure.

# Fields
* `ELBO`: The values of the evidence lower bound per iteration.
* `converged`: Boolean flag indicating whether the optimization was succesful or not.
* `n_iter`: Number of iterations run before termination.
* `tolerance`: Tolerance parameter used to determine convergence.
* `variational_posterior`: The fitted variational posterior distribution.
"""
struct VariationalOptimizationResult{T<:Real, V<:AbstractVector, A<:AbstractVIPosterior}
    ELBO::V
    converged::Bool
    n_iter::Int
    tolerance::T
    variational_posterior::A
    function VariationalOptimizationResult{T}(ELBO::V, converged::Bool, n_iter::Int, tolerance::Real, variational_posterior::A) where {T<:Real, V<:AbstractVector, A<:AbstractVIPosterior}
        return new{T, V, A}(ELBO, converged, n_iter, tolerance, variational_posterior)
    end
end

# Print method:
function Base.show(io::IO, ::MIME"text/plain", varoptinf::VariationalOptimizationResult{T, V}) where {T, V}
    status_msg = ifelse(
        converged(varoptinf),
        " Converged in $(n_iter(varoptinf)) iterations.",
        " Failed to converge in $(n_iter(varoptinf)) iterations."
    )
    println(io, nameof(typeof(varoptinf)), "{", T, "} object.")
    println(io, status_msg)
    println(io, " Tolerance level for convergence: ", tolerance(varoptinf))
    print(io, "Posterior: ", nameof(typeof(varoptinf.variational_posterior)))
    nothing
end

Base.show(io::IO, varoptinf::VariationalOptimizationResult) = show(io, MIME("text/plain"), varoptinf)

"""
    posterior(varoptinf::VariationalOptimizationResult{T}) where {T} -> AbstractVIPosterior{T}

Get the fitted variational posterior distribution.
"""
posterior(varoptinf::VariationalOptimizationResult) = varoptinf.variational_posterior

"""
    n_iter(varoptinf::VariationalOptimizationResult{T}) where {T} -> Int

Get the number of iterations used to fit the variational posterior distribution.
"""
n_iter(varoptinf::VariationalOptimizationResult) = varoptinf.n_iter

"""
    elbo(varoptinf::VariationalOptimizationResult{T}) where {T} -> Vector{T}

Get the value of the evidence lower bound for each iteration of the optimization procedure.
"""
elbo(varoptinf::VariationalOptimizationResult) = varoptinf.ELBO

"""
    converged(varoptinf::VariationalOptimizationResult{T}) where {T} -> Bool

Return the convergence status of the variational optimization as a bool.
"""
converged(varoptinf::VariationalOptimizationResult) = varoptinf.converged

"""
    tolerance(varoptinf::VariationalOptimizationResult{T}) where {T} -> T

Get the value of the tolerance level used to determine convergence.
"""
tolerance(varoptinf::VariationalOptimizationResult) = varoptinf.tolerance