"""
    SHSModel{T<:Real} <: AbstractBayesDensityModel{T}

Struct representing a spline histogram smoothing model.
"""
struct SHSModel{T<:Real, A<:AbstractBSplineBasis, D<:NamedTuple} <: AbstractBayesDensityModel{T}
    data::D
    bs::A
    σ_β::T
    s_σ::T
    function SHSModel{T}(x::AbstractVector{<:Real}, K::Int=52; n_bins::Int=400, bounds::Tuple{<:Real, <:Real}=get_default_bounds(x), σ_β::Real=1e3, s_σ::Real=1e3) where {T<:Real}
        check_shskwargs(x, n_bins, bounds, σ_β, s_σ)
        n = length(x)
        T_x = T.(x)
        x_trans = (T_x .- bounds[1]) / (bounds[2] - bounds[1])
        kn_first = -0.05
        kn_last = 1.05

        # Define knot vector and create the corresponding B-spline basis:
        kn = vcat(kn_first, quantile(unique(x_trans), LinRange{T}(0, 1, K-4)), kn_last)
        bs = BSplineBasis(BSplineOrder(4), kn)

        # Get penalty matrix and find its eigendecomposition
        Ω = galerkin_matrix(bs, (Derivative(2), Derivative(2)))
        eig = eigen(Ω)
        # Keep vectors corresponding to nonzero eigenvalues
        U = eig.vectors[:, end:-1:3]
        D = Diagonal(eig.values[end:-1:3])
        LZ = U * sqrt(inv(D))

        # Compute linearly binned counts
        N = linear_binning(x_trans, n_bins, kn_first, kn_last)
        
        # Reparametrize
        bin_grid = LinRange{T}(kn_first, kn_last, n_bins+1)
        x_grid = 0.5 * (bin_grid[1:end-1] .+ bin_grid[2:end]) # midpoints of bin edges
        Z = demmler_reinsch_basis_matrix(x_grid, bs, LZ)
        C = hcat(fill(T(1), n_bins), x_grid, Z)
        data = (x = T_x, n = n, N = N, C = C, LZ = LZ, bounds = bounds)
        return new{T, typeof(bs), typeof(data)}(data, bs, T(σ_β), T(s_σ))
    end
end
SHSModel(args...; kwargs...) = SHSModel{Float64}(args...; kwargs...)

Base.eltype(::SHSModel{T, A, D}) where {T, A, D} = T

"""
    support(shs::SHSModel) -> NTuple{2, <:Real}

Get the support of the spline histogram smoother model `shs`.
"""
function BayesDensityCore.support(shs::SHSModel)
    bounds = shs.data.bounds
    smin = bounds[1] - 0.05 * (bounds[2] - bounds[1])
    smax = bounds[2] + 0.05 * (bounds[2] - bounds[1])
    return smin, smax
end

"""
    hyperparams(
        bsm::BSMModel{T}
    ) where {T} -> @NamedTuple{a_τ::T, b_τ::T, a_δ::T, b_δ::T}

Returns the hyperparameters of the B-Spline mixture model `bsm` as a `NamedTuple`.
"""
BayesDensityCore.hyperparams(shs::SHSModel) = (σ_β = shs.σ_β, s_σ = shs.s_σ)

"""
    pdf(
        bsm::SHSModel,
        params::NamedTuple,
        t::Union{Real, AbstractVector{<:Real}}
    ) -> Union{Real, Vector{<:Real}}

    pdf(
        bsm::SHSModel,
        params::AbstractVector{NamedTuple},
        t::Union{Real, AbstractVector{<:Real}}
    ) -> Matrix{<:Real}

Evaluate f(t | η) when the model parameters are equal to η.

The named tuple should contain a field named `:β`.
If the `parameters` argument does not contain a field named `:norm`, then the normalization constant will be computed using Simpson's method.
Alternatively, if `parameters` contains the field `:norm`, then this value is used instead.
"""
function Distributions.pdf(shs::SHSModel, params::NamedTuple{Names, Vals}, t::Real) where {Names, Vals<:Tuple}
    return _pdf(shs, params, [t], Val(:norm in Names))[1]
end
function Distributions.pdf(shs::SHSModel, params::AbstractVector{NamedTuple{Names, Vals}}, t::Real) where {Names, Vals<:Tuple}
    return _pdf(shs, params, [t], Val(:norm in Names))
end
function Distributions.pdf(shs::SHSModel, params::NamedTuple{Names, Vals}, t::AbstractVector{<:Real}) where {Names, Vals<:Tuple}
    return _pdf(shs, params, t, Val(:norm in Names))
end
function Distributions.pdf(shs::SHSModel, params::AbstractVector{NamedTuple{Names, Vals}}, t::AbstractVector{<:Real}) where {Names, Vals<:Tuple}
    return _pdf(shs, params, t, Val(:norm in Names))
end

# If normalization constant is provided, we do not have to compute it
function _pdf(shs::SHSModel, params::NamedTuple{Names, Vals}, t::AbstractVector{<:Real}, ::Val{true}) where {Names, Vals<:Tuple}
    bounds = shs.data.bounds
    t_trans = (t .- bounds[1]) / (bounds[2] - bounds[1])
    bs_min, bs_max = boundaries(shs.bs)
    Z = demmler_reinsch_basis_matrix(t_trans, shs.bs, shs.data.LZ)
    C = hcat(fill(1, length(t_trans)), t_trans, Z)
    l1_norm = params.norm

    # Compute linear predictor, exponentiate and normalize:
    linpreds = C * params.β
    return exp.(linpreds) / l1_norm * (bounds[2] - bounds[1]) .* ifelse.(bs_min .≤ t_trans .≤ bs_max, 1, 0)
end
function _pdf(shs::SHSModel{T, A, D}, params::AbstractVector{NamedTuple{Names, Vals}}, t::AbstractVector{S}, ::Val{true}) where {T<:Real, A, D, Names, Vals<:Tuple, S<:Real}
    R = promote_type(T, S)
    bounds = shs.data.bounds
    t_trans = (t .- bounds[1]) / (bounds[2] - bounds[1])
    bs_min, bs_max = boundaries(shs.bs)

    # Evaluate linear predictors for each value in t, params:
    linpreds = eval_linpred(shs, params, t_trans)
    l1_norm_vec = Vector{R}(undef, length(params))
    for i in eachindex(params)
        l1_norm_vec[i] = params[i].norm
    end
    
    # Normalize:
    f_samp = Matrix{R}(undef, (length(t), length(params)))
    for i in eachindex(params)
        f_samp[:,i] = exp.(linpreds[:,i]) / l1_norm_vec[i] * (bounds[2] - bounds[1]) .* ifelse.(bs_min .≤ t_trans .≤ bs_max, 1, 0)
    end
    return f_samp
end

# Normalization constant not provided, so it must be computed first.
function _pdf(shs::SHSModel, params::NamedTuple{Names, Vals}, t::AbstractVector{<:Real}, ::Val{false}) where {Names, Vals<:Tuple}
    bounds = shs.data.bounds
    t_trans = (t .- bounds[1]) / (bounds[2] - bounds[1])
    bs_min, bs_max = boundaries(shs.bs)
    Z = demmler_reinsch_basis_matrix(t_trans, shs.bs, shs.data.LZ)
    C = hcat(fill(1, length(t)), t_trans, Z)
    l1_norm = compute_norm_constants(shs, params)

    # Compute linear predictor, exponentiate and normalize:
    linpreds = C * params.β
    return exp.(linpreds) / l1_norm * (bounds[2] - bounds[1]) .* ifelse.(bs_min .≤ t_trans .≤ bs_max, 1, 0)
end
function _pdf(shs::SHSModel{T, A, D}, params::AbstractVector{NamedTuple{Names, Vals}}, t::AbstractVector{S}, ::Val{false}) where {T<:Real, A, D, Names, Vals<:Tuple, S<:Real}
    R = promote_type(T, S)
    bounds = shs.data.bounds
    t_trans = (t .- bounds[1]) / (bounds[2] - bounds[1])
    bs_min, bs_max = boundaries(shs.bs)

    # Evaluate linear predictors for each value in t, params:
    linpreds = eval_linpred(shs, params, t_trans)
    l1_norm_vec = compute_norm_constants(shs, params)
    
    # Normalize:
    f_samp = Matrix{R}(undef, (length(t), length(params)))
    for i in eachindex(params)
        f_samp[:,i] = exp.(linpreds[:,i]) / l1_norm_vec[i] * (bounds[2] - bounds[1]) .* ifelse.(bs_min .≤ t_trans .≤ bs_max, 1, 0)
    end
    return f_samp
end


function get_default_bounds(x::AbstractVector{<:Real})
    xmin, xmax = extrema(x)
    R = xmax - xmin
    return xmin - 0.05*R, xmax + 0.05*R
end

# When sampling from these models (MCMC and VI), compute the normalization constant for each of the draws so we do not (can also provide method dispatch for the pdf method so the user also has the option of passing in custom params)

function check_shskwargs(x::AbstractVector{<:Real}, n_bins::Int, bounds::Tuple{<:Real, <:Real}, σ_β::Real, s_σ::Real)
    if !isnothing(n_bins) && n_bins ≤ 1
        throw(ArgumentError("Number of bins must be a positive integer or 'nothing'."))
    end
    xmin, xmax = extrema(x)
    if bounds[1] ≥ bounds[2]
        throw(ArgumentError("Supplied upper bound must be strictly greater than the lower bound."))
    elseif bounds[1] > xmin || bounds[2] < xmax
        throw(ArgumentError("Data is not contained within supplied bounds."))
    end
    hyperpar = [σ_β, s_σ]
    hyperpar_symb = [:σ_β, :s_σ]
    for i in eachindex(hyperpar)
        if hyperpar[i] ≤ 0.0 || hyperpar[i] == Inf
            throw(ArgumentError("Hyperparameter $(hyperpar_symb[i]) must be a strictly positive finite number."))
        end
    end
end