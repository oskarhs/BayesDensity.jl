"""
    HistSmoother{T<:Real} <: AbstractBayesDensityModel{T}

Struct representing a spline histogram smoothing model.

# Constructors
    
    HistSmoother(x::AbstractVector{<:Real}; kwargs...) 

# Arguments
* `x`: The data vector.

# Keyword arguments
* `K`: B-spline basis dimension of a regular augmented spline basis. Defaults to `52`.
* `bounds`: A tuple giving the support of the B-spline mixture model.
* `n_bins`: Number of bins used to construct the histogram likelihood. Defaults to `400`.
* `σ_β`: Scale hyperparameter for 0th and 1st order (fixed effect) spline terms. Defaults to `1000.0`.
* `s_σ`: Scale hyperparameter for the (higher order) mixed effects spline terms. Defaults to `1000.0`.

# Returns
* `shs`: A histogram spline smoother object.

# Examples
```julia
julia> x = (1.0 .- (1.0 .- LinRange(0.0, 1.0, 5000)) .^(1/3)).^(1/3);

julia> shs = HistSmoother(x)
52-dimensional HistSmoother{Float64}:
Using 5000 binned observations with 400 bins.

julia> shs = HistSmoother(x; K = 80 σ_β = 1e5);
```

# Extended help

### Binning
The binning step used by the spline histogram smoother is an essential part of the model fitting procedure, and can as such not be disabled.
Using a greater number of bins means that less precision is lost due to the binning step, but makes the model fitting procedure slower due to a larger compuatational burden.
Note that the number of bins only affects the model fitting process, and does otherwise not change the returned 

### Hyperparameter selection
The hyperparameter `s_β` contols the smoothness of the resulting density estimates.
Setting this to a smaller value leads to smoother estimates.
"""
struct HistSmoother{T<:Real, A<:AbstractBSplineBasis, D<:NamedTuple} <: AbstractBayesDensityModel{T}
    data::D
    bs::A
    σ_β::T
    s_σ::T
    function HistSmoother{T}(x::AbstractVector{<:Real}; K::Int=52, n_bins::Int=400, bounds::Tuple{<:Real, <:Real}=get_default_bounds(x), σ_β::Real=1e3, s_σ::Real=1e3) where {T<:Real}
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
        data = (x = T_x, n = n, x_grid = x_grid, N = N, C = C, LZ = LZ, bounds = bounds)
        return new{T, typeof(bs), typeof(data)}(data, bs, T(σ_β), T(s_σ))
    end
end
HistSmoother(args...; kwargs...) = HistSmoother{Float64}(args...; kwargs...)

function Base.:(==)(shs1::HistSmoother, shs2::HistSmoother)
    return shs1.bs == shs2.bs && shs1.data == shs2.data && hyperparams(shs1) == hyperparams(shs2)
end

function Base.show(io::IO, ::MIME"text/plain", shs::HistSmoother{T, A, D}) where {T, A, D}
    println(io, length(shs.bs), "-dimensional ", nameof(typeof(shs)), '{', T, "}:")
    println(io, "Using ", shs.data.n, " binned observations with ", length(shs.data.N), " bins.")
    let io = IOContext(io, :compact => true, :limit => true)
        println(io, " support: ", BayesDensityCore.support(shs))
        println(io, "Hyperparameters: ")
        println(io, " σ_β = ", shs.σ_β)
        print(io, " s_σ = ", shs.s_σ)
    end
    nothing
end

Base.show(io::IO, shs::HistSmoother) = show(io, MIME("text/plain"), shs)

"""
    support(shs::HistSmoother) -> NTuple{2, <:Real}

Get the support of the spline histogram smoother model `shs`.
"""
function BayesDensityCore.support(shs::HistSmoother)
    bounds = shs.data.bounds
    bs_min = bounds[1] - 0.05 * (bounds[2] - bounds[1])
    bs_max = bounds[2] + 0.05 * (bounds[2] - bounds[1])
    return bs_min, bs_max
end

"""
    hyperparams(
        shs::HistSmoother{T}
    ) where {T} -> @NamedTuple{σ_β::T, s_σ::T}

Returns the hyperparameters of the spline histogram smoother `shs` as a `NamedTuple`.
"""
BayesDensityCore.hyperparams(shs::HistSmoother) = (σ_β = shs.σ_β, s_σ = shs.s_σ)

"""
    pdf(
        shs::HistSmoother,
        params::NamedTuple,
        t::Union{Real, AbstractVector{<:Real}}
    ) -> Union{Real, Vector{<:Real}}

    pdf(
        shs::HistSmoother,
        params::AbstractVector{NamedTuple},
        t::Union{Real, AbstractVector{<:Real}}
    ) -> Matrix{<:Real}

Evaluate ``f(t \\,|\\, \\boldsymbol{\\eta})`` for the HistSmoother `shs` when the model parameters are equal to ``\\boldsymbol{\\eta}``.

The named tuple should contain a field named `:β`.
If the `parameters` argument does not contain a field named `:norm`, then the normalization constant will be computed using Simpson's method.
Alternatively, if `parameters` contains the field `:norm`, then this value is used instead.
"""
function Distributions.pdf(shs::HistSmoother, params::NamedTuple{Names, Vals}, t::Real) where {Names, Vals<:Tuple}
    return _pdf(shs, params, [t], Val(:norm in Names))[1]
end
function Distributions.pdf(shs::HistSmoother, params::AbstractVector{NamedTuple{Names, Vals}}, t::Real) where {Names, Vals<:Tuple}
    return _pdf(shs, params, [t], Val(:norm in Names))
end
function Distributions.pdf(shs::HistSmoother, params::NamedTuple{Names, Vals}, t::AbstractVector{<:Real}) where {Names, Vals<:Tuple}
    return _pdf(shs, params, t, Val(:norm in Names))
end
function Distributions.pdf(shs::HistSmoother, params::AbstractVector{NamedTuple{Names, Vals}}, t::AbstractVector{<:Real}) where {Names, Vals<:Tuple}
    return _pdf(shs, params, t, Val(:norm in Names))
end

# If normalization constant is provided, we do not have to compute it
function _pdf(shs::HistSmoother, params::NamedTuple{Names, Vals}, t::AbstractVector{<:Real}, ::Val{true}) where {Names, Vals<:Tuple}
    bounds = shs.data.bounds
    t_trans = (t .- bounds[1]) / (bounds[2] - bounds[1])
    bs_min, bs_max = boundaries(shs.bs)
    Z = demmler_reinsch_basis_matrix(t_trans, shs.bs, shs.data.LZ)
    C = hcat(fill(1, length(t_trans)), t_trans, Z)
    l1_norm = params.norm

    # Compute linear predictor, exponentiate and normalize:
    linpreds = C * params.β
    return exp.(linpreds) / (l1_norm*(bounds[2] - bounds[1])) .* ifelse.(bs_min .≤ t_trans .≤ bs_max, 1, 0)
end
function _pdf(shs::HistSmoother{T, A, D}, params::AbstractVector{NamedTuple{Names, Vals}}, t::AbstractVector{S}, ::Val{true}) where {T<:Real, A, D, Names, Vals<:Tuple, S<:Real}
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
        f_samp[:,i] = exp.(linpreds[:,i]) / (l1_norm_vec[i] * (bounds[2] - bounds[1])) .* ifelse.(bs_min .≤ t_trans .≤ bs_max, 1, 0)
    end
    return f_samp
end

# Normalization constant not provided, so it must be computed first.
function _pdf(shs::HistSmoother, params::NamedTuple{Names, Vals}, t::AbstractVector{<:Real}, ::Val{false}) where {Names, Vals<:Tuple}
    bounds = shs.data.bounds
    t_trans = (t .- bounds[1]) / (bounds[2] - bounds[1])
    bs_min, bs_max = boundaries(shs.bs)
    Z = demmler_reinsch_basis_matrix(t_trans, shs.bs, shs.data.LZ)
    C = hcat(fill(1, length(t)), t_trans, Z)
    _, _, l1_norm = compute_norm_constants_cdf_grid(shs, params)

    # Compute linear predictor, exponentiate and normalize:
    linpreds = C * params.β
    return exp.(linpreds) / (l1_norm * (bounds[2] - bounds[1])) .* ifelse.(bs_min .≤ t_trans .≤ bs_max, 1, 0)
end
function _pdf(shs::HistSmoother{T, A, D}, params::AbstractVector{NamedTuple{Names, Vals}}, t::AbstractVector{S}, ::Val{false}) where {T<:Real, A, D, Names, Vals<:Tuple, S<:Real}
    R = promote_type(T, S)
    bounds = shs.data.bounds
    t_trans = (t .- bounds[1]) / (bounds[2] - bounds[1])
    bs_min, bs_max = boundaries(shs.bs)

    # Evaluate linear predictors for each value in t, params:
    linpreds = eval_linpred(shs, params, t_trans)
    _, _, l1_norm_vec = compute_norm_constants_cdf_grid(shs, params)
    
    # Normalize:
    f_samp = Matrix{R}(undef, (length(t), length(params)))
    for i in eachindex(params)
        f_samp[:,i] = exp.(linpreds[:,i]) / (l1_norm_vec[i] * (bounds[2] - bounds[1])) .* ifelse.(bs_min .≤ t_trans .≤ bs_max, 1, 0)
    end
    return f_samp
end

"""
    cdf(
        shs::HistSmoother,
        params::NamedTuple,
        t::Union{Real, AbstractVector{<:Real}}
    ) -> Union{Real, Vector{<:Real}}

    cdf(
        shs::HistSmoother,
        params::AbstractVector{NamedTuple},
        t::Union{Real, AbstractVector{<:Real}}
    ) -> Matrix{<:Real}

Evaluate ``F(t \\,|\\, \\boldsymbol{\\eta}) = \\int_{-\\infty}^t f(s \\,|\\, \\boldsymbol{\\eta})\\, \\text{d}s`` for the HistSmoother `shs` when the model parameters are equal to ``\\boldsymbol{\\eta}``.

The named tuple should contain a field named `:β`.
If the `parameters` argument does not contain a field named `:norm`, then the normalization constant will be computed using Simpson's method.
Alternatively, if `parameters` contains the field `:norm`, then this value is used instead.

Internally, this function computes the cdf on a regular grid, and uses linear interpolation to 
"""
function Distributions.cdf(shs::HistSmoother, params::NamedTuple{Names, Vals}, t::Real) where {Names, Vals<:Tuple}
    return _cdf(shs, params, [t], Val(:eval_grid in Names && :val_cdf in Names))[1]
end
function Distributions.cdf(shs::HistSmoother, params::AbstractVector{NamedTuple{Names, Vals}}, t::Real) where {Names, Vals<:Tuple}
    return _cdf(shs, params, [t], Val(:eval_grid in Names && :val_cdf in Names))
end
function Distributions.cdf(shs::HistSmoother, params::NamedTuple{Names, Vals}, t::AbstractVector{<:Real}) where {Names, Vals<:Tuple}
    return _cdf(shs, params, t, Val(:eval_grid in Names && :val_cdf in Names))
end
function Distributions.cdf(shs::HistSmoother, params::AbstractVector{NamedTuple{Names, Vals}}, t::AbstractVector{<:Real}) where {Names, Vals<:Tuple}
    return _cdf(shs, params, t, Val(:eval_grid in Names && :val_cdf in Names))
end

# If cdf has been evaluated on a grid, we use this to compute the linear interpolant:
function _cdf(shs::HistSmoother{T, A, D}, params::NamedTuple{Names, Vals}, t::AbstractVector{S}, ::Val{true}) where {T, A, D, Names, Vals<:Tuple, S<:Real}
    R = promote_type(T, S)
    bounds = shs.data.bounds
    bs_min, bs_max = boundaries(shs.bs)
    t_trans = (t .- bs_min) / (bs_max - bs_min)

    # Perform linear interpolation
    F_interp = interpolate(params.eval_grid, params.val_cdf, BSplineOrder(2))
    F_samp[:, i] = F_interp.(t_trans)
    F_samp[t_trans .> bs_max, i] .= one(T)
    return F_samp
end
function _cdf(shs::HistSmoother{T, A, D}, params::AbstractVector{NamedTuple{Names, Vals}}, t::AbstractVector{S}, ::Val{true}) where {T<:Real, A, D, Names, Vals<:Tuple, S<:Real}
    R = promote_type(T, S)
    bounds = shs.data.bounds
    bs_min, bs_max = boundaries(shs.bs)
    t_trans = (t .- bs_min) / (bs_max - bs_min)
    
    # Interpolate
    F_samp = zeros(R, (length(t), length(params)))
    n_intervals = length(params[1].eval_grid)-1
    for i in eachindex(params)
        F_interp = interpolate(params[i].eval_grid, params[i].val_cdf, BSplineOrder(2))
        F_samp[:, i] = F_interp.(t_trans)
        F_samp[t_trans .> bs_max, i] .= one(T)
    end
    return F_samp
end

# If cdf has not been evaluated on a grid, we must evaluate it first:
function _cdf(shs::HistSmoother{T, A, D}, params::NamedTuple{Names, Vals}, t::AbstractVector{S}, ::Val{false}) where {T<:Real, A, D, Names, Vals<:Tuple, S<:Real}
    R = promote_type(T, S)
    bounds = shs.data.bounds
    bs_min, bs_max = BayesDensityHistSmoother.support(shs)
    t_trans = (t .- bounds[1]) / (bounds[2] - bounds[1])
    eval_grid, val_cdf, _ = compute_norm_constants_cdf_grid(shs, params)

    # Interpolate
    F_interp = interpolate(eval_grid, val_cdf, BSplineOrder(2))
    F_samp = F_interp.(t_trans)
    F_samp[t_trans .> bs_max] .= one(T)
    return F_samp
end
function _cdf(shs::HistSmoother{T, A, D}, params::AbstractVector{NamedTuple{Names, Vals}}, t::AbstractVector{S}, ::Val{false}) where {T<:Real, A, D, Names, Vals<:Tuple, S<:Real}
    R = promote_type(T, S)
    bounds = shs.data.bounds
    eval_grid, val_cdf, _ = compute_norm_constants_cdf_grid(shs, params)
    bs_min, bs_max = BayesDensityHistSmoother.support(shs)
    t_trans = (t .- bounds[1]) / (bounds[2] - bounds[1])
    
    # Interpolate
    F_samp = zeros(R, (length(t), length(params)))
    for i in eachindex(params)
        F_interp = interpolate(eval_grid, val_cdf[i], BSplineOrder(2))
        F_samp[:, i] = F_interp.(t_trans)
        F_samp[t_trans .> bs_max, i] .= one(T)
    end
    return F_samp
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