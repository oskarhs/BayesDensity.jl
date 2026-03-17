"""
    LogisticGaussianProcess{T<:Real} <: AbstractBayesDensityModel{T}
    
Struct representing a logistic Gaussian process model.

# Constructors
    
    LogisticGaussianProcess(x::AbstractVector{<:Real}; kwargs...)
    LogisticGaussianProcess{T}(x::AbstractVector{<:Real}; kwargs...)

# Arguments
* `x`: The data vector.

# Keyword arguments
* `n_bins`: The number of bins at which the Gaussian process posterior is computed. Defaults to 
* `bounds`: A tuple giving the support of the B-spline mixture model.
* `prior_variance_scale`: Scale hyperparameter for the variance parameter σ². Defaults to `10.0`.
* `prior_length_scale`: Scale hyperparameter for the length parameter λ. Defaults to `1.0`.

# Returns
* `lgp`: A LogisticGaussianProcess model object

# Examples
```julia-repl
julia> x = (1.0 .- (1.0 .- LinRange(0.0, 1.0, 5000)) .^(1/3)).^(1/3);

julia> model = LogisticGaussianProcess(x)
200-dimensional LogisticGaussianProcess{Float64}:
Using 5000 binned observations on a regular grid consisting of 1187 bins.
 support: (-0.05, 1.05)
Hyperparameters:
 prior_global_shape = 1.0, prior_global_rate = 0.0002
 prior_local_shape = 1, prior_local_rate = 0.5

julia> model = LogisticGaussianProcess(x; K = 150, bounds=(0, 1), n_bins=nothing, prior_global_rate = 5e-3);
```

# Extended help

### Binned fitting
The binning step used by the logistic Gaussian process is an essential part of the model fitting procedure, and can as such not be disabled.
Using a greater number of bins means that less precision is lost due to the binning step, but makes the model fitting procedure slower due to a larger computational burden.
Note that the number of bins only affects the model fitting process, and does otherwise not affect the returned model object.

### Hyperparameter selection
The hyperparameter `prior_variance_scale` contols the variance of the underlying gaussian process.
Setting this to a larger value leads to a more diffuse prior, i.e. the prior assigns more mass to densities which are further away from the uniform.

The hyperparameter `prior_length_scale` contols the length parameters of the underlying gaussian process.
Setting this to a larger value leads to less smoothing.
"""
struct LogisticGaussianProcess{T<:Real, NT<:NamedTuple} <: AbstractBayesDensityModel{T}
    data::NT
    prior_variance_scale::T
    prior_length_scale::T
    function LogisticGaussianProcess{T}(
        x::AbstractVector{<:Real};
        n_bins::Int                  = 400,
        bounds::Tuple{<:Real,<:Real} = _get_default_bounds(x),
        prior_variance_scale::Real   = 10.0,
        prior_length_scale::Real     = 1.0
    ) where {T<:Real}
        _check_lgpkwargs(x, n_bins, bounds, prior_variance_scale, prior_length_scale)
        n = length(x)
        
        T_x = T.(x)
        xmin::T, xmax::T = bounds
        N = bin_regular(T_x, xmin, xmax, n_bins)
        x_grid = LinRange{T}(1/(2*n_bins), 1-1/(2*n_bins), n_bins)
        
        pairwise_dists = [abs2(x_grid[i] - x_grid[j]) for i in 1:n_bins, j in 1:n_bins]
        data = (x = T_x, n = n, x_grid = x_grid, N = N, pairwise_dists = pairwise_dists, bounds = bounds, n_bins = n_bins)
        return new{T, typeof(data)}(data, T(prior_variance_scale), T(prior_length_scale))
    end
end
LogisticGaussianProcess(args...; kwargs...) = LogisticGaussianProcess{Float64}(args...; kwargs...)

function Base.:(==)(lgp1::LogisticGaussianProcess, lgp2::LogisticGaussianProcess)
    return lgp1.data == lgp2.data && hyperparams(lgp1) == hyperparams(lgp2)
end

function Base.show(io::IO, ::MIME"text/plain", lgp::LogisticGaussianProcess{T}) where {T}
    println(io, nameof(typeof(lgp)), '{', T, "}:")
    println(io, "Using ", lgp.data.n, " binned observations with ", length(lgp.data.N), " bins.")
    let io = IOContext(io, :compact => true, :limit => true)
        println(io, " support: ", BayesDensityCore.support(lgp))
        println(io, "Hyperparameters: ")
        println(io, " prior_variance_scale = ", lgp.prior_variance_scale)
        print(io, " prior_length_scale = ", lgp.prior_length_scale)
    end
    nothing
end

Base.show(io::IO, lgp::LogisticGaussianProcess) = show(io, MIME("text/plain"), lgp)

function BayesDensityCore.support(lgp::LogisticGaussianProcess)
    bs_min, bs_max = lgp.data.bounds
    return bs_min, bs_max
end

function default_grid_points(lgp::LogisticGaussianProcess{T}) where {T}
    xmin, xmax = BayesDensityCore.support(lgp)
    return LinRange{T}(xmin, xmax, 2001)
end

function _get_default_bounds(x::AbstractVector{<:Real})
    xmin, xmax = extrema(x)
    R = xmax - xmin
    return xmin - 0.05*R, xmax + 0.05*R
end


"""
    hyperparams(
        lgp::LogisticGaussianProcess{T}
    ) where {T} -> @NamedTuple{prior_variance_scale::T, prior_length_scale::T}

Returns the hyperparameters of the logistic Gaussian process model `lgp` as a `NamedTuple`.
"""
BayesDensityCore.hyperparams(lgp::LogisticGaussianProcess) = (prior_variance_scale=lgp.prior_variance_scale, prior_length_scale=lgp.prior_length_scale)

"""
    pdf(
        lgp::LogisticGaussianProcess,
        params::NamedTuple,
        t::Union{Real, AbstractVector{<:Real}}
    ) -> Union{Real, Vector{<:Real}}

    pdf(
        lgp::LogisticGaussianProcess,
        params::AbstractVector{NamedTuple},
        t::Union{Real, AbstractVector{<:Real}}
    ) -> Matrix{<:Real}

Evaluate ``f(t\\, |\\, \\boldsymbol{\\eta})`` for a given `LogisticGaussianProcess` when the model parameters of the NamedTuple `params` are given by ``\\boldsymbol{\\eta}``.

The named tuple should contain a field named `:β` (untransformed sample from the Laplace approximation) or `:val_pdf` (properly normalized sample).
The dimension of β/val_pdf must match that of the supplied LogisticGaussianProcess model object.
The pdf at the given input point is approximated using linear interpolation.
"""
function Distributions.pdf(lgp::LogisticGaussianProcess, params::NamedTuple{Names, Vals}, t::Real) where {Names, Vals<:Tuple}
    _pdf(lgp, params, t, Val(:val_pdf in Names))
end
function Distributions.pdf(lgp::LogisticGaussianProcess, params::NamedTuple{Names, Vals}, t::AbstractVector{<:Real}) where {Names, Vals<:Tuple}
    _pdf(lgp, params, t, Val(:val_pdf in Names))
end
function Distributions.pdf(lgp::LogisticGaussianProcess, params::AbstractVector{NamedTuple{Names, Vals}}, t::AbstractVector{<:Real}) where {Names, Vals<:Tuple}
    _pdf(lgp, params, t, Val(:val_pdf in Names))
end
Distributions.pdf(lgp::LogisticGaussianProcess, params::AbstractVector{NamedTuple{Names, Vals}}, t::Real) where {Names, Vals<:Tuple} = pdf(lgp, params, [t])

# Compile-time dispatch
function _pdf(lgp::LogisticGaussianProcess, params::NamedTuple{Names, Vals}, t, ::Val{true}) where {Names, Vals}
    return _pdf(lgp, params.val_pdf, t)
end
function _pdf(lgp::LogisticGaussianProcess, params::NamedTuple{Names, Vals}, t, ::Val{false}) where {Names, Vals}
    # Pdf has not been evaluated
    n_bins = lgp.data.n_bins
    eval_unnorm = exp.(params.β)
    norm = sum(eval_unnorm) / n_bins
    eval_pdf_grid = eval_unnorm / norm
    return _pdf(lgp, eval_pdf_grid, t)
end
function _pdf(lgp::LogisticGaussianProcess{T}, params::AbstractVector{NamedTuple{Names, Vals}}, t::Union{S, AbstractVector{S}}, ::Val{true}) where {T<:Real, Names, Vals, S<:Real}
    # Pdf already evaluated
    n_bins = lgp.data.n_bins
    eval_pdf_grid = Matrix{promote_type(T, S)}(undef, (n_bins, length(params)))
    for i in eachindex(params)
        eval_pdf_grid[:, i] = params[i].val_pdf
    end
    return _pdf(lgp, eval_pdf_grid, t)
end
function _pdf(lgp::LogisticGaussianProcess{T}, params::AbstractVector{NamedTuple{Names, Vals}}, t::Union{S, AbstractVector{S}}, ::Val{false}) where {T<:Real, Names, Vals, S<:Real}
    # Build evaluation matrix (normalization constants are not given)
    n_bins = lgp.data.n_bins
    eval_pdf_grid = Matrix{promote_type(T, S)}(undef, (n_bins, length(params)))
    for i in eachindex(params)
        eval_unnorm = exp.(params[i].β)
        norm = sum(eval_unnorm) / n_bins
        eval_pdf_grid[:, i] = eval_unnorm / norm
    end
    return _pdf(lgp, eval_pdf_grid, t)
end

_pdf(lgp::LogisticGaussianProcess, eval_pdf_grid::AbstractVector{<:Real}, t::Real) = _pdf(lgp, eval_pdf_grid, [t])[1]
function _pdf(lgp::LogisticGaussianProcess, eval_pdf_grid::AbstractVector{<:Real}, t::AbstractVector{<:Real})
    n_bins = lgp.data.n_bins
    xmin, xmax = BayesDensityCore.support(lgp)
    x_grid = lgp.data.x_grid
    t_trans = (t .- xmin) / (xmax - xmin)
    t_eval = clamp.(t_trans, 1/(2*n_bins), 1-1/(2*n_bins))
    f_interp = interpolate(x_grid, eval_pdf_grid, BSplineOrder(2))
    f_samp = f_interp.(t_eval) * (xmax - xmin)
    return ifelse.(xmin .≤ t .≤ xmax, f_samp, 0.0)
end
function _pdf(lgp::LogisticGaussianProcess, eval_pdf_grid::AbstractMatrix{T}, t::AbstractVector{S}) where {T<:Real, S<:Real}
    n_bins = lgp.data.n_bins
    xmin, xmax = BayesDensityCore.support(lgp)
    x_grid = lgp.data.x_grid
    t_trans = (t .- xmin) / (xmax - xmin)
    t_eval = clamp.(t_trans, 1/(2*n_bins), 1-1/(2*n_bins))
    f_samp = Matrix{promote_type(T, S)}(undef, (length(t), size(eval_pdf_grid, 2)))
    for i in axes(eval_pdf_grid, 2)
        f_interp = interpolate(x_grid, eval_pdf_grid[:, i], BSplineOrder(2))
        f_samp[:,i] = ifelse.(xmin .≤ t .≤ xmax, f_interp.(t_eval) * (xmax - xmin), 0.0)
    end
    return f_samp
end


"""
    cdf(
        lgp::LogisticGaussianProcess,
        params::NamedTuple,
        t::Union{Real, AbstractVector{<:Real}}
    ) -> Union{Real, Vector{<:Real}}

    cdf(
        lgp::LogisticGaussianProcess,
        params::AbstractVector{NamedTuple},
        t::Union{Real, AbstractVector{<:Real}}
    ) -> Matrix{<:Real}

Evaluate ``F(t\\, |\\, \\boldsymbol{\\eta})`` for a given `LogisticGaussianProcess` when the model parameters of the NamedTuple `params` are given by ``\\boldsymbol{\\eta}``.

The named tuple should contain a field named `:β` (untransformed sample from the Laplace approximation) or `:val_cdf` (cdf evaluated on a regular grid).
The dimension of β/val_cdf must match that of the supplied LogisticGaussianProcess model object.
The pdf at the given input point is approximated using linear interpolation.
"""
function Distributions.cdf(lgp::LogisticGaussianProcess, params::NamedTuple{Names, Vals}, t::Real) where {Names, Vals<:Tuple}
    return _cdf(lgp, params, [t], Val(:val_cdf in Names))[1]
end
function Distributions.cdf(lgp::LogisticGaussianProcess, params::AbstractVector{NamedTuple{Names, Vals}}, t::Real) where {Names, Vals<:Tuple}
    return _cdf(lgp, params, [t], Val(:val_cdf in Names))
end
function Distributions.cdf(lgp::LogisticGaussianProcess, params::NamedTuple{Names, Vals}, t::AbstractVector{<:Real}) where {Names, Vals<:Tuple}
    return _cdf(lgp, params, t, Val(:val_cdf in Names))
end
function Distributions.cdf(lgp::LogisticGaussianProcess, params::AbstractVector{NamedTuple{Names, Vals}}, t::AbstractVector{<:Real}) where {Names, Vals<:Tuple}
    return _cdf(lgp, params, t, Val(:val_cdf in Names))
end

# If cdf has been evaluated on a grid, we use this to compute the linear interpolant:
function _cdf(lgp::LogisticGaussianProcess{T}, params::NamedTuple{Names, Vals}, t::AbstractVector{S}, ::Val{true}) where {T, Names, Vals<:Tuple, S<:Real}
    R = promote_type(T, S)
    bounds = BayesDensityCore.support(lgp)
    t_trans = (t .- bounds[1]) / (bounds[2] - bounds[1])
    n_bins = lgp.data.n_bins
    eval_grid = LinRange{R}(0, 1, n_bins+1)

    # Perform linear interpolation
    F_interp = interpolate(eval_grid, params.val_cdf, BSplineOrder(2))
    F_samp = F_interp.(t_trans)
    F_samp[t_trans .> bounds[2]] .= one(R)
    F_samp[t_trans .< bounds[1]] .= zero(R)
    return F_samp
end
function _cdf(lgp::LogisticGaussianProcess{T}, params::AbstractVector{NamedTuple{Names, Vals}}, t::AbstractVector{S}, ::Val{true}) where {T<:Real, Names, Vals<:Tuple, S<:Real}
    R = promote_type(T, S)
    bounds = BayesDensityCore.support(lgp)
    t_trans = (t .- bounds[1]) / (bounds[2] - bounds[1])
    n_bins = lgp.data.n_bins
    eval_grid = LinRange{R}(0, 1, n_bins+1)
    
    # Interpolate
    F_samp = zeros(R, (length(t), length(params)))
    for i in eachindex(params)
        F_interp = interpolate(eval_grid, params[i].val_cdf, BSplineOrder(2))
        F_samp[:, i] = F_interp.(t_trans)
        F_samp[t_trans .> bounds[2], i] .= one(R)
    end
    return F_samp
end

# If cdf has not been evaluated on a grid, we must evaluate it first:
function _cdf(lgp::LogisticGaussianProcess{T}, params::NamedTuple{Names, Vals}, t::AbstractVector{S}, ::Val{false}) where {T<:Real, Names, Vals<:Tuple, S<:Real}
    R = promote_type(T, S)
    bounds = BayesDensityCore.support(lgp)
    n_bins = lgp.data.n_bins
    t_trans = (t .- bounds[1]) / (bounds[2] - bounds[1])

    # Evaluate on grid:
    eval_unnorm = cumsum(exp.(params.β))
    eval_cdf_grid = vcat(0.0, eval_unnorm / eval_unnorm[end])
    eval_grid = LinRange{R}(0, 1, n_bins+1)

    # Interpolate
    F_interp = interpolate(eval_grid, eval_cdf_grid, BSplineOrder(2))
    F_samp = F_interp.(t_trans)
    F_samp[t_trans .> bounds[2]] .= one(R)
    return F_samp
end
function _cdf(lgp::LogisticGaussianProcess{T}, params::AbstractVector{NamedTuple{Names, Vals}}, t::AbstractVector{S}, ::Val{false}) where {T<:Real, Names, Vals<:Tuple, S<:Real}
    R = promote_type(T, S)
    bounds = BayesDensityCore.support(lgp)
    n_bins = lgp.data.n_bins
    t_trans = (t .- bounds[1]) / (bounds[2] - bounds[1])

    # Get evaluation grid for cumulative midpoint quadrature
    eval_grid = LinRange{R}(0, 1, n_bins+1)
    
    # Interpolate
    F_samp = zeros(R, (length(t), length(params)))
    for i in eachindex(params)
        eval_unnorm = cumsum(exp.(params[i].β))
        eval_cdf_grid = vcat(0.0, eval_unnorm / eval_unnorm[end])
        F_interp = interpolate(eval_grid, eval_cdf_grid, BSplineOrder(2))
        F_samp[:, i] = F_interp.(t_trans)
        F_samp[t_trans .> bounds[2], i] .= one(R)
    end
    return F_samp
end

function _check_lgpkwargs(x::AbstractVector{<:Real}, n_bins::Int, bounds::Tuple{<:Real, <:Real}, prior_variance_scale::Real, prior_length_scale::Real)
    (n_bins ≥ 1) || throw(ArgumentError("Number of bins must be a positive integer."))
    xmin, xmax = extrema(x)
    (bounds[1] < bounds[2]) || throw(ArgumentError("Supplied upper bound must be strictly greater than the lower bound."))
    (bounds[1] ≤ xmin ≤ xmax ≤ bounds[2]) || throw(ArgumentError("Data is not contained within supplied bounds."))
    hyperpar = [prior_variance_scale, prior_length_scale]
    hyperpar_symb = [:prior_variance_scale, :prior_length_scale]
    for i in eachindex(hyperpar)
        (0 < hyperpar[i] < Inf) || throw(ArgumentError("Hyperparameter $(hyperpar_symb[i]) must be a strictly positive finite number."))
    end
end