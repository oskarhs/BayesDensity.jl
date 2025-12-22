"""
    BSMModel{T<:Real} <: AbstractBayesDensityModel{T}
    
Struct representing a B-spline mixture model.

The BSMModel struct is used to generate quantities that are needed for the model fitting procedure.

# Constructors
    
    BSMModel(
        x::AbstractVector{<:Real},
        K::Int = get_default_splinedim(x);
        kwargs...
    ) 

# Arguments
* `x`: The data vector.
* `basis`: The B-spline basis in the model. Defaults to a regular (augmented) spline basis covering [minimum(x) - 0.05*R, maximum(x) + 0.05*R] where `R` is the sample range. 
* `K`: B-spline basis dimension of a regular augmented spline basis. Defaults to max(100, min(200, ⌈n/5⌉))
* `bounds`: A tuple specifying the range of the `K`-dimensional B-spline basis. Defaults to [minimum(x) - 0.05*R, maximum(x) + 0.05*R] where `R` is the sample range. 

# Keyword arguments
* `bounds`: A tuple giving the support of the B-spline mixture model.
* `n_bins`: Lower bound on the number of bins used when fitting the `BSMModel` to data. Binned fitting can be disabled by setting this equal to `nothing`. Defaults to `1000`.
* `a_τ`: Shape hyperparameter for the global smoothing parameter τ². Defaults to `1.0`.
* `b_τ`: Rate hyperparameter for the global smoothing parameter τ². Defaults to `1e-3`.
* `a_δ`: Shape hyperparameter for the local smoothing parameters δₖ². Defaults to `0.5`.
* `b_δ`: Rate hyperparameter for the local smoothing parameters δₖ². Defaults to `0.5`.

# Returns
* `bsm`: A B-Spline mixture model object.

# Examples
```julia
julia> x = (1.0 .- (1.0 .- LinRange(0.0, 1.0, 5000)) .^(1/3)).^(1/3);

julia> model = BSMModel(x)
200-dimensional BSMModel{Float64}:
Using 5000 binned observations on a regular grid consisting of 1187 bins.
 basis:  200-element BSplineBasis of order 4, domain [-0.05, 1.05]
 order:  4
 knots:  [-0.05, -0.05, -0.05, -0.05, -0.0444162, -0.0388325, -0.0332487, -0.027665, -0.0220812, -0.0164975  …  1.0165, 1.02208, 1.02766, 1.03325, 1.03883, 1.04442, 1.05, 1.05, 1.05, 1.05]

julia> model = BSMModel(x, 150; bounds=(0, 1), n_bins=nothing, b_τ = 5e-3);

julia> posterior_samples = sample(Random.Xoshiro(1), model, 5000; n_burnin = 1000);

julia> vip = varinf(model);
```

# Extended help

### Binned fitting
To disable binned fitting, one can set `n_bins=nothing`.
Note that the binning is only used as part of the model fitting procedure, and the structure of the resulting fitted model object is the same regardless of whether the binning step is performed or not.
Empirically, the results obtained from running the binned and unbinned model fitting procedures tend to be very similar.
We therefore recommend using the binned fitting procedure, due to the large improvements in model fitting speed, particularly for larger samples.

For computational reasons, the supplied number of bins is rounded up to the nearest integer such that the bin boundaries overlap with the knots of the spline basis.
This is done to ensure that at most 4 cubic splines have positive integrals over each bin.

### Hyperparameter selection
The hyperparameters `τ2` and `δ2[k]` govern the smoothness of the B-spline mixture prior through the centered random walk prior on β | τ2, δ2:

    β[k+2] = μ[k+2] + 2 {β[k+1] - μ[k+1]} - {β[k] - μ[k]} + τ * δ[k] * ϵ[k],

where ϵ[k] is standard normal.

The prior distributions of the local and global smoothing parameters are given by

    τ² ∼ InverseGamma(a_τ, b_τ)
    δₖ² ∼ InverseGamma(a_δ, b_δ),   1 ≤ k ≤ K-3.

As noninformative defaults, we suggest using `a_τ = 1`, `b_τ = 1e-3`, `a_δ = 0.5`, `b_δ = 0.5`.
To control the smoothness in the resulting density estimates, we recommend adjusting the value of `b_τ` while keeping the other hyperparameters fixed.
Setting `b_τ` to a smaller value generally yields smoother curves.
Similar priors for regression models suggest that values in the range [5e-5, 5e-3] are reasonable.
"""
struct BSMModel{T<:Real, A<:AbstractBSplineBasis, NT<:NamedTuple} <: AbstractBayesDensityModel{T}
    data::NT
    basis::A
    a_τ::T
    b_τ::T
    a_δ::T
    b_δ::T
    function BSMModel{T}(x::AbstractVector{<:Real}, K::Int = get_default_splinedim(x); bounds::Tuple{<:Real,<:Real} = get_default_bounds(x), n_bins::Union{Nothing,Int}=1000, a_τ::Real=1.0, b_τ::Real=1e-3, a_δ::Real=0.5, b_δ::Real=0.5) where {T<:Real}
        check_bsmkwargs(x, n_bins, bounds, a_τ, b_τ, a_δ, b_δ) # verify that supplied parameters make sense

        bs = BSplineBasis(BSplineOrder(4), LinRange{T}(bounds[1], bounds[2], K-2))
        K = length(bs)

        # Here: determine μ via the medians (e.g. we penalize differences away from the values that yield a uniform prior median)
        μ = compute_μ(bs)

        # Set up difference matrix:
        P = BandedMatrix((0=>fill(1, K-3), 1=>fill(-2, K-3), 2=>fill(1, K-3)), (K-3, K-1))

        T_a_τ = T(a_τ)
        T_b_τ = T(b_τ)
        T_a_δ = T(a_δ)
        T_b_δ = T(b_δ)
        T_x = T.(x)
        
        n = length(x)
        if !isnothing(n_bins)
            # Create binned B-Spline basis matrix
            B, b_ind, bincounts = create_spline_basis_matrix_binned(T_x, bs, n_bins)
            log_B = log.(B)

            data = (x = x, log_B = log_B, b_ind = b_ind, bincounts = bincounts, μ = μ, P = P, n = n)
        else
            B, b_ind = create_spline_basis_matrix(T_x, bs)
            log_B = log.(B)

            data = (x = x, log_B = log_B, b_ind = b_ind, μ = μ, P = P, n = n)
        end
        return new{T,typeof(bs),typeof(data)}(data, bs, T_a_τ, T_b_τ, T_a_δ, T_b_δ)
    end # NB! Might remove this constructor in the future (if we want to enforce the use of regularly spaced cubic splines)
    # In this cae, we will likely make bounds a keyword argument instead.
end
BSMModel(args...; kwargs...) = BSMModel{Float64}(args...; kwargs...)

function Base.:(==)(bsm1::BSMModel, bsm2::BSMModel)
    return basis(bsm1) == basis(bsm2) && bsm1.data == bsm2.data && params(bsm1) == params(bsm2)
end

BSplineKit.basis(bsm::B) where {B<:BSMModel} = bsm.basis
BSplineKit.order(bsm::B) where {B<:BSMModel} = order(bsm.basis)
BSplineKit.length(bsm::B) where {B<:BSMModel} = length(bsm.basis)
BSplineKit.knots(bsm::B) where {B<:BSMModel} = knots(bsm.basis)

"""
    support(bsm::BSMModel) -> NTuple{2, <:Real}

Get the support of the B-Spline mixture model `bsm`.
"""
BayesDensityCore.support(bsm::BSMModel) = boundaries(bsm.basis)

"""
    hyperparams(
        bsm::BSMModel{T}
    ) where {T} -> @NamedTuple{a_τ::T, b_τ::T, a_δ::T, b_δ::T}

Returns the hyperparameters of the B-Spline mixture model `bsm` as a `NamedTuple`.
"""
BayesDensityCore.hyperparams(bsm::BSMModel) = (a_τ=bsm.a_τ, b_τ=bsm.b_τ, a_δ=bsm.a_δ, b_δ=bsm.b_δ)

Base.eltype(::BSMModel{T,<:AbstractBSplineBasis,<:NamedTuple}) where {T<:Real} = T

# Print method for binned data
function Base.show(io::IO, ::MIME"text/plain", bsm::BSMModel{T, A, NamedTuple{(:x, :log_B, :b_ind, :bincounts, :μ, :P, :n), Vals}}) where {T, A, Vals}
    n_bins = length(bsm.data.b_ind)
    println(io, length(bsm), "-dimensional ", nameof(typeof(bsm)), '{', eltype(bsm), "}:")
    println(io, "Using ", bsm.data.n, " binned observations on a regular grid consisting of ", n_bins, " bins.")
    print(io, " basis:  ")
    let io = IOContext(io, :compact => true, :limit => true)
        summary(io, basis(bsm))
    end
    println(io, "\n order:  ", order(bsm))
    let io = IOContext(io, :compact => true, :limit => true)
        print(io, " knots:  ", knots(bsm))
    end
    nothing
end

# Print method for unbinned data
function Base.show(io::IO, ::MIME"text/plain", bsm::BSMModel{T, A, NamedTuple{(:x, :log_B, :b_ind, :μ, :P, :n), Vals}}) where {T, A, Vals}
    println(io, length(bsm), "-dimensional ", nameof(typeof(bsm)), '{', eltype(bsm), "}:")
    println(io, "Using ", bsm.data.n, " unbinned observations.")
    print(io, " basis:  ")
    let io = IOContext(io, :compact => true, :limit => true)
        summary(io, basis(bsm))
    end
    println(io, "\n order:  ", order(bsm))
    let io = IOContext(io, :compact => true, :limit => true)
        print(io, " knots:  ", knots(bsm))
    end
    nothing
end

Base.show(io::IO, bsm::BSMModel) = show(io, MIME("text/plain"), bsm)

"""
    pdf(
        bsm::BSMModel,
        params::Union{NamedTuple, AbstractVector{NamedTuple}},
        t::Union{Real, AbstractVector{<:Real}}
    ) -> Union{Real, Vector{<:Real}, Matrix{<:Real}}

Evaluate f(t | η) when the model parameters are equal to η.

The named tuple should contain a field named `:spline_coefs` or `:β`.
"""
function Distributions.pdf(bsm::BSMModel, params::NamedTuple{Names, Vals}, t::Real) where {Names, Vals}
    _pdf(bsm, params, t, Val(:spline_coefs in Names))
end
function Distributions.pdf(bsm::BSMModel, params::NamedTuple{Names, Vals}, t::AbstractVector{T}) where {Names, Vals, T<:Real}
    _pdf(bsm, params, t, Val(:spline_coefs in Names))
end
function Distributions.pdf(bsm::BSMModel, params::AbstractVector{NamedTuple{Names, Vals}}, t::Real) where {Names, Vals}
    _pdf(bsm, params, t, Val(:spline_coefs in Names))
end
function Distributions.pdf(bsm::BSMModel, params::AbstractVector{NamedTuple{Names, Vals}}, t::AbstractVector{T}) where {Names, Vals, T<:Real}
    _pdf(bsm, params, t, Val(:spline_coefs in Names))
end

# Compile-time dispatch
function _pdf(bsm::BSMModel, params::NamedTuple{Names, Vals}, t, ::Val{true}) where {Names, Vals}
    # Coefs given, no need to compute them
    spline_coefs = params.spline_coefs
    return _pdf(bsm, spline_coefs, t)
end
function _pdf(bsm::BSMModel, params::NamedTuple{Names, Vals}, t, ::Val{false}) where {Names, Vals}
    # Coefs not given, compute them from β
    θ = logistic_stickbreaking(params.β)
    spline_coefs = theta_to_coef(θ, basis)
    return _pdf(bsm, spline_coefs, t)
end
function _pdf(bsm::BSMModel{T, A, N}, params::AbstractVector{NamedTuple{Names, Vals}}, t::Union{S, AbstractVector{S}}, ::Val{true}) where {T<:Real, A, N, Names, Vals, S<:Real}
    # Build coefficient matrix (coefs are given)
    # TODO allow for different types here
    spline_coefs = Matrix{promote_type(T, S)}(undef, (length(bsm), length(params)))
    for i in eachindex(params)
        spline_coefs[:, i] = params[i].spline_coefs
    end
    return _pdf(bsm, spline_coefs, t)
end
function _pdf(bsm::BSMModel{T, A, N}, params::AbstractVector{NamedTuple{Names, Vals}}, t::Union{S, AbstractVector{S}}, ::Val{false}) where {T<:Real, A, N, Names, Vals, S<:Real}
    # Build coefficient matrix (coefs not given)
    # TODO allow for different types here
    spline_coefs = Matrix{promote_type(T, S)}(undef, (length(bsm), length(params)))
    for i in eachindex(params)
        θ = logistic_stickbreaking(params[i].β)
        spline_coefs[:, i] = theta_to_coef(θ, basis(bsm))
    end
    return _pdf(bsm, spline_coefs, t)
end

# Batch evalutation (for mutiple samples, it is more efficient to reuse computation of spline basis terms)
function _pdf(bsm::BSMModel, spline_coefs::AbstractMatrix{<:Real}, t::AbstractVector{<:Real})
    bs = basis(bsm)
    B_sparse = create_unnormalized_sparse_spline_basis_matrix(t, bs)
    f_samp = B_sparse * spline_coefs
    return f_samp
end
_pdf(bsm::BSMModel, spline_coefs::AbstractMatrix{<:Real}, t::Real) = _pdf(bsm, spline_coefs, [t])

# Evaluate for single sample
function _pdf(bsm::BSMModel, spline_coefs::AbstractVector{<:Real}, t::Union{Real, AbstractVector{<:Real}})
    f = Spline(basis(bsm), spline_coefs)
    return f.(t)
end

# More efficient version of the posterior mean (we only need to average the coefficients)
Distributions.mean(ps::PosteriorSamples{T, M, <:AbstractVector{NamedTuple{Names, Vals}}}, t::S) where {T<:Real, M<:BSMModel, Names, Vals, S<:Real} = _mean(ps, t, Val(:spline_coefs in Names))
Distributions.mean(ps::PosteriorSamples{T, M, <:AbstractVector{NamedTuple{Names, Vals}}}, t::S) where {T<:Real, M<:BSMModel, Names, Vals, S<:AbstractVector{<:Real}} = _mean(ps, t, Val(:spline_coefs in Names))


function _mean(ps::PosteriorSamples{T, M, <:AbstractVector}, t::S, ::Val{true}) where {T<:Real, M<:BSMModel, S<:Union{Real, AbstractVector{<:Real}}}
    mean_spline_coefs = zeros(T, length(model(ps)))
    for i in eachindex(ps.samples)
        mean_spline_coefs += ps.samples[i].spline_coefs
    end
    mean_spline_coefs /= length(ps.samples)
    return _mean(ps, mean_spline_coefs, t)
end
function _mean(ps::PosteriorSamples{T, M, <:AbstractVector}, t::S, ::Val{false}) where {T<:Real, M<:BSMModel, S<:Union{Real, AbstractVector{<:Real}}}
    mean_spline_coefs = zeros(T, length(model(ps)))
    bs = basis(model(ps))
    for i in eachindex(ps.samples)
        θ = logistic_stickbreaking(ps.samples[i].β)
        mean_spline_coefs += theta_to_coef(θ, bs)
    end
    mean_spline_coefs /= length(ps.samples)
    return _mean(ps, mean_spline_coefs, t)
end
function _mean(ps::PosteriorSamples{T, M, <:AbstractVector}, mean_spline_coefs::AbstractVector{<:Real}, t::S) where {T<:Real, M<:BSMModel, S<:Union{Real, AbstractVector{<:Real}}}
    meanfunc = Spline(basis(model(ps)), mean_spline_coefs)
    return meanfunc.(t)
end

function get_default_splinedim(x::AbstractVector{<:Real})
    n = length(x)
    return max(min(200, ceil(Int, n/10)), 100)
end

function get_default_bounds(x::AbstractVector{<:Real})
    xmin, xmax = extrema(x)
    R = xmax - xmin
    return xmin - 0.05*R, xmax + 0.05*R
end

function check_bsmkwargs(x::AbstractVector{<:Real}, n_bins::Union{Nothing,Int}, bounds::Tuple{<:Real, <:Real}, a_τ::Real, b_τ::Real, a_δ::Real, b_δ::Real)
    if !isnothing(n_bins) && n_bins ≤ 1
        throw(ArgumentError("Number of bins must be a positive integer or 'nothing'."))
    end
    xmin, xmax = extrema(x)
    if bounds[1] ≥ bounds[2]
        throw(ArgumentError("Supplied upper bound must be strictly greater than the lower bound."))
    elseif bounds[1] > xmin || bounds[2] < xmax
        throw(ArgumentError("Data is not contained within supplied bounds."))
    end
    hyperpar = [a_τ, b_τ, a_δ, b_δ]
    hyperpar_symb = [:a_τ, :b_τ, :a_δ, :b_δ]
    for i in eachindex(hyperpar)
        if hyperpar[i] ≤ 0.0 || hyperpar[i] == Inf
            throw(ArgumentError("Hyperparameter $(hyperpar_symb[i]) must be a strictly positive finite number."))
        end
    end
end