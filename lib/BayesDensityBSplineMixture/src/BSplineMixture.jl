"""
    BSplineMixture{T<:Real} <: AbstractBayesDensityModel{T}
    
Struct representing a B-spline mixture model.

# Constructors
    
    BSplineMixture(x::AbstractVector{<:Real}; kwargs...)
    BSplineMixture{T}(x::AbstractVector{<:Real}; kwargs...)

# Arguments
* `x`: The data vector.

# Keyword arguments
* `K`: B-spline basis dimension of a regular augmented spline basis. Defaults to max(100, min(200, ⌈n/5⌉))
* `bounds`: A tuple giving the support of the B-spline mixture model.
* `n_bins`: Lower bound on the number of bins used when fitting the `BSplineMixture` to data. Binned fitting can be disabled by setting this equal to `nothing`. The default setting uses unbinned fitting if `length(x) ≤ 1200` and `1000` bins otherwise.
* `prior_global_shape`: Shape hyperparameter for the global smoothing parameter τ². Defaults to `1.0`.
* `prior_global_rate`: Rate hyperparameter for the global smoothing parameter τ². Defaults to `1e-3`.
* `prior_local_shape`: Shape hyperparameter for the local smoothing parameters δₖ². Defaults to `0.5`.
* `prior_local_rate`: Rate hyperparameter for the local smoothing parameters δₖ². Defaults to `0.5`.
* `prior_stdev`: Prior standard deviation of the first two unconstrained spline parameters β₁ and β₂. Defaults to `1e5`.

# Returns
* `bsm`: A B-Spline mixture model object.

# Examples
```julia-repl
julia> x = (1.0 .- (1.0 .- LinRange(0.0, 1.0, 5000)) .^(1/3)).^(1/3);

julia> model = BSplineMixture(x)
200-dimensional BSplineMixture{Float64}:
Using 5000 binned observations on a regular grid consisting of 1187 bins.
 support: (-0.05, 1.05)
Hyperparameters:
 prior_global_shape = 1.0, prior_global_rate = 0.001
 prior_local_shape = 0.5, prior_local_rate = 0.5

julia> model = BSplineMixture(x; K = 150, bounds=(0, 1), n_bins=nothing, prior_global_rate = 5e-3);
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
The global variance parameter `τ2` and the local variance parameters `δ2[k]` govern the smoothness of the B-spline mixture prior through the centered random walk prior on β | τ2, δ2:

    β[k+2] = μ[k+2] + 2 {β[k+1] - μ[k+1]} - {β[k] - μ[k]} + τ * δ[k] * ϵ[k],

where ϵ[k] is standard normal. The first two parameters β[1] and β[2] are assigned diffuse N(0, σ²) priors.

The prior distributions of the local and global smoothing parameters are given by

    τ² ∼ InverseGamma(a_τ, b_τ)
    δₖ² ∼ InverseGamma(a_δ, b_δ),   1 ≤ k ≤ K-3.

As noninformative defaults, we suggest using `prior_global_shape = 1`, `prior_global_rate = 1e-3`, `prior_local_shape = 0.5`, `prior_local_rate = 0.5` and `prior_stdev = 1e5`.
To control the smoothness in the resulting density estimates, we recommend adjusting the value of `prior_global_rate` while keeping the other hyperparameters fixed.
Setting `prior_global_rate` to a smaller value generally yields smoother curves.
Similar priors for regression models suggest that values in the range [5e-5, 5e-3] are reasonable.
"""
struct BSplineMixture{T<:Real, A<:AbstractBSplineBasis, NT<:NamedTuple} <: AbstractBayesDensityModel{T}
    data::NT
    basis::A
    prior_global_shape::T
    prior_global_rate::T
    prior_local_shape::T
    prior_local_rate::T
    prior_stdev::T
    function BSplineMixture{T}(x::AbstractVector{<:Real};
        K::Int = _get_default_splinedim(x),
        bounds::Tuple{<:Real,<:Real} = _get_default_bounds(x),
        n_bins::Union{Nothing,Int}=_get_default_bins(x),
        prior_global_shape::Real=1.0,
        prior_global_rate::Real=1e-3,
        prior_local_shape::Real=0.5,
        prior_local_rate::Real=0.5,
        prior_stdev::Real=1e5
    ) where {T<:Real}
        _check_bsmkwargs(x, n_bins, bounds, prior_global_shape, prior_global_rate, prior_local_shape, prior_local_rate, prior_stdev) # verify that supplied parameters make sense

        bs = BSplineBasis(BSplineOrder(4), LinRange{T}(bounds[1], bounds[2], K-2))
        K = length(bs)

        # Here: determine μ via the medians (e.g. we penalize differences away from the values that yield a uniform prior median)
        μ = compute_μ(bs)

        # Set up difference matrix:
        P = BandedMatrix((0=>fill(1, K-3), 1=>fill(-2, K-3), 2=>fill(1, K-3)), (K-3, K-1))

        T_a_τ = T(prior_global_shape)
        T_b_τ = T(prior_global_rate)
        T_a_δ = T(prior_local_shape)
        T_b_δ = T(prior_local_rate)
        T_σ = T(prior_stdev)
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
        return new{T,typeof(bs),typeof(data)}(data, bs, T_a_τ, T_b_τ, T_a_δ, T_b_δ, T_σ)
    end
end
BSplineMixture(args...; kwargs...) = BSplineMixture{Float64}(args...; kwargs...)

function Base.:(==)(bsm1::BSplineMixture, bsm2::BSplineMixture)
    return basis(bsm1) == basis(bsm2) && bsm1.data == bsm2.data && hyperparams(bsm1) == hyperparams(bsm2)
end

BSplineKit.basis(bsm::BSplineMixture) = bsm.basis
BSplineKit.order(bsm::BSplineMixture) = order(bsm.basis)
BSplineKit.length(bsm::BSplineMixture) = length(bsm.basis)
BSplineKit.knots(bsm::BSplineMixture) = knots(bsm.basis)

BayesDensityCore.support(bsm::BSplineMixture{T}) where {T} = boundaries(bsm.basis)

"""
    hyperparams(
        bsm::BSplineMixture{T}
    ) where {T} -> @NamedTuple{prior_global_shape::T, prior_global_rate::T, prior_local_shape::T, prior_local_rate::T, prior_stdev::T}

Returns the hyperparameters of the B-Spline mixture model `bsm` as a `NamedTuple`.
"""
BayesDensityCore.hyperparams(bsm::BSplineMixture) = (prior_global_shape=bsm.prior_global_shape, prior_global_rate=bsm.prior_global_rate, prior_local_shape=bsm.prior_local_shape, prior_local_rate=bsm.prior_local_rate, prior_stdev=bsm.prior_stdev)

function BayesDensityCore.default_grid_points(bsm::BSplineMixture{T}) where {T}
    xmin, xmax = support(bsm)
    return LinRange{T}(xmin, xmax, 2001)
end

# Print method for binned data
function Base.show(io::IO, ::MIME"text/plain", bsm::BSplineMixture{T, A, NamedTuple{(:x, :log_B, :b_ind, :bincounts, :μ, :P, :n), Vals}}) where {T, A, Vals}
    n_bins = length(bsm.data.b_ind)
    println(io, length(bsm), "-dimensional ", nameof(typeof(bsm)), '{', eltype(bsm), "}:")
    println(io, "Using ", bsm.data.n, " binned observations on a regular grid consisting of ", n_bins, " bins.")
    let io = IOContext(io, :compact => true, :limit => true)
        println(io, " support: ", BayesDensityCore.support(bsm))
        println(io, "Hyperparameters: ")
        println(io, " prior_global_shape = " , bsm.prior_global_shape, ", prior_global_rate = ", bsm.prior_global_rate)
        println(io, " prior_local_shape = " , bsm.prior_local_shape, ", prior_local_rate = ", bsm.prior_local_rate)
        print(io, "prior_stdev = ", bsm.prior_stdev)
    end
    nothing
end

# Print method for unbinned data
function Base.show(io::IO, ::MIME"text/plain", bsm::BSplineMixture{T, A, NamedTuple{(:x, :log_B, :b_ind, :μ, :P, :n), Vals}}) where {T, A, Vals}
    println(io, length(bsm), "-dimensional ", nameof(typeof(bsm)), '{', eltype(bsm), "}:")
    println(io, "Using ", bsm.data.n, " unbinned observations.")
    let io = IOContext(io, :compact => true, :limit => true)
        println(io, " support: ", BayesDensityCore.support(bsm))
        println(io, "Hyperparameters: ")
        println(io, " prior_global_shape = " , bsm.prior_global_shape, ", prior_global_rate = ", bsm.prior_global_rate)
        println(io, " prior_local_shape = " , bsm.prior_local_shape, ", prior_local_rate = ", bsm.prior_local_rate)
        print(io, "prior_stdev = ", bsm.prior_stdev)
    end
    nothing
end

Base.show(io::IO, bsm::BSplineMixture) = show(io, MIME("text/plain"), bsm)

"""
    pdf(
        bsm::BSplineMixture,
        params::NamedTuple,
        t::Union{Real, AbstractVector{<:Real}}
    ) -> Union{Real, Vector{<:Real}}

    pdf(
        bsm::BSplineMixture,
        params::AbstractVector{NamedTuple},
        t::Union{Real, AbstractVector{<:Real}}
    ) -> Matrix{<:Real}

Evaluate ``f(t\\, |\\, \\boldsymbol{\\eta})`` for a given `BSplineMixture` when the model parameters of the NamedTuple `params` are given by ``\\boldsymbol{\\eta}``.

The named tuple should contain a field named `:spline_coefs` or `:β`.
"""
function Distributions.pdf(bsm::BSplineMixture, params::NamedTuple{Names, Vals}, t::Real) where {Names, Vals<:Tuple}
    _pdf(bsm, params, t, Val(:spline_coefs in Names))
end
function Distributions.pdf(bsm::BSplineMixture, params::NamedTuple{Names, Vals}, t::AbstractVector{T}) where {Names, Vals<:Tuple, T<:Real}
    _pdf(bsm, params, t, Val(:spline_coefs in Names))
end
function Distributions.pdf(bsm::BSplineMixture, params::AbstractVector{NamedTuple{Names, Vals}}, t::Real) where {Names, Vals<:Tuple}
    _pdf(bsm, params, t, Val(:spline_coefs in Names))
end
function Distributions.pdf(bsm::BSplineMixture, params::AbstractVector{NamedTuple{Names, Vals}}, t::AbstractVector{T}) where {Names, Vals<:Tuple, T<:Real}
    _pdf(bsm, params, t, Val(:spline_coefs in Names))
end

# Compile-time dispatch
function _pdf(bsm::BSplineMixture, params::NamedTuple{Names, Vals}, t, ::Val{true}) where {Names, Vals}
    # Coefs given, no need to compute them
    spline_coefs = params.spline_coefs
    return _pdf(bsm, spline_coefs, t)
end
function _pdf(bsm::BSplineMixture, params::NamedTuple{Names, Vals}, t, ::Val{false}) where {Names, Vals}
    # Coefs not given, compute them from β
    θ = logistic_stickbreaking(params.β)
    spline_coefs = theta_to_coef(θ, basis(bsm))
    return _pdf(bsm, spline_coefs, t)
end
function _pdf(bsm::BSplineMixture{T}, params::AbstractVector{NamedTuple{Names, Vals}}, t::Union{S, AbstractVector{S}}, ::Val{true}) where {T<:Real, Names, Vals, S<:Real}
    # Build coefficient matrix (coefs are given)
    spline_coefs = Matrix{promote_type(T, S)}(undef, (length(bsm), length(params)))
    for i in eachindex(params)
        spline_coefs[:, i] = params[i].spline_coefs
    end
    return _pdf(bsm, spline_coefs, t)
end
function _pdf(bsm::BSplineMixture{T}, params::AbstractVector{NamedTuple{Names, Vals}}, t::Union{S, AbstractVector{S}}, ::Val{false}) where {T<:Real, Names, Vals, S<:Real}
    # Build coefficient matrix (coefs not given)
    spline_coefs = Matrix{promote_type(T, S)}(undef, (length(bsm), length(params)))
    for i in eachindex(params)
        θ = logistic_stickbreaking(params[i].β)
        spline_coefs[:, i] = theta_to_coef(θ, basis(bsm))
    end
    return _pdf(bsm, spline_coefs, t)
end

# Batch evalutation (for mutiple samples, it is more efficient to reuse computation of spline basis terms)
function _pdf(bsm::BSplineMixture, spline_coefs::AbstractMatrix{<:Real}, t::AbstractVector{<:Real})
    bs = basis(bsm)
    B_sparse = create_unnormalized_sparse_spline_basis_matrix(t, bs)
    f_samp = B_sparse * spline_coefs
    return f_samp
end
_pdf(bsm::BSplineMixture, spline_coefs::AbstractMatrix{<:Real}, t::Real) = _pdf(bsm, spline_coefs, [t])

# Evaluate for single sample
function _pdf(bsm::BSplineMixture, spline_coefs::AbstractVector{<:Real}, t::Union{Real, AbstractVector{<:Real}})
    f = Spline(basis(bsm), spline_coefs)
    return f.(t)
end

# Cdf evaluation:
"""
    cdf(
        bsm::BSplineMixture,
        params::NamedTuple,
        t::Union{Real, AbstractVector{<:Real}}
    ) -> Union{Real, Vector{<:Real}}

    cdf(
        bsm::BSplineMixture,
        params::AbstractVector{NamedTuple},
        t::Union{Real, AbstractVector{<:Real}}
    ) -> Matrix{<:Real}

Evaluate ``F(t\\, |\\, \\boldsymbol{\\eta})`` for a given `BSplineMixture` when the model parameters of the NamedTuple `params` are given by ``\\boldsymbol{\\eta}``.

The named tuple should contain a field named `:spline_coefs` or `:β`.
"""
function Distributions.cdf(bsm::BSplineMixture, params::NamedTuple{Names, Vals}, t::Real) where {Names, Vals<:Tuple}
    bounds = BayesDensityCore.support(bsm)
    _cdf(bsm, params, clamp.(t, bounds[1], bounds[2]), Val(:spline_coefs in Names))
end
function Distributions.cdf(bsm::BSplineMixture, params::NamedTuple{Names, Vals}, t::AbstractVector{T}) where {Names, Vals<:Tuple, T<:Real}
    bounds = BayesDensityCore.support(bsm)
    _cdf(bsm, params, clamp.(t, bounds[1], bounds[2]), Val(:spline_coefs in Names))end
function Distributions.cdf(bsm::BSplineMixture, params::AbstractVector{NamedTuple{Names, Vals}}, t::Real) where {Names, Vals<:Tuple}
    bounds = BayesDensityCore.support(bsm)
    _cdf(bsm, params, clamp.(t, bounds[1], bounds[2]), Val(:spline_coefs in Names))
end
function Distributions.cdf(bsm::BSplineMixture, params::AbstractVector{NamedTuple{Names, Vals}}, t::AbstractVector{T}) where {Names, Vals<:Tuple, T<:Real}
    bounds = BayesDensityCore.support(bsm)
    _cdf(bsm, params, clamp.(t, bounds[1], bounds[2]), Val(:spline_coefs in Names))
end

# Compile-time dispatch
function _cdf(bsm::BSplineMixture, params::NamedTuple{Names, Vals}, t, ::Val{true}) where {Names, Vals}
    # Coefs given, no need to compute them
    spline_coefs = params.spline_coefs
    return _cdf(bsm, spline_coefs, t)
end
function _cdf(bsm::BSplineMixture, params::NamedTuple{Names, Vals}, t, ::Val{false}) where {Names, Vals}
    # Coefs not given, compute them from β
    θ = logistic_stickbreaking(params.β)
    spline_coefs = theta_to_coef(θ, basis(bsm))
    return _cdf(bsm, spline_coefs, t)
end
function _cdf(bsm::BSplineMixture{T}, params::AbstractVector{NamedTuple{Names, Vals}}, t::Union{S, AbstractVector{S}}, ::Val{true}) where {T<:Real, Names, Vals, S<:Real}
    # Build coefficient matrix (coefs are given)
    spline_coefs = Matrix{promote_type(T, S)}(undef, (length(bsm), length(params)))
    for i in eachindex(params)
        spline_coefs[:, i] = params[i].spline_coefs
    end
    return _cdf(bsm, spline_coefs, t)
end
function _cdf(bsm::BSplineMixture{T}, params::AbstractVector{NamedTuple{Names, Vals}}, t::Union{S, AbstractVector{S}}, ::Val{false}) where {T<:Real, Names, Vals, S<:Real}
    # Build coefficient matrix (coefs not given)
    spline_coefs = Matrix{promote_type(T, S)}(undef, (length(bsm), length(params)))
    for i in eachindex(params)
        θ = logistic_stickbreaking(params[i].β)
        spline_coefs[:, i] = theta_to_coef(θ, basis(bsm))
    end
    return _cdf(bsm, spline_coefs, t)
end

# Batch evalutation (for mutiple samples, it is more efficient to reuse computation of spline basis terms)
function _cdf(bsm::BSplineMixture, spline_coefs::AbstractMatrix{<:Real}, t::AbstractVector{<:Real})
    bs = basis(bsm)
    B_int = _create_integrated_sparse_spline_basis_matrix(t, bs)
    spline_coefs_int = _get_integrated_spline_coefs(bs, spline_coefs)
    F_samp = B_int * spline_coefs_int
    return F_samp
end
_cdf(bsm::BSplineMixture, spline_coefs::AbstractMatrix{<:Real}, t::Real) = _cdf(bsm, spline_coefs, [t])

# Evaluate for single sample
function _cdf(bsm::BSplineMixture, spline_coefs::AbstractVector{<:Real}, t::Union{Real, AbstractVector{<:Real}})
    F = integral(Spline(basis(bsm), spline_coefs))
    return F.(t)
end

# More efficient version of the posterior mean (we only need to average the coefficients)
for func in (:pdf, :cdf)
    @eval begin
        Distributions.mean(ps::PosteriorSamples{<:Real, <:AbstractVector, <:BSplineMixture}, ::typeof($func), t::Real) = _mean(ps, $func, t)
        Distributions.mean(ps::PosteriorSamples{<:Real, <:AbstractVector, <:BSplineMixture}, ::typeof($func), t::AbstractVector{<:Real}) = _mean(ps, $func, t)

        function _mean(ps::PosteriorSamples{T, <:AbstractVector,<:BSplineMixture}, ::typeof($func), t::S) where {T<:Real, S<:Union{Real, AbstractVector{<:Real}}}
            mean_spline_coefs = zeros(T, length(model(ps)))
            samples = ps.samples[ps.non_burnin_ind]
            for i in eachindex(samples)
                mean_spline_coefs += samples[i].spline_coefs
            end
            mean_spline_coefs /= length(samples)
            return _mean(ps, $func, mean_spline_coefs, t)
        end
    end
end
function _mean(ps::PosteriorSamples{T, <:AbstractVector,<:BSplineMixture}, ::typeof(pdf), mean_spline_coefs::AbstractVector{<:Real}, t::S) where {T<:Real, S<:Union{Real, AbstractVector{<:Real}}}
    meanfunc = Spline(basis(model(ps)), mean_spline_coefs)
    return meanfunc.(t)
end
function _mean(ps::PosteriorSamples{T, <:AbstractVector,<:BSplineMixture}, ::typeof(cdf), mean_spline_coefs::AbstractVector{<:Real}, t::S) where {T<:Real, S<:Union{Real, AbstractVector{<:Real}}}
    meanfunc = Spline(basis(model(ps)), mean_spline_coefs)
    bmin, _ = support(model(ps))
    F_bar = integral(meanfunc)
    return F_bar.(t) .- F_bar(bmin)
end

_get_default_splinedim(x::AbstractVector{<:Real}) = max(min(200, ceil(Int, length(x)/10)), 100)

function _get_default_bounds(x::AbstractVector{<:Real})
    xmin, xmax = extrema(x)
    R = xmax - xmin
    return xmin - 0.05*R, xmax + 0.05*R
end

_get_default_bins(x::AbstractVector{<:Real}) = ifelse(length(x) ≤ 1200, nothing, 1000)

function _check_bsmkwargs(x::AbstractVector{<:Real}, n_bins::Union{Nothing,Int}, bounds::Tuple{<:Real, <:Real}, prior_global_shape::Real, prior_global_rate::Real, prior_local_shape::Real, prior_local_rate::Real, prior_stdev::Real)
    (isnothing(n_bins) || n_bins ≥ 1) || throw(ArgumentError("Number of bins must be a positive integer or 'nothing'."))
    xmin, xmax = extrema(x)
    (bounds[1] < bounds[2]) || throw(ArgumentError("Supplied upper bound must be strictly greater than the lower bound."))
    (bounds[1] ≤ xmin ≤ xmax ≤ bounds[2]) || throw(ArgumentError("Data is not contained within supplied bounds."))
    hyperpar = [prior_global_shape, prior_global_rate, prior_local_shape, prior_local_rate, prior_stdev]
    hyperpar_symb = [:prior_global_shape, :prior_global_rate, :prior_local_shape, :prior_local_rate, :prior_stdev]
    for i in eachindex(hyperpar)
        (0 < hyperpar[i] < Inf) || throw(ArgumentError("Hyperparameter $(hyperpar_symb[i]) must be a strictly positive finite number."))
    end
end