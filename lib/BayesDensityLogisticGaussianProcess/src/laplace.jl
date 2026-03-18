"""
    LogisticGaussianProcessLaplacePosterior{T<:Real, A<:MvNormalCanon{T}, B<:InverseGamma{T}, M<:BSplineMixture} <: Abstractlaposterior{T}

Struct representing the Laplace approximation to the posterior distribution of a [`LogisticGaussianProcess`](@ref).

# Fields
* `q_β`: Distribution representing the multivariate normal approximation to the posterior of q*(β | σ2, λ), given optimal hyperparameters.
* `σ2`: Optimal value of the variance parameter.
* `λ`: Optimal value of the length parameter.
* `lgp`: The `LogisticGaussianProcess` to which the Laplace approximation was fit.
"""
struct LogisticGaussianProcessLaplacePosterior{T<:Real, A<:AbstractMvNormal, L<:LogisticGaussianProcess{T}} <: AbstractLaplacePosterior{T}
    q_β::A
    σ2::T
    λ::T
    lgp::L
    function LogisticGaussianProcessLaplacePosterior{T}(
        μ::AbstractVector{T},
        Σ::AbstractMatrix{T},
        σ2::T,
        λ::T,
        lgp::LogisticGaussianProcess{T}
    ) where {T<:Real}
        q_β = MvNormal(μ, Σ)
        return new{T, typeof(q_β), typeof(lgp)}(q_β, σ2, λ, lgp)
    end
end

BayesDensityCore.model(lap::LogisticGaussianProcessLaplacePosterior) = lap.lgp

function Base.show(io::IO, ::MIME"text/plain", lap::LogisticGaussianProcessLaplacePosterior{T}) where {T}
    println(io, nameof(typeof(lap)), "{", T, "} vith optimal hyperparameters:")
    let io = IOContext(io, :compact => true, :limit => true)
        println(io, " σ2 = ", lap.σ2)
        println(io, " λ = ", lap.λ)
    end
    println(io, "Model:")
    print(io, model(lap))
    nothing
end

Base.show(io::IO, lap::LogisticGaussianProcessLaplacePosterior) = show(io, MIME("text/plain"), lap)

function StatsBase.sample(
    rng::AbstractRNG,
    lgp_laplace::LogisticGaussianProcessLaplacePosterior{T},
    n_samples::Int
) where {T<:Real}
    (; q_β, lgp) = lgp_laplace
    bounds = BayesDensityCore.support(lgp)
    n_bins = lgp.data.n_bins
    samples = Vector{NamedTuple{(:β, :val_pdf, :val_cdf), Tuple{Vector{T}, Vector{T}, Vector{T}}}}(undef, n_samples)
    for i in 1:n_samples
        β = rand(rng, q_β)
        exp_β = exp.(β)
        eval_unnorm = cumsum(exp_β)
        val_pdf = n_bins * exp_β / (eval_unnorm[end]*(bounds[2] - bounds[1]))
        val_cdf = vcat(0.0, eval_unnorm / eval_unnorm[end])
        samples[i] = (β = β, val_pdf = val_pdf, val_cdf = val_cdf)
    end
    return PosteriorSamples{T}(samples, lgp, n_samples, 0)
end

"""
    laplace_approximation(
        lgp::LogisticGaussianProcess{T};
        initial_params::NamedTuple = (σ2 = T(1.0), λ = T(1.0)),
        max_iter_newton::Int       = 100,
        atol_newton::Real          = 1e-3,
        rtol_newton::Real          = 1e-3,
    ) where {T<:Real} -> LogisticGaussianProcessLaplacePosterior{T}

Find a Laplace approximation to the posterior distribution of a [`LogisticGaussianProcess`](@ref).

Estimates the hyperparameters `σ2` and `λ` by maximizing the (approximate) marginal likelihood.

# Arguments
* `lgp`: The `LogisticGaussianProcess` whose posterior we want to approximate.

# Keyword arguments
* `initial_params`: Initial values of the parameters `σ2` and `λ`, supplied as a NamedTuple.
* `max_iter_newton`: Maximal number of Newton iterations for the inner optimization. Defaults to `100`.
* `atol_newton`: Absolute tolerance used to determine convergence in the inner optimization. Defaults to `1e-3`.
* `rtol_newton`: Relative tolerance used to determine convergence in the inner optimization. Defaults to `1e-3`.

# Returns
* `vip`: A [`LogisticGaussianProcessLaplacePosterior`](@ref) object representing the variational posterior.
* `info`: A [`VariationalOptimizationResult`](@ref) describing the result of the optimization.

# Examples
```julia-repl
julia> using Random

julia> x = (1.0 .- (1.0 .- LinRange(0.0, 1.0, 5000)) .^(1/3)).^(1/3);

julia> lgp = LogisticGaussianProcess(x);

julia> vip, info = laplace_approximation(lgp);

julia> vip, info = laplace_approximation(lgp; rtol_newton=1e-4, max_iter_newton=1000);
```
"""
function laplace_approximation(
    lgp::LogisticGaussianProcess{T};
    initial_params::NamedTuple = (σ2 = T(5.0), λ = T(1.0)),
    max_iter_newton::Int       = 100,
    atol_newton::Real          = 1e-3,
    rtol_newton::Real          = 1e-3
) where {T<:Real}
    (max_iter_newton >= 1) || throw(ArgumentError("Maximum number of iterations must be positive."))
    (rtol_newton ≥ 0.0) || @warn "Relative tolerance is negative."
    (atol_newton ≥ 0.0) || @warn "Absolute tolerance is negative."
    (; σ2, λ) = initial_params
    (σ2 > 0) || throw(ArgumentError("Supplied starting value for σ2 must be positive."))
    (λ > 0) || throw(ArgumentError("Supplied starting value for λ must be positive."))

    initial_params = (σ2 = T(σ2), λ = T(λ))
    return _laplace_approximation(lgp, initial_params, max_iter_newton, atol_newton, rtol_newton)
end

function _laplace_approximation(
    lgp::LogisticGaussianProcess{T},
    initial_params::NamedTuple,
    max_iter_newton::Int,
    atol_newton::Real,
    rtol_newton::Real
) where {T<:Real}
    # Unpack
    (; data, prior_variance_scale, prior_length_scale) = lgp
    (; x, n, x_grid, N, pairwise_dists, bounds, n_bins) = data
    (; σ2, λ) = initial_params

    # Initialize hyperparameters and reusable quantities
    η0 = log.([σ2, λ]) 
    # Create initial cache object (needed to initialize β)
    cache = NewtonRhapsonCache(zeros(n_bins), zeros(n_bins), N, n, n_bins)

    # Find optimal hyperparameters using MAP estimation
    fg! = (F, G, η) -> loglik_and_grad!(F, G, η, cache, pairwise_dists, prior_variance_scale, prior_length_scale, max_iter_newton, atol_newton, rtol_newton)
    #result = Optim.optimize(NLSolversBase.only_fg!(fg!), η0, Optim.LBFGS())
    result = Optim.optimize(η -> loglik(η, cache, pairwise_dists, prior_variance_scale, prior_length_scale, max_iter_newton, atol_newton, rtol_newton), η0, Optim.NelderMead(), Optim.Options(x_abstol = 1e-3, x_reltol=1e-3, f_calls_limit=100))
    η = Optim.minimizer(result)
    σ2, λ = exp.(η)

    # Find the Laplace approximation given the optimal hyperparameters
    # Add jitter to K for better numerical conditioning.
    ∂K_∂σ2 = Symmetric(exp.(-pairwise_dists / λ)) + 1e-6 * I(n_bins)
    K = σ2 * ∂K_∂σ2
    newton_rhapson!(cache, K, max_iter_newton, atol_newton, rtol_newton) # Get the posterior mode of β
    β = cache.β

    # Get the posterior covariance matrix
    u = softmax(β)
    R = LinearOperator(T, n_bins, n_bins, false, false, (res, b) -> mul_R!(res, u, n, b), (res, b) -> mul_R_transpose!(res, u, n, b))
    B = Symmetric(Matrix(I(n_bins) + transpose(R) * K * R))
    chol_B = cholesky(B)
    Σ = Symmetric(K - Matrix(K * R) * (chol_B \ Matrix(transpose(R) * K)))


    return LogisticGaussianProcessLaplacePosterior{T}(β, Σ, σ2, λ, lgp)
end

# Struct used to store useful intermediate computations from the inner Newton-Rhapson solver.
# These are then used as starting values in the next iteration/used in the (approximate) marginal likelihood calculation
struct NewtonRhapsonCache{T<:Real, A<:AbstractVector{T}, B<:AbstractVector{Int}}
    β::A
    a::A
    N::B
    n::Int
    n_bins::Int
end

function newton_rhapson!(
    cache::NewtonRhapsonCache{T},
    K::AbstractMatrix{T},
    max_iter_newton::Int,
    atol_newton::Real,
    rtol_newton::Real
) where {T<:Real}
    (; β, N, n, n_bins) = cache
    a = Vector{T}(undef, n_bins)

    iter = 0
    converged = false
    while !converged && iter ≤ max_iter_newton
        iter = iter + 1
        u = softmax(β)

        # Compute v:
        v = n * (u .* β - u * dot(u, β)) + (N - n * u)

        # Compute linear operators:
        R = LinearOperator(T, n_bins, n_bins, false, false, (res, b) -> mul_R!(res, u, n, b), (res, b) -> mul_R_transpose!(res, u, n, b))
        #A = LinearOperator(T, n_bins, n_bins, true, true, (res, b) -> mul_A!(res, R, K, n_bins, b))
        Kv = K * v
        RtCv = transpose(R) * Kv
        # Solve Az = RtCv using the conjugate gradient method:
        #z, _ = cg(A, RtCv; atol=1e-3, rtol=1e-3)
        #z, _ = gmres(A, RtCv; atol=1e-3, rtol=1e-3)
        B = Symmetric(Matrix(I(n_bins) + transpose(R) * K * R))
        chol_B = cholesky(B)
        z = chol_B \ RtCv
        # Finally, get the new mode:
        a .= v - R * z
        β_new = K * a

        # Check convergence:
        converged = (norm(β_new - β) ≤ atol_newton + rtol_newton * norm(β))
        copyto!(β, β_new)
    end
    cache.β .= β
    cache.a .= a
    return cache
end

function mul_R!(res::AbstractVector{T}, u::AbstractVector{T}, n::Int, b::AbstractVector{T}) where {T<:Real}
    s = sqrt.(u)
    copyto!(res, sqrt(n) * (s .* b - u * dot(s, b)))
end

function mul_R_transpose!(res::AbstractVector{T}, u::AbstractVector{T}, n::Int, b::AbstractVector{T}) where {T<:Real}
    s = sqrt.(u)
    copyto!(res, sqrt(n) * (s .* b - s * dot(u, b)))
end

function mul_A!(res::AbstractVector{T}, R::LinearOperator{T}, K::AbstractMatrix{T}, n_bins::Int, b::AbstractVector{T}) where {T<:Real}
    tmp = R * b
    tmp = K * tmp
    res .= b .+ transpose(R) * tmp
end

function loglik(
    η::AbstractVector{T},
    cache::NewtonRhapsonCache{T},
    pairwise_dists::AbstractMatrix{T},
    prior_variance_scale::T,
    prior_length_scale::T,
    max_iter_newton::Int,
    atol_newton::T,
    rtol_newton::T
) where {T<:Real}
    σ2, λ = exp.(η) # Perform optimization in unconstrained space
    ∂K_∂σ2 = Symmetric(exp.(-pairwise_dists / λ)) + 1e-6 * I(cache.n_bins)
    K = σ2 * ∂K_∂σ2
    ∂K_∂λ = (pairwise_dists / λ^2) .* K
    newton_rhapson!(cache, K, max_iter_newton, atol_newton, rtol_newton)
    (; β, a, N, n, n_bins) = cache

    # Precompute shared quantities
    u = softmax(β)
    ∂logp_∂β = (N - n * u)
    R = LinearOperator(T, n_bins, n_bins, false, false, (res, b) -> mul_R!(res, u, n, b), (res, b) -> mul_R_transpose!(res, u, n, b))
    B = Symmetric(Matrix(I(n_bins) + transpose(R) * K * R))
    chol_B = cholesky(B)
    L = chol_B.L
    U = chol_B.L

    prior_terms = -log(1 + σ2 / abs2(prior_variance_scale))
    prior_terms += -log(1 + λ / abs2(prior_length_scale))
    log_ml = -1/2 * dot(a, β) + dot(β, N) - n*log(sum(exp.(β))) - logabsdet(L)[1]
    log_obj = log_ml + prior_terms
    return -log_obj
end

function loglik_and_grad!(
    F,
    G,
    η::AbstractVector{T},
    cache::NewtonRhapsonCache{T},
    pairwise_dists::AbstractMatrix{T},
    prior_variance_scale::T,
    prior_length_scale::T,
    max_iter_newton::Int,
    atol_newton::T,
    rtol_newton::T
) where {T<:Real}
    σ2, λ = exp.(η) # Perform optimization in unconstrained space
    @show σ2
    @show λ
    ∂K_∂σ2 = Symmetric(exp.(-pairwise_dists / λ)) + 1e-6 * I(cache.n_bins)
    K = σ2 * ∂K_∂σ2
    ∂K_∂λ = (pairwise_dists / λ^2) .* K
    newton_rhapson!(cache, K, max_iter_newton, atol_newton, rtol_newton)
    (; β, a, N, n, n_bins) = cache

    # Precompute shared quantities
    u = softmax(β)
    ∂logp_∂β = (N - n * u)
    R = LinearOperator(T, n_bins, n_bins, false, false, (res, b) -> mul_R!(res, u, n, b), (res, b) -> mul_R_transpose!(res, u, n, b))
    B = Symmetric(Matrix(I(n_bins) + transpose(R) * K * R))
    chol_B = cholesky(B)
    L = chol_B.L
    U = chol_B.L

    # V = inv(K + W⁻¹) = R * inv(I + transpose(R) * K * R)*transpose(R)
    # Q = inv(I + KW) = I - K*V
    # S = inv(W + K⁻¹) = K - K * R * inv(B) * transpose(R) * K = K - K * R * inv(I + R^T * K * R) * transpose(R) * K
    #tmp = L \ transpose(Matrix(R))
    #V = Matrix(R * (U \ tmp))
    R_mat = Matrix(R)
    V = R_mat * (chol_B \ transpose(R_mat))
    #C = L \ (Matrix(transpose(R) * K))
    Σ = Symmetric(K - Matrix(K * R) * (chol_B \ Matrix(transpose(R) * K)))

    
    ∂3logp_∂β3 = -n* u .* (1 .- u) .* (1.0 .- 2*u)
    #s2 = -1/2 * (Diagonal(K) - Diagonal(transpose(C) * C)) * ∂3logp_∂β3
    s2 = -1/2 * Diagonal(Σ) * ∂3logp_∂β3
    if G !== nothing
        # Derivative wrt σ2:
        # Terms from prior
        prior_term = 1/abs2(prior_variance_scale) * 1/(1 + σ2 / abs2(prior_variance_scale))
        # Explicit marginal likelihood terms:
        ∂logq_∂σ2 = 1/2*dot(a, ∂K_∂σ2, a) - 1/2 * tr(V * ∂K_∂σ2) # V=R in Rasmussen
        # Implicit terms
        b = ∂K_∂σ2 * ∂logp_∂β
        s3 = b - K*V*b
        ∂logq_∂σ2 += dot(s2, s3)
        ∂logq_∂σ2 += prior_term
        ∂logq_∂σ2 *= σ2 # Adjustment from optimizing in unconstrained space

        # Derivative wrt λ
        # Terms from prior

        # Explicit marginal likelihood terms:
        prior_term = 1/abs2(prior_length_scale) * 1/(1 + λ / abs2(prior_length_scale))
        ∂logq_∂λ = 1/2*dot(a, ∂K_∂λ, a) - 1/2 * tr(V * ∂K_∂λ) # V=R in Rasmussen
        # Implicit marginal likelihood terms
        b = ∂K_∂λ * ∂logp_∂β
        s3 = b - K*V*b
        ∂logq_∂λ += dot(s2, s3)
        ∂logq_∂λ += prior_term
        ∂logq_∂λ *= λ

        println("Grad σ2: ", ∂logq_∂σ2)
        println("Grad λ: ", ∂logq_∂λ)

        # Write result to output:
        copyto!(G, [-∂logq_∂σ2, -∂logq_∂λ])
    end
    if F !== nothing
        prior_terms = -log(1 + σ2 / abs2(prior_variance_scale))
        prior_terms += -log(1 + λ / abs2(prior_length_scale))
        log_ml = -1/2 * dot(a, β) + dot(β, N) - n*log(sum(exp.(β))) - logabsdet(L)[1]
        @show log_ml
        return -log_ml
    end
end