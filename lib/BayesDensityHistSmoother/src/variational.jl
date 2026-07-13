"""
    HistSmootherVIPosterior{T<:Real} <: AbstractVIPosterior{T}

Struct representing the variational posterior distribution of a [`HistSmoother`](@ref).

# Fields
* `q_β`: Distribution representing the optimal variational density q*(β).
* `q_σ`: Distribution representing the optimal variational density q*(σ²).
* `shs`: The `HistSmoother` to which the variational posterior was fit.
"""
struct HistSmootherVIPosterior{T<:Real, A<:MvNormal{T}, B<:InverseGamma{T}, M<:HistSmoother} <: AbstractVIPosterior{T}
    q_β::A
    q_σ::B
    shs::M
    function HistSmootherVIPosterior{T}(μ_opt::Vector{T}, Σ_opt::A, a_σ_opt::T, b_σ_opt::T, shs::M) where {T<:Real, A<:AbstractMatrix{T}, M<:HistSmoother}
        q_β = MvNormal(μ_opt, Σ_opt)
        q_σ = InverseGamma(a_σ_opt, b_σ_opt)
        return new{T,MvNormal{T},InverseGamma{T},M}(q_β, q_σ, shs)
    end
end

BayesDensityCore.model(vip::HistSmootherVIPosterior) = vip.shs

function Base.show(io::IO, ::MIME"text/plain", vip::HistSmootherVIPosterior{T, A, B}) where {T, A, B}
    println(io, nameof(typeof(vip)), "{", T, "} vith variational densities:")
    println(io, " q_β::", A, ",")
    println(io, " q_σ::", B, ",")
    println(io, "Model:")
    print(io, model(vip))
    nothing
end

Base.show(io::IO, vip::HistSmootherVIPosterior) = show(io, MIME("text/plain"), vip)

function StatsBase.sample(
    rng::AbstractRNG,
    vip::HistSmootherVIPosterior{T, A, B},
    n_samples::Int
) where {T, A, B}
    (; q_β, q_σ, shs) = vip
    samples_temp = Vector{NamedTuple{(:β, :σ2), Tuple{Vector{T}, T}}}(undef, n_samples)
    for m in 1:n_samples
        β = rand(rng, q_β)
        σ2 = rand(rng, q_σ)
        samples_temp[m] = (β = β, σ2 = σ2)
    end
    eval_grid, val_cdf, l1_norm_vec = compute_norm_constants_cdf_grid(shs, samples_temp)
    samples = Vector{NamedTuple{(:β, :σ2, :norm, :eval_grid, :val_cdf), Tuple{Vector{T}, T, T, LinRange{T}, Vector{T}}}}(undef, n_samples)
    for m in 1:n_samples
        samples[m] = (β = samples_temp[m].β, σ2 = samples_temp[m].σ2,
                      norm = l1_norm_vec[m], eval_grid = eval_grid,
                      val_cdf = val_cdf[m])
    end
    return PosteriorSamples{T}(samples, shs, n_samples, 0)
end

"""
    varinf(
        hs::HistSmoother{T};
        initial_params::NamedTuple = _get_default_initparams_varinf(hs),
        max_iter::Int              = 500,
        rtol::Real                 = 1e-5
    ) where {T} -> HistSmootherVIPosterior{T}

Find a variational approximation to the posterior distribution of a [`HistSmoother`](@ref) using semiparametric mean-field variational inference.

# Arguments
* `hs`: The `HistSmoother` whose posterior we want to approximate.

# Keyword arguments
* `initial_params`: Initial values of the VI parameters `μ_opt`, `Σ_opt` and `b_σ_opt`.
* `max_iter`: Maximal number of VI iterations. Defaults to `500`.
* `rtol`: Relative tolerance used to determine convergence. Defaults to `1e-5`.

# Returns
* `vip`: A [`HistSmootherVIPosterior`](@ref) object representing the variational posterior.
* `info`: A [`VariationalOptimizationResult`](@ref) describing the result of the optimization.

!!! note
    To run the optimization loop for a fixed number of iterations irrespective of the convergence criterion, one can set `rtol = 0.0`, and `max_iter` equal to the desired total iteration count.
    Note that setting `rtol` to a strictly negative value will issue a warning.

# Examples
```julia-repl
julia> using Random

julia> x = (1.0 .- (1.0 .- LinRange(0.0, 1.0, 5000)) .^(1/3)).^(1/3);

julia> hs = HistSmoother(x);

julia> vip, info = varinf(hs; rtol=1e-6);
```

# Extended help
## Convergence
The criterion used to determine convergence is that the relative change in the expectation of ``\\mathbb{E}(\\sigma^{-2})`` falls below the given `rtol`.
"""
function BayesDensityCore.varinf(shs::HistSmoother;
    initial_params::NamedTuple=_get_default_initparams_varinf(shs),
    max_iter::Int=500,
    rtol::Real=1e-5
)
    (max_iter >= 1) || throw(ArgumentError("Maximum number of iterations must be positive."))
    (rtol ≥ 0.0) || @warn "Relative tolerance is negative."
    return _variational_inference(shs, initial_params, max_iter, rtol)
end

function _get_default_initparams_varinf(shs::HistSmoother{T}) where {T<:Real}
    (; data, bs, prior_scale_fixed, prior_scale_random) = shs
    (; n, x_grid, N, C, LZ, bounds) = data
    # Use MixedModels.jl to find initial parameter estimate:
    Z = C[:, 3:end]
    K = size(Z, 2) + 2

    any(!=(N[1]), N) || return (μ_opt = zeros(K), Σ_opt = Diagonal(ones(K)), b_σ_opt = prior_scale_random)

    df = DataFrame(obs_ind = 1, x_grid = x_grid, N = N)
    df = hcat(df, DataFrame(Z, Symbol.("z", 1:size(Z, 2))))

    terms = "-1 +" * join(["z$i" for i in 1:K-2], " + ")
    fstr = "N ~ 1 + x_grid + zerocorr($terms | obs_ind)"
    formula = eval(Meta.parse("@formula($fstr)"))

    # Perform a few PIRLS iterations to find reasonable starting values for β:
    mixedmodel = Logging.with_logger(ConsoleLogger(stderr, Logging.Error)) do
        mixedmodel = GeneralizedLinearMixedModel(formula, df, Poisson());
        mixedmodel.optsum.maxtime = 0.5;
        fit!(mixedmodel; fast=true, progress=false);
        mixedmodel
    end

    # Extract Point estimate
    μ_opt = Vector{Float64}(undef, K)
    μ_opt[1:2] = fixef(mixedmodel)
    μ_opt[3:end] = vec(ranef(mixedmodel)[1])

    # Extract covariance estimates
    ranef_std_nt = VarCorr(mixedmodel).σρ.obs_ind.σ
    variance_component_names = [Symbol("z$k") for k in 1:K-2]
    Σ_opt = zeros(T, (K, K))
    Σ_opt[1:2, 1:2] = vcov(mixedmodel)
    for k in eachindex(variance_component_names)
        Σ_opt[k+2, k+2] = ranef_std_nt[Symbol("z$k")]^2
    end

    # Initialize q(σ²)
    b_σ_opt = T(1) * prior_scale_random + @views(tr(Σ_opt[3:end, 3:end]) + sum(abs2, μ_opt[3:end])) / 2
    
    return (μ_opt = μ_opt, Σ_opt = Σ_opt, b_σ_opt = b_σ_opt)
end

# Compute the per-bin mean rates w = exp.(Cμ + ½ diag(C Σ Cᵀ)) into `w`, allocation-free,
# reusing the buffers Cμ (=Cμ) and CΣ (=C*Σ).
function _predictor_w!(w, C, μ, Σ, Cμ, CΣ)
    mul!(Cμ, C, μ)
    mul!(CΣ, C, Σ)
    @inbounds for i in axes(C, 1)
        q = zero(eltype(w))
        for j in axes(C, 2)
            q += CΣ[i, j] * C[i, j]
        end
        w[i] = exp(Cμ[i] + q/2)
    end
    return w
end

function _variational_inference(shs::HistSmoother{T}, initial_params::NamedTuple, max_iter::Int, rtol::Real) where {T<:Real}
    (; data, bs, prior_scale_fixed, prior_scale_random) = shs
    (; n, x_grid, N, C, LZ, bounds) = data
    (; μ_opt, Σ_opt, b_σ_opt) = initial_params

    K = length(bs)
    # These stay constant throughout the optimization procedure
    a_a_opt = T(1)
    a_σ_opt = T(K - 1) / 2

    converged = false
    iter = 1
    ELBO_const_terms_sum = 1/2 * (K - one(T)) + loggamma(1/2 * (K - one(T))) - log(T(pi)) - log(prior_scale_random) - sum(loggamma.(N .+ 1)) - log(prior_scale_fixed)
    ELBO = fill(ELBO_const_terms_sum, max_iter)

    # Preallocated buffers reused across iterations
    n_bins = length(N)
    Cw = Matrix{T}(undef, n_bins, K)      # C scaled row-wise by w
    CΣ = Matrix{T}(undef, n_bins, K)      # C * Σ_opt
    Cμ = Vector{T}(undef, n_bins)         # C * μ_opt
    Nw = Vector{T}(undef, n_bins)         # N - w
    λvec = Vector{T}(undef, K)            # diagonal of Λ; only the 3:K block changes per iteration
    λvec[1] = 1/prior_scale_fixed^2
    λvec[2] = 1/prior_scale_fixed^2

    w = Vector{T}(undef, n_bins)
    _predictor_w!(w, C, μ_opt, Σ_opt, Cμ, CΣ)
    N_transpose_C = transpose(N) * C

    while !converged && iter ≤ max_iter
        # Update q(a)
        b_a_opt = a_σ_opt / b_σ_opt + 1/prior_scale_random^2

        # Update q(σ²)
        b_σ_new = a_a_opt / b_a_opt + @views(tr(Σ_opt[3:end, 3:end]) + sum(abs2, μ_opt[3:end])) / 2

        relative_change = abs(b_σ_opt/b_σ_new - 1)

        # Update q(β)
        @inbounds for k in 3:K
            λvec[k] = a_σ_opt/b_σ_opt
        end
        Λ = Diagonal(λvec)
        Cw .= C .* w
        inv_Σ_opt = transpose(C) * Cw + Λ
        Σ_opt = inv(inv_Σ_opt)
        Nw .= N .- w
        μ_opt = μ_opt + Σ_opt * (transpose(C) * Nw - Λ * μ_opt)

        # Check convergence criterion
        b_σ_opt = b_σ_new
        converged = (relative_change < rtol)

        # Compute the ELBO:
        _predictor_w!(w, C, μ_opt, Σ_opt, Cμ, CΣ)
        ELBO[iter] += N_transpose_C * μ_opt - sum(w)
        ELBO[iter] += -1/(2*prior_scale_fixed^2) * @views(sum(abs2, μ_opt[1:2]) + tr(Σ_opt[1:2, 1:2])) + 1/2 * logabsdet(Σ_opt)[1]
        ELBO[iter] += -1/2 * (K - one(T)) * log(a_a_opt/b_a_opt + 1/2 * @views(sum(abs2, μ_opt[3:end]) + tr(Σ_opt[3:end, 3:end])))
        ELBO[iter] += -log(a_σ_opt / b_σ_opt+1/prior_scale_random^2) + a_a_opt * a_σ_opt / (b_a_opt * b_σ_opt)
        
        # Increment iteration counter
        iter += 1
    end
    
    converged || @warn "Failed to meet convergence criterion in $iter iterations."
    posterior = HistSmootherVIPosterior{T}(μ_opt, Symmetric(Σ_opt), a_σ_opt, b_σ_opt, shs)
    info = VariationalOptimizationResult{T}(ELBO[1:iter-1], converged, iter-1, rtol, posterior)
    return posterior, info
end