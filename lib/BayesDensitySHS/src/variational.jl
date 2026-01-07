"""
    SHSVIPosterior{T<:Real} <: AbstractVIPosterior

Struct representing the variational posterior distribution of a [`SHSModel`](@ref).

# Fields
* `q_β`: Distribution representing the optimal variational density q*(β).
* `q_σ`: Distribution representing the optimal variational density q*(σ²).
* `shs`: The `SHSModel` to which the variational posterior was fit.

# Examples
```julia
```
"""
struct SHSVIPosterior{T<:Real, A<:MvNormalCanon{T}, B<:InverseGamma{T}, M<:BSMModel} <: AbstractVIPosterior
    q_β::A
    q_σ::B
    shs::M
    function SHSVIPosterior{T}(μ_opt::Vector{T}, Σ_opt::A, a_σ_opt::T, b_σ_opt::T, shs::M) where {T<:Real, A<:AbstractMatrix{T}, M<:BSMModel}
        q_β = MvNormal(μ_opt, Σ_opt)
        q_σ = InverseGamma(a_σ_opt, b_σ_opt)
        return new{T,MvNormalCanon{T},InverseGamma{T},M}(q_β, q_σ, q_a, shs)
    end
end

Base.eltype(::SHSVIPosterior{T, A, B, M}) where {T, A, B, M} = T
BayesDensityCore.model(vip::SHSVIPosterior) = vip.shs

function Base.show(io::IO, ::MIME"text/plain", vip::SHSVIPosterior{T, A, B, M}) where {T, A, B, M}
    println(io, nameof(typeof(vip)), "{", T, "} vith variational densities:")
    println(io, " q_β::", A, ",")
    println(io, " q_σ::", B, ",")
    println(io, "Model:")
    println(io, model(vip))
    nothing
end

Base.show(io::IO, vip::SHSVIPosterior) = show(io, MIME("text/plain"), vip)

function StatsBase.sample(rng::AbstractRNG, vip::SHSVIPosterior{T, A, B, M}, n_samples::Int) where {T, A, B, M}
    (; q_β, q_σ, shs) = vip
    samples_temp = Vector{NamedTuple{(:β, :σ2), Tuple{Vector{T}, T}}}(undef, n_samples)
    for m in 1:n_samples
        β = rand(rng, q_β)
        σ2 = rand(rng, q_σ)
        samples_temp[m] = (β = β, σ2 = σ2)
    end
    l1_norm_vec = compute_norm_constants(shs, params)
    samples = Vector{NamedTuple{(:β, :σ2, :norm), Tuple{Vector{T}, T, T}}}(undef, n_samples)
    for m in 1:n_samples
        samples[m] = (β = samples_temp[m].β, σ2 = samples_temp[m].σ2, norm = l1_norm_vec[m])
    end
    return PosteriorSamples{T}(samples, shs, n_samples, 0)
end

function BayesDensityCore.varinf(shs::SHSModel; init_params::NamedTuple=get_default_initparams(shs), max_iter::Int=500, rtol::Real=1e-5) # Also: tolerance parameters
    return _variational_inference(shs, init_params, max_iter, rtol)
end

function get_default_initparams(shs::SHSModel{T, A, D}) where {T, A, D}
    (; data, bs, σ_β, s_σ) = shs
    (; x, n, N, C, LZ, bounds) = data
    # Use MixedModels.jl to find initial parameter estimate:

    Z = C[:, 3:end]
    df = DataFrame(obs_ind = 1, x = x, N = N)
    df = hcat(df, DataFrame(z, Symbol.("z", 1:size(z, 2))))

    terms = "-1 +" * join(["z$i" for i in 1:K-2], " + ")
    fstr = "N ~ 1 + x + zerocorr($terms | obs_ind)"
    formula = eval(Meta.parse("@formula($fstr)"))

    # Perform a few PIRLS iterations to find reasonable starting values for β:
    Logging.with_logger(ConsoleLogger(stderr, Logging.Error)) do
        mixedmodel = GeneralizedLinearMixedModel(formula, df, Poisson());
        mixedmodel.optsum.maxtime = 0.5;
        fit!(mixedmodel; fast=true, verbose=false);
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
    b_σ_opt = T(1) * s_σ + @views(tr(Σ_opt[3:end, 3:end]) + sum(abs2, μ[3:end])) / 2
    
    return (μ_opt = μ_opt, Σ_opt = Σ_opt, b_σ_opt = b_σ_opt)
end

function _variational_inference(shs::SHSModel{T, A, D}, init_params::NamedTuple, max_iter::Int, rtol::Real) where {T, A, D}
    (; data, bs, σ_β, s_σ) = shs
    (; x, n, N, C, LZ, bounds) = data
    (; μ_opt, Σ_opt, b_σ_opt) = init_params

    K = length(bs)
    # These stay constant throughout the optimization procedure
    a_a_opt = T(1)
    a_σ_opt = T(K - 1) / 2

    for _ in 1:max_iter
        # Update q(a)
        b_a_opt = a_σ_opt / b_σ_opt + 1/s_σ^2

        # Update q(σ²)
        b_σ_new = a_a_opt / b_a_opt + @views(tr(Σ_opt[3:end, 3:end]) + sum(abs2, μ[3:end])) / 2

        relative_change = abs(b_σ_opt/b_σ_new - 1)

        # Update q(β)
        w = exp.(C * μ_opt + vec(sum(C * Σ_opt .* C / 2; dims=2)))
        Λ = Diagonal(vcat(fill(1/σ_β^2, 2), fill(a_σ_opt/b_σ_opt, K-2)))
        inv_Σ_opt = transpose(C) * w .* C + Λ
        Σ_opt = inv(inv_Σ_opt)
        μ_opt = μ_opt + Σ_opt * (tranpose(C) * (y - w) - Λ * μ_opt) 

        # Check convergence criterion
        b_σ_opt = b_σ_new
        if relative_change < rtol
            break
        end
    end
    return SHSVIPosterior{T}(μ_opt, Σ_opt, a_σ_opt, b_σ_opt, shs)
end