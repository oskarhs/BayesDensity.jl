"""
    BSplineMixtureVIPosterior{T<:Real, A<:MvNormalCanon{T}, B<:InverseGamma{T}, M<:BSplineMixture} <: AbstractVIPosterior{T}

Struct representing the variational posterior distribution of a [`BSplineMixture`](@ref).

# Fields
* `q_β`: Distribution representing the optimal variational density q*(β).
* `q_τ`: Distribution representing the optimal variational density q*(τ²).
* `q_δ`: Vector of distributions, with element `k` corresponding to the optimal variational density q*(δₖ²).
* `bsm`: The `BSplineMixture` to which the variational posterior was fit.

# Examples
```julia
julia> x = (1.0 .- (1.0 .- LinRange(0, 1, 5001)) .^(1/3)).^(1/3);

julia> vip = varinf(BSplineMixture(x))
BSplineMixtureVIPosterior{Float64} vith variational densities:
 q_β::Distributions.MvNormalCanon{Float64},
 q_τ::Distributions.InverseGamma{Float64},
 q_δ::Vector{Distributions.InverseGamma{Float64}}.
Model:
200-dimensional BSplineMixture{Float64}:
Using 5001 binned observations on a regular grid consisting of 1187 bins.
 support: (-0.05, 1.05)
Hyperparameters:
 a_τ = 1.0, b_τ = 0.001
 a_δ = 0.5, b_δ = 0.5

julia> mean(Random.Xoshiro(1), vip, 0.2)
0.35127443392229263
```
"""
struct BSplineMixtureVIPosterior{T<:Real, A<:MvNormalCanon{T}, B<:InverseGamma{T}, M<:BSplineMixture} <: AbstractVIPosterior{T}
    q_β::A
    q_τ::B
    q_δ::Vector{B}
    bsm::M
    function BSplineMixtureVIPosterior{T}(μ_opt::Vector{T}, inv_Σ_opt::A, a_τ_opt::T, b_τ_opt::T, a_δ_opt::Vector{T}, b_δ_opt::Vector{T}, bsm::M) where {T<:Real, A<:AbstractMatrix{T}, M<:BSplineMixture}
        K = length(basis(bsm))
        q_β = MvNormalCanon(inv_Σ_opt * μ_opt, inv_Σ_opt)
        q_τ = InverseGamma(a_τ_opt, b_τ_opt)
        q_δ = Vector{InverseGamma{T}}(undef, K-3)
        for k in 1:K-3
            q_δ[k] = InverseGamma(a_δ_opt[k], b_δ_opt[k])
        end
        return new{T,MvNormalCanon{T},InverseGamma{T},M}(q_β, q_τ, q_δ, bsm)
    end
end

BayesDensityCore.model(vip::BSplineMixtureVIPosterior) = vip.bsm

function Base.show(io::IO, ::MIME"text/plain", vip::BSplineMixtureVIPosterior{T, A, B, M}) where {T, A, B, M}
    println(io, nameof(typeof(vip)), "{", T, "} vith variational densities:")
    println(io, " q_β::", A, ",")
    println(io, " q_τ::", B, ",")
    println(io, " q_δ::", Vector{B}, ".")
    println(io, "Model:")
    println(io, model(vip))
    nothing
end

Base.show(io::IO, vip::BSplineMixtureVIPosterior) = show(io, MIME("text/plain"), vip)

function StatsBase.sample(rng::AbstractRNG, vip::BSplineMixtureVIPosterior{T, A, B, M}, n_samples::Int) where {T<:Real, A, B, M}
    (; q_β, q_τ, q_δ, bsm) = vip
    K = length(basis(bsm))
    samples = Vector{NamedTuple{(:spline_coefs, :θ, :β, :τ2, :δ2), Tuple{Vector{T}, Vector{T}, Vector{T}, T, Vector{T}}}}(undef, n_samples)
    δ2 = Vector{T}(undef, K-3)

    for m in 1:n_samples
        β = rand(rng, q_β)
        θ = max.(eps(), logistic_stickbreaking(β))
        θ = θ / sum(θ)
        spline_coefs = theta_to_coef(θ, basis(bsm))
        τ2 = rand(rng, q_τ)
        for k in 1:K-3
            δ2[k] = rand(rng, q_δ[k])
        end

        samples[m] = (spline_coefs = spline_coefs, θ = θ, β = β, τ2 = τ2, δ2 = δ2)
    end
    return PosteriorSamples{T}(samples, bsm, n_samples, 0)
end

"""
    varinf(
        bsm::BSplineMixture;
        init_params::NamedTuple=get_default_initparams(bsm),
        max_iter::Int=1000
    ) -> BSplineMixtureVIPosterior{<:Real}

Find a variational approximation to the posterior distribution of a [`BSplineMixture`](@ref) using mean-field variational inference.

# Arguments
* `bsm`: The `BSplineMixture` whose posterior we want to approximate.

# Keyword arguments
* `init_params`: Initial values of the VI parameters `μ_opt` `inv_Σ_opt`, `b_τ_opt` and `b_δ_opt`, supplied as a NamedTuple.
* `max_iter`: Maximal number of VI iterations. Defaults to `1000`.

# Returns
* `vip`: A [`BSplineMixtureVIPosterior`](@ref) object representing the variational posterior.
"""
function BayesDensityCore.varinf(bsm::BSplineMixture; init_params=get_default_initparams(bsm), max_iter::Int=1000) # Also: tolerance parameters
    return _variational_inference(bsm, init_params, max_iter)
end

# Can do something more sophisticated here at a later point in time if we get a good idea.
function get_default_initparams(bsm::BSplineMixture{T, A, NT}) where {T, A, NT}
    K = length(basis(bsm))
    P = spdiagm(K-3, K-1, 0=>fill(1, K-3), 1=>fill(-2, K-3), 2=>fill(1, K-3))
    (; a_τ, b_τ, a_δ, b_δ) = hyperparams(bsm)
    a_τ_opt = a_τ + (K-3)/2
    b_τ_opt = b_τ
    a_δ_opt = fill(a_δ + 1/2, K-3)
    b_δ_opt = fill(b_δ, K-3)

    μ_opt = compute_μ(basis(bsm))
    D = Diagonal(a_τ_opt / b_τ_opt * a_δ_opt ./ b_δ_opt)
    Q = transpose(P) * D * P
    inv_Σ_opt = Q + 0.05 * Diagonal(ones(K-1))
    return (μ_opt = μ_opt, inv_Σ_opt = inv_Σ_opt, b_τ_opt = b_τ_opt, b_δ_opt = b_δ_opt)
end

function _variational_inference(bsm::BSplineMixture{T, A, NamedTuple{(:x, :log_B, :b_ind, :bincounts, :μ, :P, :n), Vals}}, init_params::NT, max_iter::Int) where {T, A, Vals, NT}
    bs = basis(bsm)
    K = length(bs)
    (; x, log_B, b_ind, bincounts, μ, P, n) = bsm.data
    P = sparse(P) # Needed for selinv
    n_bins = length(bincounts)

    (; a_τ, b_τ, a_δ, b_δ) = hyperparams(bsm)

    (; μ_opt, inv_Σ_opt, b_τ_opt, b_δ_opt) = init_params

    # These two stay constant throughout the optimization loop.
    a_τ_opt = a_τ + (K-3)/2
    a_δ_opt = fill(a_δ + 1/2, K-3)

    non_basis_term = Vector{T}(undef, K)
    logprobs = Vector{T}(undef, 4)
    E_S = Vector{T}(undef, K)
    E_ω = Vector{T}(undef, K-1)

    # Add converence check later
    for i in 1:max_iter
        # Find the required posterior moments of q(β):
        Z, _ = selinv(inv_Σ_opt; depermute=true) # Get pentadiagonal entries of Σ*
        d0 = Vector(diag(Z)) # Vector of Σ*_{k,k}
        d1 = Vector(diag(Z, 1)) # Vector of Σ*_{k,k+1}
        d2 = Vector(diag(Z, 2)) # Vector of Σ*_{k,k+2}
        E_β = μ_opt         # Do this for enhanced readability, remove later
        E_β2 = abs2.(μ_opt) .+ d0
        E_Δ2 = abs2.(diff(diff(μ_opt - μ))) + view(d0, 3:K-1) + 4 * view(d0, 2:K-2) + view(d0, 1:K-3) - 4 * view(d1, 2:K-2) - 4 * view(d1, 1:K-3) + 2 * d2

        # Update q(δ²)
        b_δ_opt = b_δ .+ T(0.5) * E_Δ2 * a_τ_opt / b_τ_opt

        # Update q(τ²)
        b_τ_opt = b_τ + T(0.5) * sum(E_Δ2 .* a_δ_opt ./ b_δ_opt)

        # Update q(z, ω)
        E_N = zeros(T, K)
        non_basis_term[1:K-1] = cumsum(@. -log(cosh(T(0.5)*sqrt(E_β2))) - T(0.5) * E_β - log(T(2))) # Replace the β's with the corresponding expectations
        non_basis_term[K] = non_basis_term[K-1]
        non_basis_term[1:K-1] .+= E_β
        @inbounds for i in 1:n_bins
            # Compute the four nonzero probabilities:
            k0 = b_ind[i]
            for l in 1:4
                k = k0 + l - 1
                logprobs[l] = log_B[i,l] + non_basis_term[k] 
            end
            E_N[k0:k0+3] .+= bincounts[i] * softmax(logprobs)
        end
        E_S[1] = n
        E_S[2:K] .= n .- cumsum(view(E_N, 1:K-1))
        E_S = clamp.(E_S, 0.0, n)
        E_ω = @. tanh(sqrt(E_β2)/2) / (2*sqrt(E_β2)) * E_S[1:K-1]
        
        # Update q(β)
        D = Diagonal(a_τ_opt / b_τ_opt * a_δ_opt ./ b_δ_opt)
        Q = transpose(P) * D * P
        Ωκ = view(E_N, 1:K-1) - view(E_S, 1:K-1) / 2
        inv_Σ_opt = Q + Diagonal(E_ω)
        μ_opt = inv_Σ_opt \ (Q*μ + Ωκ)
    end
    return BSplineMixtureVIPosterior{T}(μ_opt, BandedMatrix(inv_Σ_opt), a_τ_opt, b_τ_opt, a_δ_opt, b_δ_opt, bsm)
end

function _variational_inference(bsm::BSplineMixture{T, A, NamedTuple{(:x, :log_B, :b_ind, :μ, :P, :n), Vals}}, init_params::NT, max_iter::Int) where {T, A, Vals, NT}
    bs = basis(bsm)
    K = length(bs)
    (; x, log_B, b_ind, μ, P, n) = bsm.data

    P = sparse(P) # Needed for selinv

    (; a_τ, b_τ, a_δ, b_δ) = hyperparams(bsm)

    (; μ_opt, inv_Σ_opt, b_τ_opt, b_δ_opt) = init_params

    # These two stay constant throughout the optimization loop.
    a_τ_opt = a_τ + (K-3)/2
    a_δ_opt = fill(a_δ + 1/2, K-3)

    non_basis_term = Vector{T}(undef, K)
    logprobs = Vector{T}(undef, 4)
    E_S = Vector{T}(undef, K)
    E_ω = Vector{T}(undef, K-1)

    # Add converence check later
    for i in 1:max_iter
        # Find the required posterior moments of q(β):
        Z, _ = selinv(inv_Σ_opt; depermute=true) # Get pentadiagonal entries of Σ*
        d0 = Vector(diag(Z)) # Vector of Σ*_{k,k}
        d1 = Vector(diag(Z, 1)) # Vector of Σ*_{k,k+1}
        d2 = Vector(diag(Z, 2)) # Vector of Σ*_{k,k+2}
        E_β = μ_opt         # Do this for enhanced readability, remove later
        E_β2 = abs2.(μ_opt) .+ d0
        E_Δ2 = abs2.(diff(diff(μ_opt - μ))) + view(d0, 3:K-1) + 4 * view(d0, 2:K-2) + view(d0, 1:K-3) - 4 * view(d1, 2:K-2) - 4 * view(d1, 1:K-3) + 2 * d2

        # Update q(δ²)
        b_δ_opt = b_δ .+ T(0.5) * E_Δ2 * a_τ_opt / b_τ_opt

        # Update q(τ²)
        b_τ_opt = b_τ + T(0.5) * sum(E_Δ2 .* a_δ_opt ./ b_δ_opt)

        # Update q(z, ω)
        E_N = zeros(T, K)
        non_basis_term[1:K-1] = cumsum(@. -log(cosh(T(0.5)*sqrt(E_β2))) - T(0.5) * E_β - log(T(2))) # Replace the β's with the corresponding expectations
        non_basis_term[K] = non_basis_term[K-1]
        non_basis_term[1:K-1] .+= E_β
        @inbounds for i in 1:n
            # Compute the four nonzero probabilities:
            k0 = b_ind[i]
            for l in 1:4
                k = k0 + l - 1
                logprobs[l] = log_B[i,l] + non_basis_term[k] 
            end
            E_N[k0:k0+3] .+= softmax(logprobs)
        end
        E_S[1] = n
        E_S[2:K] .= n .- cumsum(view(E_N, 1:K-1))
        E_S = clamp.(E_S, 0.0, n)
        E_ω = @. tanh(sqrt(E_β2)/2) / (2*sqrt(E_β2)) * E_S[1:K-1]
        
        # Update q(β)
        D = Diagonal(a_τ_opt / b_τ_opt * a_δ_opt ./ b_δ_opt)
        Q = transpose(P) * D * P
        Ωκ = view(E_N, 1:K-1) - view(E_S, 1:K-1) / 2
        inv_Σ_opt = Q + Diagonal(E_ω)
        μ_opt = inv_Σ_opt \ (Q*μ + Ωκ)
    end
    return BSplineMixtureVIPosterior{T}(μ_opt, BandedMatrix(inv_Σ_opt), a_τ_opt, b_τ_opt, a_δ_opt, b_δ_opt, bsm)
end