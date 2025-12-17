"""
    BSMVIPosterior{T<:Real, A<:AbstractMatrix, M<:BSMModel} <: AbstractVIPosterior

Struct representing the variational posterior distribution of a [`BSMModel`](@ref).

# Fields
* `μ_opt`: Mean vector of q(β).
* `inv_Σ_opt`: Precision matrix of q(β).
* `a_τ_opt`: Shape parameter of q(τ²).
* `b_τ_opt`: Rate parameter of q(τ²).
* `a_δ_opt`: Vector of shape parameters of q(δ²).
* `b_δ_opt`: Vector of rate parameters of q(δ²).
* `bsm`: The `BSMModel` to which the variational posterior was fit.
"""
struct BSMVIPosterior{T<:Real, A<:AbstractMatrix{T}, M<:BSMModel} <: AbstractVIPosterior
    μ_opt::Vector{T}
    inv_Σ_opt::A
    a_τ_opt::T
    b_τ_opt::T
    a_δ_opt::Vector{T}
    b_δ_opt::Vector{T}
    bsm::M
    function BSMVIPosterior{T}(μ_opt::Vector{T}, inv_Σ_opt::A, a_τ_opt::T, b_τ_opt::T, a_δ_opt::Vector{T}, b_δ_opt::Vector{T}, bsm::M) where {T<:Real, A<:AbstractMatrix{T}, M<:BSMModel}
        return new{T,A,M}(μ_opt, inv_Σ_opt, a_τ_opt, b_τ_opt, a_δ_opt, b_δ_opt, bsm)
    end
end

Base.eltype(::BSMVIPosterior{T, A, M}) where {T, A, M} = T
BayesDensityCore.model(vip::BSMVIPosterior) = vip.bsm

StatsBase.sample(vip::BSMVIPosterior{T, M, V}, n_samples::Int) where {T<:Real, M, V} = sample(Random.default_rng(), vip, n_samples)
function StatsBase.sample(rng::AbstractRNG, vip::BSMVIPosterior{T, M, V}, n_samples::Int) where {T<:Real, M, V}
    (; μ_opt, inv_Σ_opt, a_τ_opt, b_τ_opt, a_δ_opt, b_δ_opt, bsm) = vip
    K = length(basis(bsm))
    samples = Vector{NamedTuple{(:spline_coefs, :θ, :β, :τ2, :δ2), Tuple{Vector{T}, Vector{T}, Vector{T}, T, Vector{T}}}}(undef, n_samples)
    δ2 = Vector{T}(undef, K-3)

    potential_opt = inv_Σ_opt * μ_opt

    for m in 1:n_samples
        β = rand(rng, MvNormalCanon(potential_opt, inv_Σ_opt))
        θ = max.(eps(), logistic_stickbreaking(β))
        θ = θ / sum(θ)
        spline_coefs = theta_to_coef(θ, basis(bsm))
        τ2 = rand(rng, InverseGamma(a_τ_opt, b_τ_opt))
        for k in 1:K-3
            δ2[k] = rand(rng, InverseGamma(a_δ_opt[k], b_δ_opt[k]))
        end

        samples[m] = (spline_coefs = spline_coefs, θ = θ, β = β, τ2 = τ2, δ2 = δ2)
    end
    return PosteriorSamples{T}(samples, bsm, n_samples, 0)
end


function varinf(bsm::BSMModel; init_params=get_default_initparams(bsm), max_iter::Int=500) # Also: tolerance parameters
    return _variational_inference(bsm, init_params, max_iter)
end

# Can do something more sophisticated here at a later point in time if we get a good idea.
function get_default_initparams(bsm::BSMModel{T, A, NT}) where {T, A, NT}
    K = length(basis(bsm))
    P = spdiagm(K-3, K-1, 0=>fill(1, K-3), 1=>fill(-2, K-3), 2=>fill(1, K-3))
    a_τ_opt, b_τ_opt, a_δ, b_δ = hyperparams(bsm)
    a_δ_opt = fill(a_δ, K-3)
    b_δ_opt = fill(b_δ, K-3)
    μ_opt = compute_μ(basis(bsm), T)
    D = Diagonal(a_τ_opt / b_τ_opt * a_δ_opt ./ b_δ_opt)
    Q = transpose(P) * D * P
    inv_Σ_opt = Q + 0.05 * Diagonal(ones(K-1))
    return (μ_opt = μ_opt, inv_Σ_opt = inv_Σ_opt, a_τ_opt = a_τ_opt, b_τ_opt = b_τ_opt, a_δ_opt = a_δ_opt, b_δ_opt = b_δ_opt)
end

function _variational_inference(bsm::BSMModel{T, A, NamedTuple{(:x, :log_B, :b_ind, :bincounts, :μ, :P, :n), Vals}}, init_params::NT, max_iter::Int) where {T, A, Vals, NT}
    bs = basis(bsm)
    K = length(bs)
    (; x, log_B, b_ind, bincounts, μ, P, n) = bsm.data
    P = sparse(P) # Needed for selinv
    n_bins = length(bincounts)

    a_τ, b_τ, a_δ, b_δ = hyperparams(bsm)

    (; μ_opt, inv_Σ_opt, a_τ_opt, b_τ_opt, a_δ_opt, b_δ_opt) = init_params

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
        E_β = copy(μ_opt)         # Do this for enhanced readability, remove later
        E_β2 = abs2.(μ_opt) .+ d0
        E_Δ2 = abs2.(diff(diff(μ_opt - μ))) + view(d0, 3:K-1) + 4 * view(d0, 2:K-2) + view(d0, 1:K-3) - 4 * view(d1, 2:K-2) - 4 * view(d1, 1:K-3) + 2 * d2

        # Update q(δ²)
        a_δ_opt = fill(a_δ + T(0.5), K-3)
        b_δ_opt = b_δ .+ T(0.5) * E_Δ2 * a_τ_opt / b_τ_opt

        # Update q(τ²)
        a_τ_opt = a_τ + T(0.5) * (K-3)
        b_τ_opt = b_τ + T(0.5) * sum(E_Δ2 .* a_δ_opt ./ b_δ_opt)

        # Update q(z, ω)
        E_N = zeros(T, K)
        non_basis_term[1:K-1] = cumsum(@. -log(cosh(T(0.5)*sqrt(E_β2))) - T(0.5) * E_β - log(T(2))) # Replace the β's with the corresponding expectations
        non_basis_term[K] = non_basis_term[K-1]
        non_basis_term[1:K-1] .+= E_β
        for i in 1:n_bins
            # Compute the four nonzero probabilities:
            k0 = b_ind[i]
            for l in 1:4
                k = k0 + l - 1
                logprobs[l] = log_B[i,l] + non_basis_term[k] 
            end
            probs = softmax(logprobs)
            E_N[k0:k0+3] .+= bincounts[i] * probs
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
    return BSMVIPosterior{T}(μ_opt, BandedMatrix(inv_Σ_opt), a_τ_opt, b_τ_opt, a_δ_opt, b_δ_opt, bsm)
end

#= K = 200
P = BandedMatrix((0=>fill(1, K-3), 1=>fill(-2, K-3), 2=>fill(1, K-3)), (K-3, K-1))
Q = transpose(P) * P + Diagonal(ones(199))

P = spdiagm(K-3, K-1, 0=>fill(1, K-3), 1=>fill(-2, K-3), 2=>fill(1, K-3))
Q = transpose(P) * P + Diagonal(ones(199))

begin
    Z, _ = selinv(Q; depermute=true)
    d0 = Vector(diag(Z))
    d1 = Vector(diag(Z, 1))
    d2 = Vector(diag(Z, 2))
end
 =#