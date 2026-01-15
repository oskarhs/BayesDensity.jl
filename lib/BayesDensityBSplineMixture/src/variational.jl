"""
    BSplineMixtureVIPosterior{T<:Real, A<:MvNormalCanon{T}, B<:InverseGamma{T}, M<:BSplineMixture} <: AbstractVIPosterior{T}

Struct representing the variational posterior distribution of a [`BSplineMixture`](@ref).

# Fields
* `q_β`: Distribution representing the optimal variational density q*(β).
* `q_τ`: Distribution representing the optimal variational density q*(τ²).
* `q_δ`: Vector of distributions, with element `k` corresponding to the optimal variational density q*(δₖ²).
* `bsm`: The `BSplineMixture` to which the variational posterior was fit.
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
    print(io, model(vip))
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
        rtol::Real=1e-6
    ) -> BSplineMixtureVIPosterior{<:Real}

Find a variational approximation to the posterior distribution of a [`BSplineMixture`](@ref) using mean-field variational inference.

# Arguments
* `bsm`: The `BSplineMixture` whose posterior we want to approximate.

# Keyword arguments
* `init_params`: Initial values of the VI parameters `μ_opt` `inv_Σ_opt`, `b_τ_opt` and `b_δ_opt`, supplied as a NamedTuple.
* `max_iter`: Maximal number of VI iterations. Defaults to `2000`.
* `rtol`: Relative tolerance used to determine convergence. Defaults to `1e-6`.

# Returns
* `vip`: A [`BSplineMixtureVIPosterior`](@ref) object representing the variational posterior.
"""
function BayesDensityCore.varinf(bsm::BSplineMixture; init_params=get_default_initparams(bsm), max_iter::Int=2000, rtol::Real=1e-6) # Also: tolerance parameters
    (max_iter >= 1) || throw(ArgumentError("Maximum number of iterations must be positive."))
    (rtol > 0.0) || @warn "Relative tolerance is not positive."
    return _variational_inference(bsm, init_params, max_iter, rtol)
end

# Can do something more sophisticated here at a later point in time if we get a good idea.
function get_default_initparams(bsm::BSplineMixture{T, A, NT}) where {T, A, NT}
    K = length(basis(bsm))
    P = bsm.data.P
    #P = spdiagm(K-3, K-1, 0=>fill(1, K-3), 1=>fill(-2, K-3), 2=>fill(1, K-3))
    (; a_τ, b_τ, a_δ, b_δ, σ) = hyperparams(bsm)
    Q0 = Diagonal(vcat([1/σ^2, 1/σ^2], zeros(T, K-3)))
    a_τ_opt = a_τ + (K-3)/2
    b_τ_opt = b_τ
    a_δ_opt = fill(a_δ + 1/2, K-3)
    b_δ_opt = fill(b_δ, K-3)

    μ_opt = compute_μ(basis(bsm))
    D = Diagonal(a_τ_opt / b_τ_opt * a_δ_opt ./ b_δ_opt)
    Q = transpose(P) * D * P + Q0
    inv_Σ_opt = Q + 0.05 * Diagonal(ones(K-1))
    return (μ_opt = μ_opt, inv_Σ_opt = inv_Σ_opt, b_τ_opt = b_τ_opt, b_δ_opt = b_δ_opt)
end

function _variational_inference(bsm::BSplineMixture{T, A, NamedTuple{(:x, :log_B, :b_ind, :bincounts, :μ, :P, :n), Vals}}, init_params::NT, max_iter::Int, rtol::Real) where {T, A, Vals, NT}
    bs = basis(bsm)
    K = length(bs)
    (; x, log_B, b_ind, bincounts, μ, P, n) = bsm.data
    n_bins = length(bincounts)

    # Get hyperparameters
    (; a_τ, b_τ, a_δ, b_δ, σ) = hyperparams(bsm)
    Q0 = Diagonal(vcat([1/σ^2, 1/σ^2], zeros(T, K-3)))

    (; μ_opt, inv_Σ_opt, b_τ_opt, b_δ_opt) = init_params
    # Find the required posterior moments of q(β):
    Z, _ = selinv(sparse(inv_Σ_opt); depermute=true) # Get pentadiagonal entries of Σ*
    d0 = Vector(diag(Z)) # Vector of Σ*_{k,k}
    d1 = Vector(diag(Z, 1)) # Vector of Σ*_{k,k+1}
    d2 = Vector(diag(Z, 2)) # Vector of Σ*_{k,k+2}
    E_β = copy(μ_opt)         # Do this for enhanced readability, remove later
    E_β2 = abs2.(μ_opt) .+ d0
    E_Δ2 = abs2.(diff(diff(μ_opt - μ))) + view(d0, 3:K-1) + 4 * view(d0, 2:K-2) + view(d0, 1:K-3) - 4 * view(d1, 2:K-2) - 4 * view(d1, 1:K-3) + 2 * d2

    # These two stay constant throughout the optimization loop.
    a_τ_opt = a_τ + (K-3)/2
    a_δ_opt = fill(a_δ + 1/2, K-3)

    non_basis_term = Vector{T}(undef, K)
    logprobs = Vector{T}(undef, 4)
    E_S = Vector{T}(undef, K)
    E_ω = Vector{T}(undef, K-1)

    ELBO = Vector{T}(undef, max_iter)
    ELBO_last = one(T)

    # Optimization loop
    iter = 1
    converged = false
    while !converged && iter ≤ max_iter
        # Update q(δ²)
        b_δ_opt = b_δ .+ T(0.5) * E_Δ2 * a_τ_opt / b_τ_opt

        # Update q(τ²)
        b_τ_opt = b_τ + T(0.5) * sum(E_Δ2 .* a_δ_opt ./ b_δ_opt)

        # Update q(z, ω)
        E_N = zeros(T, K)
        non_basis_term[1:K-1] = cumsum(@. -log(cosh(T(0.5)*sqrt(E_β2))) - T(0.5) * E_β - log(T(2))) # Replace the β's with the corresponding expectations
        non_basis_term[K] = non_basis_term[K-1]
        non_basis_term[1:K-1] .+= E_β
        last_term = zero(T)
        @inbounds for i in 1:n_bins
            # Compute the four nonzero probabilities:
            k0 = b_ind[i]
            for l in 1:4
                k = k0 + l - 1
                logprobs[l] = log_B[i,l] + non_basis_term[k] 
            end
            probs = softmax(logprobs)
            E_N[k0:k0+3] .+= bincounts[i] * probs
            last_term += bincounts[i] * sum(probs .* (log_B[i,:] - log.(probs)))
        end
        E_S[1] = n
        E_S[2:K] .= n .- cumsum(view(E_N, 1:K-1))
        E_S = clamp.(E_S, 0.0, n)
        E_ω = @. tanh(sqrt(E_β2)/2) / (2*sqrt(E_β2)) * E_S[1:K-1]
        # Compute KL divergence to prior:
        KL_ω = view(E_S, 1:K-1) .* log.(cosh.(sqrt.(E_β2) / 2)) - E_β2 .* E_ω/2 # NB! Before q(beta) is updated
        
        # Update q(β)
        D = Diagonal(a_τ_opt / b_τ_opt * a_δ_opt ./ b_δ_opt)
        Q = transpose(P) * D * P + Q0
        Ωκ = view(E_N, 1:K-1) - view(E_S, 1:K-1) / 2
        inv_Σ_opt = Q + Diagonal(E_ω)
        h2 = Q*μ
        h1 = h2 + Ωκ
        μ_opt = inv_Σ_opt \ h1

        # Compute ELBO:
        KL_τ2 = (a_τ_opt - a_τ) * digamma(a_τ_opt) - loggamma(a_τ_opt) + loggamma(a_τ) + a_τ*(log(b_τ_opt) - log(b_τ)) + a_τ_opt/b_τ_opt * (b_τ - b_τ_opt)
        KL_δ2 = @. (a_δ_opt - a_δ) * digamma(a_δ_opt) - loggamma(a_δ_opt) + loggamma(a_δ) + a_δ*(log(b_δ_opt) - log(b_δ)) + a_δ_opt / b_δ_opt * (b_δ - b_δ_opt)
        KL_β = kldivergence(h1, inv_Σ_opt, h2, Q)
        # Find the required posterior moments of q(β):
        Z, _ = selinv(sparse(inv_Σ_opt); depermute=true) # Get pentadiagonal entries of Σ*
        d0 = Vector(diag(Z)) # Vector of Σ*_{k,k}
        d1 = Vector(diag(Z, 1)) # Vector of Σ*_{k,k+1}
        d2 = Vector(diag(Z, 2)) # Vector of Σ*_{k,k+2}
        E_β = copy(μ_opt)         # Do this for enhanced readability, remove later
        E_β2 = abs2.(μ_opt) .+ d0
        E_Δ2 = abs2.(diff(diff(μ_opt - μ))) + view(d0, 3:K-1) + 4 * view(d0, 2:K-2) + view(d0, 1:K-3) - 4 * view(d1, 2:K-2) - 4 * view(d1, 1:K-3) + 2 * d2

        nonprob_term = sum(-view(E_S, 1:K-1) * log(T(2)) + Ωκ .* μ_opt - E_ω .* E_β2 / 2)

        ELBO[iter] = - KL_τ2 - sum(KL_δ2) - KL_β - sum(KL_ω) + nonprob_term + last_term
        converged = ((ELBO[iter]-ELBO_last)/abs(ELBO[iter]) < rtol) && iter ≥ 2
        ELBO_last = ELBO[iter]
        iter += 1
    end

    converged || @warn "Failed to meet convergence criterion in $(iter-1) iterations."
    return BSplineMixtureVIPosterior{T}(μ_opt, inv_Σ_opt, a_τ_opt, b_τ_opt, a_δ_opt, b_δ_opt, bsm), ELBO[1:iter-1]
end

function _variational_inference(bsm::BSplineMixture{T, A, NamedTuple{(:x, :log_B, :b_ind, :μ, :P, :n), Vals}}, init_params::NT, max_iter::Int, rtol::Real) where {T, A, Vals, NT}
    bs = basis(bsm)
    K = length(bs)
    (; x, log_B, b_ind, μ, P, n) = bsm.data

    # Get prior hyperparameters
    (; a_τ, b_τ, a_δ, b_δ, σ) = hyperparams(bsm)
    Q0 = Diagonal(vcat([1/σ^2, 1/σ^2], zeros(T, K-3)))

    (; μ_opt, inv_Σ_opt, b_τ_opt, b_δ_opt) = init_params
    # Find the required posterior moments of q(β):
    Z, _ = selinv(sparse(inv_Σ_opt); depermute=true) # Get pentadiagonal entries of Σ*
    d0 = Vector(diag(Z)) # Vector of Σ*_{k,k}
    d1 = Vector(diag(Z, 1)) # Vector of Σ*_{k,k+1}
    d2 = Vector(diag(Z, 2)) # Vector of Σ*_{k,k+2}
    E_β = copy(μ_opt)         # Do this for enhanced readability, remove later
    E_β2 = abs2.(μ_opt) .+ d0
    E_Δ2 = abs2.(diff(diff(μ_opt - μ))) + view(d0, 3:K-1) + 4 * view(d0, 2:K-2) + view(d0, 1:K-3) - 4 * view(d1, 2:K-2) - 4 * view(d1, 1:K-3) + 2 * d2

    # These two stay constant throughout the optimization loop.
    a_τ_opt = a_τ + (K-3)/2
    a_δ_opt = fill(a_δ + 1/2, K-3)

    non_basis_term = Vector{T}(undef, K)
    logprobs = Vector{T}(undef, 4)
    E_S = Vector{T}(undef, K)
    E_ω = Vector{T}(undef, K-1)

    ELBO = Vector{T}(undef, max_iter)
    ELBO_last = one(T)

    # Optimization loop
    iter = 1
    converged = false
    while !converged && iter ≤ max_iter
        # Update q(δ²)
        b_δ_opt = b_δ .+ T(0.5) * E_Δ2 * a_τ_opt / b_τ_opt

        # Update q(τ²)
        b_τ_opt = b_τ + T(0.5) * sum(E_Δ2 .* a_δ_opt ./ b_δ_opt)

        # Update q(z, ω)
        E_N = zeros(T, K)
        non_basis_term[1:K-1] = cumsum(@. -log(cosh(T(0.5)*sqrt(E_β2))) - T(0.5) * E_β - log(T(2))) # Replace the β's with the corresponding expectations
        non_basis_term[K] = non_basis_term[K-1]
        non_basis_term[1:K-1] .+= E_β
        last_term = zero(T)
        @inbounds for i in 1:n
            # Compute the four nonzero probabilities:
            k0 = b_ind[i]
            for l in 1:4
                k = k0 + l - 1
                logprobs[l] = log_B[i,l] + non_basis_term[k] 
            end
            probs = softmax(logprobs)
            E_N[k0:k0+3] .+= probs
            last_term += sum(probs .* (log_B[i,:] - log.(probs)))
        end
        E_S[1] = n
        E_S[2:K] .= n .- cumsum(view(E_N, 1:K-1))
        E_S = clamp.(E_S, 0.0, n)
        E_ω = @. tanh(sqrt(E_β2)/2) / (2*sqrt(E_β2)) * E_S[1:K-1]
        # Compute the required KL-divergence
        KL_ω = view(E_S, 1:K-1) .* log.(cosh.(sqrt.(E_β2) / 2)) - E_β2 .* E_ω/2 # NB! Before q(beta) is updated
        
        # Update q(β)
        D = Diagonal(a_τ_opt / b_τ_opt * a_δ_opt ./ b_δ_opt)
        Q = transpose(P) * D * P + Q0
        Ωκ = view(E_N, 1:K-1) - view(E_S, 1:K-1) / 2
        inv_Σ_opt = Q + Diagonal(E_ω)
        h2 = Q*μ
        h1 = Q*μ + Ωκ
        μ_opt = inv_Σ_opt \ h1

        # Compute ELBO:
        KL_τ2 = (a_τ_opt - a_τ) * digamma(a_τ_opt) - loggamma(a_τ_opt) + loggamma(a_τ) + a_τ*(log(b_τ_opt) - log(b_τ)) + a_τ_opt/b_τ_opt * (b_τ - b_τ_opt)
        KL_δ2 = @. (a_δ_opt - a_δ) * digamma(a_δ_opt) - loggamma(a_δ_opt) + loggamma(a_δ) + a_δ*(log(b_δ_opt) - log(b_δ)) + a_δ_opt / b_δ_opt * (b_δ - b_δ_opt)
        KL_β = kldivergence(h1, inv_Σ_opt, h2, Q)
        # Find the required posterior moments of q(β):
        Z, _ = selinv(sparse(inv_Σ_opt); depermute=true) # Get pentadiagonal entries of Σ*
        d0 = Vector(diag(Z)) # Vector of Σ*_{k,k}
        d1 = Vector(diag(Z, 1)) # Vector of Σ*_{k,k+1}
        d2 = Vector(diag(Z, 2)) # Vector of Σ*_{k,k+2}
        E_β = copy(μ_opt)         # Do this for enhanced readability, remove later
        E_β2 = abs2.(μ_opt) .+ d0
        E_Δ2 = abs2.(diff(diff(μ_opt - μ))) + view(d0, 3:K-1) + 4 * view(d0, 2:K-2) + view(d0, 1:K-3) - 4 * view(d1, 2:K-2) - 4 * view(d1, 1:K-3) + 2 * d2

        nonprob_term = sum(-view(E_S, 1:K-1) * log(T(2)) + Ωκ .* μ_opt - E_ω .* E_β2 / 2)

        ELBO[iter] = - KL_τ2 - sum(KL_δ2) - KL_β - sum(KL_ω) + nonprob_term + last_term
        converged = ((ELBO[iter]-ELBO_last)/abs(ELBO[iter]) < rtol) && iter ≥ 2
        ELBO_last = ELBO[iter]
        iter += 1
    end

    converged || @warn "Failed to meet convergence criterion in $(iter-1) iterations."
    return BSplineMixtureVIPosterior{T}(μ_opt, BandedMatrix(inv_Σ_opt), a_τ_opt, b_τ_opt, a_δ_opt, b_δ_opt, bsm), ELBO[1:iter-1]
end