
StatsBase.sample(bsm::BSMModel, n_samples::Int; kwargs...) = sample(Random.default_rng(), bsm, n_samples; kwargs...)

function StatsBase.sample(rng::AbstractRNG, bsm::BSMModel, n_samples::Int; n_burnin::Int = min(1000, div(n_samples, 5)))
    return sample_posterior(rng, bsm, n_samples, n_burnin)
end

# To do: make a multithreaded version (also one for unbinned data)
function sample_posterior(rng::AbstractRNG, bsm::BSMModel{T, A, NamedTuple{(:x, :log_B, :b_ind, :bincounts, :μ, :P, :n), Vals}}, n_samples::Int, n_burnin::Int) where {T, A, Vals}
    basis = BSplineKit.basis(bsm)
    K = length(basis)
    (; log_B, b_ind, bincounts, μ, P, n) = bsm.data
    n_bins = length(bincounts)

    # Prior Hyperparameters
    a_τ, b_τ, a_δ, b_δ = hyperparams(bsm)
    
    # Store draws
    β = copy(μ)
    τ2 = one(T)                # Global smoothing parameter
    δ2 = Vector{T}(undef, K-3) # Local smoothing parameters
    ω = Vector{T}(undef, K-1)  # PolyaGamma variables

    logprobs = Vector{T}(undef, 4)  # class label probabilities

    #θ = Vector{T}(undef, K) # Mixture probabilities
    θ = max.(eps(), logistic_stickbreaking(β))
    θ = θ / sum(θ)
    log_θ = log.(θ)
    
    # Initialize vector of samples
    samples = Vector{NamedTuple{(:spline_coefs, :θ, :β, :τ2, :δ2), Tuple{Vector{T}, Vector{T}, Vector{T}, T, Vector{T}}}}(undef, n_samples)
    spline_coefs = theta_to_coef(θ, basis)
    samples[1] = (spline_coefs = spline_coefs, θ = θ, β = β, τ2 = τ2, δ2 = δ2)

    for m in 2:n_samples

        # Update δ2: (some inefficiencies here, but okay for now)
        for k in 1:K-3
            a_δ_k_new = a_δ + T(0.5)
            b_δ_k_new = b_δ + T(0.5) * abs2( β[k+2] -  μ[k+2] - ( 2*(β[k+1] - μ[k+1]) - (β[k] - μ[k]) )) / τ2
            δ2[k] = rand(rng, InverseGamma(a_δ_k_new, b_δ_k_new))
        end

        # Update τ2
        a_τ_new = a_τ + T(0.5) * (K - 3)
        b_τ_new = b_τ
        for k in 1:K-3
            b_τ_new += T(0.5) * abs2( β[k+2] -  μ[k+2] - ( 2*(β[k+1] - μ[k+1]) - (β[k] - μ[k]) )) / δ2[k]
        end
        τ2 = rand(rng, InverseGamma(a_τ_new, b_τ_new))
        #τ2 = 0.01

        # Update z (N and S)
        N = zeros(Int, K)               # class label counts (of z[i]'s)
        for i in 1:n_bins
            # Compute the four nonzero probabilities:
            k0 = b_ind[i]
            for l in 1:4
                k = k0 + l - 1
                #= if k != K
                    #sumterm = sum(@. -log(cosh(T(0.5)*β[1:k, m-1])) - T(0.5) * β[1:k, m-1] - log(T(2)))
                    sumterm = sum(@. -log(cosh(T(0.5)*β[k0+1:k, m-1])) - T(0.5) * β[k0+1:k, m-1] - log(T(2)))
                    logprobs[l] = log_B[i, l] + β[k, m-1] + sumterm
                else
                    sumterm = sum(@. -log(cosh(T(0.5)*β[k0+1:K-1, m-1])) - T(0.5) * β[k0+1:K-1, m-1] - log(T(2)))
                    logprobs[l] = log_B[i, l] + sumterm
                end =#
                logprobs[l] = log_B[i,l] + log_θ[k] 
            end
            probs = softmax(logprobs)
            counts = rand(rng, Multinomial(bincounts[i], probs))
            N[k0:k0+3] .+= counts
        end
        # Update ω
        # Compute N and S
        S = n .- cumsum(vcat(0, N[1:K-1]))
        for k in 1:K-1
            c_k_new = S[k]
            d_k_new = β[k]
            ω[k] = rand(rng, PolyaGammaHybridSampler(c_k_new, d_k_new))
        end

        # Update β
        # Compute the Q matrix
        D = Diagonal(1 ./(τ2*δ2))
        Q = transpose(P) * D * P
        # Compute the Ω matrix (Note: Q + D retains the banded structure!)
        Ω = Diagonal(ω)
        inv_Σ_new = Ω + Q
        # Compute inv(Σ_new) * μ_new
        canon_mean_new = Q * μ + (N[1:K-1] - S[1:K-1]/2)
        # Sample β
        β = rand(rng, MvNormalCanon(canon_mean_new, inv_Σ_new))

        # Record θ
        θ = max.(eps(), logistic_stickbreaking(β))
        θ = θ / sum(θ)
        log_θ = log.(θ)

        # Compute coefficients in terms of unnormalized B-spline basis
        spline_coefs = theta_to_coef(θ, basis)
        samples[m] = (spline_coefs = spline_coefs, θ = θ, β = β, τ2 = τ2, δ2 = δ2)
    end
    return PosteriorSamples{T}(samples, bsm, n_samples, n_burnin)
end


function sample_posterior(rng::AbstractRNG, bsm::BSMModel{T, A, NamedTuple{(:x, :log_B, :b_ind, :μ, :P, :n), Vals}}, n_samples::Int, n_burnin::Int) where {T, A, Vals}
    basis = BSplineKit.basis(bsm)
    K = length(basis)
    (; log_B, b_ind, μ, P, n) = bsm.data

    # Prior Hyperparameters
    a_τ, b_τ, a_δ, b_δ = hyperparams(bsm)
    
    # Store draws
    β = copy(μ)
    τ2 = one(T)                # Global smoothing parameter
    δ2 = Vector{T}(undef, K-3) # Local smoothing parameters
    ω = Vector{T}(undef, K-1)  # PolyaGamma variables

    logprobs = Vector{T}(undef, 4)  # class label probabilities

    #θ = Vector{T}(undef, K) # Mixture probabilities
    θ = max.(eps(), logistic_stickbreaking(β))
    θ = θ / sum(θ)
    log_θ = log.(θ)
    
    # Initialize vector of samples
    samples = Vector{NamedTuple{(:spline_coefs, :θ, :β, :τ2, :δ2), Tuple{Vector{T}, Vector{T}, Vector{T}, T, Vector{T}}}}(undef, n_samples)
    spline_coefs = theta_to_coef(θ, basis)
    samples[1] = (spline_coefs = spline_coefs, θ = θ, β = β, τ2 = τ2, δ2 = δ2)

    for m in 2:n_samples

        # Update δ2: (some inefficiencies here, but okay for now)
        for k in 1:K-3
            a_δ_k_new = a_δ + T(0.5)
            b_δ_k_new = b_δ + T(0.5) * abs2( β[k+2] -  μ[k+2] - ( 2*(β[k+1] - μ[k+1]) - (β[k] - μ[k]) )) / τ2
            δ2[k] = rand(rng, InverseGamma(a_δ_k_new, b_δ_k_new))
            #δ2[k] = 1.0
        end

        # Update τ2
        a_τ_new = a_τ + T(0.5) * (K - 3)
        b_τ_new = b_τ
        for k in 1:K-3
            b_τ_new += T(0.5) * abs2( β[k+2] -  μ[k+2] - ( 2*(β[k+1] - μ[k+1]) - (β[k] - μ[k]) )) / δ2[k]
        end
        τ2 = rand(rng, InverseGamma(a_τ_new, b_τ_new))
        #τ2 = 0.01

        # Update z (N and S)
        N = zeros(Int, K)               # class label counts (of z[i]'s)
        for i in 1:n
            # Compute the four nonzero probabilities:
            k0 = b_ind[i]
            for l in 1:4
                k = k0 + l - 1
                #= if k != K
                    #sumterm = sum(@. -log(cosh(T(0.5)*β[1:k, m-1])) - T(0.5) * β[1:k, m-1] - log(T(2)))
                    sumterm = sum(@. -log(cosh(T(0.5)*β[k0+1:k, m-1])) - T(0.5) * β[k0+1:k, m-1] - log(T(2)))
                    logprobs[l] = log_B[i, l] + β[k, m-1] + sumterm
                else
                    sumterm = sum(@. -log(cosh(T(0.5)*β[k0+1:K-1, m-1])) - T(0.5) * β[k0+1:K-1, m-1] - log(T(2)))
                    logprobs[l] = log_B[i, l] + sumterm
                end =#
                logprobs[l] = log_B[i,l] + log_θ[k] 
            end
            probs = softmax(logprobs)
            counts = rand(rng, Multinomial(1, probs))
            N[k0:k0+3] .+= counts
        end
        # Update ω
        # Compute N and S
        S = n .- cumsum(vcat(0, N[1:K-1]))
        for k in 1:K-1
            c_k_new = S[k]
            d_k_new = β[k]
            ω[k] = rand(rng, PolyaGammaHybridSampler(c_k_new, d_k_new))
        end

        # Update β
        # Compute the Q matrix
        D = Diagonal(1 ./(τ2*δ2))
        Q = transpose(P) * D * P
        # Compute the Ω matrix (Note: Q + D retains the banded structure!)
        Ω = Diagonal(ω)
        inv_Σ_new = Ω + Q
        # Compute inv(Σ_new) * μ_new
        canon_mean_new = Q * μ + (N[1:K-1] - S[1:K-1]/2)
        # Sample β
        β[:, m] = rand(rng, MvNormalCanon(canon_mean_new, inv_Σ_new))

        # Record θ
        θ = max.(eps(), logistic_stickbreaking(β))
        θ = θ / sum(θ)
        log_θ = log.(θ)
        
        # Compute coefficients in terms of unnormalized B-spline basis
        spline_coefs = theta_to_coef(θ, basis)
        samples[m] = (spline_coefs = spline_coefs, θ = θ, β = β, τ2 = τ2, δ2 = δ2)
    end
    return PosteriorSamples{T}(samples, bsm, n_samples, n_burnin)
end