using BayesDensityCore
using CairoMakie
using Distributions
using Random
using SpecialFunctions # For the digamma-function
using StatsBase

struct BernsteinDensity{T<:Real, D<:NamedTuple} <: AbstractBayesDensityModel{T}
    data::D # NamedTuple holding data
    K::Int  # Basis dimension
    a::T    # Symmetric Dirichlet parameter.
    function BernsteinDensity{T}(x::AbstractVector{<:Real}, K::Int; a::Real=1.0) where {T<:Real}
        φ_x = Matrix{T}(undef, (length(x), K))
        for i in eachindex(x)
            for k in 1:K
                φ_x[i, k] = pdf(Beta(k, K - k + 1), x[i])
            end
        end
        data = (x = x, n = length(x), φ_x = φ_x)
        return new{T, typeof(data)}(data, K, T(a))
    end
end
BernsteinDensity(args...; kwargs...) = BernsteinDensity{Float64}(args...; kwargs...) # For convenience
Base.:(==)(bd1::BernsteinDensity, bd2::BernsteinDensity) = bd1.data == bd2.data && bd1.K == bd2.K && bd1.a == bd2.a

BayesDensityCore.default_grid_points(::BernsteinDensity{T}) where {T} = LinRange{T}(0, 1, 2001)

function Distributions.pdf(bdm::BernsteinDensity{T, D}, params::NamedTuple, t::S) where {T<:Real, D, S<:Real}
    K = bdm.K
    (; θ) = params
    f = zero(promote_type(T, S))
    for k in 1:K
        f += θ[k] * pdf(Beta(k, K - k + 1), t)
    end
    return f
end

function Distributions.cdf(bdm::BernsteinDensity{T, D}, params::NamedTuple, t::S) where {T<:Real, D, S<:Real}
    K = bdm.K
    (; θ) = params
    f = zero(promote_type(T, S))
    for k in 1:K
        f += θ[k] * cdf(Beta(k, K - k + 1), t)
    end
    return f
end

BayesDensityCore.support(::BernsteinDensity{T, D}) where {T, D} = (T(0.0), T(1.0))
BayesDensityCore.hyperparams(bdm::BernsteinDensity) = (a = bdm.a,)

function StatsBase.sample(rng::AbstractRNG, bdm::BernsteinDensity{T, D}, n_samples::Int; n_burnin=min(div(length(x), 5), 1000), init_params::NamedTuple=(θ = fill(1/K, K),)) where {T, D}
    (; K, data, a) = bdm
    (; x, n, φ_x) = data

    a_vec = fill(a, K) # Dirichlet prior parameter

    θ = T.(init_params.θ) # Initialize θ as the uniform vector
    probs = Vector{T}(undef, K) # Vector used to store intermediate calculations of p(zᵢ|θ, x)

    # Store samples as a vector of NamedTuples
    samples = Vector{NamedTuple{(:θ,), Tuple{Vector{Float64}}}}(undef, n_samples)

    for m in 1:n_samples
        N = zeros(Int, K) # N[k] = number of z[i] equal to k.
        for i in 1:n
            for k in 1:K
                probs[k] = θ[k] * φ_x[i, k]
            end
            probs = probs / sum(probs)
            N .+= rand(rng, Multinomial(1, probs)) # sample zᵢ ∼ p(zᵢ|θ, x)
        end
        θ = rand(rng, Dirichlet(a_vec + N)) # sample θ ∼ p(θ|z, x)
        samples[m] = (θ = θ,) # store the current value of θ
    end
    return PosteriorSamples{T}(samples, bdm, n_samples, n_burnin)
end

d_true = Kumaraswamy(2, 5) # Simulate some data from a density supported on [0, 1]
rng = Xoshiro(1) # for reproducibility
x = rand(rng, d_true, 1000)

K = 20
bdm = BernsteinDensity(x, K) # Create Bernstein density model object (a = 1)
ps = sample(rng, bdm, 1_000; n_burnin=500) # Run MCMC

t = LinRange(0, 1, 1001) # Grid for plotting

fig = Figure(size=(600, 320))
ax1 = Axis(fig[1,1], xlabel="x", ylabel="Density")
plot!(ax1, ps, t, label="MCMC") # Plot the posterior mean and credible bands:
lines!(ax1, t, pdf(d_true, t), label="Truth", color=:black) # Also plot truth for comparison

ax2 = Axis(fig[1,2], xlabel="x", ylabel="Cumulative distribution")
plot!(ax2, ps, cdf, t, label="MCMC")
lines!(ax2, t, cdf(d_true, t), label="Truth", color=:black)

Legend(fig[1,3], ax1, framevisible=false)

save(joinpath("src", "assets", "bernstein_tutorial", "bernstein_mcmc.svg"), fig)

struct BernsteinDensityVIPosterior{T<:Real, D<:Dirichlet{T}, M<:BernsteinDensity} <: AbstractVIPosterior{T}
    q_θ::D
    model::M
    function BernsteinDensityVIPosterior{T}(r::AbstractVector{<:Real}, model::M) where {T<:Real, M<:BernsteinDensity}
        a = hyperparams(model).a
        K = model.K
        q_θ = Dirichlet{T}(fill(a, K) + r)
        return new{T, Dirichlet{T}, M}(q_θ, model)
    end
end

BayesDensityCore.model(vip::BernsteinDensityVIPosterior) = vip.model

function StatsBase.sample(rng::AbstractRNG, vip::BernsteinDensityVIPosterior{T,D, M}, n_samples::Int) where {T, D, M}
    q_θ = vip.q_θ
    samples = Vector{NamedTuple{(:θ,), Tuple{Vector{Float64}}}}(undef, n_samples)
    for m in 1:n_samples
        θ = rand(rng, q_θ)
        samples[m] = (θ = θ,)
    end
    # Note that we return independent samples here, so burn-in is not needed
    return PosteriorSamples{T}(samples, model(vip), n_samples, 0)
end

function Bernstein_ELBO(model::BernsteinDensity{T, D}, r::AbstractVector{<:Real}, ω::AbstractMatrix{<:Real}) where {T, D}
    (; data, K, a) = model
    (; x, n, φ_x) = data
    logφ_x = log.(φ_x)
    ELBO = loggamma(a*K) - loggamma(a*K+n)
    ELBO += sum(loggamma.(r .+ a)) - K*loggamma(a)
    for k in 1:K
        for i in 1:n
            ELBO += ω[i,k]*(logφ_x[i,k] - log(ω[i,k]))
        end
    end
    return ELBO
end


function BayesDensityCore.varinf(model::BernsteinDensity{T, D}; max_iter::Int=1000, rtol::Real=1e-4) where {T, D}
    (; data, K, a) = model
    (; x, n, φ_x) = data

    # Initialize the latent variables ω[i,k] = q(z_i = k) to 1/K:
    ω = fill(T(1/K), (n, K))
    r = fill(a + n/K, K)

    # CAVI optimization loop
    ELBO_prev = T(-1)
    ELBO = Vector{T}(undef, max_iter)
    converged = false
    iter = 1
    while !converged && iter <= max_iter
        # Update q(θ)
        r = fill(a, K) + vec(sum(ω, dims=1))

        # Update q(z)
        for i in 1:n
            # Compute q(z_i = k) up to proportionality
            for k in 1:K
               ω[i,k] = φ_x[i,k] * exp(digamma(a + r[k])) 
            end
            # Normalize so that the rows of ω sum to 1:
            ω[i,:] = ω[i,:] / sum(ω[i,:])
        end

        # Check if the procedure has converged:
        ELBO[iter] = Bernstein_ELBO(model, r, ω)

        # Run at least two iterations
        converged = (abs(ELBO_prev - ELBO[iter]) / ELBO_prev <= rtol) && iter > 1
        ELBO_prev = ELBO[iter]
        iter += 1
    end

    # Print a warning if the procedure fails to converge within the maximum number of iterations
    converged || @warn "Maximum number of iterations reached."
    
    posterior = BernsteinDensityVIPosterior{T}(r, model)
    info = VariationalOptimizationResult{T}(ELBO[1:iter-1], converged, iter-1, rtol, posterior)
    return posterior, info
end

# Estimate model:
vip, info = varinf(bdm)

# Plot the estimated pdf and cdf
t = LinRange(0, 1, 1001) # Grid for plotting

fig = Figure(size=(600, 320))
ax1 = Axis(fig[1,1], xlabel="x", ylabel="Density")
plot!(ax1, vip, t, label="VI") # Plot the posterior mean and credible bands:
lines!(ax1, t, pdf(d_true, t), label="Truth", color=:black) # Also plot truth for comparison

ax2 = Axis(fig[1,2], xlabel="x", ylabel="Cumulative distribution")
plot!(ax2, vip, cdf, t, label="VI")
lines!(ax2, t, cdf(d_true, t), label="Truth", color=:black)

Legend(fig[1,3], ax1, framevisible=false)

save(joinpath("src", "assets", "bernstein_tutorial", "bernstein_varinf.svg"), fig)

# Plot ELBO 
fig = Figure(size=(400, 350))
ax = Axis(fig[1,1], xlabel="Iteration", ylabel="ELBO")
lines!(ax, info)
save(joinpath("src", "assets", "bernstein_tutorial", "bernstein_elbo.svg"), fig)
