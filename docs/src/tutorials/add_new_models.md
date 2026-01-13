# Implementing new Bayesian density estimators

The following page provides a tutorial on how to implement new Bayesian density estimators compatible with the BayesDensity.jl-package.
Prior to reading this tutorial, one should have already familiarized oneself with the general package API, for instance by reading the [General API](@ref) documentation.

In order to be able to follow this tutorial, it is advantageous to have some prior exposure to data-augmentation schemes, Gibbs sampling and mean-field variational inference.
A good introduction to all three topics can be found in [Bishop2006pattern](@citet).

!!! note
    The focus of the following tutorial is to present how one can implement new Bayesian models in a `BayesDensity`-compatible way. As a result, the implementation presented here is by no means optimal in terms of computational efficiency or numerical stability for this particular example.

## Bayesian inference for Bernstein densities.

For our tutorial, we will illustrate by focusing on a Bayesian Bernstein-type density estimator for data supported on the unit interval.[^1] This section provides the theoretical background for the model we will later implement as an example, and can be skipped by readers who are more interested in the details of the implementation itself.

[^1]:
    A Bayesian Bernstein-type estimator, where the number ``K`` of mixture components is treated as a further random variable has been proposed by [Petrone1999bernstein](@citet).

Given a positive integer ``K``, we say that ``f`` is a Bernstein density if we can write
```math
f(x) = \sum_{k=1}^K \theta_k\, \varphi_k(x), \quad x\in [0, 1],
```
where ``\varphi_k(\cdot)`` is the density of the ``\mathrm{Beta}(k, K-k+1)``-distribution and ``\boldsymbol{\theta} \in \{\boldsymbol{\vartheta}\in [0,1]^K \colon \sum_k \vartheta_k = 1 \}``.
The corresponding cumulative distribution function ``F`` is then
```math
F(x) = \sum_{k=1}^K \theta_k\, \int_{0}^x\varphi_k(t)\, \text{d}t, \quad x\in [0, 1],
```
owing to the linearity of the integral.

Our main motivation for considering the Bernstein model is that under mild regularity conditions on the true density ``f_0``, the Bernstein density ``f`` can approximate ``f_0`` to an arbitrary degree of precision with respect to a suitable metric, such as the total variation distance, provided ``K`` is sufficiently large.

For a Bayesian treatment of the Bernstein density model, we impose a ``\mathrm{Dirichlet}(\boldsymbol{a})``-prior distribution on ``\boldsymbol{\theta}``, where ``\boldsymbol{a} = (a,a, \ldots, a)`` for some ``a>0``.
Given an observed independent and identically distributed sample ``\boldsymbol{x} = (x_1, x_2, \ldots, x_n)``, the likelihood of the observed sample under the Bernstein model for ``f`` is
```math
p(\boldsymbol{x}\,|\, \boldsymbol{\theta}) = \prod_{i=1}^n \sum_{k=1}^K \theta_k\, \varphi_k(x_i).
```
The form taken by the likelihood function above makes Bayesian inference challenging due to the fact that the resulting posterior distribution is analytically intractable. However, by augmenting the data with latent variables ``\boldsymbol{z} \in \{1,2,\ldots, K\}^n``, it is possible to perform posterior inference very efficiently through Gibbs sampling or mean-field VI. Another possible way of defining the Bernstein density model is to let ``x_i \, |\, \boldsymbol{\theta}, z_i = k \sim \varphi_k`` and ``p(z_i = k) = \theta_k`` for all ``i``, as this leads to the same likelihood function as previously when the ``z_i`` are marginalized out.`

Under this data augmentation strategy it can then be shown that joint posterior of ``\boldsymbol{\theta}, \boldsymbol{z}`` is
```math
p(\boldsymbol{\theta}, \boldsymbol{z}\, |\, \boldsymbol{x}) \propto \prod_{k = 1}^K \theta_k^{N_k + a - 1} \prod_{i=1}^n \prod_{k=1}^K \varphi_k(x_i)^{\mathbf{1}_{\{k\}}(z_i)},
```
where ``\mathbf{1}_{\{k\}}(\cdot)`` is the indicator function and ``N_k = \sum_{i=1}^n \mathbf{1}_{\{k\}}(z_i)``.

### Gibbs sampling
To write down a Gibbs sampler for this model, we need to derive the full conditional distributions of ``\boldsymbol{\theta}`` and ``\boldsymbol{z}``. In this case, direct inspection of the joint posterior shows that
```math
p(\boldsymbol{\theta}\, |\, \boldsymbol{z}, \boldsymbol{x}) = \mathrm{Dirichlet}(\boldsymbol{a} + \boldsymbol{N}),
```
where ``\boldsymbol{N} = (N_1, N_2, \ldots, N_K)``. The full conditional distributions for ``\boldsymbol{z}`` are 

```math
p(\boldsymbol{z}\, |\, \boldsymbol{\theta}, \boldsymbol{x}) \propto \prod_{i=1}^n \prod_{k=1}^K \big\{\theta_k\,\varphi_k(x_i)\big\}^{\mathbf{1}_{\{k\}}(z_i)}.
```
Hence, we see that ``z_1, \ldots, z_n`` are independent given ``\boldsymbol{\theta}, \boldsymbol{x}``, with ``p(z_i = k \,|\, \boldsymbol{\theta}, \boldsymbol{x}) \propto \theta_k\, \varphi_k(x_i)``.

### Variational inference
For the Bernstein density , it is relatively straightforward to implement a mean-field variational inference scheme. Here, we approximate the joint posterior ``p(\boldsymbol{\theta}, \boldsymbol{z}\,|\, \boldsymbol{x})`` via a distribution ``q`` which satisfies
```math
q(\boldsymbol{\theta}, \boldsymbol{z}) = q(\boldsymbol{\theta})\, q(\boldsymbol{z}).
```
It can be shown (see e.g. [Ormerod2010explaining](@citet)) that the optimal ``q`` densities are given by
```math
\begin{aligned}
  q(\boldsymbol{\theta}) &\propto \exp\big\{\mathbb{E}_{\boldsymbol{z}}\big[\log p(\boldsymbol{\theta}, \boldsymbol{z})\big]\big\},\\
  q(\boldsymbol{z}) &\propto \exp\big\{\mathbb{E}_{\boldsymbol{\theta}}\big[\log p(\boldsymbol{\theta}, \boldsymbol{z})\big]\big\},
\end{aligned}
```
where the expectations are taken with respect to ``q(\boldsymbol{\theta})`` and ``q(\boldsymbol{z})``, respectively. This result leads to the iterative coordinate-wise ascent variational inference algorithm (CAVI) for finding the optimal ``q`` densities, where we cyclically update ``q(\boldsymbol{\theta})`` and ``q(\boldsymbol{z})`` until some convergence criterion has been met. An oft-used convergence criterion for this purpose is the evidence lower bound, (ELBO):
```math
\mathrm{ELBO}(q) = \exp \Big\{\mathbb{E}_{\boldsymbol{\theta}, \boldsymbol{z}}\Big(\log \frac{p(\boldsymbol{x}, \boldsymbol{\theta}, \boldsymbol{z})}{q(\boldsymbol{\theta}, \boldsymbol{z})}\Big)\Big\}.
```

For our particular example it can be shown that the optimal ``q`` densities are:
- ``q(\boldsymbol{\theta})`` is the ``\mathrm{Dirichlet}(\boldsymbol{a} + \boldsymbol{r})``-density, where ``r_k = \sum_{i=1}^n q(z_i = k)``.
- ``q(\boldsymbol{z})`` is the pmf of a categorical distribution on ``\{1,2,\ldots, K\}`` with ``q(z_i = k) \propto \varphi_k(x_i)\, \exp\big\{\psi(a_k + r_k)\big\}``, where ``\psi(\cdot)`` denotes the [digamma function](https://en.wikipedia.org/wiki/Digamma_function).

An expression for the ELBO of this model is as follows:
```math
\begin{aligned}
  \mathrm{ELBO}(q) &= \sum_{i=1}^n \sum_{k=1}^K q(z_i = k) \big\{\log b_k(x_i) - \log q(z_i = k)\big\} \\ &+ \sum_{k=1}^K \big\{\log \Gamma(a + r_k) - \log \Gamma(a)\big\} \\ &- \log \Gamma(aK+n) + \log \Gamma(aK)
\end{aligned}
```

## Implementation
We start by importing the required packages:
```@example Bernstein; continued = true
using BayesDensityCore, Distributions, Random, StatsBase
```

### Model struct and pdf

The first step to implementing the Bernstein density model in a `BayesDensity`-compatible way is to define a model struct which is a subtype of [`AbstractBayesDensityModel`](@ref):

```@example Bernstein; continued = true
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
```
In the above implementation, we store the values of ``\varphi_k(x_i)`` for ``1 \leq i \leq n`` and ``1 \leq k \leq K``, as these values are reused repeatedly in the model fitting processes later.

In order to be able use all the functionality of `BayesDensityCore`, we also need to implement an equality method for our new type.
In this case, this is just a matter of checking that all the fields of two such objects are equal:
```@example Bernstein; continued = true
Base.:(==)(bd1::BernsteinDensity, bd2::BernsteinDensity) = bd1.data == bd2.data && bd1.K == bd2.K && bd1.a == bd2.a
```

Next, we implement a method that calculates the pdf of the model when the parameters of the model are given.
The [`pdf`](@ref) method should always receive the model object as the first argument, the parameters as the second argument and the point(s) at which the density should be evaluated as the third.
In the implementation presented below, we take in a `NamedTuple` with a single field named `θ` which represents the mixture probabilities.

```@example Bernstein; continued = true
function Distributions.pdf(bdm::BernsteinDensity{T, D}, params::NamedTuple, t::S) where {T<:Real, D, S<:Real}
    K = bdm.K
    (; θ) = params
    f = zero(promote_type(T, S))
    for k in 1:K
        f += θ[k] * pdf(Beta(k, K - k + 1), t)
    end
    return f
end
```

The `BayesDensityCore` module provides generic fallback methods for the cases where `params` is given as a Vector of NamedTuples and when `t` is a vector.
However, as noted in the general API, it is recommended that most models provide specialized methods for vectors of parameters and vectors of evaluation points,
as it is often possible to implement batch evaluation more efficiently, e.g. by leveraging BLAS calls instead of loops, when the parameters and the evaluation grid are provided in batches.

Next, we need to implement the cdf method. Owing to the nice structure of the cdf ``F`` in this example, this is no more complicated than implementing the pdf:
```@example Bernstein; continued = true
function Distributions.cdf(bdm::BernsteinDensity{T, D}, params::NamedTuple, t::S) where {T<:Real, D, S<:Real}
    K = bdm.K
    (; θ) = params
    f = zero(promote_type(T, S))
    for k in 1:K
        f += θ[k] * cdf(Beta(k, K - k + 1), t)
    end
    return f
end
```

In general, it is good practice to also implement the [`support`](@ref) and [`hyperparams`](@ref) methods for new models.
Note that for the Bernstein density model, the support is always equal to the unit interval, and the only hyperparameter is the scalar value `a` (here, we treat `K` as fixed).
Hence, the following provides appropriate implementations of the aforementioned methods:
```@example Bernstein; continued = true
BayesDensityCore.support(::BernsteinDensity{T, D}) where {T, D} = (T(0.0), T(1.0))
BayesDensityCore.hyperparams(bdm::BernsteinDensity) = (a = bdm.a,)
```

### Gibbs sampler

We now turn our attention to implementing the Gibbs sampler itself.
All `BayesDensity`-compatible Markov chain Monte Carlo samplers should overload the [`sample`](@ref) method.
This function should always take in a random seed as the first argument, the density model object as the second argument and the total number of samples (including burn-in) as the third argument.
In addition, the number of burn-in samples must be provided as a keyword argument.

The `sample` method should always return an object of type `PosteriorSamples` in order to be compatible with the rest of the package.
The samples generated during the MCMC routine should be stored in a subtype of `AbstractVector`, where the type of the elements are compatible with the function signature for the implemented `pdf` method.
Since our implementation of the `pdf` method takes in a NamedTuple as the `parameters` argument, we store the generated samples in a vector of NamedTuples in the implementation shown below:

```@example Bernstein; continued = true
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
```

The above implementation allows the user to supply the initial value of ``\theta`` used when performing the first iteration of the Gibbs sampler, via a `NamedTuple` to match the data structure we use to store the samples.

!!! note
    The convention adopted by the current set of `BayesDensity` models is that when during an MCMC run, only model pararameters should be stored, and not auxilliary variables which are only introduced in order to facilitate efficient computation.
    In this case, we therefore do not store the ``z_i`` in the model object returned by this method.

#### Example usage

Having implemented the model struct and the `pdf`- and `sample` methods, we can run the MCMC algorithm and perform posterior inference as with any of the other density esitmators implemented in this package:

```@example Bernstein
d_true = Kumaraswamy(2, 5) # Simulate some data from a density supported on [0, 1]
rng = Xoshiro(1) # for reproducibility
x = rand(rng, d_true, 1000)

K = 20
bdm = BernsteinDensity(x, K) # Create Bernstein density model object (a = 1)
ps = sample(rng, bdm, 1_000; n_burnin=500) # Run MCMC

median(ps, 0.5) # Compute the posterior median of f(0.5)
```

For instance, we can visualize the posterior fit by plotting the posterior means of ``f(t)`` and ``F(t)`` along with 95 % pointwise credible bands:
```@example Bernstein
using CairoMakie
t = LinRange(0, 1, 1001) # Grid for plotting

fig = Figure(size=(670, 320))
ax1 = Axis(fig[1,1], xlabel="x", ylabel="Density")
plot!(ax1, ps, t, label="MCMC") # Plot the posterior mean and credible bands:
lines!(ax1, t, pdf(d_true, t), label="Truth", color=:black) # Also plot truth for comparison

ax2 = Axis(fig[1,2], xlabel="x", ylabel="Cumulative distribution")
plot!(ax2, ps, cdf, label="MCMC")
lines!(ax2, t, cdf(d_true, t), label="Truth", color=:black)

Legend(fig[1,3], ax1, framevisible=false)

fig
```

### Mean-field variational inference
Before getting started on implementing a variational inference algorithm, we first need to define a new struct that represents the variational posterior distribution.
To make the resulting variational posterior compatible with the `BayesDensityCore` interface, the new struct should be a subtype of [`AbstractVIPosterior`](@ref).

Although not strictly required in order to make the variational posterior struct `BayesDensity`-compatible,
it is customary to have the variational posterior distribution store the variational densities as `Distributions`-objects,
in addition to storing the original model object.

The implementation below stores the variational density ``q(\boldsymbol{\theta})``, along with the Bernstein-density model object.
We also create a default constructor which takes in the vector ``\boldsymbol{r}`` resulting from the CAVI algorithm, along with the original model object:

```@example Bernstein; continued=true
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
```

It is also recommended to implement the [`model`](@ref) method, so that the user can easily extract the model to which the variational posterior was fitted:
```@example Bernstein; continued=true
BayesDensityCore.model(vip::BernsteinDensityVIPosterior) = vip.model
```

Next, we need to implement a method for generating samples from the variational posterior distribution, i.e. sampling from ``q(\boldsymbol{\theta})``.
This is achieved by implementing the [`sample`](@ref) method:
```@example Bernstein; continued=true
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
```

Having implemented a struct for storing the variational posterior, we can now turn our attention to the optimization procedure itself.
To start, we implement the ELBO, which we will need to determine convergence later.
Note that in the implementation below, we assume that the values of ``q(z_i = k)`` are stored in a ``n \times K`` matrix ``\omega``, so that ``\omega_{i,k} = q(z_i = k)``.
```@example Bernstein; continued=true
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
```

Finally, we are ready to implement the optimization procedure itself by overloading [`varinf`](@ref).
```@example Bernstein; continued=true
using SpecialFunctions # For the digamma-function

function BayesDensityCore.varinf(model::BernsteinDensity{T, D}; max_iter::Int=1000, rtol::Real=1e-4) where {T, D}
    (; data, K, a) = model
    (; x, n, φ_x) = data

    # Initialize the latent variables ω[i,k] = q(z_i = k) to 1/K:
    ω = fill(T(1/K), (n, K))
    r = fill(a + n/K, K)

    # CAVI optimization loop
    ELBO_prev = T(-1)
    ELBO = T(0)
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
        ELBO = Bernstein_ELBO(model, r, ω)

        # Run at least two iterations
        converged = (abs(ELBO_prev - ELBO) / ELBO_prev <= rtol) && iter > 1
        ELBO_prev = ELBO
        iter += 1
    end

    # Print a warning if the procedure fails to converge within the maximum number of iterations
    if !converged
        @warn "Maximum number of iterations reached."
    end

    return BernsteinDensityVIPosterior{T}(r, model)
end
```

#### Example usage

Having implemented the `varinf` method, we can now perform variational inference for the `BernsteinDensity` model just as easily as for any other `BayesDensityCore`-compatible model.
The example below shows how to fit a variational posterior to the Bernstein model for a simulated dataset:

```@example Bernstein
d_true = Kumaraswamy(2, 5) # Simulate some data from a density supported on [0, 1]
rng = Xoshiro(1) # for reproducibility
x = rand(rng, d_true, 1_000)

K = 20
bdm = BernsteinDensity(x, K) # Create Bernstein density model object (a = 1)
vip = varinf(bdm) # Compute the variational posterior.

mean(vip, 0.2) # Compute the posterior mean of f(0.2)
```

For instance, we can visualize the variational posterior fit by displaying the (variational) posterior means of ``f(t)`` and ``F(t)`` along with 95 % pointwise credible bands:
```@example Bernstein
using CairoMakie
t = LinRange(0, 1, 1001) # Grid for plotting

fig = Figure(size=(670, 320))
ax1 = Axis(fig[1,1], xlabel="x", ylabel="Density")
plot!(ax1, vip, t, label="VI") # Plot the posterior mean and credible bands:
lines!(ax1, t, pdf(d_true, t), label="Truth", color=:black) # Also plot truth for comparison

ax2 = Axis(fig[1,2], xlabel="x", ylabel="Cumulative distribution")
plot!(ax2, vip, cdf, label="VI")
lines!(ax2, t, cdf(d_true, t), label="Truth", color=:black)

Legend(fig[1,3], ax1, framevisible=false)

fig
```