# Implementing new Bayesian density estimators

The following page provides a tutorial on how to implement new Bayesian density estimators compatible with the BayesDensity.jl-package.
Prior to reading this tutorial, one should have already familiarized oneself with the general package API, for instance by reading the [`general API documentation`].

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

Our main motivation for considering the Bernstein model is that under mild regularity conditions on the true density ``f_0``, the Bernstein density ``f`` can approximate ``f_0`` to an arbitrary degree of precision with respect to a suitable metric, such as the total variation distance, provided ``K`` is sufficiently large.

For a Bayesian treatment of the Bernstein density model, we impose a ``\mathrm{Dirichlet}(\boldsymbol{a})``-prior distribution on ``\boldsymbol{\theta}``, where ``\boldsymbol{a} = (a,a, \ldots, a)`` for some ``a>0``. Given an observed independent and identically distributed sample ``\boldsymbol{x} = (x_1, x_2, \ldots, x_n)``, the likelihood of the observed sample under the Bernstein model for ``f`` is
```math
p(\boldsymbol{x}\,|\, \boldsymbol{\theta}) = \prod_{i=1}^n \sum_{k=1}^K \theta_k\, \varphi_k(x_i).
```
The form taken by the likelihood function above makes Bayesian inference challenging due to the fact that the resulting posterior distribution is analytically intractable. However, by augmenting the data with latent variables ``\boldsymbol{z} \in \{1,2,\ldots, K\}^n``, it is possible to perform posterior inference very efficiently through Gibbs sampling or mean-field VI. Another possible way of defining the Bernstein density model is to let ``x_i \, |\, \boldsymbol{\theta}, z_i = k \sim \varphi_k`` and ``p(z_i = k) = \theta_k`` for all ``i``, as this leads to the same likelihood function as previously when the ``z_i`` are marginalized out.`

Under this data augmentation strategy it can then be shown that joint posterior of ``\boldsymbol{\theta}, \boldsymbol{z}`` is
```math
p(\boldsymbol{\theta}, \boldsymbol{z}\, |\, \boldsymbol{x}) \propto \prod_{k = 1}^K \theta_k^{N_k + a - 1} \prod_{i=1}^n \prod_{k=1}^K \varphi_k(x_i)^{\mathbf{1}_{\{k\}}(z_i)},
```
where ``N_k = \sum_{i=1}^n \mathbf{1}_{\{k\}}(z_i)``.

### Gibbs sampling
To write down a Gibbs sampler for this model, we need to derive the full conditional distributions of ``\boldsymbol{\theta}`` and ``\boldsymbol{z}``. In this case, direct inspection of the joint posterior shows that
```math
p(\boldsymbol{\theta}\, |\, \boldsymbol{z}, \boldsymbol{x}) = \mathrm{Dirichlet}(\boldsymbol{a} + \boldsymbol{N}),
```
where ``\boldsymbol{N} = (N_1, N_2, \ldots, N_K)``. The full conditional distributions for ``\boldsymbol{z}`` are 

```math
p(\boldsymbol{z}\, |\, \boldsymbol{\theta}, \boldsymbol{x}) \propto \prod_{i=1}^n \prod_{k=1}^K \big\{\theta_k\,\varphi_k(x_i)\big\}^{\mathbf{1}_{\{k\}}(z_i)}.
```
Hence, we see that ``z_1, \ldots, z_n`` are independent given ``\boldsymbol{\theta}, \boldsymbol{x}``, with ``p(z_i \,|\, \boldsymbol{\theta}, \boldsymbol{x}) \propto \theta_k\, \varphi_k(x_i)``.

## Implementation
We start by importing the required packages:
```@example Bernstein; continued = true
using BayesDensityCore, Random, Distributions, StatsBase
```

### Model struct and pdf

The first step to implementing the Bernstein density model in a `BayesDensity`-compatible way is to define a model struct which is a subtype of [`AbstractBayesDensityModel`](@ref): 

```@example Bernstein; continued = true
struct BDModel{T<:Real, D} <: AbstractBayesDensityModel{T}
    data::D # NamedTuple holding data
    K::Int  # Basis dimension
    a::T    # Dirichlet prior concentration parameter.
    function BDModel{T}(x::AbstractVector{<:Real}, K::Int; a::Real=1.0) where {T<:Real}
        data = (x = x, n = length(x))
        return new{T, typeof(data)}(data, K, T(a))
    end
end
BDModel(args...; kwargs...) = BDModel{Float64}(args...; kwargs...) # For convenience
```

Next, we implement a method that calculates the pdf of the model when the parameters of the model are given.
The [`pdf`](@ref) method should always receive the model object as the first argument, the parameters as the second argument and the point(s) at which the density should be evaluated as the third.
In the implementation presented below, we take in a NamedTuple with a single field named `θ` which represents the mixture probabilities.

```@example Bernstein; continued = true
function Distributions.pdf(bdm::BDModel{T, D}, params::NamedTuple, t::S) where {T<:Real, D, S<:Real}
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
However, as noted in the general API, it is recommended that most models provide specialized methods for vectors of parameters and vectors of evaluation points, as it is often possible to implement batch evaluation more efficiently (e.g. by leveraging BLAS calls instead of loops) when the parameters and the evaluation grid are provided in batches.

In general, it is good practice to also implement the [`support`](@ref) and [`hyperparams`](@ref) methods for new models.
Note that for the Bernstein density model, the support is always equal to the unit interval, and the only hyperparameter is the scalar value `a` (here, we treat `K` as fixed).
Hence, the following provides appropriate implementations of the aforementioned methods:
```@example Bernstein; continued = true
BayesDensityCore.support(::BDModel{T, D}) where {T, D} = (T(0.0), T(1.0))
BayesDensityCore.hyperparams(bdm::BDModel) = (a = bdm.a,)
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
function StatsBase.sample(rng::AbstractRNG, bdm::BDModel{T, D}, n_samples::Int; n_burnin=min(div(length(x), 5), 1000)) where {T, D}
    (; K, data, a) = bdm
    (; x, n) = data

    a_vec = fill(a, K) # Dirichlet prior parameter

    θ = fill(T(1/K), K) # Initialize θ as the uniform vector
    probs = Vector{T}(undef, K) # Vector used to store intermediate calculations of p(zᵢ|θ, x)

    # Store samples as a vector of NamedTuples
    samples = Vector{NamedTuple{(:θ,), Tuple{Vector{Float64}}}}(undef, n_samples)

    for m in 1:n_samples
        N = zeros(Int, K) # N[k] = number of z[i] equal to k.
        for i in 1:n
            for k in 1:K
                probs[k] = θ[k] * pdf(Beta(k, K - k + 1), x[i])
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

!!! note
    The convention adopted by the current set of `BayesDensity` models is that when during an MCMC run, only model pararameters should be stored, and not auxilliary variables which are only introduced in order to facilitate efficient computation. In this case, we therefore do not store the ``z_i`` in the model object returned by this method.

Having implemented the model struct and the `pdf`- and `sample` methods, we can run the MCMC algorithm and perform posterior inference as with any of the other density esitmators implemented in this package:

```@example Bernstein
d_true = Kumaraswamy(2, 5) # Simulate some data from a density supported on [0, 1]
rng = Xoshiro(1) # for reproducibility
x = rand(rng, d_true, 3_000)

K = 25
bdm = BDModel(x, K) # Create Bernstein density model object (a = 1)
ps = sample(rng, bdm, 3_000; n_burnin=500) # Run MCMC

median(ps, 0.5) # Compute the posterior median of f(0.5)
```

For instance, we can visualize how close the estimated by plotting the posterior mean of ``f(t)`` along with a 95 % credible band:
```@example Bernstein
using CairoMakie
t = LinRange(0, 1, 1001) # Grid for plotting

fig = Figure()
ax = Axis(fig[1,1], xlabel="x", ylabel="Density")
plot!(ax, ps, t, label="Bernstein estimate") # Plot the posterior mean and credible bands:
lines!(ax, t, pdf(d_true, t), label="True density", color=:black)
axislegend(ax, framevisible=false)
fig
```

### TODOs

- TODO: Write down the  VI updates. Should probably also find the ELBO in the VI case.
- TODO: Add plots where we show the fallback cdf method, once this has been implemented.
- TODO: Write the actual tutorial. First as a separate script. Then just copy the contents here.