# Implementing new Bayesian density estimators

The following page provides a tutorial on how one can implement new Bayesian density estimators compatible with the BayesDensity.jl-package. In order to be able to follow this tutorial, it is advantageous to have some prior exposure to data-augmentation schemes, Gibbs sampling and mean-field variational inference. A good introduction to all three topics can be found in [Bishop2006pattern](@citet).

## Bayesian inference for Bernstein densities.

For our tutorial, we will illustrate by focusing on a Bayesian Bernstein-type density estimator for data supported on the unit interval.[^1] This section provides the theoretical background for the model we will later implement as an example, and can be skipped by readers who are more interested in the details of the implementation itself.

[^1]:
    A Bayesian Bernstein-type estimator, where the number ``K`` of mixture components is treated as a further random variable has been proposed by [Petrone1999bernstein](@cite).

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
To write down a Gibbs sampler for this model, we need to derive the full conditional distributions of ``\boldsymbol{\theta}`` and ``\boldsymbol{z}``. In this case, direct inspection of the joint posterior yields
```math
p(\boldsymbol{\theta}\, |\, \boldsymbol{z}, \boldsymbol{x}) = \mathrm{Dirichlet}(\boldsymbol{a} + \boldsymbol{N}),
```
where ``\boldsymbol{N} = (N_1, N_2, \ldots, N_K)``. The full conditional distributions for ``\boldsymbol{z}`` are 

```math
p(\boldsymbol{z}\, |\, \boldsymbol{\theta}, \boldsymbol{x}) \propto \prod_{i=1}^n \prod_{k=1}^K \big\{\theta_k\,\varphi_k(x_i)\big\}^{\mathbf{1}_{\{k\}}(z_i)} 
```

TODO: Write down the full conditionals and the VI updates. Should probably also find the ELBO in the VI case.

TODO: Write the actual tutorial. First as a separate script. Then just copy the contents here.