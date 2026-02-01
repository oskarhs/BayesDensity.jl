# A primer on Bayesian nonparametric density estimation

The following page provides a short introduction to nonparametric density estimation, with a particular focus on Bayesian approaches.
The topic of nonparametric density estimation is vast, and we do as such only aim to provide a rather broad overview here.
For an in-depth and quite accessible introduction to frequentist approaches in nonparametric density estimation, we recommend [Scott1992multivariate](@citet).
A good introduction to the Bayesian perspective, with a focus on practical computation, can be found in [Gelman2013bayesian](@citet).

% TODO: Explain basics behind smoothing parameters. Also a very brief intro to MCMC. ALso show how MCMC output elvolves with increasing iterations through an animation. Then smack bang VI

## Introduction

The goal of density estimation is to estimate an unknown density based on observed data. Given an independent and identically distributed sample ``x_1, x_2, \ldots, x_n`` from a density ``f``, the goal is to construct an estimate ``\hat{f}`` that is close to the true density. Parametric approaches to density estimation assume that the density belongs to a parametric family of distribution, such as the normal family of distributions, indexed by a parameter of low to moderate dimension. This approach is generally highly efficient if the true density belongs to or is well-approximated by a member of the postulated parametric family. However, in cases where the parametric family at hand is misspecified, the quality of the estimator ``\hat{f}`` is often of low quality.

In constrast, nonparametric approaches to density estimation strive to make less restrictive assumptions on the true density ``f``. This aim is acheived by employing richer classes of models which are able to approximate the true density under minimal smoothness assumptions. Unlike their parametric counterparts, most nonparametric density estimators are built on high- or even infinite-dimensional parameter spaces.

Although the increased flexibility of nonparametric approaches is an attractive feature, it can be quite challenging to construct well-working nonparametric density estimators owing to the high-dimensional parameter spaces. In particular, the high dimensionality of the parameter space introduces the need for some form of regularization, typically through the introduction of one or several smoothing parameters, so that the resulting estimates do not end up being to bumpy. As an illustration, consider the well-known (frequentist) kernel density estimator:
```math
\hat{f}(x)\frac{1}{nh}\sum_{i=1}^n \phi\Big(\frac{x - x_i}{h}\Big),
```
where ``\phi(\cdot)`` is the probability density function of the standard normal distribution and the bandwidth ``h > 0`` is a parameter which controls the smoothness of the estimate.
To illustrate the role of the smoothing parameter, we generated a sample of size ``n = 1000`` from the mixture density ``f(x) = 0.4 \phi\big((x+0.2)/0.25\big) / 0.25 + 0.6 \phi\big((x-0.5)/0.15\big) / 0.15``, and fitted a kernel density estimator for three different choices of the bandwidth parameter. The resulting density estimates are shown in the figure below:

![kernel density bandwidth](../assets/introduction_to_density_estimation/kernel.svg)

For the smallest choice of the bandwidth parameter, the estimate ``\hat{f}`` ends up being very wiggly, severly distorting the shape of ``f`` with many small bumps. On the other hand, the largest bandwidth results in far too much smoothing, and the resulting density estimate fails to resemble ``f``. A good choice of the bandwidth parameter needs to strike a balance between these two extremes.

In practical applications, the true density is of course not known, and one has to select the smoothing parameter in a data-driven manner. Frequentist approaches to selecting the smoothing parameter are often based on optimizing some criterion that balances representational capacity against model complexity. Examples include the optimization of a cross-validation criterion or the minimization of an asymptotic expansion of a risk function.

## The Bayesian approach
Bayesian statistics offers a fundamentally different approach to nonparametric density estimation. Here, the density ``f`` is itself treated as a random quantity, and is assigned a prior distribution ``p(f)``. Density estimates in Bayesian models are based on the posterior distribution,
```math
p(f\,|\, x_1, \ldots, x_n) \propto p(f)\prod_{i=1}^n f(x_i).
```

Unlike frequentist approaches to smoothing parameter selection, where they are typically chosen via an optimization, the Bayesian approach implicitly introduces smoothing through the prior distribution. The key challenge to constructing successful Bayesian nonparametric density estimators is to find a model with a very large representational capacity, combined with a prior distribution that results in smooth density estimates.

### Computation
A key challenge in the Bayesian approach to density estimation is that for most genuinely nonparametric models, the posterior distribution ``p(f\,|\, x_1, \ldots, x_n)`` is often not tractable analytically. Instead, inference is often based on Monte Carlo approximations.

However, the high- to infinite dimensional nature of nonparametric procedures further complicate this issue, as the techniques employed for approximate inference in lower-dimensional Bayesian models often do not scale well to higher dimensions.