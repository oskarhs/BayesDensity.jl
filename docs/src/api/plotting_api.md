# Plotting

Graphical displays are a powerful tool for providing informative vizual summaries of the result of a given Bayesian inference procedure for a univariate density. `BayesDensity` makes it easy to plot posterior summaries for ``f`` using the results from Markov chain Monte Carlo sampling or variational inference through its extensions for the [Makie.jl](https://github.com/MakieOrg/Makie.jl) and [Plots.jl](https://github.com/JuliaPlots/Plots.jl) packages.

In addition to documenting the plotting-related public API, this page also showcases the plotting capabilities of the `BayesDensityCore` package through examples. Although we will not delve deep into implementational details here, some familiarity with `Makie.jl` or `Plots.jl` is an advantage when reading this part of the documentation.

The following sections are structured so that the `Makie`- and the `Plots`-portions of the tutorial can be read independently of one another. As such, there is no need for a `Makie` power-user to read the `Plots` sections of this page.

## Plotting with Makie.jl
To show the plotting-capabilities of the Makie extension, we start by importing the required packages and fit a `BayesDensity` model to some simulated data:
```@example Makie; continued = true
using BayesDensityHistSmoother, CairoMakie, Distributions, Random
rng = Random.Xoshiro(1)

# Simulate some data from the "Claw" density
d_true = MixtureModel(vcat(Normal(0, 1) ,[Normal(0.5*j, 0.1) for j in -2:2]), [0.5, 0.1, 0.1, 0.1, 0.1, 0.1])
x = rand(rng, d_true, 5000)

# Fit the model via MCMC and VI
histsmoother = HistSmoother(x)
posterior_sample = sample(rng, histsmoother, 1100)
vi_posterior = varinf(histsmoother)
```

In general, the available plot method for [`PosteriorSamples`](@ref) and [`AbstractVIPosterior`](@ref) objects has the following signature:
```julia
plot(
    ps::Union{PosteriorSamples, AbstractVIPosterior},
    [func = ::typeof(pdf)],
    [t::AbstractVector{<:Real}];
    ci::Bool = true,
    level::Real = 0.95,
    estimate::Symbol = :mean,
    kwargs...
)
```
The first argument to `plot` is the posterior distribution, fitted either via Markov chain Monte Carlo or variational inference.
The second (optional) positional argument indicates whether to plot estimates of the `pdf` or the `cdf`. By default, the estimated pdf is shown.
The third (optional) positional argument is the grid at which the `pdf` or `cdf` is evaluated to draw the grid.
The `ci` keyword is a boolean, controlling whether or not a credible interval should be drawn (enabled by default).
To control the level of the drawn credible interval, set the `level` keyword argument to the desired confidence level.

The example shown below illustrates how 

```@example Makie
t = LinRange(-3.5, 3.5, 4001)

# Create figure, axes
fig = Figure(size=(670, 670))
ax1 = Axis(fig[1,1], xlabel="x", ylabel="Density")
ax2 = Axis(fig[1,2], xlabel="x", ylabel="Density")
ax3 = Axis(fig[2,1], xlabel="x", ylabel="Cumulative density")
ax4 = Axis(fig[2,2], xlabel="x", ylabel="Cumulative density")

plot!(ax1, posterior_sample, color=:red, strokecolor=:red)

plot!(ax2, vi_posterior, pdf, t; ci=false,
      estimate=:median, label="Estimate") # NB! Supplying pdf is redundant
lines!(ax2, t, pdf(d_true, t), color=:black, label="True pdf",
       linestyle=:dash)
axislegend(ax2; position=:lt, framevisible=false)

plot!(ax3, posterior_sample, cdf, level=0.99, color=:red, strokecolor=:red)

# Compare estimated cdf to the true cdf
plot!(ax4, posterior_sample, cdf, ci=false, label="Estimate")
lines!(ax4, t, cdf(d_true, t),
      color = :black, label="True cdf", linestyle=:dash)
axislegend(ax4; position=:lt, framevisible=false)
xlims!(ax4, -2.2, 2.2)

fig
```

!!! note
    When estimating extreme quantiles, the variance of Monte Carlo estimates tends to increase.
    As a result, when the value of `level` is very close to `1`, the resulting credible intervals often exhibit considerable jitter.
    An effective remedy to this issue is to simply generate more samples from the posterior distribution.

## Plotting with Plots.jl