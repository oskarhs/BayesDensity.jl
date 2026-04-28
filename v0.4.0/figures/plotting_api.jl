
using BayesDensityHistSmoother, CairoMakie, Distributions, Random
rng = Random.Xoshiro(1)

# Simulate some data from the "Claw" density
d_true = MixtureModel(
    vcat(Normal(0, 1), [Normal(0.5*j, 0.1) for j in -2:2]),
    [0.5, 0.1, 0.1, 0.1, 0.1, 0.1]
)
x = rand(rng, d_true, 1000)

# Fit the model via MCMC and VI
histsmoother = HistSmoother(x)
posterior_sample = sample(rng, histsmoother, 1100)
vi_posterior, info = varinf(histsmoother)

t = LinRange(-3.5, 3.5, 4001)

# Create figure, axes
fig = Figure(size=(550, 550))
ax1 = Axis(fig[1,1], xlabel="x", ylabel="Density")
ax2 = Axis(fig[1,2], xlabel="x", ylabel="Density")
ax3 = Axis(fig[2,1], xlabel="x", ylabel="Cumulative density")
ax4 = Axis(fig[2,2], xlabel="x", ylabel="Cumulative density")

# Plot estimated density and CI from MCMC samples
plot!(ax1, posterior_sample, color=:red, strokecolor=:red, label="Estimate (CI)", alpha=0.1)
ylims!(ax1, 0.0, 0.85)

# Compare the posterior median of the VI fit to the true density (without CI)
plot!(ax2, vi_posterior, pdf, t; ci=false,
      estimate=median, label="Estimate") # NB! Supplying pdf is redundant here
lines!(ax2, t, pdf(d_true, t), color=:black, label="True pdf",
       linestyle=:dash)
xlims!(ax2, -2.5, 2.5)
ylims!(ax2, 0.0, 0.85)

# Plot the estimated cdf and the CI
plot!(ax3, posterior_sample, cdf, level=0.99, color=:red, strokecolor=:red, label="Estimate (CI)")

# Compare estimated cdf of the VI fit to the true cdf (without CI)
plot!(ax4, vi_posterior, cdf, ci=false, label="Estimate")
lines!(ax4, t, cdf(d_true, t),
      color = :black, label="True cdf", linestyle=:dash)
xlims!(ax4, -2.2, 2.2)

for ax in (ax1, ax2, ax3, ax4)
    axislegend(ax; position=:lt, framevisible=false, labelsize=10)
end

save(joinpath("src", "assets", "plotting_api", "makie.svg"), fig)

### Example

import Plots

# Create subplots
p1 = Plots.plot(xlabel="x", ylabel="Density")
p2 = Plots.plot(xlabel="x", ylabel="Density")
p3 = Plots.plot(xlabel="x", ylabel="Cumulative density")
p4 = Plots.plot(xlabel="x", ylabel="Cumulative density")

# Plot estimated density and CI from MCMC samples
Plots.plot!(p1, posterior_sample, color=:red, fillcolor=:red,
      label="Estimate (CI)", fillalpha=0.1)
Plots.ylims!(p1, 0.0, 0.7)

# Compare the posterior median of the VI fit to the true density (without CI)
Plots.plot!(p2, vi_posterior, pdf, t; ci=false,
      estimate=median, label="Estimate") # NB! Supplying pdf is redundant here
Plots.plot!(p2, t, pdf(d_true, t), color=:black, label="True pdf",
       linestyle=:dash)
Plots.xlims!(p2, -2.5, 2.5)
Plots.ylims!(p2, 0.0, 0.7)

# Plot the estimated cdf and the CI
Plots.plot!(p3, posterior_sample, cdf, level=0.99, color=:red, fillcolor=:red, label="Estimate (CI)")

# Compare estimated cdf of the VI fit to the true cdf (without CI)
Plots.plot!(p4, vi_posterior, cdf, ci=false, label="Estimate")
Plots.plot!(p4, t, cdf(d_true, t),
      color = :black, label="True cdf", linestyle=:dash)
Plots.xlims!(p4, -2.2, 2.2)

p = Plots.plot(p1, p2, p3, p4, layout=(2,2), size=(550, 550))

Plots.savefig(p, joinpath("src", "assets", "plotting_api", "plots.svg"))