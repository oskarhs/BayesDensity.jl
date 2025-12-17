using BayesDensity
using Plots, Random, Distributions, StatsBase, PGFPlotsX, BSplineKit

rng = Random.Xoshiro(1)
d_true = MixtureModel([Normal(-0.2, 0.25), Normal(0.5, 0.15)], [0.4, 0.6])
#d_true = MixtureModel(vcat(Normal(0, 1) ,[Normal(0.5*j, 0.1) for j in -2:2]), [0.5, 0.1, 0.1, 0.1, 0.1, 0.1])

x = rand(rng, d_true, 1000)
x = clamp.(x, -0.96, 0.96)


R = maximum(x) - minimum(x)

bsm = BSMModel(x, (-1.0, 1.0))
model_fit = sample(rng, bsm, 10000, n_burnin=1000)
t = LinRange(-0.975, 0.975, 3001)

qs = [0.005, 0.5, 0.995]
quants = quantile(model_fit, t, qs)
low, med, up = (quants[:,i] for i in eachindex(qs))

# NB! Estimates of rare quantiles are a bit noizy in this case, so we smooth them a bit to make the appearance of the logo a bit nicer!

λ = 2e-3
S_low = fit(BSplineOrder(4), t, low, λ)
low1 = S_low.(t)

S_up = fit(BSplineOrder(4), t, up, λ)
up1 = S_up.(t)

# Increase distance between red, green and purple curves:

low2 = med + 1.7*(low1 - med) .- 0.05
up2 = med + 1.7*(up1 - med) .+ 0.05

M = maximum(up)

juliared   = "{rgb,1:red,0.796; green,0.235; blue,0.2}"
juliagreen = "{rgb,1:red,0.22; green,0.596; blue,0.149}"
juliapurple= "{rgb,1:red,0.584; green,0.345; blue,0.698}"

axis = @pgf Axis(
               {
        axis_lines="none"
        },
        Plot({line_width = "4.5pt", color = juliared}, Table(x = t, y = med)),
        Plot({line_width = "4.5pt", color = juliagreen}, Table(x = t, y = up2)),
        Plot({line_width = "4.5pt", color = juliapurple}, Table(x = t, y = low2)),
       )

PGFPlotsX.pgfsave(joinpath(@__DIR__, "logo.svg"), axis)