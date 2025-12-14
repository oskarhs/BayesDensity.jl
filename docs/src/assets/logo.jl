using BayesianDensityEstimation
using Plots, Random, Distributions, StatsBase, PGFPlotsX, BSplineKit

rng = Random.Xoshiro(1)
d_true = MixtureModel([Normal(-0.2, 0.25), Normal(0.5, 0.15)], [0.4, 0.6])
#d_true = MixtureModel(vcat(Normal(0, 1) ,[Normal(0.5*j, 0.1) for j in -2:2]), [0.5, 0.1, 0.1, 0.1, 0.1, 0.1])

x = rand(rng, d_true, 1000)
x = clamp.(x, -0.96, 0.96)

function bin_regular_nochmal(x::AbstractVector{T}, xmin::T, xmax::T, M::Int, right::Bool) where {T<:Real}
    R = xmax - xmin
    bincounts = zeros(Int, M)
    edges_inc = M/R
    if right
        for val in x
            idval = min(M-1, floor(Int, (val-xmin)*edges_inc+eps())) + 1
            bincounts[idval] += 1.0
        end
    else
        for val in x
            idval = max(0, floor(Int, (val-xmin)*edges_inc-eps())) + 1
            bincounts[idval] += 1.0
        end
    end
    return bincounts
end

K = 100
N = bin_regular_nochmal(x, minimum(x), maximum(x), K, true)

R = maximum(x) - minimum(x)

bsm = BSMModel(x, (-1.0, 1.0))
model_fit = sample(bsm, 5000, n_burnin=1000)
t = LinRange(-0.975, 0.975, 3001)

qs = [0.005, 0.5, 0.995]
quants = quantile(model_fit, t, qs)
low, med, up = (quants[:,i] for i in eachindex(qs))

# NB! Estimates of rare quantiles are a bit noizy in this case, so we smooth them a bit to make the appearance of the logo a bit nicer!

λ = 1e-3
S_low = fit(BSplineOrder(4), t, low, λ)
low = S_low.(t)

S_up = fit(BSplineOrder(4), t, up, λ)
up = S_up.(t)

M = maximum(up)

juliared   = "{rgb,1:red,0.796; green,0.235; blue,0.2}"
juliagreen = "{rgb,1:red,0.22; green,0.596; blue,0.149}"
juliapurple= "{rgb,1:red,0.584; green,0.345; blue,0.698}"

axis = @pgf Axis(
               {
        axis_lines="none",
        xmin = minimum(t) - 0.05,
        xmax = maximum(t) + 0.05,
        ymin = -0.05*M,
        ymax = 1.05*M
        },
        Plot({line_width = "3.0pt", color = juliared}, Table(x = t, y = med)),
        Plot({line_width = "3.0pt", color = juliagreen}, Table(x = t, y = up)),
        Plot({line_width = "3.0pt", color = juliapurple}, Table(x = t, y = low)),
       )

PGFPlotsX.pgfsave(joinpath(@__DIR__, "logo.svg"), axis)