using BayesianDensityEstimation
using Plots, Random, Distributions, StatsBase, PGFPlotsX, LinearAlgebra

push!(PGFPlotsX.CUSTOM_PREAMBLE, raw"\usepgfplotslibrary{fillbetween}")
push!(PGFPlotsX.CUSTOM_PREAMBLE,
       raw"""
       \usepackage{xcolor}
        \definecolor{juliapurple}{rgb}{0.584, 0.345, 0.698}
        \definecolor{juliared}{rgb}{0.796, 0.235,0.2}
       \definecolor{juliagreen}{rgb}{0.22, 0.596, 0.149}
       """)
push!(PGFPlotsX.CUSTOM_PREAMBLE, raw"\usetikzlibrary{arrows.meta}")

rng = Random.Xoshiro(1)
d_true = MixtureModel([Normal(-0.2, 0.25), Normal(0.5, 0.15)], [0.4, 0.6])

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

H = fit(Histogram, x, LinRange(minimum(x), maximum(x), K+1))
H = normalize(H, mode=:pdf)

M = maximum(up)

juliared   = "{rgb,1:red,0.796; green,0.235; blue,0.2}"
juliagreen = "{rgb,1:red,0.22; green,0.596; blue,0.149}"
juliapurple= "{rgb,1:red,0.584; green,0.345; blue,0.698}"


#= axis = @pgf Axis(
    {
        axis_lines="none",
        xmin = minimum(t) - 0.05,
        xmax = maximum(t) + 0.05,
        ymin = -0.04*M,
        ymax = 1.04*M
    },
    Plot({ybar_interval, color=juliapurple, opacity=0.5, fill=juliapurple, fill_opacity=0.25}, Coordinates(H.edges[1], vcat(H.weights, 0))),
    Plot({line_width = "1.8pt", color = juliared}, Table(x = t, y = med)),

    Plot({ "name path=upper", no_marks, draw="none"}, Coordinates(t, up)),
    Plot({ "name path=lower", no_marks, draw="none"}, Coordinates(t, low)),
    Plot({ draw = "none", fill = juliagreen, fill_opacity = 0.5 },
    raw"fill between [of=lower and upper]")
)

PGFPlotsX.pgfsave(joinpath(@__DIR__, "logo.svg"), axis)


axis = @pgf Axis(
           {
               axis_lines="none",
               xmin = minimum(x) - 0.12*R,
               xmax = maximum(x) + 0.12*R,
               ymin = -0.04*M,
               ymax = 1.04*M
           },
           Plot({line_width = "1.8pt", color = juliared}, Table(x = t, y = med)),

           Plot({ "name path=upper", no_marks, draw="none"}, Coordinates(t, up)),
           Plot({ "name path=lower", no_marks, draw="none"}, Coordinates(t, low)),
           Plot({ draw = "none", fill = raw"juliagreen!60"},
           raw"fill between [of=lower and upper]"),

           raw"\draw[-{Stealth}, line width=0.95mm, color=juliapurple!90]
         (axis cs:-1.1, 0) -- (axis cs:1.2, 0);",
            raw"\draw[-{Stealth}, line width=0.95mm, color=juliapurple!90]
         (axis cs:-1.0, -0.1) -- (axis cs:-1.0, 1.9);"
       )

PGFPlotsX.pgfsave(joinpath(@__DIR__, "logo_nohist.svg"), axis) =#

# Shaded CI's look like shit
axis = @pgf Axis(
               {
        axis_lines="none",
        xmin = minimum(t) - 0.05,
        xmax = maximum(t) + 0.05,
        ymin = -0.05*M,
        ymax = 1.05*M
        },
        Plot({line_width = "2.2pt", color = juliared}, Table(x = t, y = med)),
        Plot({line_width = "2.2pt", color = juliagreen}, Table(x = t, y = up)),
        Plot({line_width = "2.2pt", color = juliapurple}, Table(x = t, y = low)),
       )

PGFPlotsX.pgfsave(joinpath(@__DIR__, "logo.svg"), axis)
