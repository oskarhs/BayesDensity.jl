#include(joinpath(@__DIR__ ,"BayesianDensityEstimationBSM.jl"))

#using BayesianDensityEstimationBSM
using Plots

using Random, Distributions

harp_means = [0.0, 5.0, 15.0, 30.0, 60.0]
harp_sds = [0.5, 1.0, 2.0, 4.0, 8.0]


rng = Random.default_rng()
#d_true = Laplace()
#d_true = LogNormal()
#d_true = Normal()
#d_true = SymTriangularDist()
d_true = MixtureModel([Normal(-0.2, 0.25), Normal(0.5, 0.15)], [0.4, 0.6])
#d_true = MixtureModel([Normal(0, 1), Normal(0, 0.1)], [2/3, 1/3])
#d_true = MixtureModel(vcat(Normal(0, 1) ,[Normal(0.5*j, 0.1) for j in -2:2]), [0.5, 0.1, 0.1, 0.1, 0.1, 0.1])
# d_true = MixtureModel([Normal(harp_means[i], harp_sds[i]) for i in eachindex(harp_means)], fill(0.2, 5))
#d_true = Beta(1.2, 1.2)

x = rand(rng, d_true, 1000)

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

t = LinRange(-1, 1, 3001)
plot(t, pdf(d_true, t))
bar!(LinRange(minimum(x), maximum(x), K+1), K * N / (R*sum(N)), alpha=0.2)


#bsm = BayesBSpline.BSMModel(x, BSplineBasis(BSplineOrder(4), LinRange(minimum(x), maximum(x), 98)))
bsm = BSMModel(x)

#bsm2 = BayesBSpline.BSMModel(x, (0, 1))
#bsm = BayesBSpline.BSMModel(x, 200, (0,1))
#bsm = BayesBSpline.BSMModel(x; n_bins=nothing)

R = maximum(x) - minimum(x)
#x = LinRange(0, 1, 1000)

# Run the Gibbs sampler
n_samples = 5000
n_burnin = 1000
@time bsmc = BayesianDensityEstimationBSM.sample_posterior(rng, bsm, n_samples, n_burnin)

# Plotting
bs = BSplineKit.basis(bsmc.model)
K = length(bs)

#t = LinRange(0, 1, 10001)
#t_orig = minimum(x) .+ R*t
t = LinRange(boundaries(basis(bsm))[1], boundaries(basis(bsm))[2], 2001)
#kdest = kde(x; bandwidth=PosteriorStats.isj_bandwidth(x))
#kdest = PosteriorStats.kde_reflected(x, bounds=(0,1))

qs = [0.025, 0.5, 0.975]
quants = quantile(bsmc, t, qs)
low, med, up = (quants[:,i] for i in eachindex(qs))


#= p = Plots.plot()
Plots.plot!(p, t, mean.(bsmc, t), color=:black, lw=1.2, label="Posterior mean")
Plots.plot!(p, t, med, color=:blue, lw=1.2, label="Posterior median")
#Plots.plot!(p, t, low, color=:green, ls=:dash, label="95% CI", alpha=0.5)
#Plots.plot!(p, t, up, color=:green, label="", ls=:dash, alpha=0.5)
#Plots.plot!(p, t, up, fillrange=low, fillcolor=:green, fillalpha=0.2, label="", color=:transparent)

#Plots.plot!(kdest.x, kdest.density, color=:grey, label="KDE", lw=1.2)
Plots.plot!(p, t, pdf(d_true, t), color=:red, label="True", lw=1.2, alpha=0.5)
#xlims!(p, -2.5, 2.5)
p


quantile(bsmc, 0.0, [0.2, 0.8])
median(bsmc, 0.0) =#