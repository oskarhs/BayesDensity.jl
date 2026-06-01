using BayesDensityBSplineMixture
using CairoMakie
using CodecXz
using Downloads
using Random
using RData

# Set seed for reproducibility
rng = Xoshiro(1984)

# Read rdata file
url  = "https://raw.githubusercontent.com/cran/densEstBayes/master/data/incomeUK.rda"
path = Downloads.download(url)
data = load(path)["incomeUK"]

# Create a model object
bsm = BSplineMixture(data)

# Sample from the posterior
ps1 = sample(rng, bsm, 1500; n_burnin=500)

# Diagnose the single chain
fig1 = BayesDensityCore.Makie.check_chains(ps1; grid=quantile(data, (1:5)/6), include_burnin=true)
save(joinpath("src", "assets", "diagnostics_example", "diagnostics_example1.svg"), fig1)

# Compute the effective sample size and R-hat diagnostics for the single chain
using MCMCDiagnosticTools # Needed to load the package extension

ess_single = MCMCDiagnosticTools.ess(ps1; grid = quantile(data, (1:5)/6))
rhat_single = MCMCDiagnosticTools.rhat(ps1; grid = quantile(data, (1:5)/6))
println("Effective sample size for single chain: ", round.(ess_single; sigdigits=4))
println("R-hat for single chain: ", round.(rhat_single; sigdigits=4))

# Fit four chains
using Distributions # For initialization of the chains
ps_chains = Vector{Any}(undef, 4)
for i in eachindex(ps_chains)
    init_params = (β = rand(rng, Normal(0, 1), length(bsm)-1), τ2 = rand(rng, InverseGamma(1, 2e-4)))
    ps_chains[i] = sample(rng, bsm, 4500; n_burnin=500, initial_params=init_params)
end

fig2 = BayesDensityCore.Makie.check_chains(ps_chains...; grid=quantile(data, (1:5)/6), include_burnin=true)
save(joinpath("src", "assets", "diagnostics_example", "diagnostics_example2.svg"), fig2)

# Compute the effective sample size and R-hat diagnostics for the four chains
ess_chains = MCMCDiagnosticTools.ess(ps_chains...; grid = quantile(data, (1:5)/6))
rhat_chains = MCMCDiagnosticTools.rhat(ps_chains...; grid = quantile(data, (1:5)/6))
println("Effective sample size for four chains: ", round.(ess_chains; sigdigits=4))
println("R-hat for four chains: ", round.(rhat_chains; sigdigits=4))