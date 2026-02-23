using BayesDensityBSplineMixture
using CairoMakie
using DataFrames
using Random
using RDatasets

# Set seed
rng = Xoshiro(1984)

# Load dataset and extract average hourly earnings by sex:
wage_data = dataset("Ecdat", "CPSch3")
male_wages = wage_data[wage_data.Sex .== "male", :].AHE
female_wages = wage_data[wage_data.Sex .== "female", :].AHE

# Find the male and female wages relative to their respective group averages:
male_rel_wages = male_wages / mean(male_wages)
female_rel_wages = female_wages / mean(female_wages)
joint_rel_wages = vcat(male_rel_wages, female_rel_wages)

# Plot density estimates for the two groups:
model_male = BSplineMixture(male_rel_wages; prior_global_rate=1e-4, bounds=(0.0, 3.5))
model_female = BSplineMixture(female_rel_wages; prior_global_rate=1e-4, bounds=(0.0, 3.5))
model_joint  = BSplineMixture(joint_rel_wages; prior_global_rate=1e-4, bounds=(0.0, 3.5))

# Fit a variational approximation to the posterior.
male_viposterior, male_info = varinf(model_male; rtol=1e-10)
female_viposterior, female_info = varinf(model_female; rtol=1e-10)
joint_viposterior, joint_info = varinf(model_joint; rtol=1e-10)

function compute_waic(ps::PosteriorSamples)
    # Note that the original data to which a `BSplineMixture`
    # object was fit is stored under `bsm.data.x`.
    logpdfs = log.(pdf(ps, model(ps).data.x))
    lppd = sum(log.(mapslices(mean, exp.(logpdfs); dims=2)))
    effpar = sum(vec(mapslices(var, logpdfs; dims=2)))
    return -2 * (lppd - effpar)
end

# Get loglikelihoods of each observation
waic_male = compute_waic(sample(rng, male_viposterior, 10_000))
waic_female = compute_waic(sample(rng, female_viposterior, 10_000))
waic_joint = compute_waic(sample(rng, joint_viposterior, 10_000))
println("WAIC separate: ", waic_male + waic_female)
println("WAIC joint: ", waic_joint)


# Plot the two estimated densities for men and women for comparison
fig = Figure(size=(550, 380))
ax = Axis(fig[1,1], xlabel="Relative wage", ylabel="Density", xlabelsize=20, ylabelsize=20)
plot!(ax, male_viposterior, label="Male wages", ci=false, strokecolor=:blue, estimate=median)
plot!(ax, female_viposterior, label="Female wages", ci=false, strokecolor=:black, linestyle=:dash, estimate=median)
Legend(fig[1,2], ax, framevisible=false)
save(joinpath("src", "assets", "model_selection", "relative_wages_by_sex.svg"), fig)