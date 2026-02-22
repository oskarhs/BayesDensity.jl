using BayesDensityBSplineMixture
using DataFrames
using Distributions
using Random
using RDatasets

# Set seed
rng = Xoshiro(1984)

# Load train and test datasets
train = dataset("MASS", "Pima.tr2")
test = dataset("MASS", "Pima.te")

# Split the training dataset into positive and negative observations
train_positive = train[train.Type .== "Yes", :]
train_negative = train[train.Type .== "No", :]

# Compute sample proportions
prior_positive = nrow(train_positive) / nrow(train)
prior_negative = nrow(train_negative) / nrow(train)

# Fit univariate density estimates to each covariate
predictor_names = [:NPreg, :Glu, :BP, :Skin, :BMI, :Ped, :Age]
densities_positive = []
densities_negative = []

for predictor in predictor_names
    fit_positive = sample(
        rng,
        BSplineMixture(collect(skipmissing(train_positive[:,predictor])); prior_global_rate=1e-4),
        10_000
    )
    fit_negative = sample(
        rng,
        BSplineMixture(collect(skipmissing(train_negative[:,predictor])); prior_global_rate=1e-4),
        10_000
    )
    push!(densities_positive, fit_positive)
    push!(densities_negative, fit_negative)
end

# Compute predictions:
predicted_class = Vector{String}(undef, nrow(test))
predicted_proba = Vector{Float64}(undef, nrow(test)) # Prob of positive result
for i in 1:nrow(test)
    posterior_positive = prior_positive
    posterior_negative = prior_negative
    for j in eachindex(predictor_names)
        posterior_positive *= median(densities_positive[j], test[i, predictor_names[j]])
        posterior_negative *= median(densities_negative[j], test[i, predictor_names[j]])
    end
    predicted_proba[i] = posterior_positive / (posterior_positive + posterior_negative)
    predicted_class[i] = ifelse(posterior_positive > posterior_negative, "Yes", "No")
end

# Compute accuracies:
acc = mean(test.Type .== predicted_class)

# Now consider a parametric naive Bayes classifier with gaussian margins
# Fit univariate density estimates to each covariate
predictor_names = [:NPreg, :Glu, :BP, :Skin, :BMI, :Ped, :Age]
densities_positive = []
densities_negative = []

for predictor in predictor_names
    fit_positive = fit(Normal, collect(skipmissing(train_positive[:,predictor])))
    fit_negative = fit(Normal, collect(skipmissing(train_negative[:,predictor])))
    push!(densities_positive, fit_positive)
    push!(densities_negative, fit_negative)
end

# Compute predictions:
predicted_class_normal = Vector{String}(undef, nrow(test))
predicted_proba_normal = Vector{Float64}(undef, nrow(test)) # Prob of positive result
for i in 1:nrow(test)
    posterior_positive = prior_positive
    posterior_negative = prior_negative
    for j in eachindex(predictor_names)
        posterior_positive *= pdf(densities_positive[j], test[i, predictor_names[j]])
        posterior_negative *= pdf(densities_negative[j], test[i, predictor_names[j]])
    end
    predicted_proba_normal[i] = posterior_positive / (posterior_positive + posterior_negative)
    predicted_class_normal[i] = ifelse(posterior_positive > posterior_negative, "Yes", "No")
end

# Compute accuracies:
acc_normal = mean(test.Type .== predicted_class_normal)