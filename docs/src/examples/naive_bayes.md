# Naive Bayes

This example illustrates how `BayesDensity` can be used for classification tasks as part of a naive Bayes classifier. Suppose that we observe a set of binary outcome variables ``C_i \in \{0, 1\}`` for ``i=1,\ldots, n``, and that we are interested in classifying each observation based on a set of ``d``-dimensional covariates ``\boldsymbol{x}_i = (x_{i,1}, \ldots, x_{i,d})^\top``. One possible approach to this classification task is to assume generative model for the class membership, where the observations have been generated according to density ``f_0`` if ``C_i = 0`` and density ``f_1`` if ``C_i = 1``. The probability of belonging to a given class can then be computed via Bayes' theorem,

```math
P(C_i = c\,|\, \boldsymbol{x}_i) \propto P(C_i = c)\, f_c(\boldsymbol{x}_i), \quad c = 0, 1.
```

The Bayes classifier in the above setting is to predict that each ``C_i`` is equal to the class which maximizes the posterior probability. Both the prior class probabilities ``P(C_i = c)`` and the class-specific densities ``f_0`` and ``f_1`` are in general not known and need to be estimated from the observed sample. A simple estimate of the prior class probabilities are the corresponding sample proportions, while the class-specific densities can be fit via either parametric or nonparametric methods.

Assuming a nonparametric for the densities ``f_c`` immediately stands out as an attractive option in this case, as one does not have to postulate a parametric family of distributions to which the component densities belong. However, in cases where the dimension of the covariate vector is moderate relative to the number of observations, nonparametric density estimation is notoriously difficult, a phenomenon often referred to as [the curse of dimensionality](https://en.wikipedia.org/wiki/Curse_of_dimensionality). A solution to this issue is to use a so-called [naive Bayes classifier](https://en.wikipedia.org/wiki/Naive_Bayes_classifier). This class of classifiers make the simplifying assumption that the densities of the features ``x_j`` are independent conditional on the class membership, i.e. we can write  ``f_c(\boldsymbol{x}_i) = \prod_{j=1}^d f_{c, j}(x_{i,j})``. Under this assumption, the problem of fitting the class-specific densities ``f_c`` reduces to that of fitting multiple univariate density estimators. Although the independence assumption is often unrealistic in practice, it often ends up performing better than the alternative due to the inefficiency of nonparametric density estimation in higher dimensions.

## A real-data example
To illustrate the naive Bayes classifier in practice, we consider the Pima indians dataset from the MASS R package, which is easily available through `RDatasets.jl`. This dataset consists of diabetes measurements from a population of women who were at least 21 years old, of Pima Indian heritage and living near Phoenix, Arizona. The dataset also includes a total of ``7`` numeric covariates. In order to evaluate the out-of-sample performance of our classifier, we split the dataset into a training and a test set. The version of the data set that we will be using here contains missing values, which is a problem when attempting to estimate the joint density of the covariates. In contrast, this does not pose a problem for naive Bayes since the marginal densities are estimated separately, and we do as such not have to discard data point ``x_{i,j}`` when estimating the density ``f_{c,j}`` when the value of ``x_{i,l}`` is missing. Due to the fact that there are only ``68`` individuals with a positive outcome in the training dataset for which we have complete data, a fully nonparametric approach to estimating ``f_0`` and ``f_1`` is unlikely to work well. Here, we fit a [`BSplineMixture`](@ref) model to the marginals ``f_{c,j}`` of the component densities, and use the resulting posterior medians as point estimates of ``f_{c,j}(x_{i,j})``.

```julia
using BayesDensityBSplineMixture
using DataFrames
using Random
using RDatasets

# Set seed
rng = Xoshiro(1984)

# Load train and test datasets
train = dataset("MASS", "Pima.tr2") # NB! This data set contains some missing values
test = dataset("MASS", "Pima.te")

# Split the training dataset into positive and negative observations
train_positive = train[train.Type .== "Yes", :]
train_negative = train[train.Type .== "No", :]

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
    # Filter missing values before estimating the densities:
    x_positive_filtered = collect(skipmissing(train_positive[:,predictor]))
    x_negative_filtered = collect(skipmissing(train_negative[:,predictor]))
    fit_positive = sample(
        rng,
        BSplineMixture(x_positive_filtered; prior_global_rate=1e-4),
        10_000
    )
    fit_negative = sample(
        rng,
        BSplineMixture(x_negative_filtered; prior_global_rate=1e-4),
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
```

The resulting accuracy on the test set is ``0.789``. This is a significant improvement over the baseline classifier that always predicts a negative result (i.e. the majority class in the training set), which here resulted in a test set accuracy of ``0.672``, showing that the naive Bayes classifier improves significantly over the simple majority classification rule. For comparison, we also fitted a parametric naive Bayes classifier to the same training dataset, where the marginal distributions are assumed to be normal, which is the original naive Bayes approach [Hastie2009Elements](@citep). This yielded a test set accuracy of ``0.783``, slightly lower than that obtained with our nonparametric approach.