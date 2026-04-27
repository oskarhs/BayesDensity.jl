"""
    quantile_bisect(
        bdm::AbstractBayesDensityModel,
        parameters::NamedTuple,
        p::Real,
        lower_bound::Real,
        upper_bound::Real
    )

Compute the `p`-quantile of the Bayesian density model `bdm` for given parameters using the bisection method.

Note: This function is indended for developer use and is not exported.
To compute quantiles for given model parameters, use [`quantile`](@ref) instead.

This function is intended as a helper function for developers to implement the [`quantile`](@ref) method for models for which the cdf is easily computed,
but where no closed-form expression for the quantile function is available.
In order for this method to work, the last two positional arguments must satisfy `lower_bound <= cdf(bdm, parameters, p) <= upper_bound`.
This method checks that `0 < p < 1`, and throws an error if this condition is not satisfied.
It is the responsibility of the developer to ensure that this condition holds prior to calling `quantile_bisect`.
"""
function quantile_bisect(
    bdm::AbstractBayesDensityModel,
    parameters::NamedTuple,
    p::Real,
    lower_bound::Real,
    upper_bound::Real
)
    # Check if p ∈ (0, 1):
    (0 < p < 1) || throw(DomainError(p, "Quantiles are undefined for `p` outside the interval (0, 1)."))

    T = promote_type(eltype(bdm), typeof(p), typeof(lower_bound), typeof(upper_bound))
    tol = cbrt(eps(float(T)))^2
    cdf_lower = cdf(bdm, parameters, lower_bound)
    cdf_upper = cdf(bdm, parameters, upper_bound)

    while upper_bound - lower_bound > tol
        midpoint = (lower_bound + upper_bound) / 2
        cdf_midpoint = cdf(bdm, parameters, midpoint)
        if cdf_midpoint < p
            lower_bound = midpoint
            cdf_lower = cdf_midpoint
        else
            upper_bound = midpoint
            cdf_upper = cdf_midpoint
        end
    end
    return (lower_bound + upper_bound) / 2
end