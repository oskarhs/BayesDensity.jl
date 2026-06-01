module Plots

"""
    BayesDensityCore.Plots.check_chains(
        ps::PosteriorSamples...;
        [grid::Union{Real, AbstractVector{<:Real}}],
        include_burnin::Bool = true,
        lags::AbstractUnitRange{<:Integer} = 1:40
    )

Generate traceplots, autocorrelation plots and a running mean plot for the posterior samples of the density ``f``, evaluated at each point in `grid`.
When called with multiple `PosteriorSamples` objects, one line per object is added to each plot, allowing visual comparison of chains across multiple runs.

!!! note
    This function generates one plot per value in the supplied grid.

# Arguments
* `ps`: A [`BayesDensityCore.PosteriorSamples`](@ref) object.

# Keyword arguments
* `grid`: The grid of values for which the posterior density is evaluated. Defaults to 5 evenly spaced points lying between the end points of [`BayesDensityCore.default_grid_points`](@ref)
* `include_burnin`: A boolean indicating whether or not the burn-in samples should be dropped in the trace- and running mean plots. Defaults to `true`.
* `lags`: The lags at which the autocorrelation function should be evaluated. Defaults to `1:30`.

# Examples
```julia-repl
julia> using Random

julia> x = (1.0 .- (1.0 .- LinRange(0.0, 1.0, 5000)) .^(1/3)).^(1/3);

julia> hs = HistSmoother(x);

julia> ps = sample(Xoshiro(1), hs, 1100);

julia> check_chains(ps, [0.2, 0.5, 0.8]);
```
"""
function check_chains end

export check_chains

end # submodule