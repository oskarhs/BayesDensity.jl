module BayesianDensityEstimationCorePlotsExt

using BayesianDensityEstimationCore
using Plots

@recipe function f(bds::BayesianDensitySamples)
    seriestype --> :line
    color --> :black
    fillcolor --> :green
    fillalpha --> 0.2

    ci = get(plotattributes, :ci, true)
    level = get(plotattributes, :level, 0.95)

    if ci && !(0 < level < 1)
        throw(ArgumentError("Level of credible intervals must lie in the interval (0, 1)."))
    end

    st_map = Dict(
        :line => :line,
        :path => :line
    )
    seriestype := get(st_map, plotattributes[:seriestype], plotattributes[:seriestype])
    if !(plotattributes[:seriestype] in [:line])
        throw(ArgumentError("Seriestype :$(plotattributes[:seriestype]) not supported for objects of type BayesianDensitySamples."))
    end

    xmin, xmax = extrema(model(bds).data.x)
    R = xmax - xmin
    x = LinRange(xmin - 0.05*R, xmax + 0.05*R, 2001)
    y = mean(bds, x)

    @series begin # Plot the posterior mean/median
        seriestype := :line
        color := plotattributes[:color]
        x, y
    end
    

    if ci
        qs = [level/2, 1 - level/2]
        quants = quantile(bds, x, qs)
        lower, upper = (quants[:,i] for i in eachindex(qs))
        @series begin
            fillrange := lower        # fill from lower to upper
            fillcolor := plotattributes[:fillcolor]
            fillalpha := plotattributes[:fillalpha]
            #color := :transparent      # no line color for CI
            label := "Huh?"
            x, upper
        end
    end
    nothing
end

end # module