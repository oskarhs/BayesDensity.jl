module BayesianDensityEstimationCoreMakieExt

using BayesianDensityEstimationCore
using Makie
import BayesianDensityEstimationCore: linebandplot, linebandplot!

function Makie.convert_arguments(P::Type{<:AbstractPlot}, bds::BayesianDensityEstimationCore.BayesianDensitySamples)
    xmin, xmax = extrema(model(bds).data.x)
    R = xmax - xmin
    t = LinRange(xmin - 0.05*R, xmax + 0.05*R, 2001)
    Makie.to_plotspec(P, Makie.convert_arguments(P, bds, t))
end

Makie.@recipe LineBandPlot (bds, x) begin
    Makie.mixin_colormap_attributes()...
    Makie.mixin_generic_plot_attributes()...
    
    color = @inherit patchcolor
    alpha = 0.3
    strokecolor = @inherit linecolor
    strokewidth = @inherit linewidth
    linestyle = nothing
    estimate = :mean
    ci = true
    level = 0.95
end

function Makie.plot!(plot::LineBandPlot{<:Tuple{<:BayesianDensityEstimationCore.BayesianDensitySamples, <:AbstractVector}})

    map!(plot, [:bds, :x, :estimate, :ci, :level], [:est, :lower, :upper]) do bds, x, estimate, ci, level
        if estimate == :mean
            #est = Point2f.(x, mean(bds, x))
            est = mean(bds, x)
            if ci
                α = 1 - level
                qs = [α/2, 1 - α/2]
                quants = quantile(bds, x, qs)
                lower, upper = (quants[:,i] for i in eachindex(qs))
            else
                lower = copy(est)
                upper = copy(est)
            end
        elseif estimate == :median
            if ci
                α = 1 - level
                qs = [α/2, 0.5, 1 - α/2]
                quants = quantile(bds, x, qs)
                lower, est, upper = (quants[:,i] for i in eachindex(qs))
            else
                est = median(bds, x)
                lower = copy(est)
                upper = copy(est)
            end
        else
            throw(ArgumentError("Supplied estimate, $estimate, is not supported."))
        end
        return Point2f.(x, est), Point2f.(x, lower), Point2f.(x, upper)
    end

    if plot.ci[]
        band!(
            plot, plot.lower, plot.upper, color = plot.color, alpha=plot.alpha
        )
    end
    lines!(
        plot, plot.est, color = plot.strokecolor, linewidth = plot.strokewidth,
        inspectable = plot.inspectable, visible = plot.visible
    )
    return plot
end

Makie.plottype(::BayesianDensitySamples) = LineBandPlot
Makie.plottype(::BayesianDensitySamples, ::AbstractVector{<:Real}) = LineBandPlot

end # module