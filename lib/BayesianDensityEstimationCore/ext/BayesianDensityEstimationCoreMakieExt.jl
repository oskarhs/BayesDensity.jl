module BayesianDensityEstimationCoreMakieExt

using BayesianDensityEstimationCore
using Makie
import BayesianDensityEstimationCore: linebandplot, linebandplot!

Makie.convert_arguments(P::Type{<:AbstractPlot}, ps::AbstractVIPosterior) = Makie.convert_arguments(P, sample(ps, 1000))
Makie.convert_arguments(P::Type{<:AbstractPlot}, ps::AbstractVIPosterior, t::AbstractVector{<:Real}) = Makie.convert_arguments(P, sample(ps, 1000), t)


function Makie.convert_arguments(P::Type{<:AbstractPlot}, ps::PosteriorSamples)
    xmin, xmax = extrema(model(ps).data.x)
    R = xmax - xmin
    t = LinRange(xmin - 0.05*R, xmax + 0.05*R, 2001)
    Makie.to_plotspec(P, Makie.convert_arguments(P, ps, t))
end

Makie.@recipe LineBandPlot (ps, x) begin
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

function Makie.plot!(plot::LineBandPlot{<:Tuple{<:PosteriorSamples, <:AbstractVector}})

    map!(plot, [:ps, :x, :estimate, :ci, :level], [:est, :lower, :upper]) do ps, x, estimate, ci, level
        if estimate == :mean
            #est = Point2f.(x, mean(ps, x))
            est = mean(ps, x)
            if ci
                α = 1 - level
                qs = [α/2, 1 - α/2]
                quants = quantile(ps, x, qs)
                lower, upper = (quants[:,i] for i in eachindex(qs))
            else
                lower = copy(est)
                upper = copy(est)
            end
        elseif estimate == :median
            if ci
                α = 1 - level
                qs = [α/2, 0.5, 1 - α/2]
                quants = quantile(ps, x, qs)
                lower, est, upper = (quants[:,i] for i in eachindex(qs))
            else
                est = median(ps, x)
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

Makie.plottype(::PosteriorSamples) = LineBandPlot
Makie.plottype(::PosteriorSamples, ::AbstractVector{<:Real}) = LineBandPlot

end # module