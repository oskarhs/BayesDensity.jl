module BayesDensityCoreMakieExt

using BayesDensityCore
using Makie
import BayesDensityCore: linebandplot, linebandplot!

for func in (:pdf, :cdf)
    @eval begin
        Makie.convert_arguments(P::Type{<:AbstractPlot}, ps::AbstractVIPosterior, ::typeof($func)) = Makie.convert_arguments(P, sample(ps, 1000), $func)
        Makie.convert_arguments(P::Type{<:AbstractPlot}, ps::AbstractVIPosterior, ::typeof($func), t::AbstractVector{<:Real}) = Makie.convert_arguments(P, sample(ps, 1000), $func, t)
        Makie.plottype(::AbstractVIPosterior, ::typeof($func)) = LineBandPlot
        Makie.plottype(::AbstractVIPosterior, ::typeof($func), ::AbstractVector{<:Real}) = LineBandPlot
    end
end
# Make pdf the default (i.e. the same as the PosteriorSamples default.)
Makie.convert_arguments(P::Type{<:AbstractPlot}, ps::AbstractVIPosterior) = Makie.convert_arguments(P, sample(ps, 1000))
Makie.convert_arguments(P::Type{<:AbstractPlot}, ps::AbstractVIPosterior, t::AbstractVector{<:Real}) = Makie.convert_arguments(P, sample(ps, 1000), t)
Makie.plottype(::AbstractVIPosterior) = LineBandPlot
Makie.plottype(::AbstractVIPosterior, ::AbstractVector{<:Real}) = LineBandPlot

Makie.@recipe LineBandPlot (ps, func, x) begin
    Makie.mixin_colormap_attributes()...
    Makie.mixin_generic_plot_attributes()...
    
    color = @inherit patchcolor
    alpha = 0.25
    strokecolor = @inherit patchstrokecolor
    strokewidth = @inherit linewidth
    linestyle = nothing
    estimate = mean
    ci = true
    level = 0.95
    cycle = [[:color, :strokecolor] => :patchcolor]
end

for func in (:pdf, :cdf)
    @eval begin
        function Makie.convert_arguments(P::Type{<:AbstractPlot}, ps::PosteriorSamples, ::typeof($func))
            xmin, xmax = extrema(model(ps).data.x)
            R = xmax - xmin
            t = LinRange(xmin - 0.05*R, xmax + 0.05*R, 2001)
            Makie.to_plotspec(P, Makie.convert_arguments(P, ps, $func, t))
        end

        function Makie.plot!(plot::LineBandPlot{<:Tuple{<:PosteriorSamples, <:typeof($func), <:AbstractVector}})

            map!(plot, [:ps, :x, :estimate, :ci, :level], [:est, :lower, :upper]) do ps, x, estimate, ci, level
                if estimate == mean
                    est = mean(ps, $func, x)
                    if ci
                        α = 1 - level
                        qs = [α/2, 1 - α/2]
                        quants = quantile(ps, $func, x, qs)
                        lower, upper = (quants[:,i] for i in eachindex(qs))
                    else
                        lower = copy(est)
                        upper = copy(est)
                    end
                elseif estimate == median
                    if ci
                        α = 1 - level
                        qs = [α/2, 0.5, 1 - α/2]
                        quants = quantile(ps, $func, x, qs)
                        lower, est, upper = (quants[:,i] for i in eachindex(qs))
                    else
                        est = median(ps, $func, x)
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
        
        Makie.plottype(::PosteriorSamples, ::typeof($func)) = LineBandPlot
        Makie.plottype(::PosteriorSamples, ::typeof($func), ::AbstractVector{<:Real}) = LineBandPlot
    end
end

# Make plotting the pdf the default behavior
Makie.convert_arguments(P::Type{<:AbstractPlot}, ps::PosteriorSamples) = Makie.convert_arguments(P, ps, pdf)
Makie.convert_arguments(P::Type{<:AbstractPlot}, ps::PosteriorSamples, t::AbstractVector{<:Real}) = Makie.convert_arguments(P, ps, pdf, t)

Makie.plottype(::PosteriorSamples) = LineBandPlot
Makie.plottype(::PosteriorSamples, ::AbstractVector{<:Real}) = LineBandPlot

Makie.convert_arguments(P::Type{<:PointBased}, varinfopt::VariationalOptimizationResult) = convert_arguments(P, collect(1:n_iter(varinfopt)), elbo(varinfopt))
Makie.plottype(::VariationalOptimizationResult) = Makie.Lines

end # module