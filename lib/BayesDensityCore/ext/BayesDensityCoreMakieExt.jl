module BayesDensityCoreMakieExt

using BayesDensityCore
using Makie
using StatsBase
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
            t = default_grid_points(model(ps))
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
                inspectable = plot.inspectable, visible = plot.visible, linestyle = plot.linestyle
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

Makie.convert_arguments(P::Type{<:AbstractPlot}, varinfopt::VariationalOptimizationResult) = convert_arguments(P, collect(1:n_iter(varinfopt)), elbo(varinfopt))
Makie.plottype(::VariationalOptimizationResult) = Makie.Lines

# check_chains
function BayesDensityCore.Makie.check_chains(
    ps::PosteriorSamples...;
    grid::Union{Real, AbstractVector{<:Real}} = BayesDensityCore._default_check_chains_grid(ps[1]),
    include_burnin::Bool                      = false,
    lags::AbstractUnitRange{<:Integer}        = 1:40
)
    # Check for equality of model objects
    all(model(ps[1]) == model(post) for post in ps) || throw(ArgumentError("The supplied PosteriorSamples objects were not fitted to the same model."))

    labels = ["Chain $i" for i in eachindex(ps)]

    fig = Figure(size = (400 * 3, 250 * length(grid)))

    # Column headers
    Label(fig[0, 2], "Trace";           tellwidth=false, fontsize=22, font=:bold)
    Label(fig[0, 3], "Autocorrelation"; tellwidth=false, fontsize=22, font=:bold)
    Label(fig[0, 4], "Running Mean";    tellwidth=false, fontsize=22, font=:bold)

    ax_trace_dum = nothing
    for i in eachindex(grid)
        t_label = "t = $(round(grid[i], sigdigits=3))"

        # Row label
        Label(fig[i, 1], t_label; tellheight=false, rotation=π/2, fontsize=18, padding=(0, 0, 0, 0))

        # Axes
        ax_trace = Makie.Axis(fig[i, 2])
        ax_acf   = Makie.Axis(fig[i, 3])
        ax_mean  = Makie.Axis(fig[i, 4])

        ax_trace_dum = ax_trace

        for (j, post) in enumerate(ps)
            pdf_eval_acf = pdf(model(post), samples(post; include_burnin=false), grid)
            pdf_eval     = pdf(model(post), samples(post; include_burnin=include_burnin), grid)
            n_samples    = size(pdf_eval, 2)
            acf          = transpose(autocor(transpose(pdf_eval_acf), lags))
            running_mean = mapslices(x -> cumsum(x) ./ (1:length(x)), pdf_eval; dims=2)
            
            # Trace plot
            lines!(ax_trace, 1:n_samples, pdf_eval[i, :]; label=labels[j])
            # Autocorrelation plot
            lines!(ax_acf, collect(lags), acf[i, :]; label=labels[j])
            hlines!(ax_acf, [0.0]; color=(:black, 0.2), linestyle=:dot)
            # Running mean plot
            lines!(ax_mean, 1:n_samples, running_mean[i, :]; label=labels[j])
        end
    end
    if length(ps) >= 2
        Legend(fig[length(grid) + 1, 1:4], ax_trace_dum; orientation=:horizontal, tellwidth=false, labelsize=20)
    end

    # Tighten gap between row labels and plots
    colgap!(fig.layout, 1, 5)

    return fig
end

end # module