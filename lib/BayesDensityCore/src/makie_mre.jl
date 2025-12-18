using Makie, CairoMakie

struct MyType
    x::Vector{Float64}
    a::Vector{Float64}
    b::Vector{Float64}
    c::Vector{Float64}
end

function Makie.convert_arguments(P::Type{<:AbstractPlot}, mt::MyType)
    Makie.to_plotspec(P, Makie.convert_arguments(P, mt.x, mt.a, mt.b, mt.c))
end

Makie.@recipe LineBandPlot (x, a, b, c) begin
    Makie.mixin_colormap_attributes()...
    Makie.mixin_generic_plot_attributes()...
    
    color = @inherit patchcolor
    alpha = 0.3
    strokecolor = @inherit patchstrokecolor
    strokewidth = @inherit linewidth
    linestyle = nothing
    show_band = true
    cycle = [[:color, :strokecolor] => :patchcolor]
end

function Makie.plot!(plot::LineBandPlot{<:Tuple{<:AbstractVector, <:AbstractVector, <:AbstractVector, <:AbstractVector}})

    map!(plot, [:x, :a, :b, :c], [:line, :band_lower, :band_upper]) do x, a, b, c
        return Point2f.(x, a), Point2f.(x, b), Point2f.(x, c)
    end

    if plot.show_band[]
        band!(
            plot, plot.band_lower, plot.band_upper, color = plot.color, alpha=plot.alpha
        )
    end
    lines!(
        plot, plot.line, color = plot.strokecolor, linewidth = plot.strokewidth,
        inspectable = plot.inspectable, visible = plot.visible
    )
    return plot
end

Makie.plottype(::MyType) = LineBandPlot

x1 = collect(0:0.01:1)
mt1 = MyType(x1, x1, x1 .- 1, x1 .+1)
mt2 = MyType(x1, x1 .+ 3, x1 .+ 2, x1 .+4)


fig = Figure()
ax = Axis(fig[1,1])
plot!(ax, mt1, label="mt1")
plot!(ax, mt2, label="mt2")
axislegend(ax)
fig