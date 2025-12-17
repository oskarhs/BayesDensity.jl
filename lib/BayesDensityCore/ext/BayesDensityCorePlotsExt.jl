module BayesDensityCorePlotsExt

using BayesDensityCore
using Plots

@recipe function f(vip::AbstractVIPosterior)
    return sample(vip, 1000)
end

@recipe function f(vip::AbstractVIPosterior, t::AbstractVector{<:Real})
    return sample(vip, 1000), t
end

@recipe function f(ps::PosteriorSamples)
    xmin, xmax = extrema(model(ps).data.x)
    R = xmax - xmin
    x = LinRange(xmin - 0.05*R, xmax + 0.05*R, 2001)
    return ps, x
end

@recipe function f(ps::PosteriorSamples, x::AbstractVector{<:Real})
    seriestype --> :line
    color --> :black
    fillcolor --> RGB(0.22, 0.596, 0.149) # JuliaGreen
    fillalpha --> 0.3
    label --> ""
    estimate --> :mean

    # Allow the user to pass a tuple of two strings
    if typeof(plotattributes[:label]) <: Tuple{<:AbstractString, <:AbstractString}
        label1 = plotattributes[:label][1]
        label2 = plotattributes[:label][2]
    elseif typeof(plotattributes[:label]) <: AbstractString
        label1 = plotattributes[:label]
        label2 = ""
    else
        throw(ArgumentError("label keyword must be a single string or a tuple of strings."))
    end


    ci = get(plotattributes, :ci, true)
    level = get(plotattributes, :level, 0.95)
    α = 1 - level

    if ci && !(0 < level < 1)
        throw(ArgumentError("Level of credible intervals must lie in the interval (0, 1)."))
    end

    st_map = Dict(
        :line => :line,
        :path => :line
    )
    seriestype := get(st_map, plotattributes[:seriestype], plotattributes[:seriestype])
    if !(plotattributes[:seriestype] in [:line])
        throw(ArgumentError("Seriestype :$(plotattributes[:seriestype]) not supported for objects of type PosteriorSamples."))
    end

    if plotattributes[:estimate] == :mean
        y = mean(ps, x)
        if ci
            qs = [α/2, 1 - α/2]
            quants = quantile(ps, x, qs)
            lower, upper = (quants[:,i] for i in eachindex(qs))
        end
    elseif plotattributes[:estimate] == :median
        if ci
            qs = [α/2, 0.5, 1 - α/2]
            quants = quantile(ps, x, qs)
            lower, y, upper = (quants[:,i] for i in eachindex(qs))
        else
            y = median(ps, x)
        end
    end

    @series begin # Plot the posterior mean/median
        seriestype := :line
        color := plotattributes[:color]
        label := label1
        x, y
    end

    if ci
        @series begin
            seriestype := :line
            fillrange := lower        # fill from lower to upper
            fillcolor := plotattributes[:fillcolor]
            fillalpha := plotattributes[:fillalpha]
            color := :transparent      # no line color for CI
            label := label2
            x, upper
        end
    end
    nothing
end

end # module