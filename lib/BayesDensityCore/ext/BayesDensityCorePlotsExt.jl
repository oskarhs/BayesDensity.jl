module BayesDensityCorePlotsExt

using BayesDensityCore
using Plots

# To plot posterior functionals, we simulate from the posterior:
for func in (:pdf, :cdf)
    @eval begin
        @recipe function f(vip::AbstractVIPosterior, ::typeof($func))
            return sample(vip, 1000), $func
        end
        @recipe function f(vip::AbstractVIPosterior, ::typeof($func), t::AbstractVector{<:Real})
            return sample(vip, 1000), $func, t
        end
    end 
end
# Fall back to default behavior for PosteriorSamples (i.e. plotting the pdf)
@recipe function f(vip::AbstractVIPosterior)
    return vip, pdf
end
@recipe function f(vip::AbstractVIPosterior, t::AbstractVector{<:Real})
    return vip, pdf, t
end

for func in (:pdf, :cdf)
    @eval begin
        @recipe function f(ps::PosteriorSamples, ::typeof($func)) # Plotting when no grid is given
            xmin, xmax = extrema(model(ps).data.x)
            R = xmax - xmin
            x = LinRange(xmin - 0.05*R, xmax + 0.05*R, 2001)
            return ps, $func, x
        end
    end
end
# Make plotting the pdf default behavior:
@recipe function f(ps::PosteriorSamples, x::AbstractVector{<:Real})
    return ps, pdf, x
end
@recipe function f(ps::PosteriorSamples)
    return ps, pdf
end

for func in (:pdf, :cdf)
    @eval begin
        @recipe function f(ps::PosteriorSamples, ::typeof($func), x::AbstractVector{<:Real})
            seriestype --> :line
            color --> :auto
            fillcolor --> :auto
            fillalpha --> 0.25
            label --> ""
            estimate --> mean

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

            if plotattributes[:estimate] == mean
                y = mean(ps, $func, x)
                if ci
                    qs = [α/2, 1 - α/2]
                    quants = quantile(ps, $func, x, qs)
                    lower, upper = (quants[:,i] for i in eachindex(qs))
                end
            elseif plotattributes[:estimate] == median
                if ci
                    qs = [α/2, 0.5, 1 - α/2]
                    quants = quantile(ps, $func, x, qs)
                    lower, y, upper = (quants[:,i] for i in eachindex(qs))
                else
                    y = median(ps, $func, x)
                end
            end

            if ci
                @series begin
                    seriestype := :line
                    ribbon := (y - lower, upper - y)        # fill from lower to upper
                    fillalpha := plotattributes[:fillalpha]
                    fillcolor := plotattributes[:fillcolor]
                    color := plotattributes[:color]      # no line color for CI
                    label := plotattributes[:label]
                    x, y
                end
            else
                @series begin # Plot the posterior mean/median
                seriestype := :line
                color := plotattributes[:color]
                label := plotattributes[:label]
                x, y
            end
            end
            nothing
        end
    end
end

@recipe function f(varinfopt::VariationalOptimizationResult)
    1:n_iter(varinfopt), elbo(varinfopt)
end

end # module