module BayesDensityCoreMCMCDiagnosticToolsExt

using BayesDensityCore
using MCMCDiagnosticTools

for (fname, quantity) in (
    (:ess,  "the effective sample size"),
    (:rhat, "the R-hat (potential scale reduction factor) diagnostic"),
    (:mcse, "the Monte Carlo standard error")
)
    docstr = """
            MCMCDiagnosticTools.$fname(
                ps::PosteriorSamples...;
                [grid::Union{Real, AbstractVector{<:Real}}],
                kwargs...
            ) -> AbstractVector{<:Real}

        Compute $quantity for the posterior samples of the density ``f``, evaluated at each point in `grid`.
        Returns a vector of length `length(grid)`.

        # Arguments
        * `ps`: One or more [`BayesDensityCore.PosteriorSamples`](@ref) object(s) for which $quantity is computed. The `PosteriorSamples` objects must be obtained from the same model object and must have the same number of samples.

        # Keyword arguments
        * `grid`: The grid of values for which the posterior density is evaluated. Defaults to the 5 evenly spaced points lying between the end points of [`BayesDensityCore.default_grid_points`]
        * `kwargs...`: Additional keyword arguments passed to [`MCMCDiagnosticTools.$fname`](@extref MCMCDiagnosticTools.$fname).
        """

    @eval begin
        @doc $docstr
        function MCMCDiagnosticTools.$fname(
            ps::PosteriorSamples{T}...;
            grid::Union{S, AbstractVector{S}} = BayesDensityCore._default_check_chains_grid(ps[1]),
            kwargs...,
        ) where {T<:Real, S<:Real}
            all(model(ps[1]) == model(post) for post in ps) ||
                throw(ArgumentError("The supplied PosteriorSamples objects were not fitted to the same model."))
            all(length(samples(post)) == length(samples(ps[1])) for post in ps) ||
                throw(ArgumentError("The supplied PosteriorSamples objects do not have the same number of samples."))

            # Evaluate the pdf
            pdf_eval = Array{promote_type(T, S),3}(
                undef, length(samples(ps[1])), length(ps), length(grid),
            )
            for (i, post) in enumerate(ps)
                pdf_eval[:, i, :] .= transpose(pdf(post, grid))
            end

            # Compute the diagnostic
            return MCMCDiagnosticTools.$fname(pdf_eval; kwargs...)
        end
    end
end

end # module