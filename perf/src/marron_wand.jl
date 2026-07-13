# The first 10 normal-mixture test densities of Marron & Wand (1992), "Exact Mean Integrated
# Squared Error", Annals of Statistics 20(2). Encoded as Distributions.MixtureModel of Normals
# (Normal takes the standard deviation, so variance σ² is written as std σ).

using Distributions

"""
    marron_wand_densities() -> Vector{NamedTuple}

The first 10 Marron–Wand densities as `(name, dist::MixtureModel)` entries, in the canonical order.
"""
function marron_wand_densities()
    return [
        (name = "01_gaussian",
         dist = MixtureModel([Normal(0.0, 1.0)], [1.0])),
        (name = "02_skewed_unimodal",
         dist = MixtureModel([Normal(0.0, 1.0), Normal(0.5, 2/3), Normal(13/12, 5/9)],
                             [1/5, 1/5, 3/5])),
        (name = "03_strongly_skewed",
         dist = MixtureModel([Normal(3*((2/3)^l - 1), (2/3)^l) for l in 0:7], fill(1/8, 8))),
        (name = "04_kurtotic_unimodal",
         dist = MixtureModel([Normal(0.0, 1.0), Normal(0.0, 0.1)], [2/3, 1/3])),
        (name = "05_outlier",
         dist = MixtureModel([Normal(0.0, 1.0), Normal(0.0, 0.1)], [1/10, 9/10])),
        (name = "06_bimodal",
         dist = MixtureModel([Normal(-1.0, 2/3), Normal(1.0, 2/3)], [1/2, 1/2])),
        (name = "07_separated_bimodal",
         dist = MixtureModel([Normal(-1.5, 0.5), Normal(1.5, 0.5)], [1/2, 1/2])),
        (name = "08_skewed_bimodal",
         dist = MixtureModel([Normal(0.0, 1.0), Normal(1.5, 1/3)], [3/4, 1/4])),
        (name = "09_trimodal",
         dist = MixtureModel([Normal(-1.2, 3/5), Normal(1.2, 3/5), Normal(0.0, 1/4)],
                             [9/20, 9/20, 1/10])),
        (name = "10_claw",
         dist = MixtureModel(vcat(Normal(0.0, 1.0), [Normal(l/2 - 1, 0.1) for l in 0:4]),
                             vcat(1/2, fill(1/10, 5)))),
    ]
end

"""
    density_grid(mm::MixtureModel, ngrid::Int) -> Vector{Float64}

A fixed evaluation grid derived only from the density's *definition* (component means ± 4·std),
so it is identical across data sizes, chains, and package versions — a prerequisite for comparing
stored density estimates across versions on a common grid.
"""
function density_grid(mm::MixtureModel, ngrid::Int)
    cs = components(mm)
    lo = minimum(mean(c) - 4*std(c) for c in cs)
    hi = maximum(mean(c) + 4*std(c) for c in cs)
    return collect(range(lo, hi; length = ngrid))
end
