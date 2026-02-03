using BayesDensity
using BayesDensityBSplineMixture
using BayesDensityCore
using BayesDensityFiniteGaussianMixture
using BayesDensityHistSmoother
using BayesDensityPitmanYorMixture
using Documenter, DocumenterCitations
using Random

bib = CitationBibliography(joinpath(@__DIR__, "bibliography.bib"), style=:authoryear)

#= DocMeta.setdocmeta!(BayesDensity, :DocTestSetup, :(
    using BayesDensityBSplineMixture;
    using BayesDensityCore;
    using BayesDensityFiniteGaussianMixture;
    using BayesDensityHistSmoother;
    using BayesDensityHistSmoother
    ); recursive=true) =#

makedocs(;
    modules=[
        BayesDensity,
        BayesDensityBSplineMixture,
        BayesDensityCore,
        BayesDensityFiniteGaussianMixture,
        BayesDensityHistSmoother,
        BayesDensityPitmanYorMixture
    ],
    authors="Oskar Høgberg Simensen",
    sitename="BayesDensity.jl",
    format=Documenter.HTML(; assets=["assets/favicon.ico"]),
    pages=[
        "Home" => "index.md",
        "A primer on Bayesian nonparametric density estimation" => "density_estimation_primer.md",
        "API" => [
            "api/general_api.md",
            "api/plotting_api.md" # Also add a subpage here with methods api
        ],
        "Methods" => [
            "methods/index.md",
            "methods/BSplineMixture.md",
            "methods/HistSmoother.md",
            "methods/PitmanYorMixture.md",
            "methods/FiniteGaussianMixture.md",
            "methods/RandomFiniteGaussianMixture.md"
        ],
        "Tutorials" => [
            "tutorials/add_new_models.md"
        ],
        "Contributing" => "contributing.md",
        "References" => "references.md"
    ],
    plugins = [bib],
    checkdocs=:none
)

deploydocs(;
    repo="github.com/oskarhs/BayesDensity.jl",
    devbranch="main",
)