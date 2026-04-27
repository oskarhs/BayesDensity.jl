using BayesDensity
using BayesDensityBSplineMixture
using BayesDensityCore
using BayesDensityFiniteGaussianMixture
using BayesDensityHistSmoother
using BayesDensityPitmanYorMixture
using BayesDensityRandomBernsteinPoly
using Documenter
using DocumenterCitations
using DocumenterInterLinks
using Random

using TOML

const pkg_version = TOML.parsefile(joinpath(@__DIR__, "../lib/BayesDensity/Project.toml"))["version"]

bib = CitationBibliography(joinpath(@__DIR__, "bibliography.bib"), style=:authoryear)

#= DocMeta.setdocmeta!(BayesDensity, :DocTestSetup, :(
    using BayesDensityBSplineMixture;
    using BayesDensityCore;
    using BayesDensityFiniteGaussianMixture;
    using BayesDensityHistSmoother;
    using BayesDensityHistSmoother
    ); recursive=true) =#

links = DocumenterInterLinks.InterLinks(
    "Distributions" => "https://juliastats.org/Distributions.jl/stable/",
    "StatsBase" => "https://juliastats.org/StatsBase.jl/stable/",
)

makedocs(;
    modules=[
        BayesDensity,
        BayesDensityBSplineMixture,
        BayesDensityCore,
        BayesDensityFiniteGaussianMixture,
        BayesDensityHistSmoother,
        BayesDensityPitmanYorMixture,
        BayesDensityRandomBernsteinPoly
    ],
    authors="Oskar Høgberg Simensen",
    sitename="BayesDensity.jl",
    format=Documenter.HTML(;
        assets=["assets/favicon.ico"],
        inventory_version=pkg_version
        ),
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
            "methods/RandomFiniteGaussianMixture.md",
            "methods/RandomBernsteinPoly.md"
        ],
        "Examples" => [
            "examples/naive_bayes.md",
            "examples/model_selection.md"
        ],
        "Tutorials" => [
            "tutorials/add_new_models.md"
        ],
        "Contributing" => "contributing.md",
        "References" => "references.md"
    ],
    plugins = [bib, links],
    checkdocs=:none
)

deploydocs(;
    repo="github.com/oskarhs/BayesDensity.jl",
    devbranch="main",
)