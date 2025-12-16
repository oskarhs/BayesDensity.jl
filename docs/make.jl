using BayesianDensityEstimation
using Documenter

DocMeta.setdocmeta!(BayesianDensityEstimation, :DocTestSetup, :(using BayesianDensityEstimation); recursive=true)

makedocs(;
    modules=[BayesianDensityEstimation],
    authors="Oskar Høgberg Simensen",
    sitename="BayesDensity.jl",
    format=Documenter.HTML(;
        canonical="https://oskarhs.github.io/BayesianDensityEstimation.jl",
        edit_link="master",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
        "API" => [
            "api/general_api.md",
            "api/plotting_api.md" # Also add a subpage here with methods api
        ],
        "Contributing" => "contributing.md"
    ],
)

deploydocs(;
    repo="github.com/oskarhs/BayesianDensityEstimation.jl",
    devbranch="master",
)
