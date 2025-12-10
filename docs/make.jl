using BayesianDensityEstimation
using Documenter

DocMeta.setdocmeta!(BayesianDensityEstimation, :DocTestSetup, :(using BayesianDensityEstimation); recursive=true)

makedocs(;
    modules=[BayesianDensityEstimation],
    authors="Oskar Høgberg Simensen",
    sitename="BayesianDensityEstimation.jl",
    format=Documenter.HTML(;
        canonical="https://oskarhs.github.io/BayesianDensityEstimation.jl",
        edit_link="master",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/oskarhs/BayesianDensityEstimation.jl",
    devbranch="master",
)
