using BayesDensity
using Documenter, DocumenterCitations

bib = CitationBibliography(joinpath(@__DIR__, "bibliography.bib"), style=:authoryear)

DocMeta.setdocmeta!(BayesDensity, :DocTestSetup, :(using BayesDensity); recursive=true)

makedocs(;
    modules=[BayesDensity],
    authors="Oskar Høgberg Simensen",
    sitename="BayesDensity.jl",
    format=Documenter.HTML(; assets=["assets/favicon.ico"]),
    pages=[
        "Home" => "index.md",
        "API" => [
            "api/general_api.md",
            "api/plotting_api.md" # Also add a subpage here with methods api
        ],
        "Contributing" => "contributing.md",
        "References" => "references.md"
    ],
    plugins = [bib]
)

deploydocs(;
    repo="github.com/oskarhs/BayesDensity.jl",
    devbranch="master",
)
