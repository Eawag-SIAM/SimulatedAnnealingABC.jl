using SimulatedAnnealingABC
using Documenter

DocMeta.setdocmeta!(SimulatedAnnealingABC, :DocTestSetup, :(using SimulatedAnnealingABC); recursive=true)

makedocs(;
    modules=[SimulatedAnnealingABC],
    authors="Andreas Scheidegger and contributors",
    repo="https://github.com/Eawag-SIAM/SimulatedAnnealingABC.jl/blob/{commit}{path}#{line}",
    sitename="SimulatedAnnealingABC.jl",
    checkdocs=:exports,
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://Eawag-SIAM.github.io/SimulatedAnnealingABC.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "index.md",
        "usage.md",
        "Example" => "example.md",
        "API" => "api.md",
        "Related packages" => "related.md"
    ],
)

deploydocs(;
    repo="github.com/Eawag-SIAM/SimulatedAnnealingABC.jl",
    devbranch="main",
)
