using SimulatedAnnealingABC
using Documenter

DocMeta.setdocmeta!(SimulatedAnnealingABC, :DocTestSetup, :(using SimulatedAnnealingABC); recursive=true)

makedocs(;
    modules=[SimulatedAnnealingABC],
    authors="Andreas Scheidegger",
    repo="https://github.com/Eawag-SIAM/SimulatedAnnealingABC.jl/blob/{commit}{path}#{line}",
    sitename="SimulatedAnnealingABC.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://Eawag-SIAM.github.io/SimulatedAnnealingABC.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/Eawag-SIAM/SimulatedAnnealingABC.jl",
    devbranch="main",
)
