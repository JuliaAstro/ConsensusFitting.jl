using ConsensusFitting
using Documenter

DocMeta.setdocmeta!(ConsensusFitting, :DocTestSetup, :(using ConsensusFitting); recursive=true)

makedocs(;
    modules=[ConsensusFitting],
    authors="cgarling <chris.t.garling@gmail.com> and contributors",
    sitename="ConsensusFitting.jl",
    format=Documenter.HTML(;
        canonical="https://JuliaAstro.org/ConsensusFitting.jl/stable/",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
    doctest=false,
    linkcheck=true,
    warnonly=[:missing_docs, :linkcheck],
)

deploydocs(;
    repo="github.com/JuliaAstro/ConsensusFitting.jl.git",
    devbranch="main",
)
