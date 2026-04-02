using ConsensusFitting
using Random
using Test

# Run doctests first
using Documenter: DocMeta, doctest
DocMeta.setdocmeta!(ConsensusFitting, :DocTestSetup, :(using ConsensusFitting); recursive=true)
doctest(ConsensusFitting)

@testset "ConsensusFitting.jl" begin

    include("RANSAC_tests.jl")

end
