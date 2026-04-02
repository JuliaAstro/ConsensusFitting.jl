module ConsensusFitting

using Random: AbstractRNG, default_rng, randperm

export ransac

include("RANSAC.jl")

end
