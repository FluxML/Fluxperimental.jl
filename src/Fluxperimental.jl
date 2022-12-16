module Fluxperimental

using Flux

include("split_join.jl")
export Split, Join

include("train.jl")
export shinkansen!

include("chain.jl")

include("compact.jl")

include("preallocated.jl")
export pre, nopre

end # module Fluxperimental
