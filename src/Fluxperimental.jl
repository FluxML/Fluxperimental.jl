module Fluxperimental

using Flux

include("split_join.jl")
export Split, Join

include("train.jl")
export shinkansen!

include("magic.jl")

end # module Fluxperimental
