module Fluxperimental

using Flux

include("split_join.jl")
export Split, Join

include("train.jl")
export shinkansen!


include("chain.jl")

include("compact.jl")
export @compact

include("noshow.jl")
export NoShow

include("new_recur.jl")

end # module Fluxperimental
