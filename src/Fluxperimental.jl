module Fluxperimental

using Flux

include("split_join.jl")
export Split, Join

include("train.jl")
export shinkansen!


include("chain.jl")

include("compact.jl")

include("noshow.jl")
export NoShow

include("autostruct.jl")
export @autostruct

include("new_recur.jl")

include("mooncake.jl")
export Moonduo

end # module Fluxperimental
