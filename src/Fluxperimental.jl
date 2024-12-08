module Fluxperimental

using Flux

# include("split_join.jl")  # crashes because of https://github.com/FluxML/Flux.jl/issues/2545
# export Split, Join

include("train.jl")
export shinkansen!

include("reactant.jl")
export Reactor

include("chain.jl")

# include("compact.jl")
# export @compact

# include("noshow.jl")
# export NoShow

include("autostruct.jl")
export @autostruct

# include("new_recur.jl")

include("mooncake.jl")
export Moonduo

end # module Fluxperimental
