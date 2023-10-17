using Test
using Flux, Fluxperimental

@testset "Fluxperimental.jl" begin
  include("split_join.jl")

  include("chain.jl")

  include("recur.jl")

  include("compact.jl")
  include("noshow.jl")

  include("new_recur.jl")

end
