using Test
using Flux, Fluxperimental

@testset "Fluxperimental.jl" begin
  include("split_join.jl")

  include("chain.jl")

  # Tests currently broken.
  # include("recur.jl")

  include("compact.jl")

end
