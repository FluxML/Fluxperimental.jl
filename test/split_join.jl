@testset "Split + Join" begin

  model = Chain(
              Dense(10 => 5),
              Split(Dense(5 => 1, tanh), Dense(5 => 3, tanh), Dense(5 => 2))
             ) |> gpu
  @test model(gpu(rand(10))) isa Tuple{AbstractVector, AbstractVector, AbstractVector}

  model2 = Chain(
              Join(vcat,
                   Chain(Dense(1 => 5, relu), Dense(5 => 1)), # branch 1
                   Dense(1 => 2),                             # branch 2
                   Dense(1 => 1)                              # branch 3
                  ),
              Dense(4 => 1)
             ) |> gpu

  xs = map(gpu, (rand(1), rand(1), rand(1)))
  @test model2(xs) isa AbstractVector

end
