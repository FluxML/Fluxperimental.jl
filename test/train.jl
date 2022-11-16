import Flux, Fluxperimental, Optimisers

@testset "shinkansen!" begin

  X = repeat(hcat(digits.(0:3, base=2, pad=2)...), 1, 32)
  Y = Flux.onehotbatch(xor.(eachrow(X)...), 0:1)

  model = Flux.Chain(Flux.Dense(2 => 3, Flux.sigmoid), Flux.BatchNorm(3), Flux.Dense(3 => 2))
  state = Optimisers.setup(Optimisers.Adam(0.1, (0.7, 0.95)), model)

  Fluxperimental.shinkansen!(model, X, Y; state, epochs=100) do m, x, y
      Flux.logitcrossentropy(m(x), y)
  end

  @test all((Flux.softmax(model(X)) .> 0.5) .== Y)

  model = Flux.Chain(Flux.Dense(2 => 3, Flux.sigmoid), Flux.BatchNorm(3), Flux.Dense(3 => 2))
  state = Optimisers.setup(Optimisers.Adam(0.1, (0.7, 0.95)), model)

  Fluxperimental.shinkansen!(model, X, Y; state, epochs=100, batchsize=16, shuffle=true) do m, x, y
      Flux.logitcrossentropy(m(x), y)
  end

  @test all((Flux.softmax(model(X)) .> 0.5) .== Y)

end

