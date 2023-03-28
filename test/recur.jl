

# we need to include rnn gardient test. Currently only doing simple RNNs with implicit

@testset "RNN gradients-implicit" begin
  cell = Flux.RNNCell(1, 1, identity)
  layer = Flux.Recur(cell)
  layer.cell.Wi .= 5.0
  layer.cell.Wh .= 4.0
  layer.cell.b .= 0.0f0
  layer.cell.state0 .= 7.0
  x = [[2.0f0], [3.0f0]]

  # theoretical primal gradients
  primal =
    layer.cell.Wh .* (layer.cell.Wh * layer.cell.state0 .+ x[1] .* layer.cell.Wi) .+
    x[2] .* layer.cell.Wi
  ∇Wi = x[1] .* layer.cell.Wh .+ x[2]
  ∇Wh = 2 .* layer.cell.Wh .* layer.cell.state0 .+ x[1] .* layer.cell.Wi
  ∇b = layer.cell.Wh .+ 1
  ∇state0 = layer.cell.Wh .^ 2

  Flux.reset!(layer)
  ps = Flux.params(layer)
  e, g = Flux.withgradient(ps) do
    out = [layer(xi) for xi in x]
    sum(out[2])
  end

  @test primal[1] ≈ e
  @test ∇Wi ≈ g[ps[1]]
  @test ∇Wh ≈ g[ps[2]]
  @test ∇b ≈ g[ps[3]]
  @test ∇state0 ≈ g[ps[4]]


  nm_layer = Fluxperimental.NM_Recur(cell)
  ps = Flux.params(nm_layer)
  e, g = Flux.withgradient(ps) do
    l, out = Fluxperimental._apply_to_layer(nm_layer, x)
    sum(out[2])
  end
  
  @test primal[1] ≈ e
  @test ∇Wi ≈ g[ps[1]]
  @test ∇Wh ≈ g[ps[2]]
  @test ∇b ≈ g[ps[3]]
  @test ∇state0 ≈ g[ps[4]]
end

