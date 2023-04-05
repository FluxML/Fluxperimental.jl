
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

  nm_layer = Fluxperimental.NM_Recur(cell; return_sequence = true)
  ps = Flux.params(nm_layer)
  e, g = Flux.withgradient(ps) do
    l, out = Fluxperimental.apply(nm_layer, x)
    sum(out[2])
  end
  
  @test primal[1] ≈ e
  @test ∇Wi ≈ g[ps[1]]
  @test ∇Wh ≈ g[ps[2]]
  @test ∇b ≈ g[ps[3]]
  @test ∇state0 ≈ g[ps[4]]
end

@testset "RNN gradients-implicit-partial sequence" begin
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

  nm_layer = Fluxperimental.NM_Recur(cell; return_sequence = false)
  ps = Flux.params(nm_layer)
  e, g = Flux.withgradient(ps) do
    l, out = Fluxperimental.apply(nm_layer, x)
    sum(out)
  end
  
  @test primal[1] ≈ e
  @test ∇Wi ≈ g[ps[1]]
  @test ∇Wh ≈ g[ps[2]]
  @test ∇b ≈ g[ps[3]]
  @test ∇state0 ≈ g[ps[4]]
end

@testset "RNN gradients-explicit partial sequence" begin


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



  nm_layer = Fluxperimental.NM_Recur(cell; return_sequence = false)
  e, g = Flux.withgradient(nm_layer) do layer
    r_l = Fluxperimental.reset(layer)
    l, out = Fluxperimental.apply(r_l, x)
    sum(out)
  end
  grads = g[1][:cell]

  @test primal[1] ≈ e

  if VERSION < v"1.7"
    @test ∇Wi ≈ grads[:Wi]
    @test ∇Wh ≈ grads[:Wh]
    @test ∇b ≈ grads[:b]
    @test ∇state0 ≈ grads[:state0]
  else
    @test ∇Wi ≈ grads[:Wi]
    @test ∇Wh ≈ grads[:Wh]
    @test ∇b ≈ grads[:b]
    @test ∇state0 ≈ grads[:state0]
  end
end
