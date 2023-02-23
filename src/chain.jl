

# Some experiments with chain to start removing the need for recur to be mutable.
# As per the conversation in the recurrent network rework issue.

# Main difference between this and the _applychain function is we return a new chain
# with the internal state modified as well as the output of applying x to the chain.
apply(chain::Flux.Chain, x) = begin
  layers, out = apply(chain.layers, x)
  Flux.Chain(layers), out
end

apply(layers::NamedTuple{NMS, TPS}, x) where {NMS, TPS} = begin
  layers, out = apply(Tuple(layers), x)
  NamedTuple{NMS}(layers), out
end

function apply(layers::AbstractVector, x)  # type-unstable path, helps compile times
  new_layers = typeof(layers)(undef, length(layers))
  for (idx, f) in enumerate(layers)
    new_layers[idx], x = apply(f, x)
  end
  new_layers, x
end

# Generated function returns a tuple of args and the last output of the network.
@generated function apply(layers::Tuple{Vararg{<:Any,N}}, x) where {N}
  x_symbols = vcat(:x, [gensym() for _ in 1:N])
  l_symbols = [gensym() for _ in 1:N]
  calls = [:(($(l_symbols[i]), $(x_symbols[i+1])) = apply(layers[$i], $(x_symbols[i]))) for i in 1:N]
  push!(calls, :(return tuple($(l_symbols...)), $(x_symbols[end])))
  Expr(:block, calls...)
end

apply(layer, x) = layer, layer(x)


