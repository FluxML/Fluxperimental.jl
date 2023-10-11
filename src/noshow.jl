
"""
    NoShow(layer)
    NoShow(string, layer)

This alters printing (for instance at the REPL prompt) to let you hide the complexity
of some part of a Flux model. It has no effect on the actual running of the model.

By default it prints `NoShow(...)` instead of the given layer.
If you provide a string, it prints that instead -- it can be anything,
but it may make sense to print the name of a function which will
re-create the same structure.

# Examples

```jldoctest
julia> Chain(Dense(2 => 3), NoShow(Parallel(vcat, Dense(3 => 4), Dense(3 => 5))), Dense(9 => 10))
Chain(
  Dense(2 => 3),                        # 9 parameters
  NoShow(...),                          # 36 parameters
  Dense(9 => 10),                       # 100 parameters
)                   # Total: 8 arrays, 145 parameters, 1.191 KiB.

julia> PseudoLayer((i,o)::Pair) = NoShow(
                                    "PseudoLayer(\$i => \$o)",
                                    Parallel(+, Dense(i => o, relu), Dense(i => o, tanh)),
                                  )
PseudoLayer (generic function with 1 method)

julia> Chain(Dense(2 => 3), PseudoLayer(3 => 10), Dense(9 => 10))
Chain(
  Dense(2 => 3),                        # 9 parameters
  PseudoLayer(3 => 10),                 # 80 parameters
  Dense(9 => 10),                       # 100 parameters
)                   # Total: 8 arrays, 189 parameters, 1.379 KiB.
```
"""
struct NoShow{T}
    str::String
    layer::T
end

NoShow(layer) = NoShow("", layer)

Flux.@functor NoShow

(no::NoShow)(x...) = no.layer(x...)

Base.show(io::IO, no::NoShow) = print(io, isempty(no.str) ? "NoShow(...)" : no.str)

Flux._show_leaflike(::NoShow) = true  # I think this is right
Flux._show_children(::NoShow) = (;)   # Seems to be needed?

function Base.show(io::IO, ::MIME"text/plain", m::NoShow)
  if get(io, :typeinfo, nothing) === nothing  # e.g., top level of REPL
    Flux._big_show(io, m)
  elseif !get(io, :compact, false)  # e.g., printed inside a Vector, but not a matrix
    Flux._layer_show(io, m)
  else
    show(io, m)
  end
end
