using Flux: withgradient, DataLoader
using Optimisers: Optimisers

#=

This grew out of explicit-mode upgrade here:
https://github.com/FluxML/Flux.jl/pull/2082

=#

"""
    shinkansen!(loss, model, data...; state, epochs=1, [batchsize, keywords...])

This is a re-design of `train!` with a less-weird set of arguments.

* The loss function must accept the remaining arguments: `loss(model, data...)`
* The optimiser state from `setup` must be passed to the keyword `state`.

By default it calls `gradient(loss, model, data...)` just like that.
Same order as the arguments. If you specify `epochs = 100`, then it will do this 100 times.

But if you specify `batchsize = 32`, then it first makes a `DataLoader`,
and uses that to generate smaller arrays to feed to `gradient`.
All other keywords are passed to `DataLoader`, e.g. to shuffle batches.

Returns the loss from every call.

```
X = repeat(hcat(digits.(0:3, base=2, pad=2)...), 1, 32)
Y = Flux.onehotbatch(xor.(eachrow(X)...), 0:1)

model = Chain(Dense(2 => 3, sigmoid), BatchNorm(3), Dense(3 => 2))
state = Flux.setup(Adam(0.1, (0.7, 0.95)), model)

shinkansen!(model, X, Y; state, epochs=100, batchsize=16, shuffle=true) do m, x, y
    Flux.logitcrossentropy(m(x), y)
end

all((softmax(model(X)) .> 0.5) .== Y)
```
"""
function shinkansen!(loss::Function, model, data::AbstractArray...; state, epochs=1, batchsize=nothing, kw...)
  if batchsize != nothing
    eplosses = Vector{Float32}[]

    for e in 1:epochs
      eplosses = Float32[]
      for d in DataLoader(data; batchsize, kw...)
        l, (g, _...) = withgradient(loss, model, d...)
        isfinite(l) || error("loss is $l")
        Optimisers.update!(state, model, g)
        push!(eplosses, l)
      end
      push!(losses, eplosses)
    end

    return losses
  else
    losses = Float32[]

    for e in 1:epochs
      l, (g, _...) = withgradient(loss, model, data...)
      isfinite(l) || error("loss is $l")
      Optimisers.update!(state, model, g)
      push!(losses, l)
    end

    return losses
  end
end
