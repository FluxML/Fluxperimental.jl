"""
    Moonduo(x, [dx])

This stores both an object `x` and its gradient `dx`,
with `dx` in the format used by Mooncake.jl. This is automatically allocated
when you call `Moonduo(x)`.

This serves the same purpose as Enzyme.jl's `Duplicated` type.
Both of these AD engines prefer that space for the gradient be pre-allocated.
"""
struct Moonduo{X,DX}
  val::X
  dval::DX
end

function Moonduo(args...)
  if length(args)==1
    error("The method `Moonduo(x)` is only available when Mooncake.jl is loaded!")
  else
    error("The only legal methods are `Moonduo(x)` and `Moonduo(x, dx)`.")
  end
end

Optimisers.trainable(m::Moonduo) = (; m.val)

Flux.@layer :expand Moonduo

function (m::Moonduo)(x...)
    Zygote.isderiving() && error("""`Moonduo(flux_model)` is only for use with Mooncake.jl.
            Calling `Zygote.gradient` directly on such a wrapped model is not supported.
            You may have accidentally called `Flux.gradient(loss, Moonduo(model), x)` without wrapping `x`.""")
    m.val(x...)
end

function _moonstrip end

function Flux._show_pre_post(obj::Moonduo)
    nrm = Flux.norm(destructure(_moonstrip(obj.dval))[1])
    str = repr(round(nrm; sigdigits=3))
    "Moonduo(", "  # norm(∇) ≈ $str\n) "
end
