"""
    Moonduo(x, [dx])

This stores both an object `x` and its gradient `dx`,
with `dx` in the format used by Mooncake.jl. This is automatically allocated
when you call `Moonduo(x)`.

This serves the same purpose as Enzyme.jl's `Duplicated` type.
Both of these AD engines prefer that space for the gradient be pre-allocated.

Maybe this is like Mooncake.CoDual, except that it's marked private and seems discouraged:
https://github.com/compintell/Mooncake.jl/issues/275

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

(m::Moonduo)(x...) = m.val(x...)

function _moonstrip end

function Flux._show_pre_post(obj::Moonduo)
    nrm = Flux.norm(destructure(_moonstrip(obj.dval))[1])
    str = repr(round(nrm; sigdigits=3))
    "Moonduo(", "  # norm(∇) ≈ $str\n) "
end
