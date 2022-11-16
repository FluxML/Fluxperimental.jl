#=

These layers are from
https://fluxml.ai/Flux.jl/stable/models/advanced/

=#

# custom split layer
struct Split{T}
  paths::T
end

Split(paths...) = Split(paths)

Flux.@functor Split

(m::Split)(x::AbstractArray) = map(f -> f(x), m.paths)


# custom join layer
struct Join{T, F}
  combine::F
  paths::T
end

# allow Join(op, m1, m2, ...) as a constructor
Join(combine, paths...) = Join(combine, paths)

Flux.@functor Join

(m::Join)(xs::Tuple) = m.combine(map((f, x) -> f(x), m.paths, xs)...)
(m::Join)(xs...) = m(xs)
