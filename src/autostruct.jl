
"""
    @autostruct function MyLayer(d)

This is a macro for easily defining new layers.

Recall that Flux layer is a `struct` which may contains parameter arrays.
Usually, the steps to make a new one are:
1. Define a `struct MyLayer` with the desired fields,
   and tell Flux to look inside with `@layer MyLayer` (or on earlier versions, `@functor`).
2. Define a constructor function like `MyLayer(d::Int)`,
   which initialises the parameters (say to `randn(d, d)`)
   and returns an instance of the `struct`, some `m::MyLayer`.
3. Define the forward pass, by making the struct callable: `(m::MyLayer)(x) = ...`

This macro handles step 1, given the function in step 2 together. You still do step 3.

If you change the name or the fields, then the `struct` definition is ahtomatically replaced.
This works because it defines has an auto-generated name, which is `== MyLayer`.

## Example

```julia
@autostruct function MyModel(d::Int)
   dense1, dense2 = [Dense(d=>d, tanh) for _ in 1:2]    # arbitrary code here, not just keyword-like
   dense2.bias[:] .= 1/d
   return MyModel(dense1, dense2)  # this must be very simple, no = signs allowed (return optional)
end

function (m::MyModel)(x)  # forward pass looks just like a normal struct
  y = m.dense1(x)
  z = m.dense2(y)
  (x .+ y .+ z)./3
end

Flux.trainable(m::MyModel) = (; m.dense1)  # if necessary, restrict which fields are trainable

MyModel(2) isa MyModel  # true
```

Here, the macro expands to these steps:

```julia
struct var"MyModel#001"{T1, T2}
  dense1::T1
  dense2::T2  # fields always use type parameters, for performance
end

Flux.@layer var"MyModel#001"

MyModel = var"MyModel#001"  # the number is incremented only when re-run with different field names

function var"MyModel#001"(d::Int)
   dense1, dense2 = [Dense(d=>d, tanh) for _ in 1:2]  # your constructor code
   dense2.bias[:] .= 1/d
   return var"MyModel#001"(dense1, dense2)
end

Base.show(io::IO, ::var"MyModel#001") = print(io, "MyModel(...)")  # can't easily infer input d
```

For comparison, the use of `@compact` to do much the same thing looks like this -- shorter,
but further from being ordinary Julia code.

```julia
function MyModel2(d::Int)
    dense1, dense2 = [Dense(d=>d, tanh) for _ in 1:2]
    dense2.bias[:] .= 1/d
    @compact(; dense1, dense2) do x
        y = m.dense1(x)
        z = m.dense2(y)
        (x .+ y .+ z)./3
    end
end

MyModel2(2) isa Fluxperimental.CompactLayer  # no easy struct type
```
"""
macro autostruct(ex)
    esc(_autostruct(ex))
end

const DEFINE = Dict{Expr, Tuple}()

function _autostruct(expr)
    Meta.isexpr(expr, :function) || throw("Expected a function definition, like `@autostruct function MyStruct(...); ...`")
    fun = expr.args[1].args[1]
    ret = expr.args[2].args[end]
    if Meta.isexpr(ret, :return)
        ret = only(ret.args)
    end
    Meta.isexpr(ret, :call) || throw("Last line of `@autostruct function $fun` must return `$fun(field1, field2, ...)`")
    ret.args[1] === fun || throw("Last line of `@autostruct function $fun` must return `$fun(field1, field2, ...)`")
    for ex in ret.args
        ex isa Symbol || throw("Last line of `@autostruct function $fun` must return `$fun(field1, field2, ...)` with only symbols, got $ex")
    end
    name, defex = get!(DEFINE, ret) do  # If we've seen same `ret` before, get it from dict
        str = "$fun(...)"
        name = gensym(fun)
        fields = map(enumerate(ret.args[2:end])) do (i, field)
            type = Symbol("T#", i)
            :($field::$type)
        end
        types = map(f -> f.args[2], fields)
        ex = quote
            struct $name{$(types...)}
                $(fields...)
            end
            $Flux.@layer $name
            $Base.show(io::IO, _::$name) = $print(io, $str)
            $fun = $name
        end
        (name, ex)
    end
    expr.args[1].args[1] = name  # this is the generated struct name
    newret = deepcopy(ret)
    newret.args[1] = name
    expr.args[2].args[end] = newret
    quote
        $(defex.args...)  # struct definition
        $expr  # constructor function
    end
end
