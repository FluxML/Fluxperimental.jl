
"""
```julia
@autostruct function MyModel(d::Int)
   dense1, dense2 = [Dense(d=>d, tanh) for _ in 1:2]    # arbitrary code here, not just keyword-like
   dense2.bias[:] .= 1/d
   return MyModel(dense1, dense2)  # demand this be very simple, no = signs allowed (return optional)
end
```
expands to
```julia
struct var"MyModel#001"{T1, T2}  # the number is incremented only when re-run with different field names
  dense1::T1
  dense2::T2
end

Flux.@layer var"MyModel#001"  # or the equivalent definitions

function var"MyModel#001"(d::Int)
   dense1, dense2 = [Dense(d=>d, tanh) for _ in 1:2]
   dense2.bias[:] .= 1/d
   return var"MyModel#001"(dense1, dense2)
end

Base.show(io::IO, ::var"MyModel#001") = print(io, "MyModel(...)")  # can't easily infer input d

MyModel = var"MyModel#001"  # maybe this can't be const
```
and would be accompanied by
```julia
function (m::MyModel)(x)  # forward pass looks just like a normal struct
  y = m.dense1(x)
  z = m.dense2(y)
  (x .+ y .+ z)./3
end
```
"""
macro autostruct(ex)
    esc(_autostruct(ex))
end

const DEFINE = Dict{Expr, Tuple}()
const COUNT = Ref(0)

function _autostruct(expr)
    Meta.isexpr(expr, :function) || error("expected a function!")
    ret = expr.args[2].args[end]
    if Meta.isexpr(ret, :return)
        ret = only(ret.args)
    end
    Meta.isexpr(ret, :call) || error("last line...")
    for ex in ret.args
        ex isa Symbol || error("expected a symbol, got $ex")
    end
    name, defex = get!(DEFINE, ret) do  # If we've seen same `ret` before, get it from dict
        c = COUNT[] += 1
        fun = ret.args[1]
        str = "$fun(...)"
        name = Symbol(fun, '#', c)
        fields = map(enumerate(ret.args[2:end])) do (i, field)
            type = Symbol("T#", i)
            :($field::$type)
        end
        types = map(f -> f.args[2], fields)
        ex = quote
            struct $name{$(types...)}
                $(fields...)
            end
            Flux.@layer $name
            Base.show(io::IO, _::$name) = print(io, $str)
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
