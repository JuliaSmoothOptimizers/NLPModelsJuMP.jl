"""
    replace!(ex, x)

Walk the expression `ex` and substitute in the actual variables `x`.
"""
function replace!(ex, x)
  if isa(ex, Expr)
    for (i, arg) in enumerate(ex.args)
      if isa(arg, Expr)
        if arg.head == :ref && arg.args[1] == :x
          ex.args[i] = x[arg.args[2].value]
        else
          replace!(arg, x)
        end
      end
    end
  end
end
