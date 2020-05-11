"Problem 10 in the Hock-Schittkowski suite"
function lls()

  model = Model()

  @variable(model, x[i=1:2], start=0.0)

  @NLconstraint(model, x[1] + x[2] ≥ 0)

  @expression(model, F1, x[1] - x[2])
  @expression(model, F2, x[1] + x[2] - 2)
  @expression(model, F3, x[2] - 2)

  return MathOptNLSModel(model, [F1, F2, F3], name="lls")
end
