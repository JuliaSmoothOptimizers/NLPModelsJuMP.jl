function mgh01()

  model = Model()

  @variable(model, x[i=1:2])
  setvalue(x[1], -1.2)
  setvalue(x[2],  1.0)

  @NLexpression(model, F1, 10 * (x[2] - x[1]^2))
  @NLexpression(model, F2, 1 - x[1])

  return MathProgNLSModel(model, [F1, F2])
end
