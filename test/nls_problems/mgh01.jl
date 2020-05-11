function mgh01()

  model = Model()

  @variable(model, x[i=1:2])
  set_start_value(x[1], -1.2)
  set_start_value(x[2],  1.0)

  @expression(model, F1, 1 - x[1])
  @NLexpression(model, F2, 10 * (x[2] - x[1]^2))

  return MathOptNLSModel(model, [F1, F2], name="mgh01")
end
