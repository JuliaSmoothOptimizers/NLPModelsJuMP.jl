function bndrosenbrock()
  model = Model()
  lvar = [-1; -2]
  uvar = [0.8; 2]
  @variable(model, uvar[i] ≥ x[i = 1:2] ≥ lvar[i])
  set_start_value(x[1], -1.2)
  set_start_value(x[2], 1.0)

  @expression(model, F1, 1 - x[1])
  @NLexpression(model, F2, 10 * (x[2] - x[1]^2))

  return MathOptNLSModel(model, [F1, F2], name = "bndrosenbrock")
end
