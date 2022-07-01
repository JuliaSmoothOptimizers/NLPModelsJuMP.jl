"Hock-Schittkowski problem 43 in NLS format without constants in the objective"
function hs43()
  model = Model()
  @variable(model, x[1:4], start = 0.0)
  @expression(model, F1, x[1] - 5 / 2)
  @expression(model, F2, x[2] - 5 / 2)
  @expression(model, F3, sqrt(2) * (x[3] - 21 / 4))
  @expression(model, F4, x[4] + 7 / 2)
  @NLconstraint(model, 8 - x[1]^2 - x[2]^2 - x[3]^2 - x[4]^2 - x[1] + x[2] - x[3] + x[4] ≥ 0.0)
  @NLconstraint(model, 10 - x[1]^2 - 2 * x[2]^2 - x[3]^2 - 2 * x[4]^2 + x[1] + x[4] ≥ 0.0)
  @NLconstraint(model, 5 - 2 * x[1]^2 - x[2]^2 - x[3]^2 - 2 * x[1] + x[2] + x[4] ≥ 0.0)

  return MathOptNLSModel(model, [F1; F2; F3; F4], name = "hs43")
end
