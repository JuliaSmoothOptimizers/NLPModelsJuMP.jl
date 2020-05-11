function nlshs20()

  model = Model()

  lvar = [-0.5; -Inf]
  uvar = [0.5; Inf]
  x0 = [-2.0; 1.0]
  @variable(model, lvar[i] ≤ x[i=1:2] ≤ uvar[i], start=x0[i])

  @NLconstraint(
    model,
    x[1] + x[2]^2 ≥ 0
  )
  @NLconstraint(
    model,
    x[1]^2 + x[2] ≥ 0
  )
  @NLconstraint(
    model,
    x[1]^2 + x[2]^2 ≥ 1
  )

  @expression(model, F1, 1 - x[1])
  @NLexpression(model, F2, 10 * (x[2] - x[1]^2))

  return MathOptNLSModel(model, [F1, F2], name="nlshs20")
end
