"Hock-Schittkowski problem 30 in NLS format"
function hs30()

  model = Model()
  lvar = [1.0; -10.0; -10.0]
  @variable(model, lvar[i] ≤ x[i=1:3] ≤ 10, start=1.0)
  @expression(model, F[i=1:3], x[i])
  @NLconstraint(model, x[1]^2 + x[2]^2 ≥ 1.0)

  return MathOptNLSModel(model, F, name="hs30")
end
