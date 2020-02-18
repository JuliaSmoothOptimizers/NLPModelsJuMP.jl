"Problem 10 in the Hock-Schittkowski suite"
function lls()

  model = Model()

  @variable(model, x[i=1:2], start=0.0)

  @constraint(
    model,
    x[1] + x[2] ≥ 0
  )

  @NLexpression(model, F1, x[1] - x[2])
  @NLexpression(model, F2, x[1] + x[2] - 2)
  @NLexpression(model, F3, x[2] - 2)

  return MathProgNLSModel(model, [F1, F2, F3])
end
