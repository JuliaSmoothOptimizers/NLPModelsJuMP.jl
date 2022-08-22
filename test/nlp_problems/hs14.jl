"Problem 14 in the Hock-Schittkowski suite"
function hs14()
  nlp = Model()

  @variable(nlp, x[i = 1:2])
  set_start_value(x[1], 2)
  set_start_value(x[2], 2)

  @objective(nlp, Min, (x[1] - 2)^2 + (x[2] - 1)^2)

  @NLconstraint(nlp, -x[1]^2 / 4 - x[2]^2 + 1 ≥ 0)

  @constraint(nlp, x[1] - 2 * x[2] + 1 == 0)

  return nlp
end
