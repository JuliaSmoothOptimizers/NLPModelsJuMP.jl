"Problem 10 in the Hock-Schittkowski suite"
function hs10()

  nlp = Model()

  @variable(nlp, x[i=1:2])
  set_start_value(x[1], -10.0)
  set_start_value(x[2],  10.0)

  @NLobjective(
    nlp,
    Min,
    x[1] - x[2]
  )

  @NLconstraint(
    nlp,
    -3 * x[1]^2 + 2 * x[1] * x[2] - x[2]^2 ≥ -1
  )

  return nlp
end
