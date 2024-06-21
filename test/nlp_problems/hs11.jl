"Problem 11 in the Hock-Schittkowski suite"
function hs11()
  nlp = Model()

  @variable(nlp, x[i = 1:2])
  set_start_value(x[1], 4.9)
  set_start_value(x[2], 0.1)

  @objective(nlp, Min, (x[1] - 5)^2 + x[2]^2 - 25)

  @constraint(nlp, -x[1]^2 + x[2] ≥ 0)

  return nlp
end
