function linsv()
  nlp = Model()

  @variable(nlp, x[i = 1:2])

  @constraint(nlp, x[1] + x[2] >= 3)
  @constraint(nlp, x[2] - 1 >= 0)

  @objective(nlp, Min, x[1])

  return nlp
end
