function quadcon()
  nlp = Model()

  @variable(nlp, x[1:3])

  @constraint(nlp, [x[1], x[1]^2 + x[1] * x[2] + 3.0] in MOI.Nonnegatives(2))
  @constraint(nlp, [x[2], x[2]^2 + x[2] * x[3] + 4.0] in MOI.Nonpositives(2))
  @constraint(nlp, [x[3], x[3]^2 + x[3] * x[1] + 5.0] in MOI.Zeros(2))

  @objective(nlp, Min, x[1] * x[2] * exp(x[3]))

  return nlp
end
