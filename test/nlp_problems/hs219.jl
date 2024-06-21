function hs219(args...; kwargs...)
  nlp = Model()
  x0 = [10, 10, 10, 10]
  @variable(nlp, x[i = 1:4], start = x0[i])

  @constraint(nlp, x[1]^2 - x[2] - x[4]^2 == 0)
  @NLconstraint(nlp, x[2] - x[1]^3 - x[3]^2 == 0)

  @NLobjective(nlp, Min, -x[1])

  return nlp
end
