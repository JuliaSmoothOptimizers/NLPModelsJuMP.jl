function nlsqc()
  nls = Model()

  @variable(nls, x[1:3])

  @constraint(nls, [x[1], x[1]^2 + x[1] * x[2] + 3.0] in MOI.Nonnegatives(2))
  @constraint(nls, [x[2], x[2]^2 + x[2] * x[3] + 4.0] in MOI.Nonpositives(2))
  @constraint(nls, [x[3], x[3]^2 + x[3] * x[1] + 5.0] in MOI.Zeros(2))

  @NLexpression(nls, F[i = 1:3], x[i]^2 - i^2)

  return MathOptNLSModel(nls, F, name = "nlsqc")
end
