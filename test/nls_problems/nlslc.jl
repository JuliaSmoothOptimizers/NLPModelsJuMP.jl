function nlslc()

  nls = Model()

  @variable(nls, x[i=1:15])

  A = [1.0 2.0; 3.0 4.0]
  b = [5.0; 6.0]
  @constraint(nls, A * x[1:2] + b in MOI.Nonnegatives(2))

  B = diagm([3.0 * i for i = 3:5])
  c = [1.0; 2.0; 3.0]
  @constraint(nls, B * x[3:5] - c in MOI.Nonpositives(3))

  C = [0.0 -2.0; 4.0 0.0]
  d = [1.0; -1.0]
  @constraint(nls, C * x[6:7] + d in MOI.Zeros(2)) 
  
  @constraint(nls, -10.0 ≤ b' * x[8:9] + 1.0 ≤ 10.0)
  
  @constraint(nls, c' * x[10:12] + 2.0 ≥ 3.0)
  
  @constraint(nls, d' * x[13:14] - 4.0 ≤ 12.0)

  @constraint(nls, 15.0 * x[15] - 21.0 == 1.0)

  @NLexpression(nls, F[i=1:15], x[i]^2 - i^2)

  return MathOptNLSModel(nls, F, name="nlslc")
end
