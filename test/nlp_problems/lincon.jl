function lincon()

  nlp = Model()

  @variable(nlp, x[i=1:15])

  A = [1.0 2.0; 3.0 4.0]
  b = [5.0; 6.0]
  @constraint(nlp, A * x[1:2] + b in MOI.Nonnegatives(2))

  B = diagm([3.0 * i for i = 3:5])
  c = [1.0; 2.0; 3.0]
  @constraint(nlp, B * x[3:5] - c in MOI.Nonpositives(3))

  C = [0.0 -2.0; 4.0 0.0]
  d = [1.0; -1.0]
  @constraint(nlp, C * x[6:7] + d in MOI.Zeros(2)) 
  
  @constraint(nlp, -10.0 ≤ b' * x[8:9] + 1.0 ≤ 10.0)
  
  @constraint(nlp, c' * x[10:12] + 2.0 ≥ 3.0)
  
  @constraint(nlp, d' * x[13:14] - 4.0 ≤ 12.0)

  @constraint(nlp, 15.0 * x[15] - 21.0 == 1.0)

  @NLobjective(
    nlp,
    Min,
    sum(i + x[i]^4 for i = 1:15)
  )

  return nlp
end
