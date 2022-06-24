println()
println("Testing MathOptNLPModel")

@printf(
  "%-15s  %4s  %4s  %10s  %10s  %10s\n",
  "Problem",
  "nvar",
  "ncon",
  "|f(x₀)|",
  "‖∇f(x₀)‖",
  "‖c(x₀)‖"
)
# Test that every problem can be instantiated.
for prob in Symbol.(lowercase.(nlp_problems ∪ ["nohesspb", "hs61", "hs100"]))
  prob_fn = eval(prob)
  nlp = MathOptNLPModel(prob_fn(), hessian = (prob != :nohesspb))
  n = nlp.meta.nvar
  m = nlp.meta.ncon
  x = nlp.meta.x0
  fx = abs(obj(nlp, x))
  ngx = norm(grad(nlp, x))
  ncx = m > 0 ? @sprintf("%10.4e", norm(cons(nlp, x))) : "NA"
  @printf("%-15s  %4d  %4d  %10.4e  %10.4e  %10s\n", prob, n, m, fx, ngx, ncx)
end
println()

function hs219(args...; kwargs...)
  nlp = Model()
  x0 = [10, 10, 10, 10]
  @variable(nlp, x[i = 1:4], start = x0[i])

  @constraint(nlp, x[1]^2 - x[2] - x[4]^2 == 0)
  @NLconstraint(nlp, x[2] - x[1]^3 - x[3]^2 == 0)

  @NLobjective(
    nlp,
    Min,
    -x[1]
  )

  return nlp
end

@testset "Testing quadratic constraints with JuMP" begin
  g(x) = [-1., 0., 0., 0.]
  Hess(x) = zeros(4, 4)
  function Hess(x, y) 
    H = zeros(4, 4)
    H[1, 1] = 2 * y[1] - y[2] * 6 * x[1]
    H[3, 3] = - 2 * y[2]
    H[4, 4] = - 2 * y[1]
    return H
  end
  J(x) = [
      2x[1] -1 0 -2x[4];
      -3x[1]^2 1 -2x[3] 0
  ]

  jump = hs219()
  nlp = MathOptNLPModel(jump)
  x1 = nlp.meta.x0
  y1 = ones(nlp.meta.ncon)
  v1 = 2 * ones(nlp.meta.nvar)
  @test jac(nlp, x1) ≈ J(x1)
  @test hess(nlp, x1) ≈ Hess(x1)
  @test hess(nlp, x1, y1) ≈ Hess(x1, y1)
  @test hprod(nlp, x1, x1) ≈ Hess(x1) * x1
  @test hprod(nlp, x1, y1, v1) ≈ Hess(x1, y1) * v1
end
