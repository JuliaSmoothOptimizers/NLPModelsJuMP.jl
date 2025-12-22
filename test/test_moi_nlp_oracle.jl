@testset "Testing Oracle vs Non-Oracle NLP Models: $prob_oracle" for prob_oracle in
                                                                     extra_nlp_oracle_problems

  prob_no_oracle = replace(prob_oracle, "_oracle" => "")
  prob_fn_no_oracle = eval(prob_no_oracle |> Symbol)
  prob_fn_oracle = eval(prob_oracle |> Symbol)

  nlp_no_oracle = MathOptNLPModel(prob_fn_no_oracle(), hessian = true)
  nlp_oracle = MathOptNLPModel(prob_fn_oracle(), hessian = true)

  n = nlp_no_oracle.meta.nvar
  m = nlp_no_oracle.meta.ncon
  x = nlp_no_oracle.meta.x0

  # Objective value
  fx_no_oracle = obj(nlp_no_oracle, x)
  fx_with_oracle = obj(nlp_oracle, x)
  @test isapprox(fx_no_oracle, fx_with_oracle; atol = 1e-8, rtol = 1e-8)

  # Gradient of objective
  ngx_no_oracle = grad(nlp_no_oracle, x)
  ngx_with_oracle = grad(nlp_oracle, x)
  @test isapprox(ngx_no_oracle, ngx_with_oracle; atol = 1e-8, rtol = 1e-8)

  # Constraint values (up to ordering)
  ncx_no_oracle = cons(nlp_no_oracle, x)
  ncx_with_oracle = cons(nlp_oracle, x)
  @test isapprox(sort(ncx_no_oracle), sort(ncx_with_oracle); atol = 1e-8, rtol = 1e-8)

  # Jacobian: compare J'J, which is invariant to row permutations
  J_no = jac(nlp_no_oracle, x)
  J_with = jac(nlp_oracle, x)

  G_no = Matrix(J_no)' * Matrix(J_no)
  G_with = Matrix(J_with)' * Matrix(J_with)

  @test isapprox(G_no, G_with; atol = 1e-8, rtol = 1e-8)

  # Hessian of the objective: use y = 0 so constraints don't enter
  λ = zeros(m)
  H_no = hess(nlp_no_oracle, x, λ)
  H_with = hess(nlp_oracle, x, λ)

  @test isapprox(Matrix(H_no), Matrix(H_with); atol = 1e-8, rtol = 1e-8)
end
