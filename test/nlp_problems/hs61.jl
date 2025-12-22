"HS61 model"
function hs61(args...; kwargs...)
  nlp = Model()
  @variable(nlp, x[i = 1:3], start = 0)

  @constraint(nlp, 3 * x[1] - 2 * x[2]^2 - 7 == 0)
  @constraint(nlp, 4 * x[1] - x[3]^2 - 11 == 0)

  @NLobjective(nlp, Min, 4 * x[1]^2 + 2 * x[2]^2 + 2 * x[3]^2 - 33 * x[1] + 16 * x[2] - 24 * x[3])

  return nlp
end

"HS61 model with both constraints as VectorNonlinearOracle"
function hs61_oracle(args...; kwargs...)
  model = Model()
  @variable(model, x[i = 1:3], start = 0)

  @NLobjective(model, Min, 4 * x[1]^2 + 2 * x[2]^2 + 2 * x[3]^2 - 33 * x[1] + 16 * x[2] - 24 * x[3])

  # First equality: 3*x1 - 2*x2^2 - 7 == 0
  # Canonical form for MathOptNLPModel:
  #   f1(x) = 3*x1 - 2*x2^2
  #   l1 = u1 = 7
  set1 = MOI.VectorNonlinearOracle(;
    dimension = 2,           # inputs: x1, x2
    l = [7.0],
    u = [7.0],
    eval_f = (ret, xv) -> begin
      # xv = [x1, x2]
      ret[1] = 3.0 * xv[1] - 2.0 * xv[2]^2
    end,
    # ∇f1 = [3, -4*x2]
    jacobian_structure = [(1, 1), (1, 2)],
    eval_jacobian = (ret, xv) -> begin
      ret[1] = 3.0             # d f1 / d x1
      ret[2] = -4.0 * xv[2]    # d f1 / d x2
    end,
    # Hessian of f1:
    # ∂²f1/∂x1² = 0
    # ∂²f1/∂x2² = -4
    # (mixed derivatives are 0)
    hessian_lagrangian_structure = [(2, 2)],
    eval_hessian_lagrangian = (ret, xv, μ) -> begin
      # Hessian of μ[1] * f1(x)
      ret[1] = μ[1] * (-4.0)   # (2,2)
    end,
  )

  # Second equality: 4*x1 - x3^2 - 11 == 0
  # Canonical form:
  #   f2(x) = 4*x1 - x3^2
  #   l2 = u2 = 11
  set2 = MOI.VectorNonlinearOracle(;
    dimension = 2,           # inputs: x1, x3
    l = [11.0],
    u = [11.0],
    eval_f = (ret, xv) -> begin
      # xv = [x1, x3]
      ret[1] = 4.0 * xv[1] - xv[2]^2
    end,
    # ∇f2 = [4, -2*x3]
    jacobian_structure = [(1, 1), (1, 2)],
    eval_jacobian = (ret, xv) -> begin
      ret[1] = 4.0             # d f2 / d x1
      ret[2] = -2.0 * xv[2]    # d f2 / d x3
    end,
    # Hessian of f2:
    # ∂²f2/∂x1² = 0
    # ∂²f2/∂x3² = -2
    hessian_lagrangian_structure = [(2, 2)],
    eval_hessian_lagrangian = (ret, xv, μ) -> begin
      ret[1] = μ[1] * (-2.0)   # (2,2)
    end,
  )

  # Equality constraints as oracles
  @constraint(model, [x[1], x[2]] in set1)
  @constraint(model, [x[1], x[3]] in set2)

  return model
end
