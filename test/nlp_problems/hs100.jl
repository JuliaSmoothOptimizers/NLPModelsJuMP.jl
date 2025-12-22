function hs100(args...; kwargs...)
  nlp = Model()
  x0 = [1, 2, 0, 4, 0, 1, 1]
  @variable(nlp, x[i = 1:7], start = x0[i])

  @NLconstraint(nlp, 127 - 2 * x[1]^2 - 3 * x[2]^4 - x[3] - 4 * x[4]^2 - 5 * x[5] ≥ 0)
  @constraint(nlp, 282 - 7 * x[1] - 3 * x[2] - 10 * x[3]^2 - x[4] + x[5] ≥ 0)
  @constraint(nlp, -196 + 23 * x[1] + x[2]^2 + 6 * x[6]^2 - 8 * x[7] ≤ 0)
  @constraint(nlp, -4 * x[1]^2 - x[2]^2 + 3 * x[1] * x[2] - 2 * x[3]^2 - 5 * x[6] + 11 * x[7] ≥ 0)

  @NLobjective(
    nlp,
    Min,
    (x[1] - 10)^2 +
    5 * (x[2] - 12)^2 +
    x[3]^4 +
    3 * (x[4] - 11)^2 +
    10 * x[5]^6 +
    7 * x[6]^2 +
    x[7]^4 - 4 * x[6] * x[7] - 10 * x[6] - 8 * x[7]
  )

  return nlp
end

"HS100 model with the 2 middle constraints as VectorNonlinearOracle"
function hs100_oracle(args...; kwargs...)
  model = Model()
  x0 = [1, 2, 0, 4, 0, 1, 1]
  @variable(model, x[i = 1:7], start = x0[i])

  # 1st constraint: keep as NLconstraint
  @NLconstraint(model, 127 - 2 * x[1]^2 - 3 * x[2]^4 - x[3] - 4 * x[4]^2 - 5 * x[5] ≥ 0)

  # 2nd constraint as oracle:
  # Original: 282 - 7x1 - 3x2 - 10x3^2 - x4 + x5 ≥ 0
  # Canonical: f2(x) = -7x1 - 3x2 - 10x3^2 - x4 + x5, l2 = -282, u2 = +∞
  set2 = MOI.VectorNonlinearOracle(;
    dimension = 5,               # inputs: x1, x2, x3, x4, x5
    l = [-282.0],
    u = [Inf],
    eval_f = (ret, xv) -> begin
      # xv = [x1, x2, x3, x4, x5]
      ret[1] = -7.0 * xv[1] - 3.0 * xv[2] - 10.0 * xv[3]^2 - xv[4] + xv[5]
    end,
    # ∇f2 = [-7, -3, -20*x3, -1, 1]
    jacobian_structure = [(1, 1), (1, 2), (1, 3), (1, 4), (1, 5)],
    eval_jacobian = (ret, xv) -> begin
      ret[1] = -7.0              # ∂f2/∂x1
      ret[2] = -3.0              # ∂f2/∂x2
      ret[3] = -20.0 * xv[3]     # ∂f2/∂x3
      ret[4] = -1.0              # ∂f2/∂x4
      ret[5] = 1.0              # ∂f2/∂x5
    end,
    # Hessian of f2: only (3,3) = -20
    hessian_lagrangian_structure = [(3, 3)],
    eval_hessian_lagrangian = (ret, xv, μ) -> begin
      # Hessian of μ[1] * f2(x)
      ret[1] = μ[1] * (-20.0)    # (3,3)
    end,
  )
  @constraint(model, [x[1], x[2], x[3], x[4], x[5]] in set2)

  # 3rd constraint as oracle:
  # Original: -196 + 23x1 + x2^2 + 6x6^2 - 8x7 ≤ 0
  # Canonical: f3(x) = 23x1 + x2^2 + 6x6^2 - 8x7, l3 = -∞, u3 = 196
  set3 = MOI.VectorNonlinearOracle(;
    dimension = 4,               # inputs: x1, x2, x6, x7
    l = [-Inf],
    u = [196.0],
    eval_f = (ret, xv) -> begin
      # xv = [x1, x2, x6, x7]
      ret[1] = 23.0 * xv[1] + xv[2]^2 + 6.0 * xv[3]^2 - 8.0 * xv[4]
    end,
    # ∇f3 = [23, 2*x2, 12*x6, -8]
    jacobian_structure = [(1, 1), (1, 2), (1, 3), (1, 4)],
    eval_jacobian = (ret, xv) -> begin
      ret[1] = 23.0             # ∂f3/∂x1
      ret[2] = 2.0 * xv[2]     # ∂f3/∂x2
      ret[3] = 12.0 * xv[3]     # ∂f3/∂x6
      ret[4] = -8.0             # ∂f3/∂x7
    end,
    # Hessian of f3: (2,2) = 2, (3,3) = 12
    hessian_lagrangian_structure = [(2, 2), (3, 3)],
    eval_hessian_lagrangian = (ret, xv, μ) -> begin
      # Hessian of μ[1] * f3(x)
      ret[1] = μ[1] * 2.0      # (2,2)
      ret[2] = μ[1] * 12.0      # (3,3)
    end,
  )
  @constraint(model, [x[1], x[2], x[6], x[7]] in set3)

  # 4th constraint: keep as standard nonlinear
  @constraint(model, -4 * x[1]^2 - x[2]^2 + 3 * x[1] * x[2] - 2 * x[3]^2 - 5 * x[6] + 11 * x[7] ≥ 0)

  # Objective: same as original
  @NLobjective(
    model,
    Min,
    (x[1] - 10)^2 +
    5 * (x[2] - 12)^2 +
    x[3]^4 +
    3 * (x[4] - 11)^2 +
    10 * x[5]^6 +
    7 * x[6]^2 +
    x[7]^4 - 4 * x[6] * x[7] - 10 * x[6] - 8 * x[7]
  )

  return model
end
