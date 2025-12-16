"Problem 14 in the Hock-Schittkowski suite"
function hs14()
  nlp = Model()

  @variable(nlp, x[i = 1:2])
  set_start_value(x[1], 2)
  set_start_value(x[2], 2)

  @objective(nlp, Min, (x[1] - 2)^2 + (x[2] - 1)^2)

  @constraint(nlp, -x[1]^2 / 4 - x[2]^2 + 1 ≥ 0)

  @constraint(nlp, x[1] - 2 * x[2] + 1 == 0)

  return nlp
end

"Problem 14 in the Hock–Schittkowski suite, with inequality as a VectorNonlinearOracle"
function hs14_oracle()
    model = Model()

    @variable(model, x[1:2])
    set_start_value(x[1], 2)
    set_start_value(x[2], 2)

    @objective(model, Min, (x[1] - 2)^2 + (x[2] - 1)^2)

    # Inequality: -x1^2/4 - x2^2 + 1 ≥ 0
    # g(x) = -x1^2/4 - x2^2 + 1,  0 ≤ g(x) ≤ +∞
    set = MOI.VectorNonlinearOracle(;
        dimension = 2,                # 2 inputs: x1, x2
        l = [-1.0],                    # lower bound on g(x)
        u = [Inf],                    # upper bound on g(x)
        eval_f = (ret, xv) -> begin
            # ret[1] = g(x)
            ret[1] = -0.25 * xv[1]^2 - xv[2]^2
        end,
        # Jacobian of g(x): ∇g = [∂g/∂x1, ∂g/∂x2]
        # ∂g/∂x1 = -0.5 x1
        # ∂g/∂x2 = -2 x2
        jacobian_structure = [(1, 1), (1, 2)],
        eval_jacobian = (ret, xv) -> begin
            ret[1] = -0.5 * xv[1]     # d g / d x1
            ret[2] = -2.0 * xv[2]     # d g / d x2
        end,
        # Hessian of g(x):
        # ∂²g/∂x1² = -0.5
        # ∂²g/∂x1∂x2 = 0
        # ∂²g/∂x2² = -2
        #
        # We store only upper-triangular entries: (1,1), (1,2), (2,2)
        hessian_lagrangian_structure = [(1, 1), (1, 2), (2, 2)],
        eval_hessian_lagrangian = (ret, xv, μ) -> begin
            # Hessian of μ[1] * g(x)
            ret[1] = μ[1] * (-0.5)   # (1,1)
            ret[2] = μ[1] * 0.0      # (1,2)
            ret[3] = μ[1] * (-2.0)   # (2,2)
        end,
    )

    # Inequality as oracle
    @constraint(model, [x[1], x[2]] in set)

    # Equality stays as a standard nonlinear constraint:
    # x1 - 2 x2 + 1 == 0
    @constraint(model, x[1] - 2 * x[2] + 1 == 0)

    return model
end