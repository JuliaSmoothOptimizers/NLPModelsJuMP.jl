"Problem 10 in the Hock-Schittkowski suite"
function hs10()
  nlp = Model()

  @variable(nlp, x[i = 1:2])
  set_start_value(x[1], -10)
  set_start_value(x[2], 10)

  @objective(nlp, Min, x[1] - x[2])

  @constraint(nlp, -3 * x[1]^2 + 2 * x[1] * x[2] - x[2]^2 ≥ -1)

  return nlp
end

"Problem 10 in the Hock-Schittkowski suite, but the nonlinear constraint is a VectorNonlinearOracle"
function hs10_oracle()
    model = Model()

    @variable(model, x[1:2])
    set_start_value(x[1], -10)
    set_start_value(x[2], 10)

    @objective(model, Min, x[1] - x[2])

    # g(x) = -3 x1^2 + 2 x1 x2 - x2^2 + 1 ≥ 0
    # Bounds: 0 ≤ g(x) ≤ +∞
    set = MOI.VectorNonlinearOracle(;
        dimension = 2,                # number of input variables (x1, x2)
        l = [-1.0],                    # lower bound on g(x)
        u = [Inf],                    # upper bound on g(x)
        eval_f = (ret, xv) -> begin
            # ret[1] = g(x)
            ret[1] = -3 * xv[1]^2 + 2 * xv[1] * xv[2] - xv[2]^2
        end,
        # Jacobian of g(x): ∇g = [∂g/∂x1, ∂g/∂x2]
        # ∂g/∂x1 = -6 x1 + 2 x2
        # ∂g/∂x2 =  2 x1 - 2 x2
        jacobian_structure = [(1, 1), (1, 2)],  # (row, col) for each entry of ret
        eval_jacobian = (ret, xv) -> begin
            ret[1] = -6 * xv[1] + 2 * xv[2]     # d g / d x1
            ret[2] =  2 * xv[1] - 2 * xv[2]     # d g / d x2
        end,
        # Hessian of g(x) (constant):
        # ∂²g/∂x1² = -6
        # ∂²g/∂x1∂x2 = ∂²g/∂x2∂x1 = 2
        # ∂²g/∂x2² = -2
        hessian_lagrangian_structure = [(1, 1), (1, 2), (2, 2)],
        eval_hessian_lagrangian = (ret, xv, μ) -> begin
            # Hessian of μ[1] * g(x)
            ret[1] = μ[1] * (-6.0)  # (1,1)
            ret[2] = μ[1] * ( 2.0)  # (1,2)
            ret[3] = μ[1] * (-2.0)  # (2,2)
        end,
    )

    # Same constraint, but expressed as a VectorNonlinearOracle on [x[1], x[2]]
    @constraint(model, c, [x[1], x[2]] in set)

    return model
end
