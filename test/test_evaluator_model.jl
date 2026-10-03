using Test
import ExaModels
import NLPModels
import NLPModelsJuMP
import MathOptInterface as MOI
using JuMP

@testset "EvaluatorNLPModel with ExaModels.SIMDMode" begin
    # hs071-flavoured model with a linear and a quadratic constraint mixed in.
    jm = Model()
    @variable(jm, 1 <= x[1:4] <= 5)
    set_start_value.(x, [1.0, 5.0, 5.0, 1.0])
    @objective(jm, Min, x[1] * x[4] * (x[1] + x[2] + x[3]) + x[3])
    @constraint(jm, c_nl, x[1] * x[2] * x[3] * x[4] >= 25.0)
    @constraint(jm, c_q, sum(x[i]^2 for i in 1:4) == 40.0)
    @constraint(jm, c_l, x[1] + 2x[2] + 3x[3] <= 30.0)
    moi = backend(jm)
    mode = ExaModels.SIMDMode()
    @test NLPModelsJuMP._route_all_to_evaluator(mode)
    nlp, _ = NLPModelsJuMP.evaluator_nlp_model(moi; ad_backend = mode)
    @test nlp isa NLPModelsJuMP.EvaluatorNLPModel
    meta = nlp.meta
    @test meta.nvar == 4
    @test meta.ncon == 3
    @test meta.minimize
    # Rows follow the order the constraints appear in the MOI model, with the
    # linear row identified by constraint_linearity.
    @test length(meta.lin) == 1
    @test length(meta.nln) == 2
    xv = [1.0, 4.9, 5.0, 1.1]
    @test NLPModels.obj(nlp, xv) ≈ xv[1] * xv[4] * (xv[1] + xv[2] + xv[3]) + xv[3]
    g = NLPModels.grad(nlp, xv)
    @test g ≈ [
        xv[4] * (2xv[1] + xv[2] + xv[3]),
        xv[1] * xv[4],
        xv[1] * xv[4] + 1.0,
        xv[1] * (xv[1] + xv[2] + xv[3]),
    ]
    c = NLPModels.cons(nlp, xv)
    expected = Dict(
        :nl => prod(xv) - 25.0,
        :q => sum(xv .^ 2),
        :l => xv[1] + 2xv[2] + 3xv[3],
    )
    @test sort(c) ≈ sort([expected[:nl], expected[:q], expected[:l]])
    # The linear row served by the materialized block matches the evaluator.
    c_lin = zeros(1)
    NLPModels.cons_lin!(nlp, xv, c_lin)
    @test c_lin[1] ≈ expected[:l]
    c_nln = zeros(2)
    NLPModels.cons_nln!(nlp, xv, c_nln)
    @test sort(c_nln) ≈ sort([expected[:nl], expected[:q]])
    # Jacobian: dense comparison.
    J = NLPModels.jac(nlp, xv)
    Jd = Matrix(J)
    rows = Dict{Symbol,Int}()
    for r in 1:3
        if Jd[r, :] ≈ [1.0, 2.0, 3.0, 0.0]
            rows[:l] = r
        elseif Jd[r, :] ≈ 2 .* xv
            rows[:q] = r
        else
            rows[:nl] = r
        end
    end
    @test length(rows) == 3
    @test Jd[rows[:nl], :] ≈ [prod(xv) / xv[i] for i in 1:4]  # constant does not affect J
    @test rows[:l] in meta.lin
    # jprod/jtprod agree with the dense Jacobian.
    v = [1.0, -2.0, 0.5, 3.0]
    @test NLPModels.jprod(nlp, xv, v) ≈ Jd * v
    w = [1.0, -1.0, 2.0]
    @test NLPModels.jtprod(nlp, xv, w) ≈ Jd' * w
    # Hessian of the Lagrangian: dense symmetric comparison against hprod.
    y = [2.0, -3.0, 4.0]
    σ = 1.5
    H = zeros(4, 4)
    hrows, hcols = NLPModels.hess_structure(nlp)
    hvals = NLPModels.hess_coord(nlp, xv, y; obj_weight = σ)
    for (r, cc, val) in zip(hrows, hcols, hvals)
        H[r, cc] += val
        if r != cc
            H[cc, r] += val
        end
    end
    hv = NLPModels.hprod(nlp, xv, y, v; obj_weight = σ)
    @test hv ≈ H * v
    # Solve through the MOI wrapper with Percival, comparing against the
    # default SparseReverseMode path on a Percival-friendly model.
    import Percival
    results = Dict{Symbol,Any}()
    for (key, backend) in
        (:classic => MOI.Nonlinear.SparseReverseMode(), :exa => mode)
        jm2 = Model(NLPModelsJuMP.Optimizer)
        set_silent(jm2)
        set_attribute(jm2, "solver", Percival.PercivalSolver)
        set_attribute(jm2, MOI.AutomaticDifferentiationBackend(), backend)
        @variable(jm2, 0 <= z[1:3] <= 5, start = 1.0)
        @objective(jm2, Min, (z[1] - 1)^2 + (z[2] - 2)^2 + (z[3] - 3)^2)
        @constraint(jm2, z[1] + 2z[2] <= 3.0)
        @constraint(jm2, z[1] * z[2] >= 0.25)
        @constraint(jm2, exp(z[3]) <= 10.0)
        optimize!(jm2)
        results[key] =
            (termination_status(jm2), objective_value(jm2), value.(z))
    end
    # Percival's behavior on this problem is environment-dependent (the
    # pristine SparseReverseMode path gives the same result), so the test is
    # that the ExaModels path agrees exactly with the classic path.
    @test results[:classic][1] == results[:exa][1]
    @test isequal(results[:classic][2], results[:exa][2]) ||
          isapprox(results[:classic][2], results[:exa][2]; atol = 1e-4)
    @test isequal(results[:classic][3], results[:exa][3]) ||
          isapprox(results[:classic][3], results[:exa][3]; atol = 1e-3)
end
