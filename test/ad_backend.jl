module TestADBackend

using Test
import JuMP
import MathOptInterface as MOI
import NLPModels
import NLPModelsJuMP
import Percival

struct CustomAD <: MOI.Nonlinear.AbstractAutomaticDifferentiation
  models::Vector{MOI.Nonlinear.Model}
end

CustomAD() = CustomAD(MOI.Nonlinear.Model[])

MOI.Nonlinear.model(::CustomAD) = MOI.Nonlinear.ModelWithQuad(MOI.Nonlinear.Model())

function MOI.Nonlinear.Evaluator(
  model::MOI.Nonlinear.Model,
  backend::CustomAD,
  vars::Vector{MOI.VariableIndex},
)
  push!(backend.models, model)
  return MOI.Nonlinear.Evaluator(model, MOI.Nonlinear.SparseReverseMode(), vars)
end

@testset "Custom AD backend" begin
  src = JuMP.Model()
  unused = JuMP.@variable(src)
  x = JuMP.@variable(src)
  JuMP.delete(src, unused)
  JuMP.set_start_value(x, 1.5)
  JuMP.@operator(src, op_square, 1, x -> x^2)
  JuMP.@objective(src, Min, op_square(x - 2))
  JuMP.@constraint(src, exp(x) == exp(2))

  opt = NLPModelsJuMP.Optimizer()
  attr = MOI.AutomaticDifferentiationBackend()
  default = MOI.Nonlinear.SparseReverseMode()
  @test MOI.supports(opt, attr)
  @test MOI.get(opt, attr) === default
  @test MOI.is_empty(opt)
  MOI.set(opt, MOI.RawOptimizerAttribute("solver"), Percival.PercivalSolver)
  MOI.set(opt, MOI.Silent(), true)
  MOI.copy_to(opt, JuMP.backend(src))
  default_evaluator = opt.nlp.eval

  backend = CustomAD()
  MOI.set(opt, attr, backend)
  @test MOI.get(opt, attr) === backend
  @test opt.nlp.eval === default_evaluator
  index_map = MOI.copy_to(opt, JuMP.backend(src))
  @test !MOI.is_empty(opt)
  @test typeof(opt.nlp.eval) != typeof(default_evaluator)
  @test length(backend.models) == 1
  @test NLPModels.obj(opt.nlp, [1.0]) ≈ 1.0
  @test NLPModels.grad(opt.nlp, [1.0]) ≈ [-2.0]
  @test NLPModels.cons(opt.nlp, [1.0]) ≈ [exp(1.0) - exp(2.0)]
  @test Matrix(NLPModels.jac(opt.nlp, [1.0])) ≈ fill(exp(1.0), 1, 1)
  MOI.optimize!(opt)
  variable = index_map[JuMP.index(x)]
  @test MOI.get(opt, MOI.VariablePrimal(), variable) ≈ 2.0 atol = 1e-4

  MOI.empty!(opt)
  @test MOI.is_empty(opt)
  @test MOI.get(opt, attr) === backend
end

struct LegacyBackend{M} <: MOI.Nonlinear.AbstractAutomaticDifferentiation
  model::M
end
MOI.Nonlinear.model(backend::LegacyBackend) = backend.model

@testset "Direct construction and legacy NLP blocks" begin
  src = JuMP.Model()
  x = JuMP.@variable(src)
  JuMP.@objective(src, Min, x)
  backend = CustomAD()
  nlp = NLPModelsJuMP.MathOptNLPModel(src; backend)
  @test length(backend.models) == 1
  @test NLPModels.grad(nlp, [1.0]) == [1.0]

  JuMP.@NLobjective(src, Min, x^3)
  nlp = NLPModelsJuMP.MathOptNLPModel(src; backend)
  @test length(backend.models) == 1
  @test NLPModels.grad(nlp, [2.0]) == [12.0]
  unlayered = LegacyBackend(MOI.Nonlinear.Model())
  @test_throws AssertionError NLPModelsJuMP.MathOptNLPModel(src; backend = unlayered)
  JuMP.@constraint(src, x^2 >= 1)
  @test_throws AssertionError NLPModelsJuMP.MathOptNLPModel(src; backend = unlayered)
  layered = LegacyBackend(MOI.Nonlinear.model(MOI.Nonlinear.SparseReverseMode()))
  @test NLPModelsJuMP.MathOptNLPModel(src; backend = layered).eval.model === layered.model
  JuMP.@constraint(src, exp(x) >= 1)
  @test_throws "both the legacy" NLPModelsJuMP.MathOptNLPModel(src; backend)
end

@testset "Backend derivative capabilities" begin
  src = JuMP.Model()
  x = JuMP.@variable(src)
  JuMP.@objective(src, Max, sin(x))
  JuMP.@constraint(src, exp(x) >= 1)
  nlp = NLPModelsJuMP.MathOptNLPModel(src; backend = MOI.Nonlinear.SymbolicMode())
  @test !nlp.meta.minimize
  @test NLPModels.obj(nlp, [1.0]) ≈ sin(1.0)
  @test nlp.meta.hess_available
  @test !nlp.meta.hprod_available
  @test !nlp.meta.jprod_available
  @test !nlp.meta.jtprod_available
  @test Matrix(NLPModels.hess(nlp, [1.0])) ≈ fill(-sin(1.0), 1, 1)
end

@testset "Least squares with nonlinear constraints" begin
  src = JuMP.Model()
  x = JuMP.@variable(src)
  JuMP.@constraint(src, sin(x) == 0)
  nls = NLPModelsJuMP.MathOptNLSModel(src, [x])
  @test NLPModels.residual(nls, [1.0]) == [1.0]
  @test NLPModels.cons(nls, [1.0]) ≈ [sin(1.0)]
end

@testset "Quadratic rows in the backend evaluator" begin
  for legacy in (false, true), least_squares in (false, true)
    src = JuMP.Model()
    x, y = JuMP.@variable(src, [1:2])
    JuMP.@constraint(src, x + y == 5)
    JuMP.@constraint(src, [x * y + 3, x^2 - 4] in MOI.Nonnegatives(2))
    if legacy
      JuMP.@NLconstraint(src, -1 <= y^2 <= 1)
      JuMP.@NLconstraint(src, sin(x) <= 0)
    else
      oracle = MOI.VectorNonlinearOracle(;
        dimension = 1,
        l = [-1.0],
        u = [1.0],
        eval_f = (ret, u) -> (ret[1] = u[1]^2),
        jacobian_structure = [(1, 1)],
        eval_jacobian = (ret, u) -> (ret[1] = 2u[1]),
        hessian_lagrangian_structure = [(1, 1)],
        eval_hessian_lagrangian = (ret, u, μ) -> (ret[1] = 2μ[1]),
      )
      JuMP.@constraint(src, [y] in oracle)
      JuMP.@constraint(src, sin(x) <= 0)
    end
    if least_squares
      F = legacy ? JuMP.@NLexpression(src, x^2) : x
      nlp = NLPModelsJuMP.MathOptNLSModel(src, [F])
      objective_hessian = legacy ? 24.0 : 1.0
    else
      if legacy
        JuMP.@NLobjective(src, Min, x^3)
      else
        JuMP.@objective(src, Min, x^3)
      end
      nlp = NLPModelsJuMP.MathOptNLPModel(src)
      @test length(nlp.eval.model) == 2
      objective_hessian = 12.0
    end
    u, v = [2.0, 3.0], [4.0, -1.0]
    c = [5.0, 6.0, 4.0, 9.0, sin(2.0)]
    J = [1.0 1.0; 3.0 2.0; 4.0 0.0; 0.0 6.0; cos(2.0) 0.0]
    μ = [1.0, 2.0, 3.0, 4.0, 5.0]
    H = [objective_hessian + 6 - 5sin(2.0) 2.0; 2.0 8.0]
    @test nlp.meta.lcon == [5.0, -3.0, 4.0, -1.0, -Inf]
    @test nlp.meta.ucon == [5.0, Inf, Inf, 1.0, 0.0]
    @test NLPModels.cons(nlp, u) ≈ c
    @test NLPModels.cons_lin(nlp, u) ≈ c[1:1]
    @test NLPModels.cons_nln(nlp, u) ≈ c[2:end]
    @test NLPModels.jac(nlp, u) ≈ J
    @test NLPModels.jac_lin(nlp, u) ≈ J[1:1, :]
    @test NLPModels.jac_nln(nlp, u) ≈ J[2:end, :]
    @test NLPModels.jprod(nlp, u, v) ≈ J * v
    @test NLPModels.jprod_lin(nlp, u, v) ≈ J[1:1, :] * v
    @test NLPModels.jprod_nln(nlp, u, v) ≈ J[2:end, :] * v
    @test NLPModels.jtprod(nlp, u, μ) ≈ J' * μ
    @test NLPModels.jtprod_lin(nlp, u, μ[1:1]) ≈ J[1:1, :]' * μ[1:1]
    @test NLPModels.jtprod_nln(nlp, u, μ[2:end]) ≈ J[2:end, :]' * μ[2:end]
    rows, cols = NLPModels.hess_structure(nlp)
    @test all(rows .>= cols)
    @test Matrix(NLPModels.hess(nlp, u, μ)) ≈ H
    @test NLPModels.hprod(nlp, u, μ, v) ≈ H * v
    H_obj = [objective_hessian 0.0; 0.0 0.0]
    @test Matrix(NLPModels.hess(nlp, u)) ≈ H_obj
    @test NLPModels.hprod(nlp, u, v) ≈ H_obj * v
    if !least_squares
      @test Matrix(NLPModels.jth_hess(nlp, u, 2)) ≈ [0.0 1.0; 1.0 0.0]
      @test NLPModels.jth_hprod(nlp, u, v, 2) ≈ reverse(v)
      @test Matrix(NLPModels.jth_hess(nlp, u, 4)) ≈ [0.0 0.0; 0.0 2.0]
      @test NLPModels.ghjvprod(nlp, u, u, v) ≈ [0.0, 10.0, 16.0, -6.0, -8sin(2.0)]
    end
  end
end

@testset "Vector oracle with first derivatives only" begin
  src = JuMP.Model()
  unused = JuMP.@variable(src)
  x, y = JuMP.@variable(src, [1:2])
  JuMP.delete(src, unused)
  oracle = MOI.VectorNonlinearOracle(;
    dimension = 2,
    l = [-Inf, 0.0],
    u = [3.0, Inf],
    eval_f = (ret, u) -> (ret .= [u[1] * u[2], u[1]^2]),
    jacobian_structure = [(1, 1), (1, 2), (2, 1)],
    eval_jacobian = (ret, u) -> (ret .= [u[2], u[1], 2u[1]]),
  )
  c = JuMP.@constraint(src, [y, x] in oracle)
  opt = NLPModelsJuMP.Optimizer()
  @test MOI.supports_constraint(opt, MOI.VectorOfVariables, typeof(oracle))
  MOI.set(opt, MOI.AutomaticDifferentiationBackend(), CustomAD())
  MOI.set(opt, MOI.RawOptimizerAttribute("solver"), identity)
  index_map = MOI.copy_to(opt, JuMP.backend(src))
  @test MOI.is_valid(opt.nlp.eval.model, index_map[JuMP.index(c)])
  u, v = [2.0, 3.0], [1.0, -2.0]
  J = [3.0 2.0; 0.0 6.0]
  for nlp in (opt.nlp, NLPModelsJuMP.MathOptNLSModel(src, [x]))
    @test nlp.meta.lcon == oracle.l
    @test nlp.meta.ucon == oracle.u
    @test NLPModels.cons(nlp, u) == [6.0, 9.0]
    @test NLPModels.jac(nlp, u) == J
    @test NLPModels.jprod(nlp, u, v) == J * v
    @test NLPModels.jtprod(nlp, u, v) == J' * v
    @test !nlp.meta.hess_available
    @test !nlp.meta.hprod_available
  end

  JuMP.@NLconstraint(src, sin(x) <= 0)
  @test_throws "legacy NLPBlock" NLPModelsJuMP.MathOptNLPModel(src)
  @test_throws "legacy NLPBlock" NLPModelsJuMP.MathOptNLSModel(src, [x])
end

struct LegacyObjective <: MOI.AbstractNLPEvaluator end
MOI.features_available(::LegacyObjective) = [:Grad, :Hess, :HessVec]
function MOI.initialize(d::LegacyObjective, features)
  @test issubset(features, MOI.features_available(d))
end
MOI.eval_objective(::LegacyObjective, x) = x[1]^3
MOI.eval_objective_gradient(::LegacyObjective, g, x) = (g[1] = 3x[1]^2)
MOI.hessian_lagrangian_structure(::LegacyObjective) = [(1, 1)]
MOI.eval_hessian_lagrangian(::LegacyObjective, H, x, σ, μ) = (H[1] = 6σ * x[1])
MOI.eval_hessian_lagrangian_product(::LegacyObjective, Hv, x, v, σ, μ) = (Hv[1] = 6σ * x[1] * v[1])

@testset "Quadratic row with a legacy objective-only evaluator" begin
  src = JuMP.Model()
  x = JuMP.@variable(src)
  JuMP.@constraint(src, x^2 >= 1)
  MOI.set(JuMP.backend(src), MOI.NLPBlock(), MOI.NLPBlockData([], LegacyObjective(), true))
  nlp = NLPModelsJuMP.MathOptNLPModel(JuMP.backend(src))
  u = [2.0]
  @test NLPModels.obj(nlp, u) == 8
  @test NLPModels.grad(nlp, u) == [12]
  @test NLPModels.cons(nlp, u) == [4]
  @test Matrix(NLPModels.jac(nlp, u)) == fill(4, 1, 1)
  @test NLPModels.jprod(nlp, u, [3.0]) == [12]
  @test NLPModels.jtprod(nlp, u, [3.0]) == [12]
  @test Matrix(NLPModels.hess(nlp, u, [3.0])) == fill(18, 1, 1)
  @test NLPModels.hprod(nlp, u, [3.0], [4.0]) == [72]
end

@testset "Quadratic objective and an empty legacy block" begin
  src = JuMP.Model()
  unused = JuMP.@variable(src)
  x = JuMP.@variable(src)
  JuMP.delete(src, unused)
  JuMP.@objective(src, Min, x^2)
  JuMP.@constraint(src, [2x^2 + 3] in MOI.Nonpositives(1))
  JuMP.@constraint(src, [x^2 - 4] in MOI.Zeros(1))
  MOI.set(JuMP.backend(src), MOI.NLPBlock(), MOI.NLPBlockData([], LegacyObjective(), false))
  backend = CustomAD()
  nlp = NLPModelsJuMP.MathOptNLPModel(JuMP.backend(src); backend)
  @test only(backend.models) === nlp.eval.model.inner
  # MOI can return the two constraint types in either order.
  eq = only(findall(==(4.0), nlp.meta.lcon))
  le = 3 - eq
  @test nlp.meta.ucon[le] == -3
  @test NLPModels.cons(nlp, [2.0])[[eq, le]] == [4, 8]
  μ = zeros(2)
  μ[eq], μ[le] = 2, 3
  @test Matrix(NLPModels.hess(nlp, [2.0], μ)) == fill(18, 1, 1)
  @test NLPModels.hprod(nlp, [2.0], μ, [3.0]) == [54]
end

@testset "Quadratic constraint in a nonlinear-only model" begin
  model = MOI.Nonlinear.Model()
  x = MOI.VariableIndex(1)
  f = 2.0 * x * x + 1.0
  ci = NLPModelsJuMP._add_quadratic_constraint(model, f, MOI.LessThan(5.0))
  @test ci isa MOI.ConstraintIndex{MOI.ScalarNonlinearFunction, MOI.LessThan{Float64}}
  @test MOI.get(model, MOI.ConstraintSet(), ci) == MOI.LessThan(4.0)
  evaluator = MOI.Nonlinear.Evaluator(model, MOI.Nonlinear.SparseReverseMode(), [x])
  MOI.initialize(evaluator, [:Jac])
  c = zeros(1)
  MOI.eval_constraint(evaluator, c, [3.0])
  @test c == [18]
end

end
