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

end
