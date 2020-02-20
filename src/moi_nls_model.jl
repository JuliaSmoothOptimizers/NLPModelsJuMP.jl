using NLPModels, SparseArrays
import NLPModels.increment!

using JuMP, MathOptInterface
const MOI = MathOptInterface

export MathOptNLSModel

mutable struct MathOptNLSModel <: AbstractNLSModel
  meta     :: NLPModelMeta
  nls_meta :: NLSMeta
  Feval    :: Union{MOI.AbstractNLPEvaluator, Nothing}
  ceval    :: Union{MOI.AbstractNLPEvaluator, Nothing}
  counters :: NLSCounters    # Evaluation counters.

  Fjrows :: Vector{Int}      # Jacobian sparsity pattern.
  Fjcols :: Vector{Int}
  Fjvals :: Vector{Float64}  # Room for the constraints Jacobian.

  Fhrows :: Vector{Int}      # Hessian sparsity pattern.
  Fhcols :: Vector{Int}
  Fhvals :: Vector{Float64}  # Room for the Lagrangian Hessian.

  cjrows :: Vector{Int}      # Jacobian sparsity pattern.
  cjcols :: Vector{Int}
  cjvals :: Vector{Float64}  # Room for the constraints Jacobian.

  chrows :: Vector{Int}      # Hessian sparsity pattern.
  chcols :: Vector{Int}
  chvals :: Vector{Float64}  # Room for the Lagrangian Hessian.
end

"""
    MathOptNLSModel(model, F, name="Generic")

Construct a `MathOptNLSModel` from a `JuMP` model and a vector of `NonlinearExpression`.
"""
function MathOptNLSModel(cmodel :: JuMP.Model, F :: Vector{JuMP.NonlinearExpression}; name :: String="Generic")

  nvar = num_variables(cmodel)
  vars = all_variables(cmodel)
  lvar = map(var -> has_lower_bound(var) ? lower_bound(var) : -Inf, vars)
  uvar = map(var -> has_upper_bound(var) ? upper_bound(var) :  Inf, vars)
  x0   = zeros(nvar)
  for (i, val) ∈ enumerate(start_value.(vars))
    if val !== nothing
      x0[i] = val
    end
  end

  ncon = num_nl_constraints(cmodel)
  cons = cmodel.nlp_data.nlconstr
  lcon = map(con -> con.lb, cons)
  ucon = map(con -> con.ub, cons)

  @NLobjective(cmodel, Min, 0.5 * sum(Fi^2 for Fi in F))
  ceval = NLPEvaluator(cmodel)
  MOI.initialize(ceval, [:Grad, :Jac, :Hess, :HessVec, :ExprGraph])  # Add :JacVec when available

  Fmodel = JuMP.Model()
  @variable(Fmodel, x[1:nvar])
  JuMP._init_NLP(Fmodel)
  @NLobjective(Fmodel, Min, 0.0)
  Fmodel.nlp_data.user_operators = cmodel.nlp_data.user_operators
    for Fi in F
    expr = ceval.subexpressions_as_julia_expressions[Fi.index]
    replace!(expr, x)
    expr = :($expr == 0)
    JuMP.add_NL_constraint(Fmodel, expr)
  end

  nequ = num_nl_constraints(Fmodel)
  Feval = NLPEvaluator(Fmodel)
  MOI.initialize(Feval, [:Grad, :Jac, :Hess, :HessVec, :ExprGraph])  # Add :JacVec when available

  jac_struct_Fmodel = MOI.jacobian_structure(Feval)
  Fjrows = map(t -> t[1], jac_struct_Fmodel)
  Fjcols = map(t -> t[2], jac_struct_Fmodel)

  jac_struct_cmodel = MOI.jacobian_structure(ceval)
  cjrows = map(t -> t[1], jac_struct_cmodel)
  cjcols = map(t -> t[2], jac_struct_cmodel)

  hess_struct_Fmodel = MOI.hessian_lagrangian_structure(Feval)
  Fhrows = map(t -> t[1], hess_struct_Fmodel)
  Fhcols = map(t -> t[2], hess_struct_Fmodel)

  hess_struct_cmodel = MOI.hessian_lagrangian_structure(ceval)
  chrows = map(t -> t[1], hess_struct_cmodel)
  chcols = map(t -> t[2], hess_struct_cmodel)

  meta = NLPModelMeta(nvar,
                      x0=x0,
                      lvar=lvar,
                      uvar=uvar,
                      ncon=ncon,
                      y0=zeros(ncon),
                      lcon=lcon,
                      ucon=ucon,
                      nnzj=length(cjrows),
                      nnzh=length(chrows),
                      lin=[],
                      nln=collect(1:ncon),
                      minimize=objective_sense(cmodel) == MOI.MIN_SENSE,
                      islp=false,
                      name=name,
                     )

  return MathOptNLSModel(meta,
                         NLSMeta(nequ, nvar, nnzj=length(Fjrows), nnzh=length(Fhrows)),
                         Feval,
                         ceval,
                         NLSCounters(),
                         Fjrows,
                         Fjcols,
                         zeros(length(Fjrows)),  # Fjvals
                         Fhrows,
                         Fhcols,
                         zeros(length(Fhrows)),  # Fhvals
                         cjrows,
                         cjcols,
                         zeros(length(cjrows)),  # cjvals
                         chrows,
                         chcols,
                         zeros(length(chrows)),  # chvals
                        )
end

function NLPModels.residual!(nls :: MathOptNLSModel, x :: AbstractVector, Fx :: AbstractVector)
  increment!(nls, :neval_residual)
  MOI.eval_constraint(nls.Feval, Fx, x)
  return Fx
end

function NLPModels.jac_structure_residual!(nls :: MathOptNLSModel, rows :: AbstractVector{<: Integer}, cols :: AbstractVector{<: Integer})
  rows[1:nls.nls_meta.nnzj] .= nls.Fjrows
  cols[1:nls.nls_meta.nnzj] .= nls.Fjcols
  return rows, cols
end

function NLPModels.jac_coord_residual!(nls :: MathOptNLSModel, x :: AbstractVector, vals :: AbstractVector)
  increment!(nls, :neval_jac_residual)
  MOI.eval_constraint_jacobian(nls.Feval, vals, x)
  return vals
end

function NLPModels.jac_residual(nls :: MathOptNLSModel, x :: AbstractVector)
  jac_coord_residual!(nls, x, nls.Fjvals)
  return sparse(nls.Fjrows, nls.Fjcols, nls.Fjvals, nls.nls_meta.nequ, nls.meta.nvar)
end

function NLPModels.jprod_residual!(nls :: MathOptNLSModel, x :: AbstractVector, v :: AbstractVector, Jv :: AbstractVector)
  increment!(nls, :neval_jprod_residual)
  Jv .= 0.0
  MOI.eval_constraint_jacobian(nls.Feval, nls.Fjvals, x)
  for k = 1 : nls.nls_meta.nnzj
    i = nls.Fjrows[k]
    j = nls.Fjcols[k]
    Jv[i] += nls.Fjvals[k] * v[j]
  end
  return Jv
end

function NLPModels.jtprod_residual!(nls :: MathOptNLSModel, x :: AbstractVector, v :: AbstractVector, Jtv :: AbstractVector)
  increment!(nls, :neval_jtprod_residual)
  Jtv .= 0.0
  MOI.eval_constraint_jacobian(nls.Feval, nls.Fjvals, x)
  for k = 1 : nls.nls_meta.nnzj
    i = nls.Fjrows[k]
    j = nls.Fjcols[k]
    Jtv[j] += nls.Fjvals[k] * v[i]
  end
  return Jtv
end

function NLPModels.hess_structure_residual!(nls :: MathOptNLSModel, rows :: AbstractVector{<: Integer}, cols :: AbstractVector{<: Integer})
  rows[1:nls.nls_meta.nnzh] .= nls.Fhrows
  cols[1:nls.nls_meta.nnzh] .= nls.Fhcols
  return rows, cols
end

function NLPModels.hess_coord_residual!(nls :: MathOptNLSModel, x :: AbstractVector, v :: AbstractVector, vals :: AbstractVector)
  increment!(nls, :neval_hess_residual)
  MOI.eval_hessian_lagrangian(nls.Feval, vals, x, 0.0, v)
  return vals
end

function NLPModels.hess_residual(nls :: MathOptNLSModel, x :: AbstractVector, v :: AbstractVector)
  hess_coord_residual!(nls, x, v, nls.Fhvals)
  return sparse(nls.Fhrows, nls.Fhcols, nls.Fhvals, nls.meta.nvar, nls.meta.nvar)
end

function NLPModels.jth_hess_residual(nls :: MathOptNLSModel, x :: AbstractVector, i :: Int)
  increment!(nls, :neval_jhess_residual)
  y = [j == i ? 1.0 : 0.0 for j = 1:nls.nls_meta.nequ]
  n = nls.meta.nvar
  MOI.eval_hessian_lagrangian(nls.Feval, nls.Fhvals, x, 0.0, y)
  return sparse(nls.Fhrows, nls.Fhcols, nls.Fhvals, n, n)
end

function NLPModels.hprod_residual!(nls :: MathOptNLSModel, x :: AbstractVector, i :: Int, v :: AbstractVector, Hiv :: AbstractVector)
  increment!(nls, :neval_hprod_residual)
  y = [j == i ? 1.0 : 0.0 for j = 1:nls.nls_meta.nequ]
  MOI.eval_hessian_lagrangian_product(nls.Feval, Hiv, x, v, 0.0, y)
  return Hiv
end

function NLPModels.obj(nls :: MathOptNLSModel, x :: AbstractVector)
  increment!(nls, :neval_obj)
  return MOI.eval_objective(nls.ceval, x)
end

function NLPModels.grad!(nls :: MathOptNLSModel, x :: AbstractVector, g :: AbstractVector)
  increment!(nls, :neval_grad)
  MOI.eval_objective_gradient(nls.ceval, g, x)
  return g
end

function NLPModels.cons!(nls :: MathOptNLSModel, x :: AbstractVector, c :: AbstractVector)
  increment!(nls, :neval_cons)
  MOI.eval_constraint(nls.ceval, c, x)
  return c
end

function NLPModels.jac_structure!(nls :: MathOptNLSModel, rows :: AbstractVector{<: Integer}, cols :: AbstractVector{<: Integer})
  rows[1:nls.meta.nnzj] .= nls.cjrows
  cols[1:nls.meta.nnzj] .= nls.cjcols
  return (rows, cols)
end

function NLPModels.jac_coord!(nls :: MathOptNLSModel, x :: AbstractVector, vals :: AbstractVector)
  increment!(nls, :neval_jac)
  MOI.eval_constraint_jacobian(nls.ceval, vals, x)
  return vals
end

function NLPModels.jac(nls :: MathOptNLSModel, x :: AbstractVector)
  jac_coord!(nls, x, nls.cjvals)
  return sparse(nls.cjrows, nls.cjcols, nls.cjvals, nls.meta.ncon, nls.meta.nvar)
end

function NLPModels.jprod!(nls :: MathOptNLSModel, x :: AbstractVector, v :: AbstractVector, Jv :: AbstractVector)
  increment!(nls, :neval_jprod)
  Jv .= 0.0
  MOI.eval_constraint_jacobian(nls.ceval, nls.cjvals, x)
  for k = 1 : nls.meta.nnzj
    i = nls.cjrows[k]
    j = nls.cjcols[k]
    Jv[i] += nls.cjvals[k] * v[j]
  end
  return Jv
end

function NLPModels.jtprod!(nls :: MathOptNLSModel, x :: AbstractVector, v :: AbstractVector, Jtv :: AbstractVector)
  increment!(nls, :neval_jtprod)
  Jtv .= 0.0
  MOI.eval_constraint_jacobian(nls.ceval, nls.cjvals, x)
  for k = 1 : nls.meta.nnzj
    i = nls.cjrows[k]
    j = nls.cjcols[k]
    Jtv[j] += nls.cjvals[k] * v[i]
  end
  return Jtv
end

function NLPModels.hess_structure!(nls :: MathOptNLSModel, rows :: AbstractVector{<: Integer}, cols :: AbstractVector{<: Integer})
  rows[1:nls.meta.nnzh] .= nls.chrows
  cols[1:nls.meta.nnzh] .= nls.chcols
  return (rows, cols)
end

function NLPModels.hess_coord!(nls :: MathOptNLSModel, x :: AbstractVector, y :: AbstractVector, vals :: AbstractVector; obj_weight :: Float64=1.0)
  increment!(nls, :neval_hess)
  MOI.eval_hessian_lagrangian(nls.ceval, vals, x, obj_weight, y)
  return vals
end

function NLPModels.hess_coord!(nls :: MathOptNLSModel, x :: AbstractVector, vals :: AbstractVector; obj_weight :: Float64=1.0)
  increment!(nls, :neval_hess)
  MOI.eval_hessian_lagrangian(nls.ceval, vals, x, obj_weight, zeros(nls.meta.ncon))
  return vals
end

function NLPModels.hess(nls :: MathOptNLSModel, x :: AbstractVector, y :: AbstractVector; obj_weight :: Float64=1.0)
  hess_coord!(nls, x, y, nls.chvals, obj_weight=obj_weight)
  return sparse(nls.chrows, nls.chcols, nls.chvals, nls.meta.nvar, nls.meta.nvar)
end

function NLPModels.hess(nls :: MathOptNLSModel, x :: AbstractVector; obj_weight :: Float64=1.0)
  hess_coord!(nls, x, nls.chvals, obj_weight=obj_weight)
  return sparse(nls.chrows, nls.chcols, nls.chvals, nls.meta.nvar, nls.meta.nvar)
end

function NLPModels.hprod!(nls :: MathOptNLSModel, x :: AbstractVector, y :: AbstractVector, v :: AbstractVector, hv :: AbstractVector; obj_weight :: Float64=1.0)
  increment!(nls, :neval_hprod)
  MOI.eval_hessian_lagrangian_product(nls.ceval, hv, x, v, obj_weight, y)
  return hv
end

function NLPModels.hprod!(nls :: MathOptNLSModel, x :: AbstractVector, v :: AbstractVector, hv :: AbstractVector; obj_weight :: Float64=1.0)
  increment!(nls, :neval_hprod)
  MOI.eval_hessian_lagrangian_product(nls.ceval, hv, x, v, obj_weight, zeros(nls.meta.ncon))
  return hv
end
