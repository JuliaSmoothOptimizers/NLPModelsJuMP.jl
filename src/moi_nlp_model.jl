using NLPModels, SparseArrays
import NLPModels.increment!

using JuMP, MathOptInterface
const MOI = MathOptInterface

export MathOptNLPModel

mutable struct MathOptNLPModel <: AbstractNLPModel
  meta     :: NLPModelMeta
  eval     :: Union{MOI.AbstractNLPEvaluator, Nothing}
  counters :: Counters      # Evaluation counters.

  jrows :: Vector{Int}      # Jacobian sparsity pattern.
  jcols :: Vector{Int}
  jvals :: Vector{Float64}  # Room for the constraints Jacobian.

  hrows :: Vector{Int}      # Hessian sparsity pattern.
  hcols :: Vector{Int} 
  hvals :: Vector{Float64}  # Room for the Lagrangian Hessian.
end

"""
    MathOptNLPModel(model, name="Generic")

Construct a `MathOptNLPModel` from a `JuMP` model.
"""
function MathOptNLPModel(jmodel :: JuMP.Model; name :: String="Generic")

  eval = NLPEvaluator(jmodel)
  MOI.initialize(eval, [:Grad, :Jac, :Hess, :HessVec, :ExprGraph])  # Add :JacVec when available

  nvar = num_variables(jmodel)
  vars = all_variables(jmodel)
  lvar = map(var -> has_lower_bound(var) ? lower_bound(var) : -Inf, vars)
  uvar = map(var -> has_upper_bound(var) ? upper_bound(var) :  Inf, vars)
  x0   = zeros(nvar)
  for (i, val) ∈ enumerate(start_value.(vars))
    if val !== nothing
      x0[i] = val
    end
  end

  ncon = num_nl_constraints(jmodel)
  cons = jmodel.nlp_data.nlconstr
  lcon = map(con -> con.lb, cons)
  ucon = map(con -> con.ub, cons)

  jac_struct = MOI.jacobian_structure(eval)
  jrows = map(t -> t[1], jac_struct)
  jcols = map(t -> t[2], jac_struct)

  hesslag_struct = MOI.hessian_lagrangian_structure(eval)
  hrows = map(t -> t[1], hesslag_struct)
  hcols = map(t -> t[2], hesslag_struct)

  nnzj = length(jrows)
  nnzh = length(hrows)

  meta = NLPModelMeta(nvar,
                      x0=x0,
                      lvar=lvar,
                      uvar=uvar,
                      ncon=ncon,
                      y0=zeros(ncon),
                      lcon=lcon,
                      ucon=ucon,
                      nnzj=nnzj,
                      nnzh=nnzh,
                      lin=[],
                      nln=collect(1:ncon),
                      minimize=objective_sense(jmodel) == MOI.MIN_SENSE,
                      islp=false,
                      name=name,
                      )

  return MathOptNLPModel(meta,
                         eval,
                         Counters(),
                         jrows,
                         jcols,
                         zeros(nnzj),  # jvals
                         hrows,
                         hcols,
                         zeros(nnzh),  # hvals
                        )
end

function NLPModels.obj(nlp :: MathOptNLPModel, x :: AbstractVector)
  increment!(nlp, :neval_obj)
  return MOI.eval_objective(nlp.eval, x)
end

function NLPModels.grad!(nlp :: MathOptNLPModel, x :: AbstractVector, g :: AbstractVector)
  increment!(nlp, :neval_grad)
  MOI.eval_objective_gradient(nlp.eval, g, x)
  return g
end

function NLPModels.cons!(nlp :: MathOptNLPModel, x :: AbstractVector, c :: AbstractVector)
  increment!(nlp, :neval_cons)
  MOI.eval_constraint(nlp.eval, c, x)
  return c
end

function NLPModels.jac_structure!(nlp :: MathOptNLPModel, rows :: AbstractVector{<: Integer}, cols :: AbstractVector{<: Integer})
  rows[1:nlp.meta.nnzj] .= nlp.jrows
  cols[1:nlp.meta.nnzj] .= nlp.jcols
  return (rows, cols)
end

function NLPModels.jac_coord!(nlp :: MathOptNLPModel, x :: AbstractVector, vals :: AbstractVector)
  increment!(nlp, :neval_jac)
  MOI.eval_constraint_jacobian(nlp.eval, vals, x)
  return vals
end

function NLPModels.jac(nlp :: MathOptNLPModel, x :: AbstractVector)
  jac_coord!(nlp, x, nlp.jvals)
  return sparse(nlp.jrows, nlp.jcols, nlp.jvals, nlp.meta.ncon, nlp.meta.nvar)
end

function NLPModels.jprod!(nlp :: MathOptNLPModel, x :: AbstractVector, v :: AbstractVector, Jv :: AbstractVector)
  increment!(nlp, :neval_jprod)
  Jv .= 0.0
  MOI.eval_constraint_jacobian(nlp.eval, nlp.jvals, x)
  for k = 1 : nlp.meta.nnzj
    i = nlp.jrows[k]
    j = nlp.jcols[k]
    Jv[i] += nlp.jvals[k] * v[j]
  end
  return Jv
end

function NLPModels.jtprod!(nlp :: MathOptNLPModel, x :: AbstractVector, v :: AbstractVector, Jtv :: AbstractVector)
  increment!(nlp, :neval_jtprod)
  Jtv .= 0.0
  MOI.eval_constraint_jacobian(nlp.eval, nlp.jvals, x)
  for k = 1 : nlp.meta.nnzj
    i = nlp.jrows[k]
    j = nlp.jcols[k]
    Jtv[j] += nlp.jvals[k] * v[i]
  end
  return Jtv
end

# Uncomment when :JacVec becomes available in MOI.
#
# function NLPModels.jprod!(nlp :: MathOptNLPModel, x :: AbstractVector, v :: AbstractVector, jv :: AbstractVector)
#   increment!(nlp, :neval_jprod)
#   MOI.eval_constraint_jacobian_product(nlp.eval, jv, x, v)
#   return jv
# end
#
# function NLPModels.jtprod!(nlp :: MathOptNLPModel, x :: AbstractVector, v :: AbstractVector, jtv :: AbstractVector)
#   increment!(nlp, :neval_jtprod)
#   MOI.eval_constraint_jacobian_transpose_product(nlp.eval, jtv, x, v)
#   return jtv
# end

function NLPModels.hess_structure!(nlp :: MathOptNLPModel, rows :: AbstractVector{<: Integer}, cols :: AbstractVector{<: Integer})
  rows[1:nlp.meta.nnzh] .= nlp.hrows
  cols[1:nlp.meta.nnzh] .= nlp.hcols
  return (rows, cols)
end

function NLPModels.hess_coord!(nlp :: MathOptNLPModel, x :: AbstractVector, y :: AbstractVector, vals :: AbstractVector; obj_weight :: Float64=1.0)
  increment!(nlp, :neval_hess)
  MOI.eval_hessian_lagrangian(nlp.eval, vals, x, obj_weight, y)
  return vals
end

function NLPModels.hess_coord!(nlp :: MathOptNLPModel, x :: AbstractVector, vals :: AbstractVector; obj_weight :: Float64=1.0)
  increment!(nlp, :neval_hess)
  MOI.eval_hessian_lagrangian(nlp.eval, vals, x, obj_weight, zeros(nlp.meta.ncon))
  return vals
end

function NLPModels.hess(nlp :: MathOptNLPModel, x :: AbstractVector, y :: AbstractVector; obj_weight :: Float64=1.0)
  hess_coord!(nlp, x, y, nlp.hvals, obj_weight=obj_weight)
  return sparse(nlp.hrows, nlp.hcols, nlp.hvals, nlp.meta.nvar, nlp.meta.nvar)
end

function NLPModels.hess(nlp :: MathOptNLPModel, x :: AbstractVector; obj_weight :: Float64=1.0)
  hess_coord!(nlp, x, nlp.hvals, obj_weight=obj_weight)
  return sparse(nlp.hrows, nlp.hcols, nlp.hvals, nlp.meta.nvar, nlp.meta.nvar)
end

function NLPModels.hprod!(nlp :: MathOptNLPModel, x :: AbstractVector, y :: AbstractVector, v :: AbstractVector, hv :: AbstractVector; obj_weight :: Float64=1.0)
  increment!(nlp, :neval_hprod)
  MOI.eval_hessian_lagrangian_product(nlp.eval, hv, x, v, obj_weight, y)
  return hv
end

function NLPModels.hprod!(nlp :: MathOptNLPModel, x :: AbstractVector, v :: AbstractVector, hv :: AbstractVector; obj_weight :: Float64=1.0)
  increment!(nlp, :neval_hprod)
  MOI.eval_hessian_lagrangian_product(nlp.eval, hv, x, v, obj_weight, zeros(nlp.meta.ncon))
  return hv
end
