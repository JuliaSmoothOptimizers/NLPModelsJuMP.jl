using NLPModels
import NLPModels.increment!, NLPModels.decrement!

using JuMP, MathOptInterface
const MOI = MathOptInterface

export MathOptNLPModel

mutable struct MathOptNLPModel <: AbstractNLPModel
  meta     :: NLPModelMeta
  eval     :: Union{MOI.AbstractNLPEvaluator, Nothing}
  counters :: Counters
end

"""
    MathOptNLPModel(model, name="Generic")

Construct a `MathOptNLPModel` from a `JuMP` model.
"""
function MathOptNLPModel(jmodel :: JuMP.Model; name :: String="Generic")

  eval = NLPEvaluator(jmodel)
  MOI.initialize(eval, [:Grad, :Jac, :Hess, :HessVec])  # Add :JacVec when available

  nvar = num_variables(jmodel)
  vars = all_variables(jmodel)
  lvar = map(var -> has_lower_bound(var) ? lower_bound(var) : -Inf, vars)
  uvar = map(var -> has_upper_bound(var) ? upper_bound(var) :  Inf, vars)

  x0 = zeros(nvar)
  for (i, val) ∈ enumerate(start_value.(vars))
    if val !== nothing
      x0[i] = val
    end
  end

  ncon = num_nl_constraints(jmodel)
  cons = jmodel.nlp_data.nlconstr
  lcon = map(con -> con.lb, cons)
  ucon = map(con -> con.ub, cons)

  nnzj = ncon == 0 ? 0 : sum(length(con.grad_sparsity) for con in eval.constraints)
  nnzh = length(eval.objective.hess_I) + (ncon == 0 ? 0 : sum(length(con.hess_I) for con in eval.constraints))

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
                      islp=!eval.has_nlobj,
                      name=name,
                      )

  return MathOptNLPModel(meta, eval, Counters())
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
  jac_struct = MOI.jacobian_structure(nlp.eval)
  for index = 1 : nlp.meta.nnzj
    rows[index] = jac_struct[index][1]
    cols[index] = jac_struct[index][2]
  end
  return rows, cols
end

function NLPModels.jac_coord!(nlp :: MathOptNLPModel, x :: AbstractVector, vals :: AbstractVector)
  increment!(nlp, :neval_jac)
  MOI.eval_constraint_jacobian(nlp.eval, vals, x)
  return vals
end

function NLPModels.jprod!(nlp :: MathOptNLPModel, x :: AbstractVector, rows :: AbstractVector{<: Integer}, cols :: AbstractVector{<: Integer}, v :: AbstractVector, Jv :: AbstractVector)
  vals = jac_coord(nlp, x)
  decrement!(nlp, :neval_jac)
  jprod!(nlp, rows, cols, vals, v, Jv)
  return Jv
end

function NLPModels.jprod!(nlp :: MathOptNLPModel, x :: AbstractVector, v :: AbstractVector, Jv :: AbstractVector)
  rows, cols = jac_structure(nlp)
  jprod!(nlp, x, rows, cols, v, Jv)
  return Jv
end

function NLPModels.jtprod!(nlp :: MathOptNLPModel, x :: AbstractVector, rows :: AbstractVector{<: Integer}, cols :: AbstractVector{<: Integer}, v :: AbstractVector, Jtv :: AbstractVector)
  vals = jac_coord(nlp, x)
  decrement!(nlp, :neval_jac)
  jtprod!(nlp, rows, cols, vals, v, Jtv)
  return Jtv
end

function NLPModels.jtprod!(nlp :: MathOptNLPModel, x :: AbstractVector, v :: AbstractVector, Jtv :: AbstractVector)
  (rows, cols) = jac_structure(nlp)
  jtprod!(nlp, x, rows, cols, v, Jtv)
  return Jtv
end

# Uncomment when :JacVec becomes available in MOI.
#
# function NLPModels.jprod!(nlp :: MathOptNLPModel, x :: AbstractVector, v :: AbstractVector, Jv :: AbstractVector)
#   increment!(nlp, :neval_jprod)
#   MOI.eval_constraint_jacobian_product(nlp.eval, Jv, x, v)
#   return Jv
# end
#
# function NLPModels.jtprod!(nlp :: MathOptNLPModel, x :: AbstractVector, v :: AbstractVector, Jtv :: AbstractVector)
#   increment!(nlp, :neval_jtprod)
#   MOI.eval_constraint_jacobian_transpose_product(nlp.eval, Jtv, x, v)
#   return Jtv
# end

function NLPModels.hess_structure!(nlp :: MathOptNLPModel, rows :: AbstractVector{<: Integer}, cols :: AbstractVector{<: Integer})
  hesslag_struct = MOI.hessian_lagrangian_structure(nlp.eval)
  for index = 1 : nlp.meta.nnzh
    rows[index] = hesslag_struct[index][1]
    cols[index] = hesslag_struct[index][2]
  end
  return rows, cols
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
