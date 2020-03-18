using NLPModels
import NLPModels.increment!, NLPModels.decrement!

using JuMP, MathOptInterface
const MOI = MathOptInterface

export MathOptNLSModel

mutable struct MathOptNLSModel <: AbstractNLSModel
  meta     :: NLPModelMeta
  nls_meta :: NLSMeta
  Feval    :: Union{MOI.AbstractNLPEvaluator, Nothing}
  ceval    :: Union{MOI.AbstractNLPEvaluator, Nothing}
  counters :: NLSCounters
end

"""
    MathOptNLSModel(model, F, name="Generic")

Construct a `MathOptNLSModel` from a `JuMP` model and a vector of `NonlinearExpression`.
"""
function MathOptNLSModel(cmodel :: JuMP.Model, F :: Union{AbstractArray{JuMP.NonlinearExpression},
                                                          Array{<: AbstractArray{JuMP.NonlinearExpression}}
                                                         }; name :: String="Generic")

  F_is_array_of_containers = F isa Array{<: AbstractArray{JuMP.NonlinearExpression}}
  nvar = Int(num_variables(cmodel))
  vars = all_variables(cmodel)
  lvar = map(var -> has_lower_bound(var) ? lower_bound(var) : -Inf, vars)
  uvar = map(var -> has_upper_bound(var) ? upper_bound(var) :  Inf, vars)

  x0 = zeros(nvar)
  for (i, val) ∈ enumerate(start_value.(vars))
    if val !== nothing
      x0[i] = val
    end
  end

  ncon = num_nl_constraints(cmodel)
  cons = cmodel.nlp_data.nlconstr
  lcon = map(con -> con.lb, cons)
  ucon = map(con -> con.ub, cons)

  if F_is_array_of_containers
    @NLobjective(cmodel, Min, 0.5 * sum(sum(Fi^2 for Fi in FF) for FF in F))
  else
    @NLobjective(cmodel, Min, 0.5 * sum(Fi^2 for Fi in F))
  end
  ceval = NLPEvaluator(cmodel)
  MOI.initialize(ceval, [:Grad, :Jac, :Hess, :HessVec, :ExprGraph])  # Add :JacVec when available / :ExprGraph is only required here

  Fmodel = JuMP.Model()
  @variable(Fmodel, x[1:nvar])
  JuMP._init_NLP(Fmodel)
  @NLobjective(Fmodel, Min, 0.0)
  Fmodel.nlp_data.user_operators = cmodel.nlp_data.user_operators
  if F_is_array_of_containers
    for FF in F, Fi in FF
      expr = ceval.subexpressions_as_julia_expressions[Fi.index]
      replace!(expr, x)
      expr = :($expr == 0)
      JuMP.add_NL_constraint(Fmodel, expr)
    end
  else
    for Fi in F
      expr = ceval.subexpressions_as_julia_expressions[Fi.index]
      replace!(expr, x)
      expr = :($expr == 0)
      JuMP.add_NL_constraint(Fmodel, expr)
    end
  end

  nequ = Int(num_nl_constraints(Fmodel))
  Feval = NLPEvaluator(Fmodel)
  MOI.initialize(Feval, [:Grad, :Jac, :Hess, :HessVec])  # Add :JacVec when available

  Fnnzj = nequ == 0 ? 0 : sum(length(con.grad_sparsity) for con in Feval.constraints)
  Fnnzh = length(Feval.objective.hess_I) + (nequ == 0 ? 0 : sum(length(con.hess_I) for con in Feval.constraints))

  cnnzj = ncon == 0 ? 0 : sum(length(con.grad_sparsity) for con in ceval.constraints)
  cnnzh = length(ceval.objective.hess_I) + (ncon == 0 ? 0 : sum(length(con.hess_I) for con in ceval.constraints))

  meta = NLPModelMeta(nvar,
                      x0=x0,
                      lvar=lvar,
                      uvar=uvar,
                      ncon=ncon,
                      y0=zeros(ncon),
                      lcon=lcon,
                      ucon=ucon,
                      nnzj=cnnzj,
                      nnzh=cnnzh,
                      lin=[],
                      nln=collect(1:ncon),
                      minimize=objective_sense(cmodel) == MOI.MIN_SENSE,
                      islp=false,
                      name=name,
                     )

  return MathOptNLSModel(meta,
                         NLSMeta(nequ, nvar, nnzj=Fnnzj, nnzh=Fnnzh),
                         Feval,
                         ceval,
                         NLSCounters()
                        )
end

function NLPModels.residual!(nls :: MathOptNLSModel, x :: AbstractVector, Fx :: AbstractVector)
  increment!(nls, :neval_residual)
  MOI.eval_constraint(nls.Feval, Fx, x)
  return Fx
end

function NLPModels.jac_structure_residual!(nls :: MathOptNLSModel, rows :: AbstractVector{<: Integer}, cols :: AbstractVector{<: Integer})
  jac_struct_residual = MOI.jacobian_structure(nls.Feval)
  for index = 1 : nls.nls_meta.nnzj
    rows[index] = jac_struct_residual[index][1]
    cols[index] = jac_struct_residual[index][2]
  end
  return rows, cols
end

function NLPModels.jac_coord_residual!(nls :: MathOptNLSModel, x :: AbstractVector, vals :: AbstractVector)
  increment!(nls, :neval_jac_residual)
  MOI.eval_constraint_jacobian(nls.Feval, vals, x)
  return vals
end

function NLPModels.jprod_residual!(nls :: MathOptNLSModel, x :: AbstractVector, rows :: AbstractVector{<: Integer}, cols :: AbstractVector{<: Integer}, v :: AbstractVector, Jv :: AbstractVector)
  vals = jac_coord_residual(nls, x)
  decrement!(nls, :neval_jac_residual)
  jprod_residual!(nls, rows, cols, vals, v, Jv)
  return Jv
end

function NLPModels.jprod_residual!(nls :: MathOptNLSModel, x :: AbstractVector, v :: AbstractVector, Jv :: AbstractVector)
  rows, cols = jac_structure_residual(nls)
  jprod_residual!(nls, x, rows, cols, v, Jv)
  return Jv
end

function NLPModels.jtprod_residual!(nls :: MathOptNLSModel, x :: AbstractVector, rows :: AbstractVector{<: Integer}, cols :: AbstractVector{<: Integer}, v :: AbstractVector, Jtv :: AbstractVector)
  vals = jac_coord_residual(nls, x)
  decrement!(nls, :neval_jac_residual)
  jtprod_residual!(nls, rows, cols, vals, v, Jtv)
  return Jtv
end

function NLPModels.jtprod_residual!(nls :: MathOptNLSModel, x :: AbstractVector, v :: AbstractVector, Jtv :: AbstractVector)
  rows, cols = jac_structure_residual(nls)
  jtprod_residual!(nls, x, rows, cols, v, Jtv)
  return Jtv
end

# Uncomment when :JacVec becomes available in MOI.
#
# function NLPModels.jprod_residual!(nls :: MathOptNLSModel, x :: AbstractVector, v :: AbstractVector, Jv :: AbstractVector)
#   increment!(nls, :neval_jprod_residual)
#   MOI.eval_constraint_jacobian_product(nls.Feval, Jv, x, v)
#   return Jv
# end
#
# function NLPModels.jtprod_residual!(nls :: MathOptNLSModel, x :: AbstractVector, v :: AbstractVector, Jtv :: AbstractVector)
#   increment!(nls, :neval_jtprod_residual)
#   MOI.eval_constraint_jacobian_transpose_product(nls.Feval, Jtv, x, v)
#   return Jtv
# end

function NLPModels.hess_structure_residual!(nls :: MathOptNLSModel, rows :: AbstractVector{<: Integer}, cols :: AbstractVector{<: Integer})
  hess_struct_residual = MOI.hessian_lagrangian_structure(nls.Feval)
    for index = 1 : nls.nls_meta.nnzh
    rows[index] = hess_struct_residual[index][1]
    cols[index] = hess_struct_residual[index][2]
  end
  return rows, cols
end

function NLPModels.hess_coord_residual!(nls :: MathOptNLSModel, x :: AbstractVector, v :: AbstractVector, vals :: AbstractVector)
  increment!(nls, :neval_hess_residual)
  MOI.eval_hessian_lagrangian(nls.Feval, vals, x, 0.0, v)
  return vals
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
  jac_struct = MOI.jacobian_structure(nls.ceval)
  for index = 1 : nls.meta.nnzj
    rows[index] = jac_struct[index][1]
    cols[index] = jac_struct[index][2]
  end
  return rows, cols
end

function NLPModels.jac_coord!(nls :: MathOptNLSModel, x :: AbstractVector, vals :: AbstractVector)
  increment!(nls, :neval_jac)
  MOI.eval_constraint_jacobian(nls.ceval, vals, x)
  return vals
end

function NLPModels.jprod!(nls :: MathOptNLSModel, x :: AbstractVector, rows :: AbstractVector{<: Integer}, cols :: AbstractVector{<: Integer}, v :: AbstractVector, Jv :: AbstractVector)
  vals = jac_coord(nls, x)
  decrement!(nls, :neval_jac)
  jprod!(nls, rows, cols, vals, v, Jv)
  return Jv
end

function NLPModels.jprod!(nls :: MathOptNLSModel, x :: AbstractVector, v :: AbstractVector, Jv :: AbstractVector)
  rows, cols = jac_structure(nls)
  jprod!(nls, x, rows, cols, v, Jv)
  return Jv
end

function NLPModels.jtprod!(nls :: MathOptNLSModel, x :: AbstractVector, rows :: AbstractVector{<: Integer}, cols :: AbstractVector{<: Integer}, v :: AbstractVector, Jtv :: AbstractVector)
  vals = jac_coord(nls, x)
  decrement!(nls, :neval_jac)
  jtprod!(nls, rows, cols, vals, v, Jtv)
  return Jtv
end

function NLPModels.jtprod!(nls :: MathOptNLSModel, x :: AbstractVector, v :: AbstractVector, Jtv :: AbstractVector)
  (rows, cols) = jac_structure(nls)
  jtprod!(nls, x, rows, cols, v, Jtv)
  return Jtv
end

# Uncomment when :JacVec becomes available in MOI.
#
# function NLPModels.jprod!(nls :: MathOptNLSModel, x :: AbstractVector, v :: AbstractVector, Jv :: AbstractVector)
#   increment!(nls, :neval_jprod)
#   MOI.eval_constraint_jacobian_product(nls.ceval, Jv, x, v)
#   return Jv
# end
#
# function NLPModels.jtprod!(nls :: MathOptNLSModel, x :: AbstractVector, v :: AbstractVector, Jtv :: AbstractVector)
#   increment!(nls, :neval_jtprod)
#   MOI.eval_constraint_jacobian_transpose_product(nls.ceval, Jtv, x, v)
#   return Jtv
# end

function NLPModels.hess_structure!(nls :: MathOptNLSModel, rows :: AbstractVector{<: Integer}, cols :: AbstractVector{<: Integer})
  hesslag_struct = MOI.hessian_lagrangian_structure(nls.ceval)
  for index = 1 : nls.meta.nnzh
    rows[index] = hesslag_struct[index][1]
    cols[index] = hesslag_struct[index][2]
  end
  return rows, cols
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
