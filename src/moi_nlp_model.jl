export MathOptNLPModel

mutable struct MathOptNLPModel <: AbstractNLPModel
  meta     :: NLPModelMeta
  eval     :: Union{MOI.AbstractNLPEvaluator, Nothing}
  lincon   :: LinearConstraints
  linobj   :: LinearObjective
  counters :: Counters
end

"""
    MathOptNLPModel(model, name="Generic")

Construct a `MathOptNLPModel` from a `JuMP` model.
"""
function MathOptNLPModel(jmodel :: JuMP.Model; name :: String="Generic")

  nvar, lvar, uvar, x0, nnln, nl_lcon, nl_ucon = parser_JuMP(jmodel)

  eval = NLPEvaluator(jmodel)
  MOI.initialize(eval, [:Grad, :Jac, :Hess, :HessVec])  # Add :JacVec when available

  nl_nnzj = nnln == 0 ? 0 : sum(length(nl_con.grad_sparsity) for nl_con in eval.constraints)
  nl_nnzh = (eval.has_nlobj ? length(eval.objective.hess_I) : 0) + (nnln == 0 ? 0 : sum(length(nl_con.hess_I) for nl_con in eval.constraints))

  moimodel = backend(jmodel)
  nlin, linrows, lincols, linvals, lin_lcon, lin_ucon = parser_MOI(moimodel)
  constant, gradient = parser_obj_MOI(moimodel, nvar)

  lin_nnzj = length(linvals)
  lincon = LinearConstraints(linrows, lincols, linvals)
  linobj = LinearObjective(constant, gradient)

  ncon = nlin + nnln
  lcon = vcat(lin_lcon, nl_lcon)
  ucon = vcat(lin_ucon, nl_ucon)
  nnzj = lin_nnzj + nl_nnzj
  nnzh = nl_nnzh

  meta = NLPModelMeta(nvar,
                      x0=x0,
                      lvar=lvar,
                      uvar=uvar,
                      ncon=ncon,
                      nlin=nlin,
                      nnln=nnln,
                      y0=zeros(ncon),
                      lcon=lcon,
                      ucon=ucon,
                      nnzj=nnzj,
                      nnzh=nnzh,
                      lin=collect(1:nlin),
                      nln=collect(nlin+1:ncon),
                      minimize=objective_sense(jmodel) == MOI.MIN_SENSE,
                      islp=!eval.has_nlobj && (nnln == 0),
                      name=name,
                      )

  return MathOptNLPModel(meta, eval, lincon, linobj, Counters())
end

function NLPModels.obj(nlp :: MathOptNLPModel, x :: AbstractVector)
  increment!(nlp, :neval_obj)
  return nlp.eval.has_nlobj ? MOI.eval_objective(nlp.eval, x) : (dot(nlp.linobj.gradient, x) + nlp.linobj.constant)
end

function NLPModels.grad!(nlp :: MathOptNLPModel, x :: AbstractVector, g :: AbstractVector)
  increment!(nlp, :neval_grad)
  if nlp.eval.has_nlobj
    MOI.eval_objective_gradient(nlp.eval, g, x)
  else
    g .= nlp.linobj.gradient
  end
  return g
end

function NLPModels.cons!(nlp :: MathOptNLPModel, x :: AbstractVector, c :: AbstractVector)
  increment!(nlp, :neval_cons)
  if nlp.meta.nlin > 0
    coo_prod!(nlp.lincon.rows, nlp.lincon.cols, nlp.lincon.vals, x, view(c, nlp.meta.lin))
  end
  if nlp.meta.nnln > 0
    MOI.eval_constraint(nlp.eval, view(c, nlp.meta.nln), x)
  end
  return c
end

function NLPModels.jac_structure!(nlp :: MathOptNLPModel, rows :: AbstractVector{<: Integer}, cols :: AbstractVector{<: Integer})
  lin_nnzj = length(nlp.lincon.vals)
  if nlp.meta.nlin > 0
    rows[1:lin_nnzj] .= nlp.lincon.rows[1:lin_nnzj]
    cols[1:lin_nnzj] .= nlp.lincon.cols[1:lin_nnzj]
  end
  if nlp.meta.nnln > 0
    jac_struct = MOI.jacobian_structure(nlp.eval)
    for index = lin_nnzj+1 : nlp.meta.nnzj
      row, col = jac_struct[index - lin_nnzj]
      rows[index] = nlp.meta.nlin + row
      cols[index] = col
    end
  end
  return rows, cols
end

function NLPModels.jac_coord!(nlp :: MathOptNLPModel, x :: AbstractVector, vals :: AbstractVector)
  increment!(nlp, :neval_jac)
  lin_nnzj = length(nlp.lincon.vals)
  if nlp.meta.nlin > 0
    vals[1:lin_nnzj] .= nlp.lincon.vals[1:lin_nnzj]
  end
  if nlp.meta.nnln > 0
    MOI.eval_constraint_jacobian(nlp.eval, view(vals, lin_nnzj+1:nlp.meta.nnzj), x)
  end
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
  if nlp.meta.nnzh > 0
    hesslag_struct = MOI.hessian_lagrangian_structure(nlp.eval)
    for index = 1 : nlp.meta.nnzh
      rows[index] = hesslag_struct[index][1]
      cols[index] = hesslag_struct[index][2]
    end
  end
  return rows, cols
end

function NLPModels.hess_coord!(nlp :: MathOptNLPModel, x :: AbstractVector, y :: AbstractVector, vals :: AbstractVector; obj_weight :: Float64=1.0)
  increment!(nlp, :neval_hess)
  if nlp.meta.nnzh > 0
    MOI.eval_hessian_lagrangian(nlp.eval, vals, x, obj_weight, view(y, nlp.meta.nln))
  end
  return vals
end

function NLPModels.hess_coord!(nlp :: MathOptNLPModel, x :: AbstractVector, vals :: AbstractVector; obj_weight :: Float64=1.0)
  increment!(nlp, :neval_hess)
  if nlp.eval.has_nlobj
    MOI.eval_hessian_lagrangian(nlp.eval, vals, x, obj_weight, zeros(nlp.meta.nnln))
  else
    vals .= 0.0
  end
  return vals
end

function NLPModels.hprod!(nlp :: MathOptNLPModel, x :: AbstractVector, y :: AbstractVector, v :: AbstractVector, hv :: AbstractVector; obj_weight :: Float64=1.0)
  increment!(nlp, :neval_hprod)
  MOI.eval_hessian_lagrangian_product(nlp.eval, hv, x, v, obj_weight, view(y, nlp.meta.nln))
  return hv
end

function NLPModels.hprod!(nlp :: MathOptNLPModel, x :: AbstractVector, v :: AbstractVector, hv :: AbstractVector; obj_weight :: Float64=1.0)
  increment!(nlp, :neval_hprod)
  MOI.eval_hessian_lagrangian_product(nlp.eval, hv, x, v, obj_weight, zeros(nlp.meta.nnln))
  return hv
end
