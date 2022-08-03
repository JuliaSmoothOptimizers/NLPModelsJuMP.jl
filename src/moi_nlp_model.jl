export MathOptNLPModel

mutable struct MathOptNLPModel <: AbstractNLPModel{Float64, Vector{Float64}}
  meta::NLPModelMeta{Float64, Vector{Float64}}
  eval::Union{MOI.AbstractNLPEvaluator, Nothing}
  lincon::LinearConstraints
  obj::Objective
  counters::Counters
end

"""
    MathOptNLPModel(model, hessian=true, name="Generic")

Construct a `MathOptNLPModel` from a `JuMP` model.

`hessian` should be set to `false` for multivariate user-defined functions registered without hessian.
"""
function MathOptNLPModel(jmodel::JuMP.Model; hessian::Bool = true, name::String = "Generic")
  nvar, lvar, uvar, x0 = parser_JuMP(jmodel)

  nnln = num_nonlinear_constraints(jmodel)

  nl_lcon = nnln == 0 ? Float64[] : map(nl_con -> nl_con.lb, jmodel.nlp_data.nlconstr)
  nl_ucon = nnln == 0 ? Float64[] : map(nl_con -> nl_con.ub, jmodel.nlp_data.nlconstr)

  eval = jmodel.nlp_data == nothing ? nothing : NLPEvaluator(jmodel)
  (eval ≠ nothing) &&
    MOI.initialize(eval, hessian ? [:Grad, :Jac, :JacVec, :Hess, :HessVec] : [:Grad, :Jac, :JacVec])

  nl_nnzj = nnln == 0 ? 0 : sum(length(nl_con.grad_sparsity) for nl_con in eval.constraints)
  nl_nnzh =
    hessian ?
    (((eval ≠ nothing) && eval.has_nlobj) ? length(eval.objective.hess_I) : 0) +
    (nnln == 0 ? 0 : sum(length(nl_con.hess_I) for nl_con in eval.constraints)) : 0

  moimodel = backend(jmodel)
  nlin, lincon, lin_lcon, lin_ucon = parser_MOI(moimodel)

  if (eval ≠ nothing) && eval.has_nlobj
    obj = Objective("NONLINEAR", 0.0, spzeros(Float64, nvar), COO(), 0)
  else
    obj = parser_objective_MOI(moimodel, nvar)
  end

  ncon = nlin + nnln
  lcon = vcat(lin_lcon, nl_lcon)
  ucon = vcat(lin_ucon, nl_ucon)
  nnzj = lincon.nnzj + nl_nnzj
  nnzh = obj.nnzh + nl_nnzh

  meta = NLPModelMeta(
    nvar,
    x0 = x0,
    lvar = lvar,
    uvar = uvar,
    ncon = ncon,
    y0 = zeros(ncon),
    lcon = lcon,
    ucon = ucon,
    nnzj = nnzj,
    nnzh = nnzh,
    lin = collect(1:nlin),
    lin_nnzj = lincon.nnzj,
    nln_nnzj = nl_nnzj,
    minimize = objective_sense(jmodel) == MOI.MIN_SENSE,
    islp = (obj.type == "LINEAR") && (nnln == 0),
    name = name,
  )

  return MathOptNLPModel(meta, eval, lincon, obj, Counters())
end

function NLPModels.obj(nlp::MathOptNLPModel, x::AbstractVector)
  increment!(nlp, :neval_obj)
  if nlp.obj.type == "LINEAR"
    res = dot(nlp.obj.gradient, x) + nlp.obj.constant
  end
  if nlp.obj.type == "QUADRATIC"
    res =
      0.5 * coo_sym_dot(nlp.obj.hessian.rows, nlp.obj.hessian.cols, nlp.obj.hessian.vals, x, x) +
      dot(nlp.obj.gradient, x) +
      nlp.obj.constant
  end
  if nlp.obj.type == "NONLINEAR"
    res = MOI.eval_objective(nlp.eval, x)
  end
  return res
end

function NLPModels.grad!(nlp::MathOptNLPModel, x::AbstractVector, g::AbstractVector)
  increment!(nlp, :neval_grad)
  if nlp.obj.type == "LINEAR"
    g .= nlp.obj.gradient
  end
  if nlp.obj.type == "QUADRATIC"
    coo_sym_prod!(nlp.obj.hessian.rows, nlp.obj.hessian.cols, nlp.obj.hessian.vals, x, g)
    g .+= nlp.obj.gradient
  end
  if nlp.obj.type == "NONLINEAR"
    MOI.eval_objective_gradient(nlp.eval, g, x)
  end
  return g
end

function NLPModels.cons_lin!(nlp::MathOptNLPModel, x::AbstractVector, c::AbstractVector)
  increment!(nlp, :neval_cons_lin)
  coo_prod!(nlp.lincon.jacobian.rows, nlp.lincon.jacobian.cols, nlp.lincon.jacobian.vals, x, c)
  return c
end

function NLPModels.cons_nln!(nlp::MathOptNLPModel, x::AbstractVector, c::AbstractVector)
  increment!(nlp, :neval_cons_nln)
  MOI.eval_constraint(nlp.eval, c, x)
  return c
end

function NLPModels.jac_lin_structure!(
  nlp::MathOptNLPModel,
  rows::AbstractVector{<:Integer},
  cols::AbstractVector{<:Integer},
)
  for index = 1:(nlp.lincon.nnzj)
    rows[index] = nlp.lincon.jacobian.rows[index]
    cols[index] = nlp.lincon.jacobian.cols[index]
  end
  return rows, cols
end

function NLPModels.jac_nln_structure!(
  nlp::MathOptNLPModel,
  rows::AbstractVector{<:Integer},
  cols::AbstractVector{<:Integer},
)
  jac_struct = MOI.jacobian_structure(nlp.eval)
  for index = 1:(nlp.meta.nln_nnzj)
    row, col = jac_struct[index]
    rows[index] = row
    cols[index] = col
  end
  return rows, cols
end

function NLPModels.jac_lin_coord!(nlp::MathOptNLPModel, x::AbstractVector, vals::AbstractVector)
  increment!(nlp, :neval_jac_lin)
  for index = 1:(nlp.lincon.nnzj)
    vals[index] = nlp.lincon.jacobian.vals[index]
  end
  return vals
end

function NLPModels.jac_nln_coord!(nlp::MathOptNLPModel, x::AbstractVector, vals::AbstractVector)
  increment!(nlp, :neval_jac_nln)
  MOI.eval_constraint_jacobian(nlp.eval, vals, x)
  return vals
end

function NLPModels.jprod_lin!(
  nlp::MathOptNLPModel,
  x::AbstractVector,
  v::AbstractVector,
  Jv::AbstractVector,
)
  jprod_lin!(
    nlp,
    nlp.lincon.jacobian.rows,
    nlp.lincon.jacobian.cols,
    nlp.lincon.jacobian.vals,
    v,
    Jv,
  )
  return Jv
end

function NLPModels.jprod_nln!(
  nlp::MathOptNLPModel,
  x::AbstractVector,
  v::AbstractVector,
  Jv::AbstractVector,
)
  increment!(nlp, :neval_jprod_nln)
  MOI.eval_constraint_jacobian_product(nlp.eval, Jv, x, v)
  return Jv
end

function NLPModels.jtprod_lin!(
  nlp::MathOptNLPModel,
  x::AbstractVector,
  v::AbstractVector,
  Jtv::AbstractVector,
)
  jtprod_lin!(
    nlp,
    nlp.lincon.jacobian.rows,
    nlp.lincon.jacobian.cols,
    nlp.lincon.jacobian.vals,
    v,
    Jtv,
  )
  return Jtv
end

function NLPModels.jtprod_nln!(
  nlp::MathOptNLPModel,
  x::AbstractVector,
  v::AbstractVector,
  Jtv::AbstractVector,
)
  increment!(nlp, :neval_jtprod_nln)
  MOI.eval_constraint_jacobian_transpose_product(nlp.eval, Jtv, x, v)
  return Jtv
end

function NLPModels.hess_structure!(
  nlp::MathOptNLPModel,
  rows::AbstractVector{<:Integer},
  cols::AbstractVector{<:Integer},
)
  if nlp.obj.type == "QUADRATIC"
    for index = 1:(nlp.obj.nnzh)
      rows[index] = nlp.obj.hessian.rows[index]
      cols[index] = nlp.obj.hessian.cols[index]
    end
  end
  if (nlp.obj.type == "NONLINEAR") || (nlp.meta.nnln > 0)
    hesslag_struct = MOI.hessian_lagrangian_structure(nlp.eval)
    for index = (nlp.obj.nnzh + 1):(nlp.meta.nnzh)
      shift_index = index - nlp.obj.nnzh
      rows[index] = hesslag_struct[shift_index][1]
      cols[index] = hesslag_struct[shift_index][2]
    end
  end
  return rows, cols
end

function NLPModels.hess_coord!(
  nlp::MathOptNLPModel,
  x::AbstractVector,
  y::AbstractVector,
  vals::AbstractVector;
  obj_weight::Float64 = 1.0,
)
  increment!(nlp, :neval_hess)
  if nlp.obj.type == "QUADRATIC"
    for index = 1:(nlp.obj.nnzh)
      vals[index] = obj_weight * nlp.obj.hessian.vals[index]
    end
  end
  if (nlp.obj.type == "NONLINEAR") || (nlp.meta.nnln > 0)
    MOI.eval_hessian_lagrangian(
      nlp.eval,
      view(vals, (nlp.obj.nnzh + 1):(nlp.meta.nnzh)),
      x,
      obj_weight,
      view(y, nlp.meta.nln),
    )
  end
  return vals
end

function NLPModels.hess_coord!(
  nlp::MathOptNLPModel,
  x::AbstractVector,
  vals::AbstractVector;
  obj_weight::Float64 = 1.0,
)
  increment!(nlp, :neval_hess)
  if nlp.obj.type == "LINEAR"
    vals .= 0.0
  end
  if nlp.obj.type == "QUADRATIC"
    for index = 1:(nlp.obj.nnzh)
      vals[index] = obj_weight * nlp.obj.hessian.vals[index]
    end
    for index = (nlp.obj.nnzh + 1):(nlp.meta.nnzh)
      vals[index] = 0.0
    end
  end
  if nlp.obj.type == "NONLINEAR"
    MOI.eval_hessian_lagrangian(nlp.eval, vals, x, obj_weight, zeros(nlp.meta.nnln))
  end
  return vals
end

function NLPModels.hprod!(
  nlp::MathOptNLPModel,
  x::AbstractVector,
  y::AbstractVector,
  v::AbstractVector,
  hv::AbstractVector;
  obj_weight::Float64 = 1.0,
)
  increment!(nlp, :neval_hprod)
  if (nlp.obj.type == "LINEAR") && (nlp.meta.nnln == 0)
    hv .= 0.0
  end
  if (nlp.obj.type == "NONLINEAR") || (nlp.meta.nnln > 0)
    MOI.eval_hessian_lagrangian_product(nlp.eval, hv, x, v, obj_weight, view(y, nlp.meta.nln))
  end
  if nlp.obj.type == "QUADRATIC"
    nlp.meta.nnln == 0 && (hv .= 0.0)
    for k = 1:(nlp.obj.nnzh)
      i, j, c = nlp.obj.hessian.rows[k], nlp.obj.hessian.cols[k], nlp.obj.hessian.vals[k]
      hv[i] += obj_weight * c * v[j]
      if i ≠ j
        hv[j] += obj_weight * c * v[i]
      end
    end
  end
  return hv
end

function NLPModels.hprod!(
  nlp::MathOptNLPModel,
  x::AbstractVector,
  v::AbstractVector,
  hv::AbstractVector;
  obj_weight::Float64 = 1.0,
)
  increment!(nlp, :neval_hprod)
  if nlp.obj.type == "LINEAR"
    hv .= 0.0
  end
  if nlp.obj.type == "QUADRATIC"
    coo_sym_prod!(nlp.obj.hessian.rows, nlp.obj.hessian.cols, nlp.obj.hessian.vals, v, hv)
    hv .*= obj_weight
  end
  if nlp.obj.type == "NONLINEAR"
    MOI.eval_hessian_lagrangian_product(nlp.eval, hv, x, v, obj_weight, zeros(nlp.meta.nnln))
  end
  return hv
end
