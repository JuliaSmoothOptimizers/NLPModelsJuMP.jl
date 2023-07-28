export MathOptNLPModel

mutable struct MathOptNLPModel <: AbstractNLPModel{Float64, Vector{Float64}}
  meta::NLPModelMeta{Float64, Vector{Float64}}
  eval::MOI.Nonlinear.Evaluator
  lincon::LinearConstraints
  nlcon::NonLinearStructure
  obj::Objective
  counters::Counters
end

"""
    MathOptNLPModel(model, hessian=true, name="Generic")

Construct a `MathOptNLPModel` from a `JuMP` model.

`hessian` should be set to `false` for multivariate user-defined functions registered without hessian.
"""
function MathOptNLPModel(jmodel::JuMP.Model; kws...)
  _nlp_sync!(jmodel)
  return MathOptNLPModel(backend(jmodel); kws...)
end

function MathOptNLPModel(moimodel::MOI.ModelLike; kws...)
  return nlp_model(moimodel; kws...)[1]
end

function nlp_model(moimodel::MOI.ModelLike; hessian::Bool = true, name::String = "Generic")
  index_map, nvar, lvar, uvar, x0 = parser_variables(moimodel)
  nlin, lincon, lin_lcon, lin_ucon = parser_MOI(moimodel, index_map)

  nlp_data = _nlp_block(moimodel)
  nnln, nlcon, nl_lcon, nl_ucon = parser_NL(nlp_data, hessian = hessian)

  if nlp_data.has_objective
    obj = Objective("NONLINEAR", 0.0, spzeros(Float64, nvar), COO(), 0)
  else
    obj = parser_objective_MOI(moimodel, nvar, index_map)
  end

  ncon = nlin + nnln
  lcon = vcat(lin_lcon, nl_lcon)
  ucon = vcat(lin_ucon, nl_ucon)
  nnzj = lincon.nnzj + nlcon.nnzj
  nnzh = obj.nnzh + nlcon.nnzh

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
    nln_nnzj = nlcon.nnzj,
    minimize = MOI.get(moimodel, MOI.ObjectiveSense()) == MOI.MIN_SENSE,
    islp = (obj.type == "LINEAR") && (nnln == 0),
    name = name,
  )

  return MathOptNLPModel(meta, nlp_data.evaluator, lincon, nlcon, obj, Counters()), index_map
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
  view(rows, 1:(nlp.lincon.nnzj)) .= nlp.lincon.jacobian.rows
  view(cols, 1:(nlp.lincon.nnzj)) .= nlp.lincon.jacobian.cols
  return rows, cols
end

function NLPModels.jac_nln_structure!(
  nlp::MathOptNLPModel,
  rows::AbstractVector{<:Integer},
  cols::AbstractVector{<:Integer},
)
  view(rows, 1:(nlp.nlcon.nnzj)) .= nlp.nlcon.jac_rows
  view(cols, 1:(nlp.nlcon.nnzj)) .= nlp.nlcon.jac_cols
  return rows, cols
end

function NLPModels.jac_lin_coord!(nlp::MathOptNLPModel, x::AbstractVector, vals::AbstractVector)
  increment!(nlp, :neval_jac_lin)
  view(vals, 1:(nlp.lincon.nnzj)) .= nlp.lincon.jacobian.vals
  return vals
end

function NLPModels.jac_nln_coord!(nlp::MathOptNLPModel, x::AbstractVector, vals::AbstractVector)
  increment!(nlp, :neval_jac_nln)
  MOI.eval_constraint_jacobian(nlp.eval, view(vals, 1:(nlp.nlcon.nnzj)), x)
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
    view(rows, 1:(nlp.obj.nnzh)) .= nlp.obj.hessian.rows
    view(cols, 1:(nlp.obj.nnzh)) .= nlp.obj.hessian.cols
  end
  if (nlp.obj.type == "NONLINEAR") || (nlp.meta.nnln > 0)
    view(rows, (nlp.obj.nnzh + 1):(nlp.meta.nnzh)) .= nlp.nlcon.hess_rows
    view(cols, (nlp.obj.nnzh + 1):(nlp.meta.nnzh)) .= nlp.nlcon.hess_cols
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
    view(vals, 1:(nlp.obj.nnzh)) .= obj_weight .* nlp.obj.hessian.vals
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
    view(vals, 1:(nlp.obj.nnzh)) .= obj_weight .* nlp.obj.hessian.vals
    view(vals, (nlp.obj.nnzh + 1):(nlp.meta.nnzh)) .= 0.0
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
    coo_sym_add_mul!(
      nlp.obj.hessian.rows,
      nlp.obj.hessian.cols,
      nlp.obj.hessian.vals,
      v,
      hv,
      obj_weight,
    )
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
