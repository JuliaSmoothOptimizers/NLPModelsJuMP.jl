export MathOptNLPModel

mutable struct MathOptNLPModel <: AbstractNLPModel{Float64, Vector{Float64}}
  meta::NLPModelMeta{Float64, Vector{Float64}}
  eval::MOI.Nonlinear.Evaluator
  lincon::LinearConstraints
  quadcon::QuadraticConstraints
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
  nlin, lincon, lin_lcon, lin_ucon, quadcon, quad_lcon, quad_ucon = parser_MOI(moimodel, index_map, nvar)

  nlp_data = _nlp_block(moimodel)
  nnln, nlcon, nl_lcon, nl_ucon = parser_NL(nlp_data, hessian = hessian)

  if nlp_data.has_objective
    obj = Objective("NONLINEAR", 0.0, spzeros(Float64, nvar), COO(), 0)
  else
    obj = parser_objective_MOI(moimodel, nvar, index_map)
  end

  ncon = nlin + quadcon.nquad + nnln
  lcon = vcat(lin_lcon, quad_lcon, nl_lcon)
  ucon = vcat(lin_ucon, quad_ucon, nl_ucon)
  nnzj = lincon.nnzj + quadcon.nnzj + nlcon.nnzj
  nnzh = obj.nnzh + quadcon.nnzh + nlcon.nnzh

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
    nln_nnzj = quadcon.nnzj + nlcon.nnzj,
    minimize = MOI.get(moimodel, MOI.ObjectiveSense()) == MOI.MIN_SENSE,
    islp = (obj.type == "LINEAR") && (nnln == 0) && (quadcon.nquad == 0),
    name = name,
  )

  return MathOptNLPModel(meta, nlp_data.evaluator, lincon, quadcon, nlcon, obj, Counters()), index_map
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
    g .= nlp.obj.gradient
    coo_sym_add_mul!(nlp.obj.hessian.rows, nlp.obj.hessian.cols, nlp.obj.hessian.vals, x, g, 1.0)
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
  for i = 1:(nlp.quadcon.nquad)
    qcon = nlp.quadcon.constraints[i]
    c[i] = 0.5 * coo_sym_dot(qcon.A.rows, qcon.A.cols, qcon.A.vals, x, x) + dot(qcon.b, x)
  end
  if nlp.meta.nnln > nlp.quadcon.nquad
    MOI.eval_constraint(nlp.eval, view(c, (nlp.quadcon.nquad + 1):(nlp.meta.nnln)), x)
  end
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
  index = 0
  if nlp.quadcon.nquad > 0
    for i = 1:(nlp.quadcon.nquad)
      qcon = nlp.quadcon.constraints[i]
      view(rows, index+1:index+qcon.nnzg) .= i
      view(cols, index+1:index+qcon.nnzg) .= qcon.g
      index += qcon.nnzg
    end
  end
  if nlp.meta.nnln > nlp.quadcon.nquad
    ind_nnln = (nlp.quadcon.nnzj + 1):(nlp.quadcon.nnzj + nlp.nlcon.nnzj)
    view(rows, ind_nnln) .= nlp.nlcon.jac_rows
    view(cols, ind_nnln) .= nlp.nlcon.jac_cols
  end
  return rows, cols
end

function NLPModels.jac_lin_coord!(nlp::MathOptNLPModel, x::AbstractVector, vals::AbstractVector)
  increment!(nlp, :neval_jac_lin)
  view(vals, 1:(nlp.lincon.nnzj)) .= nlp.lincon.jacobian.vals
  return vals
end

function NLPModels.jac_nln_coord!(nlp::MathOptNLPModel, x::AbstractVector, vals::AbstractVector)
  increment!(nlp, :neval_jac_nln)
  index = 0
  if nlp.quadcon.nquad > 0
    view(vals, 1:nlp.quadcon.nnzj) .= 0.0
    for i = 1:(nlp.quadcon.nquad)
      qcon = nlp.quadcon.constraints[i]
      for (j, ind) in enumerate(qcon.b.nzind)
        k = qcon.dg[ind]
        vals[index+k] += qcon.b.nzval[j]
      end
      for j = 1:qcon.nnzh
        row = qcon.A.rows[j]
        col = qcon.A.cols[j]
        val = qcon.A.vals[j]
        k1 = qcon.dg[row]
        vals[index+k1] += val * x[col]
        if row != col
          k2 = qcon.dg[col]
          vals[index+k2] += val * x[row]
        end
      end
      index += qcon.nnzg
    end
  end
  if nlp.meta.nnln > nlp.quadcon.nquad
    ind_nnln = (nlp.quadcon.nnzj + 1):(nlp.quadcon.nnzj + nlp.nlcon.nnzj)
    MOI.eval_constraint_jacobian(nlp.eval, view(vals, ind_nnln), x)
  end
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
  if nlp.meta.nnln > nlp.quadcon.nquad
    ind_nnln = (nlp.quadcon.nquad + 1):(nlp.meta.nnln)
    MOI.eval_constraint_jacobian_product(nlp.eval, view(Jv, ind_nnln), x, v)
  end
  (nlp.meta.nnln == nlp.quadcon.nquad) && (Jv .= 0.0)
  if nlp.quadcon.nquad > 0
    for i = 1:(nlp.quadcon.nquad)
      # Jv[i] = (Aᵢ * x + bᵢ)ᵀ * v
      qcon = nlp.quadcon.constraints[i]
      v[i] += coo_sym_dot(qcon.A.rows, qcon.A.cols, qcon.A.vals, x, v) + dot(qcon.b, v)
    end
  end
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
  if nlp.meta.nnln > nlp.quadcon.nquad
    ind_nnln = (nlp.quadcon.nquad + 1):(nlp.meta.nnln)
    MOI.eval_constraint_jacobian_transpose_product(nlp.eval, Jtv, x, view(v, ind_nnln))
  end
  (nlp.meta.nnln == nlp.quadcon.nquad) && (Jtv .= 0.0)
  if nlp.quadcon.nquad > 0
    for i = 1:(nlp.quadcon.nquad)
      # Jtv += v[i] * (Aᵢ * x + bᵢ)
      qcon = nlp.quadcon.constraints[i]
      coo_sym_add_mul!(rows, cols, vals, x, Jtv, v[i])
      Jtv .+= v[i] .* qcon.b
    end
  end
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
  if (nlp.obj.type == "NONLINEAR") || (nlp.meta.nnln > nlp.quadcon.nquad)
    view(rows, (nlp.obj.nnzh + nlp.quadcon.nnzh + 1):(nlp.meta.nnzh)) .= nlp.nlcon.hess_rows
    view(cols, (nlp.obj.nnzh + nlp.quadcon.nnzh + 1):(nlp.meta.nnzh)) .= nlp.nlcon.hess_cols
  end
  if nlp.quadcon.nquad > 0
    index = nlp.obj.nnzh
    for i = 1:(nlp.quadcon.nquad)
      qcon = nlp.quadcon.constraints[i]
      view(rows, (index + 1):(index + qcon.nnzh)) .= qcon.A.rows
      view(cols, (index + 1):(index + qcon.nnzh)) .= qcon.A.cols
      index += qcon.nnzh
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
    view(vals, 1:(nlp.obj.nnzh)) .= obj_weight .* nlp.obj.hessian.vals
  end
  if (nlp.obj.type == "NONLINEAR") || (nlp.meta.nnln > nlp.quadcon.nquad)
    MOI.eval_hessian_lagrangian(nlp.eval,
      view(vals, (nlp.obj.nnzh + nlp.quadcon.nnzh + 1):(nlp.meta.nnzh)),
      x,
      obj_weight,
      view(y, (nlp.meta.nlin + nlp.quadcon.nquad + 1):(nlp.meta.nnln))
    )
  end
  if nlp.quadcon.nquad > 0
    index = nlp.obj.nnzh
    for i = 1:(nlp.quadcon.nquad)
      qcon = nlp.quadcon.constraints[i]
      view(vals, (index + 1):(index + qcon.nnzh)) .= y[i] .* qcon.A.vals
      index += qcon.nnzh
    end
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
    λ = zeros(nlp.meta.nnln - nlp.quadcon.nquad)  # Should be stored in the structure MathOptNLPModel
    MOI.eval_hessian_lagrangian(nlp.eval, vals, x, obj_weight, λ)
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
  if (nlp.obj.type == "NONLINEAR") || (nlp.meta.nnln > nlp.quadcon.nquad)
    λ = view(y, (nlp.meta.nlin + nlp.quadcon.nquad + 1):(nlp.meta.ncon))
    MOI.eval_hessian_lagrangian_product(nlp.eval, hv, x, v, obj_weight, λ)
  end
  if nlp.obj.type == "QUADRATIC"
    (nlp.meta.nnln == nlp.quadcon.nquad) && (hv .= 0.0)
    coo_sym_add_mul!(nlp.obj.hessian.rows, nlp.obj.hessian.cols, nlp.obj.hessian.vals, v, hv, obj_weight)
  end
  if nlp.quadcon.nquad > 0
    (nlp.obj.type == "LINEAR") && (hv .= 0.0)
    for i = 1:(nlp.quadcon.nquad)
      qcon = nlp.quadcon.constraints[i]
      coo_sym_add_mul!(qcon.A.rows, qcon.A.cols, qcon.A.vals, v, hv, y[nlp.meta.nlin + i])
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
    hv .= 0.0
    coo_sym_add_mul!(nlp.obj.hessian.rows, nlp.obj.hessian.cols, nlp.obj.hessian.vals, v, hv, obj_weight)
  end
  if nlp.obj.type == "NONLINEAR"
    λ = zeros(nlp.meta.nnln - nlp.quadcon.nquad)  # Should be stored in the structure MathOptNLPModel
    MOI.eval_hessian_lagrangian_product(nlp.eval, hv, x, v, obj_weight, λ)
  end
  return hv
end
