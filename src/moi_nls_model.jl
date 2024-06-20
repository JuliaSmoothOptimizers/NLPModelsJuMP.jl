export MathOptNLSModel

mutable struct MathOptNLSModel <: AbstractNLSModel{Float64, Vector{Float64}}
  meta::NLPModelMeta{Float64, Vector{Float64}}
  nls_meta::NLSMeta{Float64, Vector{Float64}}
  Feval::MOI.Nonlinear.Evaluator
  ceval::MOI.Nonlinear.Evaluator
  lls::Objective
  linequ::LinearEquations
  nlequ::NonLinearStructure
  lincon::LinearConstraints
  quadcon::QuadraticConstraints
  nlcon::NonLinearStructure
  counters::NLSCounters
end

"""
    MathOptNLSModel(model, F, hessian=true, name="Generic")

Construct a `MathOptNLSModel` from a `JuMP` model and a container of JuMP
`GenericAffExpr` (generated by @expression) and `NonlinearExpression` (generated by @NLexpression).

`hessian` should be set to `false` for multivariate user-defined functions registered without hessian.
"""
function MathOptNLSModel(cmodel::JuMP.Model, F; hessian::Bool = true, name::String = "Generic")
  moimodel = backend(cmodel)
  index_map, nvar, lvar, uvar, x0 = parser_variables(moimodel)

  lls, linequ, nlinequ = parser_linear_expression(cmodel, nvar, index_map, F)
  Feval, nlequ, nnlnequ = parser_nonlinear_expression(cmodel, nvar, F, hessian = hessian)

  _nlp_sync!(cmodel)
  moimodel = backend(cmodel)
  nlin, lincon, lin_lcon, lin_ucon, quadcon, quad_lcon, quad_ucon = parser_MOI(moimodel, index_map, nvar)

  nlp_data = _nlp_block(moimodel)
  nnln, nlcon, nl_lcon, nl_ucon = parser_NL(nlp_data, hessian = hessian)

  nequ = nlinequ + nnlnequ
  Fnnzj = linequ.nnzj + nlequ.nnzj
  Fnnzh = nlequ.nnzh

  ncon = nlin + quadcon.nquad + nnln
  lcon = vcat(lin_lcon, quad_lcon, nl_lcon)
  ucon = vcat(lin_ucon, quad_ucon, nl_ucon)
  cnnzj = lincon.nnzj + quadcon.nnzj + nlcon.nnzj
  cnnzh = lls.nnzh + quadcon.nnzh + nlcon.nnzh

  meta = NLPModelMeta(
    nvar,
    x0 = x0,
    lvar = lvar,
    uvar = uvar,
    ncon = ncon,
    y0 = zeros(ncon),
    lcon = lcon,
    ucon = ucon,
    nnzj = cnnzj,
    nnzh = cnnzh,
    lin = collect(1:nlin),
    lin_nnzj = lincon.nnzj,
    nln_nnzj = quadcon.nnzj + nlcon.nnzj,
    minimize = objective_sense(cmodel) == MOI.MIN_SENSE,
    islp = false,
    name = name,
  )

  return MathOptNLSModel(
    meta,
    NLSMeta(nequ, nvar, nnzj = Fnnzj, nnzh = Fnnzh, lin = collect(1:nlinequ)),
    Feval,
    nlp_data.evaluator,
    lls,
    linequ,
    nlequ,
    lincon,
    quadcon,
    nlcon,
    NLSCounters(),
  )
end

function NLPModels.residual!(nls::MathOptNLSModel, x::AbstractVector, Fx::AbstractVector)
  increment!(nls, :neval_residual)
  if nls.nls_meta.nlin > 0
    coo_prod!(
      nls.linequ.jacobian.rows,
      nls.linequ.jacobian.cols,
      nls.linequ.jacobian.vals,
      x,
      view(Fx, nls.nls_meta.lin),
    )
    view(Fx, nls.nls_meta.lin) .+= nls.linequ.constants
  end
  if nls.nls_meta.nnln > 0
    MOI.eval_constraint(nls.Feval, view(Fx, nls.nls_meta.nln), x)
  end
  return Fx
end

function NLPModels.jac_structure_residual!(
  nls::MathOptNLSModel,
  rows::AbstractVector{<:Integer},
  cols::AbstractVector{<:Integer},
)
  if nls.nls_meta.nlin > 0
    view(rows, 1:(nls.linequ.nnzj)) .= nls.linequ.jacobian.rows
    view(cols, 1:(nls.linequ.nnzj)) .= nls.linequ.jacobian.cols
  end
  if nls.nls_meta.nnln > 0
    view(rows, (nls.linequ.nnzj + 1):(nls.nls_meta.nnzj)) .= nls.nlequ.jac_rows .+ nls.nls_meta.nlin
    view(cols, (nls.linequ.nnzj + 1):(nls.nls_meta.nnzj)) .= nls.nlequ.jac_cols
    jac_struct_residual = MOI.jacobian_structure(nls.Feval)
  end
  return rows, cols
end

function NLPModels.jac_coord_residual!(
  nls::MathOptNLSModel,
  x::AbstractVector,
  vals::AbstractVector,
)
  increment!(nls, :neval_jac_residual)
  if nls.nls_meta.nlin > 0
    view(vals, 1:(nls.linequ.nnzj)) .= nls.linequ.jacobian.vals
  end
  if nls.nls_meta.nnln > 0
    MOI.eval_constraint_jacobian(
      nls.Feval,
      view(vals, (nls.linequ.nnzj + 1):(nls.nls_meta.nnzj)),
      x,
    )
  end
  return vals
end

function NLPModels.jprod_residual!(
  nls::MathOptNLSModel,
  x::AbstractVector,
  v::AbstractVector,
  Jv::AbstractVector,
)
  increment!(nls, :neval_jprod_residual)
  nls.nls_meta.nlin > 0 && (Jv .= 0.0)
  if nls.nls_meta.nnln > 0
    MOI.eval_constraint_jacobian_product(nls.Feval, view(Jv, nls.nls_meta.nln), x, v)
  end
  if nls.nls_meta.nlin > 0
    for k = 1:(nls.linequ.nnzj)
      row, col, val =
        nls.linequ.jacobian.rows[k], nls.linequ.jacobian.cols[k], nls.linequ.jacobian.vals[k]
      Jv[row] += v[col] * val
    end
  end
  return Jv
end

function NLPModels.jtprod_residual!(
  nls::MathOptNLSModel,
  x::AbstractVector,
  v::AbstractVector,
  Jtv::AbstractVector,
)
  increment!(nls, :neval_jtprod_residual)
  nls.nls_meta.nlin > 0 && (Jtv .= 0.0)
  if nls.nls_meta.nnln > 0
    MOI.eval_constraint_jacobian_transpose_product(nls.Feval, Jtv, x, view(v, nls.nls_meta.nln))
  end
  if nls.nls_meta.nlin > 0
    for k = 1:(nls.linequ.nnzj)
      row, col, val =
        nls.linequ.jacobian.rows[k], nls.linequ.jacobian.cols[k], nls.linequ.jacobian.vals[k]
      Jtv[col] += v[row] * val
    end
  end
  return Jtv
end

function NLPModels.hess_structure_residual!(
  nls::MathOptNLSModel,
  rows::AbstractVector{<:Integer},
  cols::AbstractVector{<:Integer},
)
  if nls.nls_meta.nnln > 0
    view(rows, 1:(nls.nls_meta.nnzh)) .= nls.nlequ.hess_rows
    view(cols, 1:(nls.nls_meta.nnzh)) .= nls.nlequ.hess_cols
  end
  return rows, cols
end

function NLPModels.hess_coord_residual!(
  nls::MathOptNLSModel,
  x::AbstractVector,
  v::AbstractVector,
  vals::AbstractVector,
)
  increment!(nls, :neval_hess_residual)
  if nls.nls_meta.nnln > 0
    MOI.eval_hessian_lagrangian(nls.Feval, vals, x, 0.0, view(v, nls.nls_meta.nln))
  end
  return vals
end

function NLPModels.hprod_residual!(
  nls::MathOptNLSModel,
  x::AbstractVector,
  i::Int,
  v::AbstractVector,
  Hiv::AbstractVector,
)
  increment!(nls, :neval_hprod_residual)
  if i ∈ nls.nls_meta.lin
    Hiv .= 0.0
  end
  if i ∈ nls.nls_meta.nln
    y = [j == i ? 1.0 : 0.0 for j in nls.nls_meta.nln]
    MOI.eval_hessian_lagrangian_product(nls.Feval, Hiv, x, v, 0.0, y)
  end
  return Hiv
end

function NLPModels.obj(nls::MathOptNLSModel, x::AbstractVector)
  increment!(nls, :neval_obj)
  obj = 0.0
  if nls.nls_meta.nnln > 0
    obj += MOI.eval_objective(nls.ceval, x)
  end
  if nls.nls_meta.nlin > 0
    obj +=
      0.5 * coo_sym_dot(nls.lls.hessian.rows, nls.lls.hessian.cols, nls.lls.hessian.vals, x, x) +
      dot(nls.lls.gradient, x) +
      nls.lls.constant
  end
  return obj
end

function NLPModels.grad!(nls::MathOptNLSModel, x::AbstractVector, g::AbstractVector)
  increment!(nls, :neval_grad)
  if nls.nls_meta.nnln > 0
    MOI.eval_objective_gradient(nls.ceval, g, x)
  end
  if nls.nls_meta.nlin > 0
    nls.nls_meta.nnln == 0 && (g .= 0.0)
    coo_sym_add_mul!(nls.lls.hessian.rows, nls.lls.hessian.cols, nls.lls.hessian.vals, x, g, 1.0)
    g .+= nls.lls.gradient
  end
  return g
end

function NLPModels.cons_lin!(nls::MathOptNLSModel, x::AbstractVector, c::AbstractVector)
  increment!(nls, :neval_cons_lin)
  coo_prod!(nls.lincon.jacobian.rows, nls.lincon.jacobian.cols, nls.lincon.jacobian.vals, x, c)
  return c
end

function NLPModels.cons_nln!(nls::MathOptNLSModel, x::AbstractVector, c::AbstractVector)
  increment!(nls, :neval_cons_nln)
  for i = 1:(nls.quadcon.nquad)
    qcon = nls.quadcon.constraints[i]
    c[i] = 0.5 * coo_sym_dot(qcon.A.rows, qcon.A.cols, qcon.A.vals, x, x) + dot(qcon.b, x)
  end
  if nls.meta.nnln > nls.quadcon.nquad
    MOI.eval_constraint(nls.ceval, view(c, (nls.quadcon.nquad + 1):(nls.meta.nnln)), x)
  end
  return c
end

function NLPModels.jac_lin_structure!(
  nls::MathOptNLSModel,
  rows::AbstractVector{<:Integer},
  cols::AbstractVector{<:Integer},
)
  view(rows, 1:(nls.lincon.nnzj)) .= nls.lincon.jacobian.rows
  view(cols, 1:(nls.lincon.nnzj)) .= nls.lincon.jacobian.cols
  return rows, cols
end

function NLPModels.jac_nln_structure!(
  nls::MathOptNLSModel,
  rows::AbstractVector{<:Integer},
  cols::AbstractVector{<:Integer},
)
  if nls.quadcon.nquad > 0
    index = 0
    for i = 1:(nls.quadcon.nquad)
      qcon = nls.quadcon.constraints[i]
      view(rows, index+1:index+qcon.nnzg) .= i
      view(cols, index+1:index+qcon.nnzg) .= qcon.g
      index += qcon.nnzg
    end
  end
  if nls.meta.nnln > nls.quadcon.nquad
    ind_nnln = (nls.quadcon.nnzj + 1):(nls.quadcon.nnzj + nls.nlcon.nnzj)
    view(rows, ind_nnln) .= nls.nlcon.jac_rows
    view(cols, ind_nnln) .= nls.nlcon.jac_cols
  end
  return rows, cols
end

function NLPModels.jac_lin_coord!(nls::MathOptNLSModel, x::AbstractVector, vals::AbstractVector)
  increment!(nls, :neval_jac_lin)
  view(vals, 1:(nls.lincon.nnzj)) .= nls.lincon.jacobian.vals
  return vals
end

function NLPModels.jac_nln_coord!(nls::MathOptNLSModel, x::AbstractVector, vals::AbstractVector)
  increment!(nls, :neval_jac_nln)
  if nls.quadcon.nquad > 0
    index = 0
    view(vals, 1:nls.quadcon.nnzj) .= 0.0
    for i = 1:(nls.quadcon.nquad)
      qcon = nls.quadcon.constraints[i]
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
  if nls.meta.nnln > nls.quadcon.nquad
    ind_nnln = (nls.quadcon.nnzj + 1):(nls.quadcon.nnzj + nls.nlcon.nnzj)
    MOI.eval_constraint_jacobian(nls.ceval, view(vals, ind_nnln), x)
  end
  return vals
end

function NLPModels.jprod_lin!(
  nls::MathOptNLSModel,
  x::AbstractVector,
  v::AbstractVector,
  Jv::AbstractVector,
)
  jprod_lin!(
    nls,
    nls.lincon.jacobian.rows,
    nls.lincon.jacobian.cols,
    nls.lincon.jacobian.vals,
    v,
    Jv,
  )
  return Jv
end

function NLPModels.jprod_nln!(
  nls::MathOptNLSModel,
  x::AbstractVector,
  v::AbstractVector,
  Jv::AbstractVector,
)
  increment!(nls, :neval_jprod_nln)
  if nls.meta.nnln > nls.quadcon.nquad
    ind_nnln = (nls.quadcon.nquad + 1):(nls.meta.nnln)
    MOI.eval_constraint_jacobian_product(nls.ceval, view(Jv, ind_nnln), x, v)
  end
  (nls.meta.nnln == nls.quadcon.nquad) && (Jv .= 0.0)
  if nls.quadcon.nquad > 0
    for i = 1:(nls.quadcon.nquad)
      # Jv[i] = (Aᵢ * x + bᵢ)ᵀ * v
      qcon = nls.quadcon.constraints[i]
      v[i] += coo_sym_dot(qcon.A.rows, qcon.A.cols, qcon.A.vals, x, v) + dot(qcon.b, v)
    end
  end
  return Jv
end

function NLPModels.jtprod_lin!(
  nls::MathOptNLSModel,
  x::AbstractVector,
  v::AbstractVector,
  Jtv::AbstractVector,
)
  jtprod_lin!(
    nls,
    nls.lincon.jacobian.rows,
    nls.lincon.jacobian.cols,
    nls.lincon.jacobian.vals,
    v,
    Jtv,
  )
  return Jtv
end

function NLPModels.jtprod_nln!(
  nls::MathOptNLSModel,
  x::AbstractVector,
  v::AbstractVector,
  Jtv::AbstractVector,
)
  increment!(nls, :neval_jtprod_nln)
  if nls.meta.nnln > nls.quadcon.nquad
    ind_nnln = (nls.quadcon.nquad + 1):(nls.meta.nnln)
    MOI.eval_constraint_jacobian_transpose_product(nls.ceval, Jtv, x, view(v, ind_nnln))
  end
  (nls.meta.nnln == nls.quadcon.nquad) && (Jtv .= 0.0)
  if nls.quadcon.nquad > 0
    for i = 1:(nls.quadcon.nquad)
      # Jtv += v[i] * (Aᵢ * x + bᵢ)
      qcon = nls.quadcon.constraints[i]
      coo_sym_add_mul!(rows, cols, vals, x, Jtv, v[i])
      Jtv .+= v[i] .* qcon.b
    end
  end
  return Jtv
end

function NLPModels.hess_structure!(
  nls::MathOptNLSModel,
  rows::AbstractVector{<:Integer},
  cols::AbstractVector{<:Integer},
)
  if nls.nls_meta.nlin > 0
    view(rows, 1:(nls.lls.nnzh)) .= nls.lls.hessian.rows
    view(cols, 1:(nls.lls.nnzh)) .= nls.lls.hessian.cols
  end
  if (nls.nls_meta.nnln > 0) || (nlp.meta.nnln > nlp.quadcon.nquad)
    view(rows, (nls.lls.nnzh + nlp.quadcon.nnzh + 1):(nls.meta.nnzh)) .= nls.nlcon.hess_rows
    view(cols, (nls.lls.nnzh + nlp.quadcon.nnzh + 1):(nls.meta.nnzh)) .= nls.nlcon.hess_cols
  end
  if nls.quadcon.nquad > 0
    index = nls.lls.nnzh
    for i = 1:(nls.quadcon.nquad)
      qcon = nls.quadcon.constraints[i]
      view(rows, (index + 1):(index + qcon.nnzh)) .= qcon.A.rows
      view(cols, (index + 1):(index + qcon.nnzh)) .= qcon.A.cols
      index += qcon.nnzh
    end
  end
  return rows, cols
end

function NLPModels.hess_coord!(
  nls::MathOptNLSModel,
  x::AbstractVector,
  y::AbstractVector,
  vals::AbstractVector;
  obj_weight::Float64 = 1.0,
)
  increment!(nls, :neval_hess)
  if nls.nls_meta.nlin > 0
    view(vals, 1:(nls.lls.nnzh)) .= obj_weight .* nls.lls.hessian.vals
  end
  if (nls.nls_meta.nnln > 0) || (nls.meta.nnln > nls.quadcon.nquad)
    λ = view(y, (nls.meta.nlin + nls.quadcon.nquad + 1):(nls.meta.ncon))
    MOI.eval_hessian_lagrangian(
      nls.ceval,
      view(vals, (nls.lls.nnzh + nls.quadcon.nnzh + 1):(nls.meta.nnzh)),
      x,
      obj_weight,
      λ
    )
  end
  if nls.quadcon.nquad > 0
    index = nls.lls.nnzh
    for i = 1:(nls.quadcon.nquad)
      qcon = nls.quadcon.constraints[i]
      view(vals, (index + 1):(index + qcon.nnzh)) .= y[i] .* qcon.A.vals
      index += qcon.nnzh
    end
  end
  return vals
end

function NLPModels.hess_coord!(
  nls::MathOptNLSModel,
  x::AbstractVector,
  vals::AbstractVector;
  obj_weight::Float64 = 1.0,
)
  increment!(nls, :neval_hess)
  if nls.nls_meta.nlin > 0
    view(vals, 1:(nls.lls.nnzh)) .= obj_weight .* nls.lls.hessian.vals
  end
  view(vals, (nls.lls.nnzh + 1):(nls.lls.nnzh + nls.quadcon.nnzh)) .= 0.0
  if nls.nls_meta.nnln > 0
    λ = zeros(nlp.meta.nnln - nlp.quadcon.nquad)  # Should be stored in the structure MathOptNLSModel
    MOI.eval_hessian_lagrangian(
      nls.ceval,
      view(vals, (nls.lls.nnzh + nls.quadcon.nnzh + 1):(nls.meta.nnzh)),
      x,
      obj_weight,
      λ,
    )
  else
    view(vals, (nls.lls.nnzh + nls.quadcon.nnzh + 1):(nls.meta.nnzh)) .= 0.0
  end
  return vals
end

function NLPModels.hprod!(
  nls::MathOptNLSModel,
  x::AbstractVector,
  y::AbstractVector,
  v::AbstractVector,
  hv::AbstractVector;
  obj_weight::Float64 = 1.0,
)
  increment!(nls, :neval_hprod)
  if (nls.nls_meta.nnln > 0) || (nls.meta.nnln > nls.quadcon.nquad)
    λ = view(y, (nls.meta.nlin + nls.quadcon.nquad + 1):(nls.meta.ncon))
    MOI.eval_hessian_lagrangian_product(nls.ceval, hv, x, v, obj_weight, λ)
  end
  (nls.nls_meta.nnln == 0) && (nls.meta.nnln == nls.quadcon.nquad) && (hv .= 0.0)
  if nls.nls_meta.nlin > 0
    coo_sym_add_mul!(
      nls.lls.hessian.rows,
      nls.lls.hessian.cols,
      nls.lls.hessian.vals,
      v,
      hv,
      obj_weight,
    )
  end
  if nls.quadcon.nquad > 0
    for i = 1:(nls.quadcon.nquad)
      qcon = nls.quadcon.constraints[i]
      coo_sym_add_mul!(qcon.A.rows, qcon.A.cols, qcon.A.vals, v, hv, y[nls.meta.nlin + i])
    end
  end
  return hv
end

function NLPModels.hprod!(
  nls::MathOptNLSModel,
  x::AbstractVector,
  v::AbstractVector,
  hv::AbstractVector;
  obj_weight::Float64 = 1.0,
)
  increment!(nls, :neval_hprod)
  if nls.nls_meta.nnln > 0
    λ = zeros(nls.meta.nnln - nls.quadcon.nquad)  # Should be stored in the structure MathOptNLSModel
    MOI.eval_hessian_lagrangian_product(nls.ceval, hv, x, v, obj_weight, λ)
  end
  if nls.nls_meta.nlin > 0
    (nls.nls_meta.nnln == 0) && (hv .= 0.0)
    coo_sym_add_mul!(
      nls.lls.hessian.rows,
      nls.lls.hessian.cols,
      nls.lls.hessian.vals,
      v,
      hv,
      obj_weight,
    )
  end
  return hv
end
