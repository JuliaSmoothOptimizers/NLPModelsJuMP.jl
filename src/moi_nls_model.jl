export MathOptNLSModel

mutable struct MathOptNLSModel <: AbstractNLSModel{Float64, Vector{Float64}}
  meta::NLPModelMeta{Float64, Vector{Float64}}
  nls_meta::NLSMeta{Float64, Vector{Float64}}
  Feval::Union{MOI.AbstractNLPEvaluator, Nothing}
  ceval::Union{MOI.AbstractNLPEvaluator, Nothing}
  lls::Objective
  linequ::LinearEquations
  lincon::LinearConstraints
  counters::NLSCounters
end

"""
    MathOptNLSModel(model, F, hessian=true, name="Generic")

Construct a `MathOptNLSModel` from a `JuMP` model and a container of JuMP
`GenericAffExpr` (generated by @expression) and `NonlinearExpression` (generated by @NLexpression).

`hessian` should be set to `false` for multivariate user-defined functions registered without hessian.
"""
function MathOptNLSModel(cmodel::JuMP.Model, F; hessian::Bool = true, name::String = "Generic")
  nvar, lvar, uvar, x0 = parser_JuMP(cmodel)

  nnln = num_nonlinear_constraints(cmodel)

  nl_lcon = nnln == 0 ? Float64[] : map(nl_con -> nl_con.lb, cmodel.nlp_data.nlconstr)
  nl_ucon = nnln == 0 ? Float64[] : map(nl_con -> nl_con.ub, cmodel.nlp_data.nlconstr)

  lls, linequ, nlinequ = parser_linear_expression(cmodel, nvar, F)
  ceval, Feval, nnlnequ = parser_nonlinear_expression(cmodel, nvar, F, hessian = hessian)

  nl_Fnnzj = (nnlnequ == 0 ? 0 : sum(length(con.grad_sparsity) for con in Feval.constraints))
  nl_Fnnzh = hessian ? (nnlnequ == 0 ? 0 : sum(length(con.hess_I) for con in Feval.constraints)) : 0

  nl_cnnzj = (nnln == 0 ? 0 : sum(length(con.grad_sparsity) for con in ceval.constraints))
  nl_cnnzh =
    hessian ?
    (nnlnequ == 0 ? 0 : length(ceval.objective.hess_I)) +
    (nnln == 0 ? 0 : sum(length(con.hess_I) for con in ceval.constraints)) : 0

  moimodel = backend(cmodel)
  nlin, lincon, lin_lcon, lin_ucon = parser_MOI(moimodel)

  nequ = nlinequ + nnlnequ
  Fnnzj = linequ.nnzj + nl_Fnnzj
  Fnnzh = nl_Fnnzh

  ncon = nlin + nnln
  lcon = vcat(lin_lcon, nl_lcon)
  ucon = vcat(lin_ucon, nl_ucon)
  cnnzj = lincon.nnzj + nl_cnnzj
  cnnzh = lls.nnzh + nl_cnnzh

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
    nln_nnzj = nl_cnnzj,
    minimize = objective_sense(cmodel) == MOI.MIN_SENSE,
    islp = false,
    name = name,
  )

  return MathOptNLSModel(
    meta,
    NLSMeta(nequ, nvar, nnzj = Fnnzj, nnzh = Fnnzh, lin = collect(1:nlinequ)),
    Feval,
    ceval,
    lls,
    linequ,
    lincon,
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
    Fx[nls.nls_meta.lin] .+= nls.linequ.constants
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
    rows[1:(nls.linequ.nnzj)] .= nls.linequ.jacobian.rows[1:(nls.linequ.nnzj)]
    cols[1:(nls.linequ.nnzj)] .= nls.linequ.jacobian.cols[1:(nls.linequ.nnzj)]
  end
  if nls.nls_meta.nnln > 0
    jac_struct_residual = MOI.jacobian_structure(nls.Feval)
    for index = (nls.linequ.nnzj + 1):(nls.nls_meta.nnzj)
      row, col = jac_struct_residual[index - nls.linequ.nnzj]
      rows[index] = nls.nls_meta.nlin + row
      cols[index] = col
    end
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
    vals[1:(nls.linequ.nnzj)] .= nls.linequ.jacobian.vals[1:(nls.linequ.nnzj)]
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
  rows::AbstractVector{<:Integer},
  cols::AbstractVector{<:Integer},
  v::AbstractVector,
  Jv::AbstractVector,
)
  vals = jac_coord_residual(nls, x)
  decrement!(nls, :neval_jac_residual)
  jprod_residual!(nls, rows, cols, vals, v, Jv)
  return Jv
end

function NLPModels.jprod_residual!(
  nls::MathOptNLSModel,
  x::AbstractVector,
  v::AbstractVector,
  Jv::AbstractVector,
)
  rows, cols = jac_structure_residual(nls)
  jprod_residual!(nls, x, rows, cols, v, Jv)
  return Jv
end

function NLPModels.jtprod_residual!(
  nls::MathOptNLSModel,
  x::AbstractVector,
  rows::AbstractVector{<:Integer},
  cols::AbstractVector{<:Integer},
  v::AbstractVector,
  Jtv::AbstractVector,
)
  vals = jac_coord_residual(nls, x)
  decrement!(nls, :neval_jac_residual)
  jtprod_residual!(nls, rows, cols, vals, v, Jtv)
  return Jtv
end

function NLPModels.jtprod_residual!(
  nls::MathOptNLSModel,
  x::AbstractVector,
  v::AbstractVector,
  Jtv::AbstractVector,
)
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

function NLPModels.hess_structure_residual!(
  nls::MathOptNLSModel,
  rows::AbstractVector{<:Integer},
  cols::AbstractVector{<:Integer},
)
  if nls.nls_meta.nnln > 0
    hess_struct_residual = MOI.hessian_lagrangian_structure(nls.Feval)
    for index = 1:(nls.nls_meta.nnzh)
      rows[index] = hess_struct_residual[index][1]
      cols[index] = hess_struct_residual[index][2]
    end
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
    for k = 1:(nls.lls.nnzh)
      i, j, c = nls.lls.hessian.rows[k], nls.lls.hessian.cols[k], nls.lls.hessian.vals[k]
      g[i] += c * x[j]
      if i ≠ j
        g[j] += c * x[i]
      end
    end
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
  MOI.eval_constraint(nls.ceval, c, x)
  return c
end

function NLPModels.jac_lin_structure!(
  nls::MathOptNLSModel,
  rows::AbstractVector{<:Integer},
  cols::AbstractVector{<:Integer},
)
  for index = 1:(nls.lincon.nnzj)
    rows[index] = nls.lincon.jacobian.rows[index]
    cols[index] = nls.lincon.jacobian.cols[index]
  end
  return rows, cols
end

function NLPModels.jac_nln_structure!(
  nls::MathOptNLSModel,
  rows::AbstractVector{<:Integer},
  cols::AbstractVector{<:Integer},
)
  jac_struct = MOI.jacobian_structure(nls.ceval)
  for index = 1:(nls.meta.nln_nnzj)
    row, col = jac_struct[index]
    rows[index] = row
    cols[index] = col
  end
  return rows, cols
end

function NLPModels.jac_lin_coord!(nls::MathOptNLSModel, x::AbstractVector, vals::AbstractVector)
  increment!(nls, :neval_jac_lin)
  for index = 1:(nls.lincon.nnzj)
    vals[index] = nls.lincon.jacobian.vals[index]
  end
  return vals
end

function NLPModels.jac_nln_coord!(nls::MathOptNLSModel, x::AbstractVector, vals::AbstractVector)
  increment!(nls, :neval_jac_nln)
  MOI.eval_constraint_jacobian(nls.ceval, vals, x)
  return vals
end

function NLPModels.jprod_lin!(
  nls::MathOptNLSModel,
  x::AbstractVector,
  v::AbstractVector,
  Jv::AbstractVector,
)
  jprod_lin!(nls, nls.lincon.jacobian.rows, nls.lincon.jacobian.cols, nls.lincon.jacobian.vals, v, Jv)
  return Jv
end

function NLPModels.jprod_nln!(
  nls::MathOptNLSModel,
  x::AbstractVector,
  v::AbstractVector,
  Jv::AbstractVector,
)
  increment!(nls, :neval_jprod_nln)
  MOI.eval_constraint_jacobian_product(nls.ceval, Jv, x, v)
  return Jv
end

function NLPModels.jtprod_lin!(
  nls::MathOptNLSModel,
  x::AbstractVector,
  v::AbstractVector,
  Jtv::AbstractVector,
)
  jtprod_lin!(nls, nls.lincon.jacobian.rows, nls.lincon.jacobian.cols, nls.lincon.jacobian.vals, v, Jtv)
  return Jtv
end

function NLPModels.jtprod_nln!(
  nls::MathOptNLSModel,
  x::AbstractVector,
  v::AbstractVector,
  Jtv::AbstractVector,
)
  increment!(nls, :neval_jtprod_nln)
  MOI.eval_constraint_jacobian_transpose_product(nls.ceval, Jtv, x, v)
  return Jtv
end

function NLPModels.hess_structure!(
  nls::MathOptNLSModel,
  rows::AbstractVector{<:Integer},
  cols::AbstractVector{<:Integer},
)
  if nls.nls_meta.nlin > 0
    for index = 1:(nls.lls.nnzh)
      rows[index] = nls.lls.hessian.rows[index]
      cols[index] = nls.lls.hessian.cols[index]
    end
  end
  if nls.nls_meta.nnln > 0
    hesslag_struct = MOI.hessian_lagrangian_structure(nls.ceval)
    for index = (nls.lls.nnzh + 1):(nls.meta.nnzh)
      shift_index = index - nls.lls.nnzh
      rows[index] = hesslag_struct[shift_index][1]
      cols[index] = hesslag_struct[shift_index][2]
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
    vals[1:(nls.lls.nnzh)] .= obj_weight .* nls.lls.hessian.vals
  end
  if (nls.nls_meta.nnln > 0) || (nls.meta.nnln > 0)
    MOI.eval_hessian_lagrangian(
      nls.ceval,
      view(vals, (nls.lls.nnzh + 1):(nls.meta.nnzh)),
      x,
      obj_weight,
      view(y, nls.meta.nln),
    )
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
    vals[1:(nls.lls.nnzh)] .= obj_weight .* nls.lls.hessian.vals
  end
  if nls.nls_meta.nnln > 0
    MOI.eval_hessian_lagrangian(
      nls.ceval,
      view(vals, (nls.lls.nnzh + 1):(nls.meta.nnzh)),
      x,
      obj_weight,
      zeros(nls.meta.nnln),
    )
  else
    vals[(nls.lls.nnzh + 1):(nls.meta.nnzh)] .= 0.0
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
  if (nls.nls_meta.nnln > 0) || (nls.meta.nnln > 0)
    MOI.eval_hessian_lagrangian_product(nls.ceval, hv, x, v, obj_weight, view(y, nls.meta.nln))
  end
  if nls.nls_meta.nlin > 0
    (nls.nls_meta.nnln == 0) && (nls.meta.nnln == 0) && (hv .= 0.0)
    for k = 1:(nls.lls.nnzh)
      i, j, c = nls.lls.hessian.rows[k], nls.lls.hessian.cols[k], nls.lls.hessian.vals[k]
      hv[i] += obj_weight * c * v[j]
      if i ≠ j
        hv[j] += obj_weight * c * v[i]
      end
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
    MOI.eval_hessian_lagrangian_product(nls.ceval, hv, x, v, obj_weight, zeros(nls.meta.nnln))
  end
  if nls.nls_meta.nlin > 0
    (nls.nls_meta.nnln == 0) && (hv .= 0.0)
    for k = 1:(nls.lls.nnzh)
      i, j, c = nls.lls.hessian.rows[k], nls.lls.hessian.cols[k], nls.lls.hessian.vals[k]
      hv[i] += obj_weight * c * v[j]
      if i ≠ j
        hv[j] += obj_weight * c * v[i]
      end
    end
  end
  return hv
end
