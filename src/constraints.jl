const _MathOptModel = Union{MathOptNLPModel, MathOptNLSModel}

_constraint_evaluator(nlp::MathOptNLPModel) = nlp.eval
_constraint_evaluator(nls::MathOptNLSModel) = nls.ceval

function NLPModels.cons_lin!(nlp::_MathOptModel, x::AbstractVector, c::AbstractVector)
  increment!(nlp, :neval_cons_lin)
  J = nlp.lincon.jacobian
  return coo_prod!(J.rows, J.cols, J.vals, x, c)
end

function _cons_nln!(nlp, x, c)
  if nlp.nlcon.nnln > 0
    MOI.eval_constraint(_constraint_evaluator(nlp), c, x)
  end
  return c
end

function NLPModels.cons_nln!(nlp::_MathOptModel, x::AbstractVector, c::AbstractVector)
  increment!(nlp, :neval_cons_nln)
  return _cons_nln!(nlp, x, c)
end

function NLPModels.cons!(nlp::_MathOptModel, x::AbstractVector, c::AbstractVector)
  increment!(nlp, :neval_cons)
  nlin = nlp.meta.nlin
  J = nlp.lincon.jacobian
  coo_prod!(J.rows, J.cols, J.vals, x, view(c, 1:nlin))
  _cons_nln!(nlp, x, view(c, (nlin + 1):nlp.meta.ncon))
  return c
end

function NLPModels.jac_lin_structure!(
  nlp::_MathOptModel,
  rows::AbstractVector{<:Integer},
  cols::AbstractVector{<:Integer},
)
  rows .= nlp.lincon.jacobian.rows
  cols .= nlp.lincon.jacobian.cols
  return rows, cols
end

function NLPModels.jac_nln_structure!(
  nlp::_MathOptModel,
  rows::AbstractVector{<:Integer},
  cols::AbstractVector{<:Integer},
)
  rows .= nlp.nlcon.jac_rows
  cols .= nlp.nlcon.jac_cols
  return rows, cols
end

function NLPModels.jac_structure!(
  nlp::_MathOptModel,
  rows::AbstractVector{<:Integer},
  cols::AbstractVector{<:Integer},
)
  lin = 1:nlp.lincon.nnzj
  nln = (nlp.lincon.nnzj + 1):nlp.meta.nnzj
  NLPModels.jac_lin_structure!(nlp, view(rows, lin), view(cols, lin))
  NLPModels.jac_nln_structure!(nlp, view(rows, nln), view(cols, nln))
  view(rows, nln) .+= nlp.meta.nlin
  return rows, cols
end

function NLPModels.jac_lin_coord!(nlp::_MathOptModel, x::AbstractVector, vals::AbstractVector)
  increment!(nlp, :neval_jac_lin)
  vals .= nlp.lincon.jacobian.vals
  return vals
end

function _jac_nln_coord!(nlp, x, vals)
  if nlp.nlcon.nnln > 0
    MOI.eval_constraint_jacobian(_constraint_evaluator(nlp), vals, x)
  end
  return vals
end

function NLPModels.jac_nln_coord!(nlp::_MathOptModel, x::AbstractVector, vals::AbstractVector)
  increment!(nlp, :neval_jac_nln)
  return _jac_nln_coord!(nlp, x, vals)
end

function NLPModels.jac_coord!(nlp::_MathOptModel, x::AbstractVector, vals::AbstractVector)
  increment!(nlp, :neval_jac)
  offset = nlp.lincon.nnzj
  view(vals, 1:offset) .= nlp.lincon.jacobian.vals
  _jac_nln_coord!(nlp, x, view(vals, (offset + 1):nlp.meta.nnzj))
  return vals
end

function NLPModels.jprod_lin!(
  nlp::_MathOptModel,
  x::AbstractVector,
  v::AbstractVector,
  Jv::AbstractVector,
)
  increment!(nlp, :neval_jprod_lin)
  J = nlp.lincon.jacobian
  return jprod_lin!(nlp, J.rows, J.cols, J.vals, v, Jv)
end

function _jprod_nln!(nlp, x, v, Jv)
  if nlp.nlcon.nnln > 0
    MOI.eval_constraint_jacobian_product(_constraint_evaluator(nlp), Jv, x, v)
  end
  return Jv
end

function NLPModels.jprod_nln!(
  nlp::_MathOptModel,
  x::AbstractVector,
  v::AbstractVector,
  Jv::AbstractVector,
)
  increment!(nlp, :neval_jprod_nln)
  return _jprod_nln!(nlp, x, v, Jv)
end

function NLPModels.jprod!(
  nlp::_MathOptModel,
  x::AbstractVector,
  v::AbstractVector,
  Jv::AbstractVector,
)
  increment!(nlp, :neval_jprod)
  nlin = nlp.meta.nlin
  J = nlp.lincon.jacobian
  coo_prod!(J.rows, J.cols, J.vals, v, view(Jv, 1:nlin))
  _jprod_nln!(nlp, x, v, view(Jv, (nlin + 1):nlp.meta.ncon))
  return Jv
end

function NLPModels.jtprod_lin!(
  nlp::_MathOptModel,
  x::AbstractVector,
  v::AbstractVector,
  Jtv::AbstractVector,
)
  increment!(nlp, :neval_jtprod_lin)
  J = nlp.lincon.jacobian
  return jtprod_lin!(nlp, J.rows, J.cols, J.vals, v, Jtv)
end

function _jtprod_nln!(nlp, x, v, Jtv)
  if nlp.nlcon.nnln > 0
    MOI.eval_constraint_jacobian_transpose_product(_constraint_evaluator(nlp), Jtv, x, v)
  else
    fill!(Jtv, 0)
  end
  return Jtv
end

function NLPModels.jtprod_nln!(
  nlp::_MathOptModel,
  x::AbstractVector,
  v::AbstractVector,
  Jtv::AbstractVector,
)
  increment!(nlp, :neval_jtprod_nln)
  return _jtprod_nln!(nlp, x, v, Jtv)
end

function NLPModels.jtprod!(
  nlp::_MathOptModel,
  x::AbstractVector,
  v::AbstractVector,
  Jtv::AbstractVector,
)
  increment!(nlp, :neval_jtprod)
  _jtprod_nln!(nlp, x, view(v, (nlp.meta.nlin + 1):nlp.meta.ncon), Jtv)
  J = nlp.lincon.jacobian
  for k in eachindex(J.vals)
    Jtv[J.cols[k]] += J.vals[k] * v[J.rows[k]]
  end
  return Jtv
end
