const _MathOptModel = Union{MathOptNLPModel, MathOptNLSModel}

_constraint_evaluator(nlp::MathOptNLPModel) = nlp.eval
_constraint_evaluator(nls::MathOptNLSModel) = nls.ceval

function NLPModels.cons_lin!(nlp::_MathOptModel, x::AbstractVector, c::AbstractVector)
  increment!(nlp, :neval_cons_lin)
  J = nlp.lincon.jacobian
  return coo_prod!(J.rows, J.cols, J.vals, x, c)
end

function _cons_nln!(nlp, x, c)
  offset = nlp.nlcon.nnln
  if offset > 0
    MOI.eval_constraint(_constraint_evaluator(nlp), view(c, 1:offset), x)
  end
  for (f, s) in nlp.oracles.oracles
    for i in eachindex(s.x)
      s.x[i] = x[f.variables[i].value]
    end
    rows = (offset + 1):(offset + s.set.output_dimension)
    s.set.eval_f(view(c, rows), s.x)
    offset += s.set.output_dimension
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
  offset = nlp.nlcon.nnzj
  view(rows, 1:offset) .= nlp.nlcon.jac_rows
  view(cols, 1:offset) .= nlp.nlcon.jac_cols
  row_offset = nlp.nlcon.nnln
  for (f, s) in nlp.oracles.oracles
    for (r, c) in s.set.jacobian_structure
      offset += 1
      rows[offset] = row_offset + r
      cols[offset] = f.variables[c].value
    end
    row_offset += s.set.output_dimension
  end
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
  offset = nlp.nlcon.nnzj
  if nlp.nlcon.nnln > 0
    MOI.eval_constraint_jacobian(_constraint_evaluator(nlp), view(vals, 1:offset), x)
  end
  for (f, s) in nlp.oracles.oracles
    for i in eachindex(s.x)
      s.x[i] = x[f.variables[i].value]
    end
    nnzj = length(s.set.jacobian_structure)
    s.set.eval_jacobian(view(vals, (offset + 1):(offset + nnzj)), s.x)
    offset += nnzj
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
  offset = nlp.nlcon.nnln
  if offset > 0
    MOI.eval_constraint_jacobian_product(_constraint_evaluator(nlp), view(Jv, 1:offset), x, v)
  end
  for (f, s) in nlp.oracles.oracles
    for i in eachindex(s.x)
      s.x[i] = x[f.variables[i].value]
    end
    s.set.eval_jacobian(s.nzJ, s.x)
    view(Jv, (offset + 1):(offset + s.set.output_dimension)) .= 0
    for (k, (r, c)) in enumerate(s.set.jacobian_structure)
      Jv[offset + r] += s.nzJ[k] * v[f.variables[c].value]
    end
    offset += s.set.output_dimension
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
  offset = nlp.nlcon.nnln
  if offset > 0
    MOI.eval_constraint_jacobian_transpose_product(
      _constraint_evaluator(nlp),
      Jtv,
      x,
      view(v, 1:offset),
    )
  else
    fill!(Jtv, 0)
  end
  for (f, s) in nlp.oracles.oracles
    for i in eachindex(s.x)
      s.x[i] = x[f.variables[i].value]
    end
    s.set.eval_jacobian(s.nzJ, s.x)
    for (k, (r, c)) in enumerate(s.set.jacobian_structure)
      Jtv[f.variables[c].value] += s.nzJ[k] * v[offset + r]
    end
    offset += s.set.output_dimension
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
