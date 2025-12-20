export MathOptNLPModel

mutable struct MathOptNLPModel <: AbstractNLPModel{Float64, Vector{Float64}}
  meta::NLPModelMeta{Float64, Vector{Float64}}
  eval::MOI.Nonlinear.Evaluator
  lincon::LinearConstraints
  quadcon::QuadraticConstraints
  nlcon::NonLinearStructure
  oracles::Oracles
  λ::Vector{Float64}
  hv::Vector{Float64}
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
  nlin, lincon, lin_lcon, lin_ucon, quadcon, quad_lcon, quad_ucon =
    parser_MOI(moimodel, index_map, nvar)

  nlp_data = _nlp_block(moimodel)
  nlcon, nl_lcon, nl_ucon = parser_NL(nlp_data, hessian = hessian)
  oracles = parser_oracles(moimodel)
  counters = Counters()
  λ = zeros(Float64, nlcon.nnln)  # Lagrange multipliers for hess_coord! and hprod! without y
  hv = zeros(Float64, nvar)  # workspace for ghjvprod!

  if nlp_data.has_objective
    obj = Objective("NONLINEAR", 0.0, spzeros(Float64, nvar), COO(), 0)
  else
    obj = parser_objective_MOI(moimodel, nvar, index_map)
  end

  # Total counts
  ncon = nlin + quadcon.nquad + nlcon.nnln + oracles.ncon
  lcon = vcat(lin_lcon, quad_lcon, nl_lcon, oracles.lcon)
  ucon = vcat(lin_ucon, quad_ucon, nl_ucon, oracles.ucon)
  nnzj = lincon.nnzj + quadcon.nnzj + nlcon.nnzj + oracles.nnzj
  nnzh = obj.nnzh + quadcon.nnzh + nlcon.nnzh + oracles.nnzh
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
    nln_nnzj = quadcon.nnzj + nlcon.nnzj + oracles.nnzj,
    minimize = MOI.get(moimodel, MOI.ObjectiveSense()) == MOI.MIN_SENSE,
    islp = (obj.type == "LINEAR") && (nlin == ncon),
    name = name,
    hprod_available = hessian && oracles.hessian_oracles_supported,
    hess_available = hessian && oracles.hessian_oracles_supported,
  )

  return MathOptNLPModel(meta, nlp_data.evaluator, lincon, quadcon, nlcon, oracles, λ, hv, obj, counters),
  index_map
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
  offset = 0
  if nlp.quadcon.nquad > 0
    offset += nlp.quadcon.nquad
    for i = 1:(nlp.quadcon.nquad)
      qcon = nlp.quadcon.constraints[i]
      c[i] = 0.5 * coo_sym_dot(qcon.A.rows, qcon.A.cols, qcon.A.vals, x, x) + dot(qcon.b, x)
    end
  end
  if nlp.nlcon.nnln > 0
    offset += nlp.nlcon.nnln
    index_nlcon = (nlp.quadcon.nquad + 1):(offset)
    MOI.eval_constraint(nlp.eval, view(c, ), x)
  end
  if nlp.oracles.ncon > 0
    for (f, s) in nlp.oracles.oracles
      for i in 1:s.set.input_dimension
        s.x[i] = x[f.variables[i].value]
      end
      index_oracle = (offset + 1):(offset + s.set.output_dimension)
      c_oracle = view(c, index_oracle)
      s.set.eval_f(c_oracle, s.x)
      offset += s.set.output_dimension
    end
  end
  @assert offset == nlp.meta.nnln
  return c
end

function NLPModels.cons!(nlp::MathOptNLPModel, x::AbstractVector, c::AbstractVector)
  increment!(nlp, :neval_cons)
  offset = 0
  if nlp.meta.nlin > 0
    offset += nlp.meta.nlin
    coo_prod!(nlp.lincon.jacobian.rows, nlp.lincon.jacobian.cols, nlp.lincon.jacobian.vals, x, c)
  end
  if nlp.quadcon.nquad > 0
    offset += nlp.quadcon.nquad
    for i = 1:(nlp.quadcon.nquad)
      qcon = nlp.quadcon.constraints[i]
      c[nlp.meta.nlin + i] = 0.5 * coo_sym_dot(qcon.A.rows, qcon.A.cols, qcon.A.vals, x, x) + dot(qcon.b, x)
    end
  end
  if nlp.nlcon.nnln > 0
    offset += nlp.nlcon.nnln
    MOI.eval_constraint(nlp.eval, view(c, (nlp.meta.nlin + nlp.quadcon.nquad + 1):(offset)), x)
  end
  if nlp.oracles.ncon > 0
    for (f, s) in nlp.oracles.oracles
      for i in 1:s.set.input_dimension
        s.x[i] = x[f.variables[i].value]
      end
      index_oracle = (offset + 1):(offset + s.set.output_dimension)
      c_oracle = view(c, index_oracle)
      s.set.eval_f(c_oracle, s.x)
      offset += s.set.output_dimension
    end
  end
  @assert offset == nlp.meta.ncon
  return c
end

function NLPModels.jac_lin_structure!(
  nlp::MathOptNLPModel,
  rows::AbstractVector{<:Integer},
  cols::AbstractVector{<:Integer},
)
  index_lin = 1:(nlp.lincon.nnzj)
  view(rows, index_lin) .= nlp.lincon.jacobian.rows
  view(cols, index_lin) .= nlp.lincon.jacobian.cols
  return rows, cols
end

function NLPModels.jac_nln_structure!(
  nlp::MathOptNLPModel,
  rows::AbstractVector{<:Integer},
  cols::AbstractVector{<:Integer},
)
  offset = 0
  if nlp.quadcon.nquad > 0
    for i = 1:(nlp.quadcon.nquad)
      # qcon.g is the sparsity pattern of the gradient of the quadratic constraint qcon
      qcon = nlp.quadcon.constraints[i]
      ind_quad = (offset + 1):(offset + qcon.nnzg)
      view(rows, ind_quad) .= i
      view(cols, ind_quad) .= qcon.g
      offset += qcon.nnzg
    end
  end
  @assert offset == nlp.quadcon.nnzj
  if nlp.nlcon.nnln > 0
     # non-oracle nonlinear constraints
    ind_nnln = (offset + 1):(offset + nlp.nlcon.nnzj)
    view(rows, ind_nnln) .= nlp.quadcon.nquad .+ nlp.nlcon.jac_rows
    view(cols, ind_nnln) .= nlp.nlcon.jac_cols
    offset += nlp.nlcon.nnzj
  end
  if nlp.oracles.ncon > 0
    # oracle nonlinear constraints
    row_offset = nlp.quadcon.nquad + nlp.nlcon.nnln
    for (f, s) in nlp.oracles.oracles
      for (r, c) in s.set.jacobian_structure
        offset += 1
        rows[offset] = row_offset + r
        cols[offset] = f.variables[c].value
      end
      row_offset += s.set.output_dimension
    end
    @assert row_offset == nlp.meta.nnln
  end
  @assert offset == nlp.meta.nln_nnzj
  return rows, cols
end

function NLPModels.jac_structure!(
  nlp::MathOptNLPModel,
  rows::AbstractVector{<:Integer},
  cols::AbstractVector{<:Integer},
)
  offset = 0
  if nlp.meta.nlin > 0
    index_lin = 1:(nlp.lincon.nnzj)
    view(rows, index_lin) .= nlp.lincon.jacobian.rows
    view(cols, index_lin) .= nlp.lincon.jacobian.cols
    offset += nlp.lincon.nnzj
  end
  if nlp.quadcon.nquad > 0
    for i = 1:(nlp.quadcon.nquad)
      # qcon.g is the sparsity pattern of the gradient of the quadratic constraint qcon
      qcon = nlp.quadcon.constraints[i]
      ind_quad = (offset + 1):(offset + qcon.nnzg)
      view(rows, ind_quad) .= nlp.meta.nlin .+ i
      view(cols, ind_quad) .= qcon.g
      offset += qcon.nnzg
    end
  end
  @assert offset == nlp.lincon.nnzj + nlp.quadcon.nnzj
  if nlp.nlcon.nnln > 0
    # non-oracle nonlinear constraints
    offset = nlp.lincon.nnzj + nlp.quadcon.nnzj
    ind_nnln = (offset + 1):(offset + nlp.nlcon.nnzj)
    view(rows, ind_nnln) .= nlp.meta.nlin .+ nlp.quadcon.nquad .+ nlp.nlcon.jac_rows
    view(cols, ind_nnln) .= nlp.nlcon.jac_cols
    offset += nlp.nlcon.nnzj
  end
  if nlp.oracles.ncon > 0
    # oracle nonlinear constraints
    row_offset = nlp.meta.nlin + nlp.quadcon.nquad + nlp.nlcon.nnln
    for (f, s) in nlp.oracles.oracles
      for (r, c) in s.set.jacobian_structure
        offset += 1
        rows[offset] = row_offset + r
        cols[offset] = f.variables[c].value
      end
      row_offset += s.set.output_dimension
    end
    @assert row_offset == nlp.meta.ncon
  end
  @assert offset == nlp.meta.nnzj
  return rows, cols
end

function NLPModels.jac_lin_coord!(nlp::MathOptNLPModel, x::AbstractVector, vals::AbstractVector)
  increment!(nlp, :neval_jac_lin)
  index_lin = 1:(nlp.lincon.nnzj)
  view(vals, index_lin) .= nlp.lincon.jacobian.vals
  return vals
end

function NLPModels.jac_nln_coord!(nlp::MathOptNLPModel, x::AbstractVector, vals::AbstractVector)
  increment!(nlp, :neval_jac_nln)
  offset = 0
  if nlp.quadcon.nquad > 0
    ind_quad = 1:(nlp.quadcon.nnzj)
    view(vals, ind_quad) .= 0.0
    for i = 1:(nlp.quadcon.nquad)
      qcon = nlp.quadcon.constraints[i]
      for (j, ind) in enumerate(qcon.b.nzind)
        k = qcon.dg[ind]
        vals[offset + k] += qcon.b.nzval[j]
      end
      for j = 1:(qcon.nnzh)
        row = qcon.A.rows[j]
        col = qcon.A.cols[j]
        val = qcon.A.vals[j]
        k1 = qcon.dg[row]
        vals[offset + k1] += val * x[col]
        if row != col
          k2 = qcon.dg[col]
          vals[offset + k2] += val * x[row]
        end
      end
      offset += qcon.nnzg
    end
  end
  if nlp.nlcon.nnln > 0
    ind_nnln = (offset + 1):(offset + nlp.nlcon.nnzj)
    MOI.eval_constraint_jacobian(nlp.eval, view(vals, ind_nnln), x)
    offset += nlp.nlcon.nnzj
  end
  if nlp.oracles.ncon > 0
    for (f, s) in nlp.oracles.oracles
      for i in 1:s.set.input_dimension
        s.x[i] = x[f.variables[i].value]
      end
      nnzj_oracle = length(s.set.jacobian_structure)
      ind_oracle = (offset + 1):(offset + nnzj_oracle)
      s.set.eval_jacobian(view(vals, ind_oracle), s.x)
      offset += nnzj_oracle
    end
  end
  @assert offset == nlp.meta.nln_nnzj
  return vals
end

function NLPModels.jac_coord!(nlp::MathOptNLPModel, x::AbstractVector, vals::AbstractVector)
  increment!(nlp, :neval_jac)
  offset = 0
  if nlp.meta.nlin > 0
    index_lin = 1:(nlp.lincon.nnzj)
    view(vals, index_lin) .= nlp.lincon.jacobian.vals
    offset += nlp.lincon.nnzj
  end
  if nlp.quadcon.nquad > 0
    ind_quad = (nlp.lincon.nnzj + 1):(nlp.lincon.nnzj + nlp.quadcon.nnzj)
    view(vals, ind_quad) .= 0.0
    for i = 1:(nlp.quadcon.nquad)
      qcon = nlp.quadcon.constraints[i]
      for (j, ind) in enumerate(qcon.b.nzind)
        k = qcon.dg[ind]
        vals[offset + k] += qcon.b.nzval[j]
      end
      for j = 1:(qcon.nnzh)
        row = qcon.A.rows[j]
        col = qcon.A.cols[j]
        val = qcon.A.vals[j]
        k1 = qcon.dg[row]
        vals[offset + k1] += val * x[col]
        if row != col
          k2 = qcon.dg[col]
          vals[offset + k2] += val * x[row]
        end
      end
      offset += qcon.nnzg
    end
  end
  if nlp.nlcon.nnln > 0
    ind_nnln = (offset + 1):(offset + nlp.nlcon.nnzj)
    MOI.eval_constraint_jacobian(nlp.eval, view(vals, ind_nnln), x)
    offset += nlp.nlcon.nnzj
  end
  if nlp.oracles.ncon > 0
    for (f, s) in nlp.oracles.oracles
      for i in 1:s.set.input_dimension
        s.x[i] = x[f.variables[i].value]
      end
      nnzj_oracle = length(s.set.jacobian_structure)
      ind_oracle = (offset + 1):(offset + nnzj_oracle)
      s.set.eval_jacobian(view(vals, ind_oracle), s.x)
      offset += nnzj_oracle
    end
  end
  @assert offset == nlp.meta.nnzj
  return vals
end

function NLPModels.jprod_lin!(
  nlp::MathOptNLPModel,
  x::AbstractVector,
  v::AbstractVector,
  Jv::AbstractVector,
)
  increment!(nlp, :neval_jprod_lin)
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
  if nlp.quadcon.nquad > 0
    for i = 1:(nlp.quadcon.nquad)
      # Jv[i] += (Aᵢ * x + bᵢ)ᵀ * v
      qcon = nlp.quadcon.constraints[i]
      Jv[i] = coo_sym_dot(qcon.A.rows, qcon.A.cols, qcon.A.vals, x, v) + dot(qcon.b, v)
    end
  end
  if nlp.nlcon.nnln > 0
    ind_nnln = (nlp.quadcon.nquad + 1):(nlp.quadcon.nquad + nlp.nlcon.nnln)
    MOI.eval_constraint_jacobian_product(nlp.eval, view(Jv, ind_nnln), x, v)
  end
  if nlp.oracles.ncon > 0
    for i in (nlp.quadcon.nquad + nlp.nlcon.nnln + 1):(nlp.quadcon.nquad + nlp.nlcon.nnln + nlp.oracles.ncon)
      Jv[i] = 0
    end
    row_offset = nlp.quadcon.nquad + nlp.nlcon.nnln
    for (f, s) in nlp.oracles.oracles
      for i in 1:s.set.input_dimension
        s.x[i] = x[f.variables[i].value]
      end
      s.set.eval_jacobian(s.nzJ, s.x)
      k = 0
      for (r, c) in s.set.jacobian_structure
        k += 1
        row = row_offset + r
        col = f.variables[c].value
        Jv[row] += s.nzJ[k] * v[col]
      end
      row_offset += s.set.output_dimension
    end
  end
  return Jv
end

function NLPModels.jprod!(
  nlp::MathOptNLPModel,
  x::AbstractVector,
  v::AbstractVector,
  Jv::AbstractVector,
)
  increment!(nlp, :neval_jprod)
  if nlp.meta.nlin > 0
    view(Jv, nlp.meta.lin) .= 0.0
    transpose = false
    coo_unsym_add_mul!(
      transpose,
      nlp.lincon.jacobian.rows,
      nlp.lincon.jacobian.cols,
      nlp.lincon.jacobian.vals,
      v,
      Jv,
      1.0,
    )
  end
  if nlp.quadcon.nquad > 0
    for i = 1:(nlp.quadcon.nquad)
      # Jv[i] = (Aᵢ * x + bᵢ)ᵀ * v
      qcon = nlp.quadcon.constraints[i]
      Jv[nlp.meta.nlin + i] =
        coo_sym_dot(qcon.A.rows, qcon.A.cols, qcon.A.vals, x, v) + dot(qcon.b, v)
    end
  end
  if nlp.nlcon.nnln > 0
    ind_nnln = (nlp.meta.nlin + nlp.quadcon.nquad + 1):(nlp.meta.nlin + nlp.quadcon.nquad + nlp.nlcon.nnln)
    MOI.eval_constraint_jacobian_product(nlp.eval, view(Jv, ind_nnln), x, v)
  end
  if nlp.oracles.ncon > 0
    for i in (nlp.meta.nlin + nlp.quadcon.nquad + nlp.nlcon.nnln + 1):(nlp.meta.ncon)
      Jv[i] = 0
    end
    row_offset = nlp.meta.nlin + nlp.quadcon.nquad + nlp.nlcon.nnln
    for (f, s) in nlp.oracles.oracles
      for i in 1:s.set.input_dimension
        s.x[i] = x[f.variables[i].value]
      end
      s.set.eval_jacobian(s.nzJ, s.x)
      k = 0
      for (r, c) in s.set.jacobian_structure
        k += 1
        row = row_offset + r
        col = f.variables[c].value
        Jv[row] += s.nzJ[k] * v[col]
      end
      row_offset += s.set.output_dimension
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
  increment!(nlp, :neval_jtprod_lin)
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
  if nlp.nlcon.nnln > 0
    ind_nnln = (nlp.quadcon.nquad + 1):(nlp.quadcon.nquad + nlp.nlcon.nnln)
    MOI.eval_constraint_jacobian_transpose_product(nlp.eval, Jtv, x, view(v, ind_nnln))
  end
  (nlp.nlcon.nnln == 0) && (Jtv .= 0.0)
  if nlp.quadcon.nquad > 0
    for i = 1:(nlp.quadcon.nquad)
      # Jtv += v[i] * (Aᵢ * x + bᵢ)
      qcon = nlp.quadcon.constraints[i]
      coo_sym_add_mul!(qcon.A.rows, qcon.A.cols, qcon.A.vals, x, Jtv, v[i])
      Jtv .+= v[i] .* qcon.b
    end
  end
  if nlp.oracles.ncon > 0
    row_offset = nlp.quadcon.nquad + nlp.nlcon.nnln
    for (f, s) in nlp.oracles.oracles
      for i in 1:s.set.input_dimension
        s.x[i] = x[f.variables[i].value]
      end
      s.set.eval_jacobian(s.nzJ, s.x)
      k = 0
      for (r, c) in s.set.jacobian_structure
        k += 1
        row = row_offset + r
        col = f.variables[c].value
        Jtv[col] += s.nzJ[k] * v[row]
      end
      row_offset += s.set.output_dimension
    end
  end
  return Jtv
end

function NLPModels.jtprod!(
  nlp::MathOptNLPModel,
  x::AbstractVector,
  v::AbstractVector,
  Jtv::AbstractVector,
)
  increment!(nlp, :neval_jtprod)
  if nlp.nlcon.nnln > 0
    ind_nnln = (nlp.meta.nlin + nlp.quadcon.nquad + 1):(nlp.meta.nlin + nlp.quadcon.nquad + nlp.nlcon.nnln)
    MOI.eval_constraint_jacobian_transpose_product(nlp.eval, Jtv, x, view(v, ind_nnln))
  end
  (nlp.nlcon.nnln == 0) && (Jtv .= 0.0)
  if nlp.meta.nlin > 0
    transpose = true
    coo_unsym_add_mul!(
      transpose,
      nlp.lincon.jacobian.rows,
      nlp.lincon.jacobian.cols,
      nlp.lincon.jacobian.vals,
      v,
      Jtv,
      1.0,
    )
  end
  if nlp.quadcon.nquad > 0
    for i = 1:(nlp.quadcon.nquad)
      qcon = nlp.quadcon.constraints[i]
      coo_sym_add_mul!(qcon.A.rows, qcon.A.cols, qcon.A.vals, x, Jtv, v[nlp.meta.nlin + i])
      Jtv .+= v[nlp.meta.nlin + i] .* qcon.b
    end
  end
  if nlp.oracles.ncon > 0
    row_offset = nlp.meta.nlin + nlp.quadcon.nquad + nlp.nlcon.nnln
    for (f, s) in nlp.oracles.oracles
      for i in 1:s.set.input_dimension
        s.x[i] = x[f.variables[i].value]
      end
      s.set.eval_jacobian(s.nzJ, s.x)
      k = 0
      for (r, c) in s.set.jacobian_structure
        k += 1
        row = row_offset + r
        col = f.variables[c].value
        Jtv[col] += s.nzJ[k] * v[row]
      end
      row_offset += s.set.output_dimension
    end
  end
  return Jtv
end

function NLPModels.hess_structure!(
  nlp::MathOptNLPModel,
  rows::AbstractVector{<:Integer},
  cols::AbstractVector{<:Integer},
)
  nlp.meta.hess_available || error("The function hess_structure! is not supported by this MathOptNLPModel.")
  if nlp.obj.type == "QUADRATIC"
    view(rows, 1:(nlp.obj.nnzh)) .= nlp.obj.hessian.rows
    view(cols, 1:(nlp.obj.nnzh)) .= nlp.obj.hessian.cols
  end
  index = nlp.obj.nnzh
  if nlp.quadcon.nquad > 0
    for i = 1:(nlp.quadcon.nquad)
      qcon = nlp.quadcon.constraints[i]
      view(rows, (index + 1):(index + qcon.nnzh)) .= qcon.A.rows
      view(cols, (index + 1):(index + qcon.nnzh)) .= qcon.A.cols
      index += qcon.nnzh
    end
  end
  if (nlp.obj.type == "NONLINEAR") || (nlp.nlcon.nnln > 0)
    view(rows, (index + 1):(index + nlp.nlcon.nnzh)) .= nlp.nlcon.hess_rows
    view(cols, (index + 1):(index + nlp.nlcon.nnzh)) .= nlp.nlcon.hess_cols
    index += nlp.nlcon.nnzh
  end
  if nlp.oracles.ncon > 0
    for (f, s) in nlp.oracles.oracles
      for (i, j) in s.set.hessian_lagrangian_structure
        index += 1
        rows[index] = f.variables[i].value
        cols[index] = f.variables[j].value
      end
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
  nlp.meta.hess_available || error("The function hess_coord! is not supported by this MathOptNLPModel.")
  increment!(nlp, :neval_hess)

  # 1. Quadratic objective block (if any)
  if nlp.obj.type == "QUADRATIC"
    view(vals, 1:nlp.obj.nnzh) .= obj_weight .* nlp.obj.hessian.vals
  end

  # 2. Nonlinear block (objective + JuMP @NLconstraint)
  if (nlp.obj.type == "NONLINEAR") || (nlp.nlcon.nnln > 0)
    # Multipliers for the JuMP nonlinear constraints (not the oracles)
    ind_nnln = (nlp.meta.nlin + nlp.quadcon.nquad + 1):(nlp.meta.nlin + nlp.quadcon.nquad + nlp.nlcon.nnln)
    λ_nnln = view(y, ind_nnln)
    ind_nnzh = (nlp.obj.nnzh + nlp.quadcon.nnzh + 1):(nlp.obj.nnzh + nlp.quadcon.nnzh + nlp.nlcon.nnzh)
    MOI.eval_hessian_lagrangian(
      nlp.eval,
      view(vals, ind_nnzh),
      x,
      obj_weight,
      λ_nnln,
    )
  end

  # 3. Quadratic constraint Hessian blocks
  if nlp.quadcon.nquad > 0
    index = nlp.obj.nnzh
    for i = 1:(nlp.quadcon.nquad)
      qcon = nlp.quadcon.constraints[i]
      ind = (index + 1):(index + qcon.nnzh)
      view(vals, ind) .= y[nlp.meta.nlin + i] .* qcon.A.vals
      index += qcon.nnzh
    end
  end

  # 4. Oracle Hessian blocks are appended at the very end
  if nlp.oracles.ncon > 0
    offset_λ = nlp.meta.nlin + nlp.quadcon.nquad + nlp.nlcon.nnln
    offset_nnzh = nlp.obj.nnzh + nlp.quadcon.nnzh + nlp.nlcon.nnzh

    for (f, s) in nlp.oracles.oracles
      # build local x for this oracle
      for i in 1:s.set.input_dimension
        s.x[i] = x[f.variables[i].value]
      end

      ncon_oracle = s.set.output_dimension
      index_λ = (offset_λ + 1):(offset_λ + ncon_oracle)
      λ_oracles = view(y, index_λ)

      nnzh_oracle = length(s.set.hessian_lagrangian_structure)
      index_nnzh = (offset_nnzh + 1):(offset_nnzh + nnzh_oracle)
      s.set.eval_hessian_lagrangian(view(vals, index_nnzh), s.x, λ_oracles)

      offset_λ += ncon_oracle
      offset_nnzh += nnzh_oracle
    end
    @assert offset_λ == nlp.meta.ncon
    @assert offset_nnzh == nlp.meta.nnzh
  end

  return vals
end

function NLPModels.hess_coord!(
  nlp::MathOptNLPModel,
  x::AbstractVector,
  vals::AbstractVector;
  obj_weight::Float64 = 1.0,
)
  nlp.meta.hess_available || error("The function hess_coord! is not supported by this MathOptNLPModel.")
  increment!(nlp, :neval_hess)
  if nlp.obj.type == "LINEAR"
    vals .= 0.0
  end
  if nlp.obj.type == "QUADRATIC"
    view(vals, 1:(nlp.obj.nnzh)) .= obj_weight .* nlp.obj.hessian.vals
    view(vals, (nlp.obj.nnzh + 1):(nlp.meta.nnzh)) .= 0.0
  end
  if nlp.obj.type == "NONLINEAR"
    view(vals, 1:(nlp.obj.nnzh + nlp.quadcon.nnzh)) .= 0.0
    view(vals, (nlp.obj.nnzh + nlp.quadcon.nnzh + nlp.nlcon.nnzh + 1):(nlp.meta.nnzh)) .= 0.0
    ind_nnln = (nlp.obj.nnzh + nlp.quadcon.nnzh + 1):(nlp.obj.nnzh + nlp.quadcon.nnzh + nlp.nlcon.nnzh)
    MOI.eval_hessian_lagrangian(nlp.eval, view(vals, ind_nnln), x, obj_weight, nlp.λ)
  end
  return vals
end

function NLPModels.jth_hess_coord!(
  nlp::MathOptNLPModel,
  x::AbstractVector,
  j::Integer,
  vals::AbstractVector,
)
  nlp.meta.hess_available || error("The function jth_hess_coord! is not supported by this MathOptNLPModel.")
  increment!(nlp, :neval_jhess)
  @rangecheck 1 nlp.meta.ncon j
  vals .= 0.0

  # Quadratic constraints
  if nlp.meta.nlin + 1 ≤ j ≤ nlp.meta.nlin + nlp.quadcon.nquad
    index = nlp.obj.nnzh
    for i = 1:(nlp.quadcon.nquad)
      qcon = nlp.quadcon.constraints[i]
      if j == nlp.meta.nlin + i
        view(vals, (index + 1):(index + qcon.nnzh)) .= qcon.A.vals
      end
      index += qcon.nnzh
    end
  end

  # Non-oracle nonlinear constraints
  if nlp.meta.nlin + nlp.quadcon.nquad + 1 ≤ j ≤ nlp.meta.nlin + nlp.quadcon.nquad + nlp.nlcon.nnln
    nlp.λ[j - nlp.meta.nlin - nlp.quadcon.nquad] = 1.0
    index_nnzh = (nlp.obj.nnzh + nlp.quadcon.nnzh + 1):(nlp.meta.nnzh)
    MOI.eval_hessian_lagrangian(
      nlp.eval,
      view(vals, index_nnzh),
      x,
      0.0,
      nlp.λ,
    )
    nlp.λ[j - nlp.meta.nlin - nlp.quadcon.nquad] = 0.0
  end

  # Oracle nonlinear constraints
  if nlp.meta.nlin + nlp.quadcon.nquad + nlp.nlcon.nnln + 1 ≤ j ≤ nlp.meta.ncon
    offset_λ = nlp.meta.nlin + nlp.quadcon.nquad + nlp.nlcon.nnln
    offset_nnzh = nlp.obj.nnzh + nlp.quadcon.nnzh + nlp.nlcon.nnzh
    find_oracle = false
    for (f, s) in nlp.oracles.oracles
      if !find_oracle
        ncon_oracle = s.set.output_dimension
        nnzh_oracle = length(s.set.hessian_lagrangian_structure)
        if offset_λ + 1 ≤ j ≤ offset_λ + s.set.output_dimension
          find_oracle = true
          # build local x
          for i in 1:s.set.input_dimension
            s.x[i] = x[f.variables[i].value]
          end
          # We could preallocate the vector in the structure nlp
          λ_oracle = zeros(Float64, s.set.output_dimension)
          λ_oracle[j - offset_λ] = 1.0
          index_nnzh = (offset_nnzh + 1):(offset_nnzh + nnzh_oracle)
          s.set.eval_hessian_lagrangian(
            view(vals, index_nnzh),
            s.x,
            λ_oracle,
          )
        else
          offset_λ += ncon_oracle
          offset_nnzh += nnzh_oracle
        end
      end
    end
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
  nlp.meta.hprod_available || error("The function hprod! is not supported by this MathOptNLPModel.")
  increment!(nlp, :neval_hprod)
  if (nlp.obj.type == "LINEAR") && (nlp.meta.nnln == 0)
    hv .= 0.0
  end
  if (nlp.obj.type == "NONLINEAR") || (nlp.nlcon.nnln > 0)
    ind_nnln = (nlp.meta.nlin + nlp.quadcon.nquad + 1):(nlp.meta.nlin + nlp.quadcon.nquad + nlp.nlcon.nnln)
    λ = view(y, ind_nnln)
    MOI.eval_hessian_lagrangian_product(nlp.eval, hv, x, v, obj_weight, λ)
  end
  if nlp.obj.type == "QUADRATIC"
    (nlp.nlcon.nnln == 0) && (hv .= 0.0)
    coo_sym_add_mul!(
      nlp.obj.hessian.rows,
      nlp.obj.hessian.cols,
      nlp.obj.hessian.vals,
      v,
      hv,
      obj_weight,
    )
  end
  if nlp.quadcon.nquad > 0
    (nlp.obj.type == "LINEAR") && (nlp.nlcon.nnln == 0) && (hv .= 0.0)
    for i = 1:(nlp.quadcon.nquad)
      qcon = nlp.quadcon.constraints[i]
      coo_sym_add_mul!(qcon.A.rows, qcon.A.cols, qcon.A.vals, v, hv, y[nlp.meta.nlin + i])
    end
  end
  if nlp.oracles.ncon > 0
    (nlp.obj.type == "LINEAR") && (nlp.meta.nnln == nlp.oracles.ncon) && (hv .= 0.0)
    offset_y = nlp.meta.nlin + nlp.quadcon.nquad + nlp.nlcon.nnln
    for (f, s) in nlp.oracles.oracles
      for i in 1:s.set.input_dimension
        s.x[i] = x[f.variables[i].value]
      end
      ind_y = (offset_y + 1):(offset_y + s.set.output_dimension)
      y_oracle = view(y, ind_y)
      s.set.eval_hessian_lagrangian(s.nzH, s.x, y_oracle)
      k = 0
      for (r, c) in s.set.hessian_lagrangian_structure
        k += 1
        row = f.variables[r].value
        col = f.variables[c].value
        hv[row] += s.nzH[k] * v[col]
      end
      offset_y += s.set.output_dimension
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
  nlp.meta.hprod_available || error("The function hprod! is not supported by this MathOptNLPModel.")
  increment!(nlp, :neval_hprod)
  if nlp.obj.type == "LINEAR"
    hv .= 0.0
  end
  if nlp.obj.type == "QUADRATIC"
    hv .= 0.0
    coo_sym_add_mul!(
      nlp.obj.hessian.rows,
      nlp.obj.hessian.cols,
      nlp.obj.hessian.vals,
      v,
      hv,
      obj_weight,
    )
  end
  if nlp.obj.type == "NONLINEAR"
    MOI.eval_hessian_lagrangian_product(nlp.eval, hv, x, v, obj_weight, nlp.λ)
  end
  return hv
end

function NLPModels.jth_hprod!(
  nlp::MathOptNLPModel,
  x::AbstractVector,
  v::AbstractVector,
  j::Integer,
  hv::AbstractVector,
)
  nlp.meta.hprod_available || error("The function jth_hprod! is not supported by this MathOptNLPModel.")
  increment!(nlp, :neval_jhprod)
  @rangecheck 1 nlp.meta.ncon j
  hv .= 0.0
  if nlp.meta.nlin + 1 ≤ j ≤ nlp.meta.nlin + nlp.quadcon.nquad
    qcon = nlp.quadcon.constraints[j - nlp.meta.nlin]
    coo_sym_add_mul!(qcon.A.rows, qcon.A.cols, qcon.A.vals, v, hv, 1.0)
  elseif nlp.meta.nlin + nlp.quadcon.nquad + 1 ≤ j ≤ nlp.meta.nlin + nlp.quadcon.nquad + nlp.nlcon.nnln
    nlp.λ[j - nlp.meta.nlin - nlp.quadcon.nquad] = 1.0
    MOI.eval_hessian_lagrangian_product(nlp.eval, hv, x, v, 0.0, nlp.λ)
    nlp.λ[j - nlp.meta.nlin - nlp.quadcon.nquad] = 0.0
  elseif nlp.meta.nlin + nlp.quadcon.nquad + nlp.nlcon.nnln + 1 ≤ j ≤ nlp.meta.ncon
    find_oracle = false
    offset_λ = nlp.meta.nlin + nlp.quadcon.nquad + nlp.nlcon.nnln
    for (f, s) in nlp.oracles.oracles
      if !find_oracle
        if offset_λ + 1 ≤ j ≤ offset_λ + s.set.output_dimension
          find_oracle = true
          # build local x
          for i in 1:s.set.input_dimension
            s.x[i] = x[f.variables[i].value]
          end
          # We could preallocate the vector in the structure nlp
          λ_oracle = zeros(Float64, s.set.output_dimension)
          λ_oracle[j - offset_λ] = 1.0
          s.set.eval_hessian_lagrangian(
            s.nzH,
            s.x,
            λ_oracle,
          )
          k = 0
          for (r, c) in s.set.hessian_lagrangian_structure
            k += 1
            row = f.variables[r].value
            col = f.variables[c].value
            hv[row] += s.nzH[k] * v[col]
          end
        else
          offset_λ += ncon_oracle
        end
      end
    end
  end
  return hv
end

function NLPModels.ghjvprod!(
  nlp::MathOptNLPModel,
  x::AbstractVector,
  g::AbstractVector,
  v::AbstractVector,
  ghv::AbstractVector,
)
  nlp.meta.hprod_available || error("The function ghjvprod! is not supported by this MathOptNLPModel.")
  increment!(nlp, :neval_hprod)
  ghv .= 0.0
  for i = (nlp.meta.nlin + 1):(nlp.meta.nlin + nlp.quadcon.nquad)
    qcon = nlp.quadcon.constraints[i - nlp.meta.nlin]
    ghv[i] = coo_sym_dot(qcon.A.rows, qcon.A.cols, qcon.A.vals, g, v)
  end
  for i = (nlp.meta.nlin + nlp.quadcon.nquad + 1):(nlp.meta.ncon)
    jth_hprod!(nlp, x, v, i, nlp.hv)
    decrement!(nlp, :neval_jhprod)
    ghv[i] = dot(g, nlp.hv)
  end
  return ghv
end
