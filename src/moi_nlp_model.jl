export MathOptNLPModel

mutable struct MathOptNLPModel <: AbstractNLPModel{Float64, Vector{Float64}}
  meta::NLPModelMeta{Float64, Vector{Float64}}
  eval::Union{MOI.AbstractNLPEvaluator, Nothing}
  lincon::LinearConstraints
  quadcon::QuadraticConstraints
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
  (eval ≠ nothing) && MOI.initialize(eval, hessian ? [:Grad, :Jac, :Hess, :HessVec] : [:Grad, :Jac])  # Add :JacVec when available

  nl_nnzj = nnln == 0 ? 0 : sum(length(nl_con.grad_sparsity) for nl_con in eval.constraints)
  nl_nnzh =
    hessian ?
    (((eval ≠ nothing) && eval.has_nlobj) ? length(eval.objective.hess_I) : 0) +
    (nnln == 0 ? 0 : sum(length(nl_con.hess_I) for nl_con in eval.constraints)) : 0

  moimodel = backend(jmodel)
  nlin, lincon, lin_lcon, lin_ucon, quadcon, quad_lcon, quad_ucon = parser_MOI(moimodel, nvar)

  if (eval ≠ nothing) && eval.has_nlobj
    obj = Objective("NONLINEAR", 0.0, spzeros(Float64, nvar), COO(), 0)
  else
    obj = parser_objective_MOI(moimodel, nvar)
  end

  ncon = nlin + quadcon.nquad + nnln
  lcon = vcat(lin_lcon, quad_lcon, nl_lcon)
  ucon = vcat(lin_ucon, quad_ucon, nl_ucon)
  nnzj = lincon.nnzj + quadcon.nnzj + nl_nnzj
  nnzh = obj.nnzh + quadcon.nnzh + nl_nnzh

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
    nln_nnzj = quadcon.nnzj + nl_nnzj,
    minimize = objective_sense(jmodel) == MOI.MIN_SENSE,
    islp = (obj.type == "LINEAR") && (nnln == 0) && (quadcon.nquad == 0),
    name = name,
  )

  return MathOptNLPModel(meta, eval, lincon, quadcon, obj, Counters())
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
  coo_prod!(
    nlp.lincon.jacobian.rows,
    nlp.lincon.jacobian.cols,
    nlp.lincon.jacobian.vals,
    x,
    c,
  )
  return c
end

function NLPModels.cons_nln!(nlp::MathOptNLPModel, x::AbstractVector, c::AbstractVector)
  increment!(nlp, :neval_cons_nln)
  for i = 1:(nlp.quadcon.nquad)
    qcon = nlp.quadcon[i]
    c[i] = 0.5 * coo_sym_dot(qcon.hessian.rows, qcon.hessian.cols, qcon.hessian.vals, x, x) + dot(qcon.b, x)
  end
  MOI.eval_constraint(nlp.eval, view(c, (nlp.quadcon.nquad + 1):(nlp.meta.nnln)), x)
  return c
end

function NLPModels.jac_lin_structure!(
  nlp::MathOptNLPModel,
  rows::AbstractVector{<:Integer},
  cols::AbstractVector{<:Integer},
)
  rows[1:(nlp.lincon.nnzj)] .= nlp.lincon.jacobian.rows[1:(nlp.lincon.nnzj)]
  cols[1:(nlp.lincon.nnzj)] .= nlp.lincon.jacobian.cols[1:(nlp.lincon.nnzj)]
  return rows, cols
end

function NLPModels.jac_nln_structure!(
  nlp::MathOptNLPModel,
  rows::AbstractVector{<:Integer},
  cols::AbstractVector{<:Integer},
)
  quad_nnzj, jrows, jcols = nlp.quadcon.nnzj, nlp.quadcon.jrows, nlp.quadcon.jcols
  rows[1:quad_nnzj] .= jrows
  cols[1:quad_nnzj] .= jcols
  jac_struct = MOI.jacobian_structure(nlp.eval)
  for index = (quad_nnzj + 1):(nlp.meta.nln_nnzj)
    row, col = jac_struct[index]
    rows[index] = row + nlp.quadcon.nquad
    cols[index] = col
  end
  return rows, cols
end

function NLPModels.jac_lin_coord!(nlp::MathOptNLPModel, x::AbstractVector, vals::AbstractVector)
  increment!(nlp, :neval_jac_lin)
  vals[1:(nlp.lincon.nnzj)] .= nlp.lincon.jacobian.vals[1:(nlp.lincon.nnzj)]
  return vals
end

function NLPModels.jac_nln_coord!(nlp::MathOptNLPModel, x::AbstractVector, vals::AbstractVector)
  increment!(nlp, :neval_jac_nln)
  quad_nnzj = nlp.quadcon.nnzj
  k = 0
  for i = 1:(nlp.quadcon.nquad)
    # rows of Qᵢx + bᵢ with nonzeros coefficients
    qcon = nlp.quadcon[i]
    vec = unique(qcon.hessian.rows ∪ qcon.b.nzind) # Can we improve here? Or store this information?
    nnzj = length(vec)
    res = similar(x) # Avoid extra allocation
    coo_sym_prod!(qcon.hessian.rows, qcon.hessian.cols, qcon.hessian.vals, x, res)
    vals[(k + 1):(k + nnzj)] .= res[vec] .+ qcon.b[vec]
    k += nnzj
  end
  MOI.eval_constraint_jacobian(nlp.eval, view(vals, (quad_nnzj + 1):(nlp.meta.nln_nnzj)), x)
  return vals
end

function NLPModels.jprod_lin!(
  nlp::MathOptNLPModel,
  x::AbstractVector,
  rows::AbstractVector{<:Integer},
  cols::AbstractVector{<:Integer},
  v::AbstractVector,
  Jv::AbstractVector,
)
  vals = jac_lin_coord(nlp, x)
  decrement!(nlp, :neval_jac_lin)
  jprod_lin!(nlp, rows, cols, vals, v, Jv)
  return Jv
end

function NLPModels.jprod_nln!(
  nlp::MathOptNLPModel,
  x::AbstractVector,
  rows::AbstractVector{<:Integer},
  cols::AbstractVector{<:Integer},
  v::AbstractVector,
  Jv::AbstractVector,
)
  vals = jac_nln_coord(nlp, x)
  decrement!(nlp, :neval_jac_nln)
  jprod_nln!(nlp, rows, cols, vals, v, Jv)
  return Jv
end

function NLPModels.jprod_lin!(
  nlp::MathOptNLPModel,
  x::AbstractVector,
  v::AbstractVector,
  Jv::AbstractVector,
)
  rows, cols = jac_lin_structure(nlp)
  jprod_lin!(nlp, x, rows, cols, v, Jv)
  return Jv
end

function NLPModels.jprod_nln!(
  nlp::MathOptNLPModel,
  x::AbstractVector,
  v::AbstractVector,
  Jv::AbstractVector,
)
  rows, cols = jac_nln_structure(nlp)
  jprod_nln!(nlp, x, rows, cols, v, Jv)
  return Jv
end

function NLPModels.jtprod!(
  nlp::MathOptNLPModel,
  x::AbstractVector,
  rows::AbstractVector{<:Integer},
  cols::AbstractVector{<:Integer},
  v::AbstractVector,
  Jtv::AbstractVector,
)
  vals = jac_coord(nlp, x)
  decrement!(nlp, :neval_jac)
  jtprod!(nlp, rows, cols, vals, v, Jtv)
  return Jtv
end

function NLPModels.jtprod!(
  nlp::MathOptNLPModel,
  x::AbstractVector,
  v::AbstractVector,
  Jtv::AbstractVector,
)
  (rows, cols) = jac_structure(nlp)
  jtprod!(nlp, x, rows, cols, v, Jtv)
  return Jtv
end

function NLPModels.jtprod_lin!(
  nlp::MathOptNLPModel,
  x::AbstractVector,
  rows::AbstractVector{<:Integer},
  cols::AbstractVector{<:Integer},
  v::AbstractVector,
  Jtv::AbstractVector,
)
  vals = jac_lin_coord(nlp, x)
  decrement!(nlp, :neval_jac_lin)
  jtprod_lin!(nlp, rows, cols, vals, v, Jtv)
  return Jtv
end

function NLPModels.jtprod_lin!(
  nlp::MathOptNLPModel,
  x::AbstractVector,
  v::AbstractVector,
  Jtv::AbstractVector,
)
  (rows, cols) = jac_lin_structure(nlp)
  jtprod_lin!(nlp, x, rows, cols, v, Jtv)
  return Jtv
end

function NLPModels.jtprod_nln!(
  nlp::MathOptNLPModel,
  x::AbstractVector,
  rows::AbstractVector{<:Integer},
  cols::AbstractVector{<:Integer},
  v::AbstractVector,
  Jtv::AbstractVector,
)
  vals = jac_nln_coord(nlp, x)
  decrement!(nlp, :neval_jac_nln)
  jtprod_nln!(nlp, rows, cols, vals, v, Jtv)
  return Jtv
end

function NLPModels.jtprod_nln!(
  nlp::MathOptNLPModel,
  x::AbstractVector,
  v::AbstractVector,
  Jtv::AbstractVector,
)
  (rows, cols) = jac_nln_structure(nlp)
  jtprod_nln!(nlp, x, rows, cols, v, Jtv)
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
    quad_nnzh = nlp.quadcon.nnzh
    rows[(1 + nlp.obj.nnzh):(nlp.obj.nnzh + quad_nnzh)] .= nlp.quadcon.hrows
    cols[(1 + nlp.obj.nnzh):(nlp.obj.nnzh + quad_nnzh)] .= nlp.quadcon.hcols
    hesslag_struct = MOI.hessian_lagrangian_structure(nlp.eval)
    for index = (nlp.obj.nnzh + quad_nnzh + 1):(nlp.meta.nnzh)
      shift_index = index - nlp.obj.nnzh - quad_nnzh
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
    vals[1:(nlp.obj.nnzh)] .= obj_weight .* nlp.obj.hessian.vals
  end
  if (nlp.obj.type == "NONLINEAR") || (nlp.meta.nnln > 0)
    quad_nnzh = nlp.quadcon.nnzh
    k = 0
    for i = 1:(nlp.quadcon.nquad)
      qcon = nlp.quadcon[i]
      nnzh = length(qcon.hessian.vals)
      vals[(k + 1):(k + nnzh)] .= qcon.hessian.vals .* y[nlp.meta.nlin + i]
      k += nnzh
    end
    MOI.eval_hessian_lagrangian(
      nlp.eval,
      view(vals, (nlp.obj.nnzh + quad_nnzh + 1):(nlp.meta.nnzh)),
      x,
      obj_weight,
      view(y, (nlp.meta.nlin + nlp.quadcon.nquad + 1):(nlp.meta.ncon)),
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
    vals[1:(nlp.obj.nnzh)] .= obj_weight .* nlp.obj.hessian.vals
    vals[(nlp.obj.nnzh + 1):(nlp.meta.nnzh)] .= 0.0
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
    for i = 1:(nlp.quadcon.nquad)
      qcon = nlp.quadcon[i]
      res = similar(x) # Avoid extra allocation
      coo_sym_prod!(qcon.hessian.rows, qcon.hessian.cols, qcon.hessian.vals, v, res)
      hv .+= res .* y[nlp.meta.nlin + i]
    end
    ind_nln = (nlp.meta.nlin + nlp.quadcon.nquad + 1):(nlp.meta.ncon)
    MOI.eval_hessian_lagrangian_product(nlp.eval, hv, x, v, obj_weight, view(y, ind_nln))
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
    nnln = nlp.meta.nnln - nlp.quadcon.nquad
    MOI.eval_hessian_lagrangian_product(nlp.eval, hv, x, v, obj_weight, zeros(nnln))
  end
  return hv
end
