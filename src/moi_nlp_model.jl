export MathOptNLPModel

mutable struct MathOptNLPModel <: AbstractNLPModel{Float64, Vector{Float64}}
  meta::NLPModelMeta{Float64, Vector{Float64}}
  eval::MOI.AbstractNLPEvaluator
  lincon::LinearConstraints
  nlcon::NonLinearStructure
  oracles::Oracles
  λ::Vector{Float64}
  hv::Vector{Float64}
  obj::Objective
  counters::Counters
end

"""
    MathOptNLPModel(model; hessian=true, name="Generic", backend=MOI.Nonlinear.SparseReverseMode())

Construct a `MathOptNLPModel` from a `JuMP` model. The `backend` selects the
model and evaluator for scalar nonlinear objectives and for quadratic and nonlinear constraints.
A legacy `MOI.NLPBlock` supplies its own evaluator.

`hessian` should be set to `false` for multivariate user-defined functions registered without hessian.
"""
function MathOptNLPModel(jmodel::JuMP.Model; kws...)
  _nlp_sync!(jmodel)
  return MathOptNLPModel(backend(jmodel); kws...)
end

function MathOptNLPModel(moimodel::MOI.ModelLike; kws...)
  return nlp_model(moimodel; kws...)[1]
end

function nlp_model(
  moimodel::MOI.ModelLike;
  hessian::Bool = true,
  name::String = "Generic",
  backend = MOI.Nonlinear.SparseReverseMode(),
)
  index_map, nvar, lvar, uvar, x0 = parser_variables(moimodel)
  nlin, lincon, lin_lcon, lin_ucon = parser_MOI(moimodel, index_map)

  nlp_data = _nlp_block(moimodel, backend)
  nlcon = parser_NL(nlp_data, hessian = hessian)
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
  ncon = nlin + nlcon.nnln + oracles.ncon
  lcon = vcat(lin_lcon, nlcon.nl_lcon, oracles.lcon)
  ucon = vcat(lin_ucon, nlcon.nl_ucon, oracles.ucon)
  nnzj = lincon.nnzj + nlcon.nnzj + oracles.nnzj
  nnzh = obj.nnzh + nlcon.nnzh + oracles.nnzh
  features = MOI.features_available(nlp_data.evaluator)
  has_nonlinear = nlp_data.has_objective || nlcon.nnln > 0
  jac_products = ncon > 0 && (nlcon.nnln == 0 || :JacVec in features)
  second_order = hessian && oracles.hessian_oracles_supported
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
    nln_nnzj = nlcon.nnzj + oracles.nnzj,
    minimize = MOI.get(moimodel, MOI.ObjectiveSense()) == MOI.MIN_SENSE,
    islp = (obj.type == "LINEAR") && (nlin == ncon),
    name = name,
    jprod_available = jac_products,
    jtprod_available = jac_products,
    hprod_available = second_order && (!has_nonlinear || :HessVec in features),
    hess_available = second_order && (!has_nonlinear || :Hess in features),
  )

  return MathOptNLPModel(meta, nlp_data.evaluator, lincon, nlcon, oracles, λ, hv, obj, counters),
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

function NLPModels.hess_structure!(
  nlp::MathOptNLPModel,
  rows::AbstractVector{<:Integer},
  cols::AbstractVector{<:Integer},
)
  nlp.meta.hess_available ||
    error("The function hess_structure! is not supported by this MathOptNLPModel.")
  if nlp.obj.type == "QUADRATIC"
    view(rows, 1:(nlp.obj.nnzh)) .= nlp.obj.hessian.rows
    view(cols, 1:(nlp.obj.nnzh)) .= nlp.obj.hessian.cols
  end
  index = nlp.obj.nnzh
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
  nlp.meta.hess_available ||
    error("The function hess_coord! is not supported by this MathOptNLPModel.")
  increment!(nlp, :neval_hess)

  if nlp.obj.type == "QUADRATIC"
    view(vals, 1:nlp.obj.nnzh) .= obj_weight .* nlp.obj.hessian.vals
  end

  if (nlp.obj.type == "NONLINEAR") || (nlp.nlcon.nnln > 0)
    ind_nnln = (nlp.meta.nlin + 1):(nlp.meta.nlin + nlp.nlcon.nnln)
    λ_nnln = view(y, ind_nnln)
    ind_nnzh = (nlp.obj.nnzh + 1):(nlp.obj.nnzh + nlp.nlcon.nnzh)
    MOI.eval_hessian_lagrangian(nlp.eval, view(vals, ind_nnzh), x, obj_weight, λ_nnln)
  end

  if nlp.oracles.ncon > 0
    offset_λ = nlp.meta.nlin + nlp.nlcon.nnln
    offset_nnzh = nlp.obj.nnzh + nlp.nlcon.nnzh

    for (f, s) in nlp.oracles.oracles
      # build local x for this oracle
      for i = 1:s.set.input_dimension
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
  nlp.meta.hess_available ||
    error("The function hess_coord! is not supported by this MathOptNLPModel.")
  increment!(nlp, :neval_hess)
  if nlp.obj.type == "LINEAR"
    vals .= 0.0
  end
  if nlp.obj.type == "QUADRATIC"
    view(vals, 1:(nlp.obj.nnzh)) .= obj_weight .* nlp.obj.hessian.vals
    view(vals, (nlp.obj.nnzh + 1):(nlp.meta.nnzh)) .= 0.0
  end
  if nlp.obj.type == "NONLINEAR"
    view(vals, (nlp.obj.nnzh + nlp.nlcon.nnzh + 1):(nlp.meta.nnzh)) .= 0.0
    ind_nnln = (nlp.obj.nnzh + 1):(nlp.obj.nnzh + nlp.nlcon.nnzh)
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
  nlp.meta.hess_available ||
    error("The function jth_hess_coord! is not supported by this MathOptNLPModel.")
  increment!(nlp, :neval_jhess)
  @rangecheck 1 nlp.meta.ncon j
  vals .= 0.0

  # Non-oracle nonlinear constraints
  if nlp.meta.nlin + 1 ≤ j ≤ nlp.meta.nlin + nlp.nlcon.nnln
    nlp.λ[j - nlp.meta.nlin] = 1.0
    index_nnzh = (nlp.obj.nnzh + 1):(nlp.obj.nnzh + nlp.nlcon.nnzh)
    MOI.eval_hessian_lagrangian(nlp.eval, view(vals, index_nnzh), x, 0.0, nlp.λ)
    nlp.λ[j - nlp.meta.nlin] = 0.0
  end

  # Oracle nonlinear constraints
  if nlp.meta.nlin + nlp.nlcon.nnln + 1 ≤ j ≤ nlp.meta.ncon
    offset_λ = nlp.meta.nlin + nlp.nlcon.nnln
    offset_nnzh = nlp.obj.nnzh + nlp.nlcon.nnzh
    find_oracle = false
    for (f, s) in nlp.oracles.oracles
      if !find_oracle
        ncon_oracle = s.set.output_dimension
        nnzh_oracle = length(s.set.hessian_lagrangian_structure)
        if offset_λ + 1 ≤ j ≤ offset_λ + s.set.output_dimension
          find_oracle = true
          # build local x
          for i = 1:s.set.input_dimension
            s.x[i] = x[f.variables[i].value]
          end
          # We could preallocate the vector in the structure nlp
          λ_oracle = zeros(Float64, s.set.output_dimension)
          λ_oracle[j - offset_λ] = 1.0
          index_nnzh = (offset_nnzh + 1):(offset_nnzh + nnzh_oracle)
          s.set.eval_hessian_lagrangian(view(vals, index_nnzh), s.x, λ_oracle)
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
  if (nlp.obj.type == "NONLINEAR") || (nlp.nlcon.nnln > 0)
    ind_nnln = (nlp.meta.nlin + 1):(nlp.meta.nlin + nlp.nlcon.nnln)
    λ = view(y, ind_nnln)
    MOI.eval_hessian_lagrangian_product(nlp.eval, hv, x, v, obj_weight, λ)
  else
    fill!(hv, 0.0)
  end
  if nlp.obj.type == "QUADRATIC"
    coo_sym_add_mul!(
      nlp.obj.hessian.rows,
      nlp.obj.hessian.cols,
      nlp.obj.hessian.vals,
      v,
      hv,
      obj_weight,
    )
  end
  if nlp.oracles.ncon > 0
    offset_y = nlp.meta.nlin + nlp.nlcon.nnln
    for (f, s) in nlp.oracles.oracles
      for i = 1:s.set.input_dimension
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
        if row != col
          hv[col] += s.nzH[k] * v[row]
        end
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
  nlp.meta.hprod_available ||
    error("The function jth_hprod! is not supported by this MathOptNLPModel.")
  increment!(nlp, :neval_jhprod)
  @rangecheck 1 nlp.meta.ncon j
  hv .= 0.0
  if nlp.meta.nlin < j ≤ nlp.meta.nlin + nlp.nlcon.nnln
    nlp.λ[j - nlp.meta.nlin] = 1.0
    MOI.eval_hessian_lagrangian_product(nlp.eval, hv, x, v, 0.0, nlp.λ)
    nlp.λ[j - nlp.meta.nlin] = 0.0
  elseif nlp.meta.nlin + nlp.nlcon.nnln + 1 ≤ j ≤ nlp.meta.ncon
    find_oracle = false
    offset_λ = nlp.meta.nlin + nlp.nlcon.nnln
    for (f, s) in nlp.oracles.oracles
      if !find_oracle
        if offset_λ + 1 ≤ j ≤ offset_λ + s.set.output_dimension
          find_oracle = true
          # build local x
          for i = 1:s.set.input_dimension
            s.x[i] = x[f.variables[i].value]
          end
          # We could preallocate the vector in the structure nlp
          λ_oracle = zeros(Float64, s.set.output_dimension)
          λ_oracle[j - offset_λ] = 1.0
          s.set.eval_hessian_lagrangian(s.nzH, s.x, λ_oracle)
          k = 0
          for (r, c) in s.set.hessian_lagrangian_structure
            k += 1
            row = f.variables[r].value
            col = f.variables[c].value
            hv[row] += s.nzH[k] * v[col]
            if row != col
              hv[col] += s.nzH[k] * v[row]
            end
          end
        else
          offset_λ += s.set.output_dimension
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
  nlp.meta.hprod_available ||
    error("The function ghjvprod! is not supported by this MathOptNLPModel.")
  increment!(nlp, :neval_hprod)
  ghv .= 0.0
  for i = (nlp.meta.nlin + 1):(nlp.meta.ncon)
    jth_hprod!(nlp, x, v, i, nlp.hv)
    decrement!(nlp, :neval_jhprod)
    ghv[i] = dot(g, nlp.hv)
  end
  return ghv
end
