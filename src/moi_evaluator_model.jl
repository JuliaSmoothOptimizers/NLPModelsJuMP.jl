export EvaluatorNLPModel

"""
    _route_all_to_evaluator(backend::MOI.Nonlinear.AbstractAutomaticDifferentiation)

Hook for AD-backend extensions: return `true` if all supported constraints
(affine, quadratic, nonlinear, and vector-nonlinear-oracle) should be routed
to the backend's nonlinear model built by `MOI.Nonlinear.model(backend)`,
instead of parsing the affine and quadratic constraints into
[`LinearConstraints`](@ref) and [`QuadraticConstraints`](@ref).

Backends that exploit the structure of affine and quadratic constraints
themselves (for example `ExaModels.SIMDMode`) opt in by overriding
`MOI.Nonlinear.exploits_structure`; the model is then built with
[`evaluator_nlp_model`](@ref).
"""
function _route_all_to_evaluator(
  backend::MOI.Nonlinear.AbstractAutomaticDifferentiation,
)
  return MOI.Nonlinear.exploits_structure(backend)
end

"""
    EvaluatorNLPModel <: AbstractNLPModel{Float64, Vector{Float64}}

An `AbstractNLPModel` whose rows are all served by a single
`MOI.AbstractNLPEvaluator` built from an AD backend's model stack, instead of
the [`LinearConstraints`](@ref)/[`QuadraticConstraints`](@ref)/evaluator split
of [`MathOptNLPModel`](@ref).

The `lin`/`nln` split of the meta is derived from
`MOI.Nonlinear.constraint_linearity`, and the constant Jacobian block of the
linear rows is materialized once at construction, so `cons_lin!` and
`jac_lin_coord!` never trigger the evaluator. The unsplit methods (`cons!`,
`jac_coord!`, `hess_coord!`) are single full passes over the evaluator, so
fused evaluators pay no slicing cost on them.
"""
mutable struct EvaluatorNLPModel{E <: MOI.AbstractNLPEvaluator} <:
               AbstractNLPModel{Float64, Vector{Float64}}
  meta::NLPModelMeta{Float64, Vector{Float64}}
  evaluator::E
  has_jacvec::Bool
  has_hessvec::Bool
  # Full Jacobian and Hessian structures, in evaluator order.
  jac_rows::Vector{Int}
  jac_cols::Vector{Int}
  hess_rows::Vector{Int}
  hess_cols::Vector{Int}
  # Indices of the Jacobian entries that belong to linear and to nonlinear
  # rows, and the corresponding structures with rows renumbered to 1:nlin and
  # 1:nnln.
  lin_jac_index::Vector{Int}
  nln_jac_index::Vector{Int}
  lin_jac_rows::Vector{Int}
  lin_jac_cols::Vector{Int}
  nln_jac_rows::Vector{Int}
  nln_jac_cols::Vector{Int}
  # The materialized linear block: rows `meta.lin` are `A * x + b`, and the
  # values of their Jacobian entries are the constant `lin_vals`.
  lin_A::SparseMatrixCSC{Float64, Int}
  lin_b::Vector{Float64}
  lin_vals::Vector{Float64}
  # Scratch buffers for the full-pass-and-gather methods.
  g_buffer::Vector{Float64}
  j_buffer::Vector{Float64}
  counters::Counters
end

"""
    evaluator_nlp_model(moimodel::MOI.ModelLike; ad_backend, hessian, name)

Build an [`EvaluatorNLPModel`](@ref) by feeding the objective and every
constraint of `moimodel` into `MOI.Nonlinear.model(ad_backend)`.

The rows of the model are the rows of the resulting evaluator: the layers'
rows (for example vector-nonlinear-oracle constraints) come first, followed
by the scalar constraints in the order they appear in `moimodel`.
"""
function evaluator_nlp_model(
  moimodel::MOI.ModelLike;
  ad_backend::MOI.Nonlinear.AbstractAutomaticDifferentiation,
  hessian::Bool = true,
  name::String = "Generic",
)
  index_map, nvar, lvar, uvar, x0 = parser_variables(moimodel)
  nlp_model = MOI.Nonlinear.model(ad_backend)
  vars = MOI.get(moimodel, MOI.ListOfVariableIndices())
  # Constraints. The oracle rows come before the scalar rows, whatever the
  # order in `moimodel`, because the layers' rows come first.
  oracle_cis = Tuple{MOI.ConstraintIndex, Int}[]
  scalar_cis = MOI.ConstraintIndex[]
  oracle_rows = 0
  for (F, S) in MOI.get(moimodel, MOI.ListOfConstraintTypesPresent())
    if F == MOI.VariableIndex
      continue # Variable bounds are handled by `parser_variables`.
    elseif F == MOI.VectorOfVariables && S <: MOI.VectorNonlinearOracle
      for ci in MOI.get(moimodel, MOI.ListOfConstraintIndices{F, S}())
        func = MOI.get(moimodel, MOI.ConstraintFunction(), ci)
        set = MOI.get(moimodel, MOI.ConstraintSet(), ci)
        MOI.Nonlinear.add_constraint(nlp_model, func, set)
        push!(oracle_cis, (ci, oracle_rows))
        oracle_rows += set.output_dimension
      end
    else
      for ci in MOI.get(moimodel, MOI.ListOfConstraintIndices{F, S}())
        func = MOI.get(moimodel, MOI.ConstraintFunction(), ci)
        set = MOI.get(moimodel, MOI.ConstraintSet(), ci)
        MOI.Nonlinear.add_constraint(nlp_model, func, set)
        push!(scalar_cis, ci)
      end
    end
  end
  for (ci, offset) in oracle_cis
    index_map[ci] = typeof(ci)(offset + 1)
  end
  for (k, ci) in enumerate(scalar_cis)
    index_map[ci] = typeof(ci)(oracle_rows + k)
  end
  # Objective
  F = MOI.get(moimodel, MOI.ObjectiveFunctionType())
  sense = MOI.get(moimodel, MOI.ObjectiveSense())
  if sense != MOI.FEASIBILITY_SENSE
    MOI.Nonlinear.set_objective(
      nlp_model,
      MOI.get(moimodel, MOI.ObjectiveFunction{F}()),
    )
  end
  # Evaluator
  evaluator = MOI.Nonlinear.Evaluator(nlp_model, ad_backend, vars)
  features = MOI.features_available(evaluator)
  requested = [:Grad, :Jac, :JacVec, :HessVec]
  if hessian
    push!(requested, :Hess)
  end
  requested = intersect(requested, features)
  MOI.initialize(evaluator, requested)
  has_hess = :Hess in requested
  bounds = MOI.Nonlinear.constraint_bounds(evaluator)
  ncon = length(bounds)
  lcon = [b.lower for b in bounds]
  ucon = [b.upper for b in bounds]
  linearity = MOI.Nonlinear.constraint_linearity(evaluator)
  lin = if linearity === nothing
    Int[]
  else
    findall(==(MOI.Nonlinear.LINEAR), linearity)
  end
  # Jacobian structure, partitioned by row linearity.
  jac_structure = MOI.jacobian_structure(evaluator)
  nnzj = length(jac_structure)
  jac_rows = [r for (r, _) in jac_structure]
  jac_cols = [c for (_, c) in jac_structure]
  lin_pos = zeros(Int, ncon)
  for (k, r) in enumerate(lin)
    lin_pos[r] = k
  end
  nln = findall(iszero, lin_pos)
  nln_pos = zeros(Int, ncon)
  for (k, r) in enumerate(nln)
    nln_pos[r] = k
  end
  lin_jac_index = findall(k -> lin_pos[jac_rows[k]] > 0, 1:nnzj)
  nln_jac_index = findall(k -> lin_pos[jac_rows[k]] == 0, 1:nnzj)
  lin_jac_rows = [lin_pos[jac_rows[k]] for k in lin_jac_index]
  lin_jac_cols = jac_cols[lin_jac_index]
  nln_jac_rows = [nln_pos[jac_rows[k]] for k in nln_jac_index]
  nln_jac_cols = jac_cols[nln_jac_index]
  # Materialize the linear block: its Jacobian entries are x-independent, so
  # one evaluation gives the constant values, and one evaluation of the
  # constraints at zero gives the constants.
  j_buffer = zeros(nnzj)
  g_buffer = zeros(ncon)
  lin_b = zeros(length(lin))
  lin_vals = zeros(length(lin_jac_index))
  lin_A = SparseMatrixCSC{Float64, Int}(spzeros(length(lin), nvar))
  if !isempty(lin)
    MOI.eval_constraint_jacobian(evaluator, j_buffer, x0)
    lin_vals .= j_buffer[lin_jac_index]
    lin_A = sparse(lin_jac_rows, lin_jac_cols, lin_vals, length(lin), nvar)
    MOI.eval_constraint(evaluator, g_buffer, zeros(nvar))
    lin_b .= g_buffer[lin]
  end
  # Hessian structure, forced to the lower triangle.
  hess_structure = if has_hess
    MOI.hessian_lagrangian_structure(evaluator)
  else
    Tuple{Int, Int}[]
  end
  hess_rows = [max(r, c) for (r, c) in hess_structure]
  hess_cols = [min(r, c) for (r, c) in hess_structure]
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
    nnzh = length(hess_structure),
    lin = lin,
    lin_nnzj = length(lin_jac_index),
    nln_nnzj = length(nln_jac_index),
    minimize = sense != MOI.MAX_SENSE,
    islp = false,
    name = name,
  )
  nlp = EvaluatorNLPModel(
    meta,
    evaluator,
    :JacVec in requested,
    :HessVec in requested,
    jac_rows,
    jac_cols,
    hess_rows,
    hess_cols,
    lin_jac_index,
    nln_jac_index,
    lin_jac_rows,
    lin_jac_cols,
    nln_jac_rows,
    nln_jac_cols,
    lin_A,
    lin_b,
    lin_vals,
    g_buffer,
    j_buffer,
    Counters(),
  )
  return nlp, index_map
end

function NLPModels.obj(nlp::EvaluatorNLPModel, x::AbstractVector)
  increment!(nlp, :neval_obj)
  return MOI.eval_objective(nlp.evaluator, x)
end

function NLPModels.grad!(nlp::EvaluatorNLPModel, x::AbstractVector, g::AbstractVector)
  increment!(nlp, :neval_grad)
  MOI.eval_objective_gradient(nlp.evaluator, g, x)
  return g
end

function NLPModels.cons!(nlp::EvaluatorNLPModel, x::AbstractVector, c::AbstractVector)
  increment!(nlp, :neval_cons)
  MOI.eval_constraint(nlp.evaluator, c, x)
  return c
end

function NLPModels.cons_lin!(nlp::EvaluatorNLPModel, x::AbstractVector, c::AbstractVector)
  increment!(nlp, :neval_cons_lin)
  NLPModels.coo_prod!(nlp.lin_jac_rows, nlp.lin_jac_cols, nlp.lin_vals, x, c)
  c .+= nlp.lin_b
  return c
end

function NLPModels.cons_nln!(nlp::EvaluatorNLPModel, x::AbstractVector, c::AbstractVector)
  increment!(nlp, :neval_cons_nln)
  MOI.eval_constraint(nlp.evaluator, nlp.g_buffer, x)
  c .= view(nlp.g_buffer, nlp.meta.nln)
  return c
end

function NLPModels.jac_structure!(
  nlp::EvaluatorNLPModel,
  rows::AbstractVector{<:Integer},
  cols::AbstractVector{<:Integer},
)
  rows .= nlp.jac_rows
  cols .= nlp.jac_cols
  return rows, cols
end

function NLPModels.jac_lin_structure!(
  nlp::EvaluatorNLPModel,
  rows::AbstractVector{<:Integer},
  cols::AbstractVector{<:Integer},
)
  rows .= nlp.lin_jac_rows
  cols .= nlp.lin_jac_cols
  return rows, cols
end

function NLPModels.jac_nln_structure!(
  nlp::EvaluatorNLPModel,
  rows::AbstractVector{<:Integer},
  cols::AbstractVector{<:Integer},
)
  rows .= nlp.nln_jac_rows
  cols .= nlp.nln_jac_cols
  return rows, cols
end

function NLPModels.jac_coord!(nlp::EvaluatorNLPModel, x::AbstractVector, vals::AbstractVector)
  increment!(nlp, :neval_jac)
  MOI.eval_constraint_jacobian(nlp.evaluator, vals, x)
  return vals
end

function NLPModels.jac_lin_coord!(
  nlp::EvaluatorNLPModel,
  x::AbstractVector,
  vals::AbstractVector,
)
  increment!(nlp, :neval_jac_lin)
  vals .= nlp.lin_vals
  return vals
end

function NLPModels.jac_nln_coord!(
  nlp::EvaluatorNLPModel,
  x::AbstractVector,
  vals::AbstractVector,
)
  increment!(nlp, :neval_jac_nln)
  MOI.eval_constraint_jacobian(nlp.evaluator, nlp.j_buffer, x)
  vals .= view(nlp.j_buffer, nlp.nln_jac_index)
  return vals
end

function NLPModels.jprod!(
  nlp::EvaluatorNLPModel,
  x::AbstractVector,
  v::AbstractVector,
  Jv::AbstractVector,
)
  increment!(nlp, :neval_jprod)
  if nlp.has_jacvec
    MOI.eval_constraint_jacobian_product(nlp.evaluator, Jv, x, v)
  else
    MOI.eval_constraint_jacobian(nlp.evaluator, nlp.j_buffer, x)
    NLPModels.coo_prod!(nlp.jac_rows, nlp.jac_cols, nlp.j_buffer, v, Jv)
  end
  return Jv
end

function NLPModels.jtprod!(
  nlp::EvaluatorNLPModel,
  x::AbstractVector,
  v::AbstractVector,
  Jtv::AbstractVector,
)
  increment!(nlp, :neval_jtprod)
  if nlp.has_jacvec
    MOI.eval_constraint_jacobian_transpose_product(nlp.evaluator, Jtv, x, v)
  else
    MOI.eval_constraint_jacobian(nlp.evaluator, nlp.j_buffer, x)
    NLPModels.coo_prod!(nlp.jac_cols, nlp.jac_rows, nlp.j_buffer, v, Jtv)
  end
  return Jtv
end

function NLPModels.hess_structure!(
  nlp::EvaluatorNLPModel,
  rows::AbstractVector{<:Integer},
  cols::AbstractVector{<:Integer},
)
  rows .= nlp.hess_rows
  cols .= nlp.hess_cols
  return rows, cols
end

function NLPModels.hess_coord!(
  nlp::EvaluatorNLPModel,
  x::AbstractVector,
  y::AbstractVector,
  vals::AbstractVector;
  obj_weight::Real = 1.0,
)
  increment!(nlp, :neval_hess)
  MOI.eval_hessian_lagrangian(nlp.evaluator, vals, x, obj_weight, y)
  return vals
end

function NLPModels.hess_coord!(
  nlp::EvaluatorNLPModel,
  x::AbstractVector,
  vals::AbstractVector;
  obj_weight::Real = 1.0,
)
  increment!(nlp, :neval_hess)
  MOI.eval_hessian_lagrangian(
    nlp.evaluator,
    vals,
    x,
    obj_weight,
    zeros(nlp.meta.ncon),
  )
  return vals
end

function NLPModels.hprod!(
  nlp::EvaluatorNLPModel,
  x::AbstractVector,
  y::AbstractVector,
  v::AbstractVector,
  hv::AbstractVector;
  obj_weight::Real = 1.0,
)
  increment!(nlp, :neval_hprod)
  if !nlp.has_hessvec
    error(
      "The AD backend's evaluator does not support Hessian-vector products.",
    )
  end
  MOI.eval_hessian_lagrangian_product(nlp.evaluator, hv, x, v, obj_weight, y)
  return hv
end

function NLPModels.hprod!(
  nlp::EvaluatorNLPModel,
  x::AbstractVector,
  v::AbstractVector,
  hv::AbstractVector;
  obj_weight::Real = 1.0,
)
  return NLPModels.hprod!(nlp, x, zeros(nlp.meta.ncon), v, hv, obj_weight = obj_weight)
end
