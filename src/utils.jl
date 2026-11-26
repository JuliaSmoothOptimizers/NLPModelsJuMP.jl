using LinearAlgebra, SparseArrays

using NLPModels
import NLPModels.increment!, NLPModels.decrement!

using JuMP, MathOptInterface
const MOI = MathOptInterface

# VariableIndex
const VI = MOI.VariableIndex  # VariableIndex(value)

# ScalarAffineFunctions and VectorAffineFunctions
const SAF = MOI.ScalarAffineFunction{Float64}  # ScalarAffineFunction{T}(terms, constant)
const VAF = MOI.VectorAffineFunction{Float64}  # VectorAffineFunction{T}(terms, constants)
const AF = Union{SAF, VAF}

# ScalarQuadraticFunctions and VectorQuadraticFunctions
const SQF = MOI.ScalarQuadraticFunction{Float64}  # ScalarQuadraticFunction{T}(affine_terms, quadratic_terms, constant)
const VQF = MOI.VectorQuadraticFunction{Float64}  # VectorQuadraticFunction{T}(affine_terms, quadratic_terms, constants)
const QF = Union{SQF, VQF}

# ScalarNonlinearFunction and VectorNonlinearFunction
const SNF = MOI.ScalarNonlinearFunction
const VNF = MOI.VectorNonlinearFunction
const NF = Union{SNF, VNF}

# VectorNonlinearOracle
const ORACLE = MOI.VectorNonlinearOracle{Float64}  # VectorNonlinearOracle{Float64}(input_dimension, output_dimension, l, u, eval_f, jacobian_structure, eval_jacobian, hessian_lagrangian_structure, eval_hessian_lagrangian)

# Cache of VectorNonlinearOracle
mutable struct _VectorNonlinearOracleCache
  set::MOI.VectorNonlinearOracle{Float64}
  x::Vector{Float64}
  nzJ::Vector{Float64}
  nzH::Vector{Float64}

  function _VectorNonlinearOracleCache(set::MOI.VectorNonlinearOracle{Float64})
    nnzj = length(set.jacobian_structure)
    nnzh = length(set.hessian_lagrangian_structure)
    return new(set, zeros(set.input_dimension), zeros(nnzj), zeros(nnzh))
  end
end

# AffLinSets and VecLinSets
const ALS = Union{
  MOI.EqualTo{Float64},
  MOI.GreaterThan{Float64},
  MOI.LessThan{Float64},
  MOI.Interval{Float64},
}
const VLS = Union{MOI.Nonnegatives, MOI.Nonpositives, MOI.Zeros}
const LS = Union{ALS, VLS}

# Expressions
const VF = VariableRef
const AE = GenericAffExpr{Float64, VariableRef}
const LE = Union{VF, AE}
const QE = GenericQuadExpr{Float64, VariableRef}
const NLE = NonlinearExpression

const LinQuad = Union{VI, SAF, SQF}

# Sparse matrix in coordinate format
mutable struct COO
  rows::Vector{Int}
  cols::Vector{Int}
  vals::Vector{Float64}
end

COO() = COO(Int[], Int[], Float64[])

mutable struct LinearConstraints
  jacobian::COO
  nnzj::Int
end

"""
    NonLinearStructure

Structure containing Jacobian and Hessian structures of nonlinear constraints:
- nnln: number of nonlinear constraints
- nl_lcon: lower bounds of nonlinear constraints
- nl_ucon: upper bounds of nonlinear constraints
- jac_rows: row indices of the Jacobian in Coordinate format (COO) format
- jac_cols: column indices of the Jacobian in COO format
- nnzj: number of non-zero entries in the Jacobian
- hess_rows: row indices of the Hessian in COO format
- hess_cols: column indices of the Hessian in COO format
- nnzh: number of non-zero entries in the Hessian
"""
mutable struct NonLinearStructure
  nnln::Int
  nl_lcon::Vector{Float64}
  nl_ucon::Vector{Float64}
  jac_rows::Vector{Int}
  jac_cols::Vector{Int}
  nnzj::Int
  hess_rows::Vector{Int}
  hess_cols::Vector{Int}
  nnzh::Int
end

mutable struct LinearEquations
  jacobian::COO
  constants::Vector{Float64}
  nnzj::Int
end

mutable struct Objective
  type::String
  constant::Float64
  gradient::SparseVector{Float64}
  hessian::COO
  nnzh::Int
end

"""
    Oracles

Structure containing nonlinear oracles data:
- oracles: vector of tuples (MOI.VectorOfVariables, _VectorNonlinearOracleCache)
- ncon: number of scalar constraints represented by all oracles
- lcon: lower bounds of oracle constraints
- ucon: upper bounds of oracle constraints
- nnzj: number of non-zero entries in the Jacobian of all oracles
- nnzh: number of non-zero entries in the Hessian of all oracles
- nzJ: buffer to store the nonzeros of the Jacobian for all oracles (needed for the functions jprod and jtprod)
- nzH: buffer to store the nonzeros of the Hessian for all oracles (needed for the function hprod)
- hessian_oracles_supported: support of the Hessian for all oracles
"""
mutable struct Oracles
  oracles::Vector{Tuple{MOI.VectorOfVariables, _VectorNonlinearOracleCache}}
  ncon::Int
  lcon::Vector{Float64}
  ucon::Vector{Float64}
  nnzj::Int
  nnzh::Int
  hessian_oracles_supported::Bool
end

"""
    replace!(ex, x)

Walk the expression `ex` and substitute in the actual variables `x`.
"""
function replace!(ex, x)
  if isa(ex, Expr)
    for (i, arg) in enumerate(ex.args)
      if isa(arg, Expr)
        if arg.head == :ref && arg.args[1] == :x
          ex.args[i] = x[arg.args[2].value]
        else
          replace!(arg, x)
        end
      end
    end
  end
end

"""
    coo_sym_add_mul!(rows, cols, vals, x, y, α)

Perform the update `y ← y + α * A * x` where `A` is a symmetric matrix in COO format given by `(rows, cols, vals)`.
Only one triangle of `A` should be passed.
"""
function coo_sym_add_mul!(
  rows::AbstractVector{<:Integer},
  cols::AbstractVector{<:Integer},
  vals::AbstractVector,
  x::AbstractVector,
  y::AbstractVector,
  α::Float64,
)
  nnz = length(vals)
  @inbounds for k = 1:nnz
    i, j, c = rows[k], cols[k], vals[k]
    y[i] += α * c * x[j]
    if i ≠ j
      y[j] += α * c * x[i]
    end
  end
  return y
end

"""
    coo_sym_dot(rows, cols, vals, x, y)

Compute the product `xᵀAy` of a symmetric matrix `A` given by `(rows, cols, vals)`.
Only one triangle of `A` should be passed.
"""
function coo_sym_dot(
  rows::AbstractVector{<:Integer},
  cols::AbstractVector{<:Integer},
  vals::AbstractVector,
  x::AbstractVector,
  y::AbstractVector,
)
  xᵀAy = 0.0
  nnz = length(vals)
  @inbounds for k = 1:nnz
    i, j, c = rows[k], cols[k], vals[k]
    xᵀAy += c * x[i] * y[j]
    if i ≠ j
      xᵀAy += c * x[j] * y[i]
    end
  end
  return xᵀAy
end

"""
    parser_SAF(fun, set, linrows, lincols, linvals, nlin, lin_lcon, lin_ucon, index_map)

Parse a `ScalarAffineFunction` fun with its associated set.
`linrows`, `lincols`, `linvals`, `lin_lcon` and `lin_ucon` are updated.
"""
function parser_SAF(fun, set, linrows, lincols, linvals, nlin, lin_lcon, lin_ucon, index_map)
  _index(v::MOI.VariableIndex) = index_map[v].value

  # Parse a ScalarAffineTerm{Float64}(coefficient, variable)
  for term in fun.terms
    push!(linrows, nlin + 1)
    push!(lincols, _index(term.variable))
    push!(linvals, term.coefficient)
  end

  if typeof(set) in (MOI.Interval{Float64}, MOI.GreaterThan{Float64})
    push!(lin_lcon, -fun.constant + set.lower)
  elseif typeof(set) == MOI.EqualTo{Float64}
    push!(lin_lcon, -fun.constant + set.value)
  else
    push!(lin_lcon, -Inf)
  end

  if typeof(set) in (MOI.Interval{Float64}, MOI.LessThan{Float64})
    push!(lin_ucon, -fun.constant + set.upper)
  elseif typeof(set) == MOI.EqualTo{Float64}
    push!(lin_ucon, -fun.constant + set.value)
  else
    push!(lin_ucon, Inf)
  end
end

"""
    parser_VAF(fun, set, linrows, lincols, linvals, nlin, lin_lcon, lin_ucon, index_map)

Parse a `VectorAffineFunction` fun with its associated set.
`linrows`, `lincols`, `linvals`, `lin_lcon` and `lin_ucon` are updated.
"""
function parser_VAF(fun, set, linrows, lincols, linvals, nlin, lin_lcon, lin_ucon, index_map)
  _index(v::MOI.VariableIndex) = index_map[v].value

  # Parse a VectorAffineTerm{Float64}(output_index, scalar_term)
  for term in fun.terms
    push!(linrows, nlin + term.output_index)
    push!(lincols, _index(term.scalar_term.variable))
    push!(linvals, term.scalar_term.coefficient)
  end

  if typeof(set) in (MOI.Nonnegatives, MOI.Zeros)
    append!(lin_lcon, -fun.constants)
  else
    append!(lin_lcon, -Inf * ones(set.dimension))
  end

  if typeof(set) in (MOI.Nonpositives, MOI.Zeros)
    append!(lin_ucon, -fun.constants)
  else
    append!(lin_ucon, Inf * ones(set.dimension))
  end
end

function _add_quadratic_constraint(model, f::SQF, set)
  # Preserve the NLPModels convention of moving the constant into the bounds.
  f, set = MOI.Utilities.normalize_constant(f, set; allow_modify_function = true)
  if MOI.supports_constraint(model, SQF, typeof(set))
    return MOI.add_constraint(model, f, set)
  end
  return MOI.add_constraint(model, convert(SNF, f), set)
end

function _add_quadratic_constraint(model, f::VQF, set)
  scalar_set = MOI.Utilities.scalar_set_type(typeof(set), Float64)(0.0)
  for row in MOI.Utilities.scalarize(f)
    _add_quadratic_constraint(model, row, scalar_set)
  end
  return
end

# A legacy objective with no constraints need not implement any Jacobian callbacks.
# EvaluatorWithQuad still calls them on the inner evaluator when it combines rows.
struct _ObjectiveOnlyEvaluator{E} <: MOI.AbstractNLPEvaluator
  inner::E
end

MOI.features_available(d::_ObjectiveOnlyEvaluator) =
  union(MOI.features_available(d.inner), [:Jac, :JacVec])
function MOI.initialize(d::_ObjectiveOnlyEvaluator, features)
  return MOI.initialize(d.inner, setdiff(features, [:Jac, :JacVec]))
end
MOI.eval_objective(d::_ObjectiveOnlyEvaluator, x) = MOI.eval_objective(d.inner, x)
function MOI.eval_objective_gradient(d::_ObjectiveOnlyEvaluator, g, x)
  return MOI.eval_objective_gradient(d.inner, g, x)
end
MOI.eval_constraint(::_ObjectiveOnlyEvaluator, g, x) = nothing
MOI.jacobian_structure(::_ObjectiveOnlyEvaluator) = Tuple{Int, Int}[]
MOI.eval_constraint_jacobian(::_ObjectiveOnlyEvaluator, J, x) = nothing
MOI.eval_constraint_jacobian_product(::_ObjectiveOnlyEvaluator, y, x, v) = nothing
MOI.eval_constraint_jacobian_transpose_product(::_ObjectiveOnlyEvaluator, y, x, v) = fill!(y, 0)
function MOI.hessian_lagrangian_structure(d::_ObjectiveOnlyEvaluator)
  return MOI.hessian_lagrangian_structure(d.inner)
end
function MOI.eval_hessian_lagrangian(d::_ObjectiveOnlyEvaluator, H, x, σ, μ)
  return MOI.eval_hessian_lagrangian(d.inner, H, x, σ, μ)
end
function MOI.eval_hessian_lagrangian_product(d::_ObjectiveOnlyEvaluator, Hv, x, v, σ, μ)
  return MOI.eval_hessian_lagrangian_product(d.inner, Hv, x, v, σ, μ)
end

function _quadratic_and_legacy_block(model, legacy)
  if !legacy.has_objective && isempty(legacy.constraint_bounds)
    return MOI.NLPBlockData(MOI.Nonlinear.Evaluator(model, MOI.Nonlinear.SparseReverseMode()))
  end
  if legacy.has_objective
    model.objective_sink = MOI.Nonlinear._INNER
  end
  inner =
    isempty(legacy.constraint_bounds) ? _ObjectiveOnlyEvaluator(legacy.evaluator) : legacy.evaluator
  evaluator = MOI.Nonlinear.EvaluatorWithQuad(model, inner)
  bounds = MOI.Utilities.constraint_bounds(model)
  pairs = vcat(MOI.NLPBoundsPair.(bounds.lower, bounds.upper), legacy.constraint_bounds)
  return MOI.NLPBlockData(pairs, evaluator, legacy.has_objective)
end

"""
    parser_MOI(moimodel, index_map)

Parse linear constraints of a `MOI.ModelLike`.
"""
function parser_MOI(moimodel, index_map)

  # Variables associated to linear constraints
  nlin = 0
  linrows = Int[]
  lincols = Int[]
  linvals = Float64[]
  lin_lcon = Float64[]
  lin_ucon = Float64[]

  contypes = MOI.get(moimodel, MOI.ListOfConstraintTypesPresent())
  for (F, S) in contypes
    # Ignore VectorNonlinearOracle here, we'll parse it separately
    if F == MOI.VectorOfVariables && S <: MOI.VectorNonlinearOracle{Float64}
      continue
    end
    (F == VNF) && error(
      "The function $F is not supported. Please use `.<=`, `.==`, and `.>=` in your constraints to ensure compatibility with ScalarNonlinearFunction.",
    )
    F <: AF || F <: QF || F == SNF || F == VI || error("Function $F is not supported.")
    S <: LS || error("Set $S is not supported.")

    conindices = MOI.get(moimodel, MOI.ListOfConstraintIndices{F, S}())
    for cidx in conindices
      if F == VI
        fun = MOI.get(moimodel, MOI.ConstraintFunction(), cidx)
        index_map[cidx] = MOI.ConstraintIndex{F, S}(fun.value)
        continue
      end
      index_map[cidx] = MOI.ConstraintIndex{F, S}(nlin)
      if !(F <: AF)
        continue
      end
      fun = MOI.get(moimodel, MOI.ConstraintFunction(), cidx)
      set = MOI.get(moimodel, MOI.ConstraintSet(), cidx)
      if F == SAF
        parser_SAF(fun, set, linrows, lincols, linvals, nlin, lin_lcon, lin_ucon, index_map)
        nlin += 1
      else
        parser_VAF(fun, set, linrows, lincols, linvals, nlin, lin_lcon, lin_ucon, index_map)
        nlin += set.dimension
      end
    end
  end
  coo = COO(linrows, lincols, linvals)
  lin_nnzj = length(linvals)
  lincon = LinearConstraints(coo, lin_nnzj)
  return nlin, lincon, lin_lcon, lin_ucon
end

function _nlp_block(src, backend)
  types = MOI.get(src, MOI.ListOfConstraintTypesPresent())
  operators =
    filter(attr -> attr isa MOI.UserDefinedFunction, MOI.get(src, MOI.ListOfModelAttributesSet()))
  has_objective = MOI.get(src, MOI.ObjectiveFunctionType()) <: SNF
  legacy = MOI.get(src, MOI.NLPBlock())
  if isnothing(legacy)
    dest = MOI.Nonlinear.model(backend)
  else
    if has_objective || !isempty(operators) || any(t -> first(t) == SNF, types)
      error(
        "Cannot optimize a model which contains the features from " *
        "both the legacy (macros beginning with `@NL`) and new " *
        "(`NonlinearExpr`) nonlinear interfaces. You must use one or " *
        "the other.",
      )
    end
    if !any(t -> first(t) <: QF, types)
      return legacy
    end
    dest = MOI.Nonlinear.ModelWithQuad(MOI.Nonlinear.Model())
  end
  src_vars = MOI.get(src, MOI.ListOfVariableIndices())
  vars = MOI.add_variables(dest, length(src_vars))
  index_map = Dict(zip(src_vars, vars))
  for attr in operators
    MOI.set(dest, attr, MOI.get(src, attr))
  end
  # Insert the quadratic rows first, including for backends that keep insertion order.
  for (F, S) in types
    if F <: QF
      for ci in MOI.get(src, MOI.ListOfConstraintIndices{F, S}())
        f = MOI.Utilities.map_indices(index_map, MOI.get(src, MOI.ConstraintFunction(), ci))
        _add_quadratic_constraint(dest, f, MOI.get(src, MOI.ConstraintSet(), ci))
      end
    end
  end
  if !isnothing(legacy)
    return _quadratic_and_legacy_block(dest, legacy)
  end
  for (F, S) in types
    if F == SNF
      for ci in MOI.get(src, MOI.ListOfConstraintIndices{F, S}())
        f = MOI.get(src, MOI.ConstraintFunction(), ci)
        MOI.add_constraint(
          dest,
          MOI.Utilities.map_indices(index_map, f),
          MOI.get(src, MOI.ConstraintSet(), ci),
        )
      end
    end
  end
  if has_objective
    attr = MOI.ObjectiveFunction{SNF}()
    MOI.set(dest, attr, MOI.Utilities.map_indices(index_map, MOI.get(src, attr)))
    # NLPModelMeta stores the sense. Its evaluator must return the original objective.
    sense = MOI.get(src, MOI.ObjectiveSense())
    MOI.set(dest, MOI.ObjectiveSense(), sense == MOI.MAX_SENSE ? MOI.MIN_SENSE : sense)
  end
  evaluator = MOI.Nonlinear.Evaluator(dest, backend, vars)
  bounds = MOI.Utilities.constraint_bounds(dest)
  return MOI.NLPBlockData(MOI.NLPBoundsPair.(bounds.lower, bounds.upper), evaluator, has_objective)
end

"""
    parser_NL(nlp_data; hessian)

Parse nonlinear constraints of an `nlp_data`.

Returns:
- nlcon: NonLinearStructure containing Jacobian and Hessian structures
"""
function parser_NL(nlp_data; hessian::Bool = true)
  nnln = length(nlp_data.constraint_bounds)
  nl_lcon = Float64[bounds.lower for bounds in nlp_data.constraint_bounds]
  nl_ucon = Float64[bounds.upper for bounds in nlp_data.constraint_bounds]

  eval = nlp_data.evaluator
  features = MOI.features_available(eval)
  has_hessian = hessian && :Hess in features
  init_feat = Symbol[]
  if nlp_data.has_objective
    push!(init_feat, :Grad)
  end
  if nnln > 0
    push!(init_feat, :Jac)
  end
  if has_hessian
    push!(init_feat, :Hess)
  end
  if nnln > 0 && :JacVec in features
    push!(init_feat, :JacVec)
  end
  if hessian && :HessVec in features
    push!(init_feat, :HessVec)
  end

  MOI.initialize(eval, init_feat)

  jac = nnln > 0 ? MOI.jacobian_structure(eval) : Tuple{Int, Int}[]
  jac_rows, jac_cols = getindex.(jac, 1), getindex.(jac, 2)
  nnzj = length(jac)

  hess = has_hessian ? MOI.hessian_lagrangian_structure(eval) : Tuple{Int, Int}[]
  # MOI allows either triangle. NLPModels requires the lower triangle.
  hess_rows = [max(i, j) for (i, j) in hess]
  hess_cols = [min(i, j) for (i, j) in hess]
  nnzh = length(hess)
  nlcon =
    NonLinearStructure(nnln, nl_lcon, nl_ucon, jac_rows, jac_cols, nnzj, hess_rows, hess_cols, nnzh)

  return nlcon
end

"""
    oracles = parser_oracles(moimodel)

Parse nonlinear oracles of a `MOI.ModelLike`.
"""
function parser_oracles(moimodel)
  hessian_oracles_supported = true
  oracles = Tuple{MOI.VectorOfVariables, _VectorNonlinearOracleCache}[]
  lcon = Float64[]
  ucon = Float64[]

  # We know this pair exists from ListOfConstraintTypesPresent
  for ci in MOI.get(
    moimodel,
    MOI.ListOfConstraintIndices{MOI.VectorOfVariables, MOI.VectorNonlinearOracle{Float64}}(),
  )
    f = MOI.get(moimodel, MOI.ConstraintFunction(), ci)  # ::MOI.VectorOfVariables
    set = MOI.get(moimodel, MOI.ConstraintSet(), ci)       # ::MOI.VectorNonlinearOracle{Float64}

    cache = _VectorNonlinearOracleCache(set)
    push!(oracles, (f, cache))

    # Bounds: MOI.VectorNonlinearOracle stores them internally (l, u)
    append!(lcon, set.l)
    append!(ucon, set.u)

    # Support for the Hessian
    hessian_oracles_supported = hessian_oracles_supported && !isnothing(set.eval_hessian_lagrangian)
  end

  # Number of scalar constraints represented by all oracles
  ncon = length(lcon)

  # Number of nonzeros for the Jacobian and Hessian
  nnzj = 0
  nnzh = 0
  for (_, cache) in oracles
    nnzj += length(cache.set.jacobian_structure)
    # there may or may not be Hessian info
    nnzh += length(cache.set.hessian_lagrangian_structure)
  end

  return Oracles(oracles, ncon, lcon, ucon, nnzj, nnzh, hessian_oracles_supported)
end

"""
    parser_variables(model)

Parse variables informations of a `MOI.ModelLike`.
"""
function parser_variables(model::MOI.ModelLike)
  # Number of variables and bounds constraints
  vars = MOI.get(model, MOI.ListOfVariableIndices())
  nvar = length(vars)
  lvar = zeros(nvar)
  uvar = zeros(nvar)
  # Initial solution
  x0 = zeros(nvar)
  has_start = MOI.VariablePrimalStart() in MOI.get(model, MOI.ListOfVariableAttributesSet())

  index_map = MOI.Utilities.IndexMap()
  for (i, vi) in enumerate(vars)
    index_map[vi] = MOI.VariableIndex(i)
  end

  for (i, vi) in enumerate(vars)
    lvar[i], uvar[i] = MOI.Utilities.get_bounds(model, Float64, vi)
    if has_start
      val = MOI.get(model, MOI.VariablePrimalStart(), vi)
      if val !== nothing
        x0[i] = val
      end
    end
  end

  return index_map, nvar, lvar, uvar, x0
end

"""
    parser_objective_MOI(moimodel, nvar, index_map)

Parse linear and quadratic objective of a `MOI.ModelLike`.
"""
function parser_objective_MOI(moimodel, nvar, index_map)
  _index(v::MOI.VariableIndex) = index_map[v].value

  # Variables associated to linear and quadratic objective
  type = "UNKNOWN"
  constant = 0.0
  vect = spzeros(Float64, nvar)
  rows = Int[]
  cols = Int[]
  vals = Float64[]

  fobj = MOI.get(moimodel, MOI.ObjectiveFunction{LinQuad}())

  # Single Variable
  if typeof(fobj) == VI
    type = "LINEAR"
    vect[_index(fobj)] = 1.0
  end

  # Linear objective
  if typeof(fobj) == SAF
    type = "LINEAR"
    constant = fobj.constant
    for term in fobj.terms
      vect[_index(term.variable)] += term.coefficient
    end
  end

  # Quadratic objective
  if typeof(fobj) == SQF
    type = "QUADRATIC"
    constant = fobj.constant
    for term in fobj.affine_terms
      vect[_index(term.variable)] += term.coefficient
    end
    for term in fobj.quadratic_terms
      i = _index(term.variable_1)
      j = _index(term.variable_2)
      if i ≥ j
        push!(rows, i)
        push!(cols, j)
      else
        push!(rows, j)
        push!(cols, i)
      end
      push!(vals, term.coefficient)
    end
  end
  return Objective(type, constant, vect, COO(rows, cols, vals), length(vals))
end

"""
    parser_linear_expression(cmodel, nvar, index_map, F)

Parse linear expressions of type `VariableRef` and `GenericAffExpr{Float64,VariableRef}`.
"""
function parser_linear_expression(cmodel, nvar, index_map, F)

  # Variables associated to linear expressions
  rows = Int[]
  cols = Int[]
  vals = Float64[]
  constants = Float64[]

  # Linear least squares model
  nlinequ = 0
  F_is_array_of_containers = F isa Array{<:AbstractArray}
  if F_is_array_of_containers
    @objective(cmodel, Min, 0.0 + 0.5 * sum(sum(Fi^2 for Fi in FF if isa(Fi, LE)) for FF in F))
    for FF in F, expr in FF
      isa(expr, QE) && @warn("GenericQuadExpr{Float64, VariableRef} are not supported.")
      if isa(expr, AE)
        nlinequ += 1
        for (i, key) in enumerate(expr.terms.keys)
          push!(rows, nlinequ)
          push!(cols, key.index.value)
          push!(vals, expr.terms.vals[i])
        end
        push!(constants, expr.constant)
      end
      if isa(expr, VF)
        nlinequ += 1
        push!(rows, nlinequ)
        push!(cols, expr.index.value)
        push!(vals, 1.0)
        push!(constants, 0.0)
      end
    end
  else
    @objective(cmodel, Min, 0.0 + 0.5 * sum(Fi^2 for Fi in F if isa(Fi, LE)))
    for expr in F
      isa(expr, QE) && @warn("GenericQuadExpr{Float64, VariableRef} are not supported.")
      if isa(expr, AE)
        nlinequ += 1
        for (i, key) in enumerate(expr.terms.keys)
          push!(rows, nlinequ)
          push!(cols, key.index.value)
          push!(vals, expr.terms.vals[i])
        end
        push!(constants, expr.constant)
      end
      if isa(expr, VF)
        nlinequ += 1
        push!(rows, nlinequ)
        push!(cols, expr.index.value)
        push!(vals, 1.0)
        push!(constants, 0.0)
      end
    end
  end
  moimodel = backend(cmodel)
  lls = parser_objective_MOI(moimodel, nvar, index_map)
  return lls, LinearEquations(COO(rows, cols, vals), constants, length(vals)), nlinequ
end

"""
    add_constraint_model(Fmodel, Fi)

Add the nonlinear constraint `Fi == 0` to the model `Fmodel`.
If `Fi` is an Array, then we iterate over each component.
"""
function add_constraint_model(Fmodel, Fi::NLE)
  Fmodel.nlp_model.last_constraint_index += 1
  ci = MOI.Nonlinear.ConstraintIndex(Fmodel.nlp_model.last_constraint_index)
  index = Fi.index
  Fmodel.nlp_model.constraints[ci] =
    MOI.Nonlinear.Constraint(Fmodel.nlp_model.expressions[index], MOI.EqualTo{Float64}(0.0))
  return nothing
end

function add_constraint_model(Fmodel, Fi::LE)
  return nothing
end

function add_constraint_model(Fmodel, Fi::QE)
  @warn("GenericQuadExpr{Float64, VariableRef} are not supported.")
end

function add_constraint_model(Fmodel, Fi::AbstractArray)
  for Fj in Fi
    add_constraint_model(Fmodel, Fj)
  end
end

"""
    parser_nonlinear_expression(cmodel, nvar, F; hessian)

Parse nonlinear expressions of type `NonlinearExpression`.
"""
function parser_nonlinear_expression(cmodel, nvar, F; hessian::Bool = true)

  # Nonlinear least squares model
  F_is_array_of_containers = F isa Array{<:AbstractArray}
  nnlnequ = 0
  if F_is_array_of_containers
    nnlnequ = sum(sum(isa(Fi, NLE) for Fi in FF) for FF in F)
    if nnlnequ > 0
      @NLobjective(cmodel, Min, 0.5 * sum(sum(Fi^2 for Fi in FF if isa(Fi, NLE)) for FF in F))
    end
  else
    nnlnequ = sum(isa(Fi, NLE) for Fi in F)
    if nnlnequ > 0
      @NLobjective(cmodel, Min, 0.5 * sum(Fi^2 for Fi in F if isa(Fi, NLE)))
    end
  end

  Fmodel = JuMP.Model()
  @variable(Fmodel, x[1:nvar])
  JuMP._init_NLP(Fmodel)
  if cmodel.nlp_model ≠ nothing
    Fmodel.nlp_model.expressions = cmodel.nlp_model.expressions
    Fmodel.nlp_model.operators = cmodel.nlp_model.operators
    for Fi in F
      add_constraint_model(Fmodel, Fi)
    end
  end

  Feval = NLPEvaluator(Fmodel)
  MOI.initialize(Feval, hessian ? [:Grad, :Jac, :JacVec, :Hess, :HessVec] : [:Grad, :Jac, :JacVec])

  Fjac = Feval ≠ nothing ? MOI.jacobian_structure(Feval) : Tuple{Int, Int}[]
  Fjac_rows = Feval ≠ nothing ? getindex.(Fjac, 1) : Int[]
  Fjac_cols = Feval ≠ nothing ? getindex.(Fjac, 2) : Int[]
  nl_Fnnzj = length(Fjac)

  Fhess = hessian && Feval ≠ nothing ? MOI.hessian_lagrangian_structure(Feval) : Tuple{Int, Int}[]
  Fhess_rows = hessian && Feval ≠ nothing ? getindex.(Fhess, 1) : Int[]
  Fhess_cols = hessian && Feval ≠ nothing ? getindex.(Fhess, 2) : Int[]
  nl_Fnnzh = length(Fhess)

  nlequ = NonLinearStructure(
    nnlnequ,
    Float64[],
    Float64[],
    Fjac_rows,
    Fjac_cols,
    nl_Fnnzj,
    Fhess_rows,
    Fhess_cols,
    nl_Fnnzh,
  )

  return Feval, nlequ, nnlnequ
end

function _nlp_sync!(model::JuMP.Model)
  # With the old `@NL...` macros, the nlp model of the backend is not kept in
  # sync, so re-set it here as in `JuMP.optimize!`
  # If only the new nonlinear interface using `@constraint` and `@objective` is
  # used, `nlp` is `nothing` and we don't have to do anything
  nlp = JuMP.nonlinear_model(model)
  if !isnothing(nlp)
    evaluator = MOI.Nonlinear.Evaluator(
      # `force = true` is needed if there is not NL objective or constraint
      nlp,
      MOI.Nonlinear.SparseReverseMode(),
      JuMP.index.(JuMP.all_variables(model)),
    )
    MOI.set(model, MOI.NLPBlock(), MOI.NLPBlockData(evaluator))
  end
end
