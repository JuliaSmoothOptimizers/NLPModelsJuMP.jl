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

# xᵀAx + bᵀx
mutable struct QuadraticConstraint
  A::COO
  b::SparseVector{Float64}
  g::Vector{Int}
  dg::Dict{Int, Int}
  nnzg::Int
  nnzh::Int
end

mutable struct QuadraticConstraints
  nquad::Int
  constraints::Vector{QuadraticConstraint}
  nnzj::Int
  nnzh::Int
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
    coo_unsym_add_mul!(transpose, rows, cols, vals, x, y, α)

Performs the update `y ← y + α * op(A) * x`, where `A` is an unsymmetric matrix in COO format given by `(rows, cols, vals)`.
If `transpose == true`, then `op(A) = Aᵀ`; otherwise, `op(A) = A`.
"""
function coo_unsym_add_mul!(
  transpose::Bool,
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
    if transpose
      y[j] += α * c * x[i]
    else
      y[i] += α * c * x[j]
    end
  end
  return y
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
    parser_SAF(fun, set, linrows, lincols, linvals, nlin, lin_lcon, lin_ucon)

Parse a `ScalarAffineFunction` fun with its associated set.
`linrows`, `lincols`, `linvals`, `lin_lcon` and `lin_ucon` are updated.
"""
function parser_SAF(fun, set, linrows, lincols, linvals, nlin, lin_lcon, lin_ucon)
  _index(v::MOI.VariableIndex) = v.value

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
    parser_VAF(fun, set, linrows, lincols, linvals, nlin, lin_lcon, lin_ucon)

Parse a `VectorAffineFunction` fun with its associated set.
`linrows`, `lincols`, `linvals`, `lin_lcon` and `lin_ucon` are updated.
"""
function parser_VAF(fun, set, linrows, lincols, linvals, nlin, lin_lcon, lin_ucon)
  _index(v::MOI.VariableIndex) = v.value

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

"""
    parser_SQF(fun, set, nvar, qcons, quad_lcon, quad_ucon)

Parse a `ScalarQuadraticFunction` fun with its associated set.
`qcons`, `quad_lcon`, `quad_ucon` are updated.
"""
function parser_SQF(fun, set, nvar, qcons, quad_lcon, quad_ucon)
  _index(v::MOI.VariableIndex) = v.value

  b = spzeros(Float64, nvar)
  rows = Int[]
  cols = Int[]
  vals = Float64[]

  # Parse a ScalarAffineTerm{Float64}(coefficient, variable_index)
  for term in fun.affine_terms
    b[_index(term.variable)] = term.coefficient
  end

  # Parse a ScalarQuadraticTerm{Float64}(coefficient, variable_index_1, variable_index_2)
  for term in fun.quadratic_terms
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

  if typeof(set) in (MOI.Interval{Float64}, MOI.GreaterThan{Float64})
    push!(quad_lcon, -fun.constant + set.lower)
  elseif typeof(set) == MOI.EqualTo{Float64}
    push!(quad_lcon, -fun.constant + set.value)
  else
    push!(quad_lcon, -Inf)
  end

  if typeof(set) in (MOI.Interval{Float64}, MOI.LessThan{Float64})
    push!(quad_ucon, -fun.constant + set.upper)
  elseif typeof(set) == MOI.EqualTo{Float64}
    push!(quad_ucon, -fun.constant + set.value)
  else
    push!(quad_ucon, Inf)
  end

  A = COO(rows, cols, vals)
  g = unique(vcat(rows, cols, b.nzind))  # sparsity pattern of Ax + b
  nnzg = length(g)
  # dg is a dictionary where:
  # - The key `r` specifies a row index in the vector Ax + b.
  # - The value `dg[r]` is a position in the vector (of length nnzg)
  # where the non-zero entries of the Jacobian for row `r` are stored.
  dg = Dict{Int, Int}(g[p] => p for p = 1:nnzg)
  nnzh = length(vals)
  qcon = QuadraticConstraint(A, b, g, dg, nnzg, nnzh)
  push!(qcons, qcon)
end

"""
    parser_VQF(fun, set, nvar, qcons, quad_lcon, quad_ucon)

Parse a `VectorQuadraticFunction` fun with its associated set.
`qcons`, `quad_lcon`, `quad_ucon` are updated.
"""
function parser_VQF(fun, set, nvar, qcons, quad_lcon, quad_ucon)
  _index(v::MOI.VariableIndex) = v.value

  ncon = length(fun.constants)
  for k = 1:ncon
    b = spzeros(Float64, nvar)
    rows = Int[]
    cols = Int[]
    vals = Float64[]

    # Parse a VectorAffineTerm{Float64}(output_index, scalar_term)
    for affine_term in fun.affine_terms
      if affine_term.output_index == k
        b[_index(affine_term.scalar_term.variable)] = affine_term.scalar_term.coefficient
      end
    end

    # Parse a VectorQuadraticTerm{Float64}(output_index, scalar_term)
    for quadratic_term in fun.quadratic_terms
      if quadratic_term.output_index == k
        i = _index(quadratic_term.scalar_term.variable_1)
        j = _index(quadratic_term.scalar_term.variable_2)
        if i ≥ j
          push!(rows, i)
          push!(cols, j)
        else
          push!(rows, j)
          push!(cols, i)
        end
        push!(vals, quadratic_term.scalar_term.coefficient)
      end
    end

    constant = fun.constants[k]

    if typeof(set) in (MOI.Nonnegatives, MOI.Zeros)
      append!(quad_lcon, constant)
    else
      append!(quad_lcon, -Inf)
    end

    if typeof(set) in (MOI.Nonpositives, MOI.Zeros)
      append!(quad_ucon, -constant)
    else
      append!(quad_ucon, Inf)
    end

    A = COO(rows, cols, vals)
    g = unique(vcat(rows, cols, b.nzind))  # sparsity pattern of Ax + b
    nnzg = length(g)
    # dg is a dictionary where:
    # - The key `r` specifies a row index in the vector Ax + b.
    # - The value `dg[r]` is a position in the vector (of length nnzg)
    # where the non-zero entries of the Jacobian for row `r` are stored.
    dg = Dict{Int, Int}(g[p] => p for p = 1:nnzg)
    nnzh = length(vals)
    qcon = QuadraticConstraint(A, b, g, dg, nnzg, nnzh)
    push!(qcons, qcon)
  end
end

"""
    parser_MOI(moimodel, variables)

Parse linear constraints of a `MOI.ModelLike`.
"""
function parser_MOI(moimodel, variables)

  # Number of variables
  nvar = length(variables)

  # Variables associated to linear constraints
  nlin = 0
  linrows = Int[]
  lincols = Int[]
  linvals = Float64[]
  lin_lcon = Float64[]
  lin_ucon = Float64[]

  # Variables associated to quadratic constraints
  nquad = 0
  qcons = QuadraticConstraint[]
  quad_lcon = Float64[]
  quad_ucon = Float64[]

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

    (F == VI) && continue
    conindices = MOI.get(moimodel, MOI.ListOfConstraintIndices{F, S}())
    for cidx in conindices
      fun = MOI.get(moimodel, MOI.ConstraintFunction(), cidx)
      set = MOI.get(moimodel, MOI.ConstraintSet(), cidx)
      if typeof(fun) <: SAF
        parser_SAF(fun, set, linrows, lincols, linvals, nlin, lin_lcon, lin_ucon)
        nlin += 1
      end
      if typeof(fun) <: VAF
        parser_VAF(fun, set, linrows, lincols, linvals, nlin, lin_lcon, lin_ucon)
        nlin += set.dimension
      end
      if typeof(fun) <: SQF
        parser_SQF(fun, set, nvar, qcons, quad_lcon, quad_ucon)
        nquad += 1
      end
      if typeof(fun) <: VQF
        parser_VQF(fun, set, nvar, qcons, quad_lcon, quad_ucon)
        nquad += set.dimension
      end
    end
  end
  coo = COO(linrows, lincols, linvals)
  lin_nnzj = length(linvals)
  lincon = LinearConstraints(coo, lin_nnzj)
  quad_nnzj = 0
  quad_nnzh = 0
  for i = 1:nquad
    quad_nnzj += qcons[i].nnzg
    quad_nnzh += qcons[i].nnzh
  end
  quadcon = QuadraticConstraints(nquad, qcons, quad_nnzj, quad_nnzh)

  return nlin, lincon, lin_lcon, lin_ucon, quadcon, quad_lcon, quad_ucon
end

# Affine or quadratic, nothing to do
_nlp_model(::MOI.Nonlinear.Model, ::MOI.ModelLike, ::Type, ::Type) = false

function _nlp_model(dest::MOI.Nonlinear.Model, src::MOI.ModelLike, F::Type{SNF}, S::Type)
  has_nonlinear = false
  for ci in MOI.get(src, MOI.ListOfConstraintIndices{F, S}())
    MOI.Nonlinear.add_constraint(
      dest,
      MOI.get(src, MOI.ConstraintFunction(), ci),
      MOI.get(src, MOI.ConstraintSet(), ci),
    )
    has_nonlinear = true
  end
  return has_nonlinear
end

function _nlp_model(model::MOI.ModelLike)::Union{Nothing, MOI.Nonlinear.Model}
  nlp_model = MOI.Nonlinear.Model()
  has_nonlinear = false
  for attr in MOI.get(model, MOI.ListOfModelAttributesSet())
    if attr isa MOI.UserDefinedFunction
      has_nonlinear = true
      args = MOI.get(model, attr)
      MOI.Nonlinear.register_operator(nlp_model, attr.name, attr.arity, args...)
    end
  end
  for (F, S) in MOI.get(model, MOI.ListOfConstraintTypesPresent())
    has_nonlinear |= _nlp_model(nlp_model, model, F, S)
  end
  F = MOI.get(model, MOI.ObjectiveFunctionType())
  if F <: SNF
    MOI.Nonlinear.set_objective(nlp_model, MOI.get(model, MOI.ObjectiveFunction{F}()))
    has_nonlinear = true
  end
  if !has_nonlinear
    return nothing
  end
  return nlp_model
end

function _nlp_block(model::MOI.ModelLike)
  # Old interface with `@NL...`
  nlp_data = MOI.get(model, MOI.NLPBlock())
  # New interface with `@constraint` and `@objective`
  nlp_model = _nlp_model(model)
  vars = MOI.get(model, MOI.ListOfVariableIndices())
  if isnothing(nlp_data)
    if isnothing(nlp_model)
      evaluator =
        MOI.Nonlinear.Evaluator(MOI.Nonlinear.Model(), MOI.Nonlinear.SparseReverseMode(), vars)
      nlp_data = MOI.NLPBlockData(evaluator)
    else
      backend = MOI.Nonlinear.SparseReverseMode()
      evaluator = MOI.Nonlinear.Evaluator(nlp_model, backend, vars)
      nlp_data = MOI.NLPBlockData(evaluator)
    end
  else
    if !isnothing(nlp_model)
      error(
        "Cannot optimize a model which contains the features from " *
        "both the legacy (macros beginning with `@NL`) and new " *
        "(`NonlinearExpr`) nonlinear interfaces. You must use one or " *
        "the other.",
      )
    end
  end
  return nlp_data
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
  MOI.initialize(eval, hessian ? [:Grad, :Jac, :JacVec, :Hess, :HessVec] : [:Grad, :Jac, :JacVec])

  jac = MOI.jacobian_structure(eval)
  jac_rows, jac_cols = getindex.(jac, 1), getindex.(jac, 2)
  nnzj = length(jac)

  hess = hessian ? MOI.hessian_lagrangian_structure(eval) : Tuple{Int, Int}[]
  hess_rows = hessian ? getindex.(hess, 1) : Int[]
  hess_cols = hessian ? getindex.(hess, 2) : Int[]
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
    set = MOI.get(moimodel, MOI.ConstraintSet(), ci)     # ::MOI.VectorNonlinearOracle{Float64}

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
  variables = MOI.get(model, MOI.ListOfVariableIndices())
  nvar = length(variables)
  lvar = zeros(nvar)
  uvar = zeros(nvar)

  # Initial solution
  x0 = zeros(nvar)
  has_start = MOI.VariablePrimalStart() in MOI.get(model, MOI.ListOfVariableAttributesSet())

  jump_variables = Dict{String,Int}()
  sizehint!(jump_variables, nvar)

  for vi in variables
    i = vi.value
    name = MOI.get(model, MOI.VariableName(), vi)
    jump_variables[name] = i

    lvar[i], uvar[i] = MOI.Utilities.get_bounds(model, Float64, vi)
    if has_start
      val = MOI.get(model, MOI.VariablePrimalStart(), vi)
      if val !== nothing
        x0[i] = val
      end
    end
  end

  return jump_variables, variables, nvar, lvar, uvar, x0
end

"""
    parser_objective_MOI(moimodel, variables)

Parse linear and quadratic objective of a `MOI.ModelLike`.
"""
function parser_objective_MOI(moimodel, variables)
  _index(v::MOI.VariableIndex) = v.value

  # Number of variables
  nvar = length(variables)

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
    parser_linear_expression(cmodel, variables, F)

Parse linear expressions of type `VariableRef` and `GenericAffExpr{Float64,VariableRef}`.
"""
function parser_linear_expression(cmodel, variables, F)

  # Number of variables
  nvar = length(variables)

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
  lls = parser_objective_MOI(moimodel, variables)
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
    parser_nonlinear_expression(cmodel, variables, F; hessian)

Parse nonlinear expressions of type `NonlinearExpression`.
"""
function parser_nonlinear_expression(cmodel, variables, F; hessian::Bool = true)

  # Number of variables
  nvar = length(variables)

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
