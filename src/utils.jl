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

mutable struct NonLinearStructure
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

Update of the form `y ← y + αAx` where `A` is a symmetric matrix given by `(rows, cols, vals)`.
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

"""
    parser_SQF(fun, set, nvar, qcons, quad_lcon, quad_ucon, index_map)

Parse a `ScalarQuadraticFunction` fun with its associated set.
`qcons`, `quad_lcon`, `quad_ucon` are updated.
"""
function parser_SQF(fun, set, nvar, qcons, quad_lcon, quad_ucon, index_map)
  _index(v::MOI.VariableIndex) = index_map[v].value

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
    parser_VQF(fun, set, nvar, qcons, quad_lcon, quad_ucon, index_map)

Parse a `VectorQuadraticFunction` fun with its associated set.
`qcons`, `quad_lcon`, `quad_ucon` are updated.
"""
function parser_VQF(fun, set, nvar, qcons, quad_lcon, quad_ucon, index_map)
  _index(v::MOI.VariableIndex) = index_map[v].value

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
    parser_MOI(moimodel, index_map, nvar)

Parse linear constraints of a `MOI.ModelLike`.
"""
function parser_MOI(moimodel, index_map, nvar)

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
    F <: AF ||
      F <: QF ||
      F == MOI.ScalarNonlinearFunction ||
      F == VI ||
      error("Function $F is not supported.")
    S <: LS || error("Set $S is not supported.")

    conindices = MOI.get(moimodel, MOI.ListOfConstraintIndices{F, S}())
    for cidx in conindices
      fun = MOI.get(moimodel, MOI.ConstraintFunction(), cidx)
      if F == VI
        index_map[cidx] = MOI.ConstraintIndex{F, S}(fun.value)
        continue
      else
        index_map[cidx] = MOI.ConstraintIndex{F, S}(nlin)
      end
      set = MOI.get(moimodel, MOI.ConstraintSet(), cidx)
      if typeof(fun) <: SAF
        parser_SAF(fun, set, linrows, lincols, linvals, nlin, lin_lcon, lin_ucon, index_map)
        nlin += 1
      end
      if typeof(fun) <: VAF
        parser_VAF(fun, set, linrows, lincols, linvals, nlin, lin_lcon, lin_ucon, index_map)
        nlin += set.dimension
      end
      if typeof(fun) <: SQF
        parser_SQF(fun, set, nvar, qcons, quad_lcon, quad_ucon, index_map)
        nquad += 1
      end
      if typeof(fun) <: VQF
        parser_VQF(fun, set, nvar, qcons, quad_lcon, quad_ucon, index_map)
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
function _nlp_model(::Union{Nothing, MOI.Nonlinear.Model}, ::MOI.ModelLike, ::Type, ::Type) end

function _nlp_model(
  dest::Union{Nothing, MOI.Nonlinear.Model},
  src::MOI.ModelLike,
  F::Type{<:Union{MOI.ScalarNonlinearFunction, MOI.VectorNonlinearFunction}},
  S::Type,
)
  for ci in MOI.get(src, MOI.ListOfConstraintIndices{F, S}())
    if isnothing(dest)
      dest = MOI.Nonlinear.Model()
    end
    MOI.Nonlinear.add_constraint(
      dest,
      MOI.get(src, MOI.ConstraintFunction(), ci),
      MOI.get(src, MOI.ConstraintSet(), ci),
    )
  end
  return dest
end

function _nlp_model(model::MOI.ModelLike)
  nlp_model = nothing
  for (F, S) in MOI.get(model, MOI.ListOfConstraintTypesPresent())
    nlp_model = _nlp_model(nlp_model, model, F, S)
  end
  F = MOI.get(model, MOI.ObjectiveFunctionType())
  if F <: MOI.ScalarNonlinearFunction
    if isnothing(nlp_model)
      nlp_model = MOI.Nonlinear.Model()
    end
    attr = MOI.ObjectiveFunction{F}()
    MOI.Nonlinear.set_objective(nlp_model, MOI.get(model, attr))
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
  nlcon = NonLinearStructure(jac_rows, jac_cols, nnzj, hess_rows, hess_cols, nnzh)

  return nnln, nlcon, nl_lcon, nl_ucon
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

  nlequ = NonLinearStructure(Fjac_rows, Fjac_cols, nl_Fnnzj, Fhess_rows, Fhess_cols, nl_Fnnzh)

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
