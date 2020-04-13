using LinearAlgebra, SparseArrays

using NLPModels
import NLPModels.increment!, NLPModels.decrement!

using JuMP, MathOptInterface
const MOI = MathOptInterface

# ScalarAffineFunctions and VectorAffineFunctions
const SAF = MOI.ScalarAffineFunction{Float64}
const VAF = MOI.VectorAffineFunction{Float64}
const AF  = Union{SAF, VAF}

# AffLinSets and VecLinSets
const ALS = Union{MOI.EqualTo{Float64}, MOI.GreaterThan{Float64}, MOI.LessThan{Float64}, MOI.Interval{Float64}}
const VLS = Union{MOI.Nonnegatives, MOI.Nonpositives, MOI.Zeros}
const LS  = Union{ALS, VLS}

const SV  = MOI.SingleVariable
const SQF = MOI.ScalarQuadraticFunction{Float64}
const OBJ = Union{SV, SAF, SQF}

mutable struct COO
  rows :: Vector{Int}
  cols :: Vector{Int}
  vals :: Vector{Float64}
end

COO() = COO(Int[], Int[], Float64[])

mutable struct LinearConstraints
  jacobian :: COO
  nnzj     :: Int
end

mutable struct Objective
  type     :: String
  constant :: Float64
  gradient :: SparseVector{Float64}
  hessian  :: COO
  nnzh     :: Int
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
    coo_sym_dot(rows, cols, vals, x, y)

Compute the product `xᵀAy` of a symmetric matrix `A` given by `(rows, cols, vals)`
Only one triangle of `A` should be passed.
"""
function coo_sym_dot(rows :: AbstractVector{<: Integer}, cols :: AbstractVector{<: Integer}, vals :: AbstractVector, x :: AbstractVector, y :: AbstractVector)
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

  # Parse a ScalarAffineTerm{Float64}(coefficient, variable_index)
  for term in fun.terms
    push!(linrows, nlin + 1)
    push!(lincols, term.variable_index.value)
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

  # Parse a VectorAffineTerm{Float64}(output_index, scalar_term)
  for term in fun.terms
    push!(linrows, nlin + term.output_index)
    push!(lincols, term.scalar_term.variable_index.value)
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
    parser_MOI(moimodel)

Parse linear constraints of a `MOI.ModelLike`.
"""
function parser_MOI(moimodel)

  # Variables associated to linear constraints
  nlin     = 0
  linrows  = Int[]
  lincols  = Int[]
  linvals  = Float64[]
  lin_lcon = Float64[]
  lin_ucon = Float64[]

  contypes = MOI.get(moimodel, MOI.ListOfConstraints())
  for (F, S) in contypes
    F == MOI.SingleVariable && continue
    F <: AF || @warn("Function $F is not supported.")
    S <: LS || @warn("Set $S is not supported.")

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
    end
  end
  coo = COO(linrows, lincols, linvals)
  nnzj = length(linvals)
  lincon = LinearConstraints(coo, nnzj)

  return nlin, lincon, lin_lcon, lin_ucon
end

"""
    parser_JuMP(jmodel)

Parse variables informations and non-linear constaints of a `JuMP.Model`.
"""
function parser_JuMP(jmodel)

  # Number of variables and bounds constraints
  nvar = Int(num_variables(jmodel))
  vars = all_variables(jmodel)
  lvar = map(var -> has_lower_bound(var) ? lower_bound(var) : -Inf, vars)
  uvar = map(var -> has_upper_bound(var) ? upper_bound(var) :  Inf, vars)

  # Initial solution
  x0 = zeros(nvar)
  for (i, val) ∈ enumerate(start_value.(vars))
    if val !== nothing
      x0[i] = val
    end
  end

  # Variables associated to non-linear constraints
  nnln = num_nl_constraints(jmodel)
  nl_cons = jmodel.nlp_data.nlconstr
  nl_lcon = map(nl_con -> nl_con.lb, nl_cons)
  nl_ucon = map(nl_con -> nl_con.ub, nl_cons)

  return nvar, lvar, uvar, x0, nnln, nl_lcon, nl_ucon
end

"""
    parser_objective_MOI(moimodel, nvar)

Parse linear and quadratic objective of a `MOI.ModelLike`.
"""
function parser_objective_MOI(moimodel, nvar)

  # Variables associated to linear and quadratic objective
  type = "UNKNOWN"
  constant = 0.0
  vect = spzeros(Float64, nvar)
  rows = Int[]
  cols = Int[]
  vals = Float64[]

  fobj = MOI.get(moimodel, MOI.ObjectiveFunction{OBJ}())

  # Single Variable
  if typeof(fobj) == SV
    type = "LINEAR"
    vect[fobj.variable.value] = 1.0
  end

  # Linear objective
  if typeof(fobj) == SAF
    type = "LINEAR"
    constant = fobj.constant
    for term in fobj.terms
      vect[term.variable_index.value] = term.coefficient
    end
  end

  # Quadratic objective
  if typeof(fobj) == SQF
    type = "QUADRATIC"
    constant = fobj.constant
    for term in fobj.affine_terms
      vect[term.variable_index.value] = term.coefficient
    end
    for term in fobj.quadratic_terms
      i = term.variable_index_1.value
      j = term.variable_index_2.value
      if i ≥ j
        push!(rows, i)
        push!(cols, j)
      else
        push!(cols, j)
        push!(rows, i)
      end
      push!(vals, term.coefficient)
    end
  end
  return Objective(type, constant, vect, COO(rows, cols, vals), length(vals))
end
