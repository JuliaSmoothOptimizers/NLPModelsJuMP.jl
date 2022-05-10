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
const AF  = Union{SAF, VAF}

# ScalarQuadraticFunctions and VectorQuadraticFunctions
const SQF = MOI.ScalarQuadraticFunction{Float64}  # ScalarQuadraticFunction{T}(affine_terms, quadratic_terms, constant)
const VQF = MOI.VectorQuadraticFunction{Float64}  # VectorQuadraticFunction{T}(affine_terms, quadratic_terms, constants)
const QF  = Union{SQF, VQF}

# AffLinSets and VecLinSets
const ALS = Union{
  MOI.EqualTo{Float64},
  MOI.GreaterThan{Float64},
  MOI.LessThan{Float64},
  MOI.Interval{Float64},
}
const VLS = Union{MOI.Nonnegatives, MOI.Nonpositives, MOI.Zeros}
const LS = Union{ALS, VLS}

# Objective
const VI = MOI.VariableIndex
const OBJ = Union{VI, SAF, SQF}

# Coordinate Matrix
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

mutable struct QuadraticConstraint
  hessian::COO
  vec::Vector{Int}
  b::SparseVector{Float64}
end

mutable struct QuadraticConstraints
  qcons::Vector{QuadraticConstraint}
  nquad::Int
  nnzj::Int
  jrows::Vector{Int}
  jcols::Vector{Int}
  nnzh::Int
  set::Set{Tuple{Int,Int}}
end

Base.getindex(qcon::QuadraticConstraints, i::Integer) = qcon.qcons[i]
Base.length(qcon::QuadraticConstraints) = qcon.nquad

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
    coo_sym_dot(rows, cols, vals, x, y)

Compute the product `xᵀAy` of a symmetric matrix `A` given by `(rows, cols, vals)`
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
    jacobian_quad(qcons)

`qcons` is a vector of `QuadraticConstraint` where each constraint has the form ½xᵀQᵢx + xᵀbᵢ.
Compute the sparsity pattern of the jacobian [Q₁x + b₁; ...; Qₚx + bₚ]ᵀ of `qcons`.
This function also allocates `qcons[i].vec`.
"""
function jacobian_quad(qcons)
  jrows = Int[]
  jcols = Int[]
  nquad = length(qcons)
  for i = 1 : nquad
    # rows of Qᵢx + bᵢ with nonzeros coefficients
    qcons[i].vec = unique(qcons[i].hessian.rows ∪ qcons[i].b.nzind)
    for elt ∈ qcons[i].vec
      push!(jcols, elt)
      push!(jrows, i)
    end
  end
  nnzj = length(jrows)
  return nnzj, jrows, jcols
end

"""
    hessian_quad(qcons)

`qcons` is a vector of `QuadraticConstraint` where each constraint has the form ½xᵀQᵢx + xᵀbᵢ.
Compute the sparsity pattern of the hessian ΣᵢQᵢ of `qcons`.
"""
function hessian_quad(qcons)
  set = Set{Tuple{Int,Int}}()
  nquad = length(qcons)
  for i = 1 : nquad
    con = qcons[i]
    for tuple ∈ zip(con.hessian.rows, con.hessian.cols)
      # Only disctinct tuples are stored in the set
      push!(set, tuple)
    end
  end
  return set
end

function hessian_structure(set)
  nnzh = length(set)
  hrows = zeros(Int, nnzh)
  hcols = zeros(Int, nnzh)
  for (index,tuple) in enumerate(set)
    hrows[index] = tuple[1]
    hcols[index] = tuple[2]
  end
  return hrows, hcols
end

"""
    parser_SAF(fun, set, linrows, lincols, linvals, nlin, lin_lcon, lin_ucon)

Parse a `ScalarAffineFunction` fun with its associated set.
`linrows`, `lincols`, `linvals`, `lin_lcon` and `lin_ucon` are updated.
"""
function parser_SAF(fun, set, linrows, lincols, linvals, nlin, lin_lcon, lin_ucon)

  # Parse a ScalarAffineTerm{Float64}(coefficient, variable)
  for term in fun.terms
    push!(linrows, nlin + 1)
    push!(lincols, term.variable.value)
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
    push!(lincols, term.scalar_term.variable.value)
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
    parser_SQF(fun, set, qcons, quad_lcon, quad_ucon)

Parse a `ScalarQuadraticFunction` fun with its associated set.
`qcons`, `quad_lcon`, `quad_ucon` are updated.
"""
function parser_SQF(fun, set, nvar, qcons, quad_lcon, quad_ucon)

  b    = spzeros(Float64, nvar)
  rows = Int[]
  cols = Int[]
  vals = Float64[]

  # Parse a ScalarAffineTerm{Float64}(coefficient, variable_index)
  for term in fun.affine_terms
    b[term.variable.value] = term.coefficient
  end

  # Parse a ScalarQuadraticTerm{Float64}(coefficient, variable_index_1, variable_index_2)
  for term in fun.quadratic_terms
    i = term.variable_1.value
    j = term.variable_2.value
    if i ≥ j
      push!(rows, i)
      push!(cols, j)
    else
      push!(cols, j)
      push!(rows, i)
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

  nnzh = length(vals)
  qcon = QuadraticConstraint(COO(rows, cols, vals), Int[], b)
  push!(qcons, qcon)
end

"""
    parser_MOI(moimodel, nvar)

Parse linear and quadratic constraints of a `MOI.ModelLike`.
"""
function parser_MOI(moimodel, nvar)

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
    F == VI && continue
    F <: AF || F <: SQF || @warn("Function $F is not supported.")
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
      if typeof(fun) <: SQF
        parser_SQF(fun, set, nvar, qcons, quad_lcon, quad_ucon)
        nquad += 1
      end
    end
  end
  coo = COO(linrows, lincols, linvals)
  nnzj = length(linvals)
  lincon = LinearConstraints(coo, nnzj)

  nnzj, jrows, jcols = jacobian_quad(qcons)
  set = hessian_quad(qcons)
  nnzh = length(set)
  quadcon = QuadraticConstraints(qcons, nquad, nnzj, jrows, jcols, nnzh, set)

  return nlin, lincon, lin_lcon, lin_ucon, quadcon, quad_lcon, quad_ucon
end

"""
    parser_JuMP(jmodel)

Parse variables informations of a `JuMP.Model`.
"""
function parser_JuMP(jmodel)

  # Number of variables and bounds constraints
  nvar = Int(num_variables(jmodel))
  vars = all_variables(jmodel)
  lvar = map(
    var -> is_fixed(var) ? fix_value(var) : (has_lower_bound(var) ? lower_bound(var) : -Inf),
    vars,
  )
  uvar = map(
    var -> is_fixed(var) ? fix_value(var) : (has_upper_bound(var) ? upper_bound(var) : Inf),
    vars,
  )

  # Initial solution
  x0 = zeros(nvar)
  for (i, val) ∈ enumerate(start_value.(vars))
    if val !== nothing
      x0[i] = val
    end
  end

  return nvar, lvar, uvar, x0
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
  if typeof(fobj) == VI
    type = "LINEAR"
    vect[fobj.value] = 1.0
  end

  # Linear objective
  if typeof(fobj) == SAF
    type = "LINEAR"
    constant = fobj.constant
    for term in fobj.terms
      vect[term.variable.value] = term.coefficient
    end
  end

  # Quadratic objective
  if typeof(fobj) == SQF
    type = "QUADRATIC"
    constant = fobj.constant
    for term in fobj.affine_terms
      vect[term.variable.value] = term.coefficient
    end
    for term in fobj.quadratic_terms
      i = term.variable_1.value
      j = term.variable_2.value
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

"""
    parser_linear_expression(cmodel, nvar, F)

Parse linear expressions of type `GenericAffExpr{Float64,VariableRef}`.
"""
function parser_linear_expression(cmodel, nvar, F)

  # Variables associated to linear expressions
  rows = Int[]
  cols = Int[]
  vals = Float64[]
  constants = Float64[]

  # Linear least squares model
  nlinequ = 0
  F_is_array_of_containers = F isa Array{<:AbstractArray}
  if F_is_array_of_containers
    @objective(
      cmodel,
      Min,
      0.0 +
      0.5 *
      sum(sum(Fi^2 for Fi in FF if typeof(Fi) == GenericAffExpr{Float64, VariableRef}) for FF in F)
    )
    for FF in F, expr in FF
      if typeof(expr) == GenericAffExpr{Float64, VariableRef}
        nlinequ += 1
        for (i, key) in enumerate(expr.terms.keys)
          push!(rows, nlinequ)
          push!(cols, key.index.value)
          push!(vals, expr.terms.vals[i])
        end
        push!(constants, expr.constant)
      end
    end
  else
    @objective(
      cmodel,
      Min,
      0.0 + 0.5 * sum(Fi^2 for Fi in F if typeof(Fi) == GenericAffExpr{Float64, VariableRef})
    )
    for expr in F
      if typeof(expr) == GenericAffExpr{Float64, VariableRef}
        nlinequ += 1
        for (i, key) in enumerate(expr.terms.keys)
          push!(rows, nlinequ)
          push!(cols, key.index.value)
          push!(vals, expr.terms.vals[i])
        end
        push!(constants, expr.constant)
      end
    end
  end
  moimodel = backend(cmodel)
  lls = parser_objective_MOI(moimodel, nvar)
  return lls, LinearEquations(COO(rows, cols, vals), constants, length(vals)), nlinequ
end

"""
    parser_nonlinear_expression(cmodel, nvar, F)

Parse nonlinear expressions of type `NonlinearExpression`.
"""
function parser_nonlinear_expression(cmodel, nvar, F; hessian::Bool = true)

  # Nonlinear least squares model
  nnlnequ = 0
  F_is_array_of_containers = F isa Array{<:AbstractArray}
  if F_is_array_of_containers
    nnlnequ = sum(sum(typeof(Fi) == NonlinearExpression for Fi in FF) for FF in F)
    if nnlnequ > 0
      @NLobjective(
        cmodel,
        Min,
        0.5 * sum(sum(Fi^2 for Fi in FF if typeof(Fi) == NonlinearExpression) for FF in F)
      )
    end
  else
    nnlnequ = sum(typeof(Fi) == NonlinearExpression for Fi in F)
    if nnlnequ > 0
      @NLobjective(cmodel, Min, 0.5 * sum(Fi^2 for Fi in F if typeof(Fi) == NonlinearExpression))
    end
  end
  ceval = cmodel.nlp_data == nothing ? nothing : NLPEvaluator(cmodel)
  (ceval ≠ nothing) &&
    (nnlnequ == 0) &&
    MOI.initialize(ceval, hessian ? [:Grad, :Jac, :Hess, :HessVec] : [:Grad, :Jac])  # Add :JacVec when available
  (ceval ≠ nothing) &&
    (nnlnequ > 0) &&
    MOI.initialize(
      ceval,
      hessian ? [:Grad, :Jac, :Hess, :HessVec, :ExprGraph] : [:Grad, :Jac, :ExprGraph],
    )  # Add :JacVec when available

  if nnlnequ == 0
    Feval = nothing
  else
    Fmodel = JuMP.Model()
    @variable(Fmodel, x[1:nvar])
    JuMP._init_NLP(Fmodel)
    @objective(Fmodel, Min, 0.0)
    Fmodel.nlp_data.user_operators = cmodel.nlp_data.user_operators
    if F_is_array_of_containers
      for FF in F, Fi in FF
        if typeof(Fi) == NonlinearExpression
          expr = ceval.subexpressions_as_julia_expressions[Fi.index]
          replace!(expr, x)
          expr = :($expr == 0)
          JuMP.add_nonlinear_constraint(Fmodel, expr)
        end
      end
    else
      for Fi in F
        if typeof(Fi) == NonlinearExpression
          expr = ceval.subexpressions_as_julia_expressions[Fi.index]
          replace!(expr, x)
          expr = :($expr == 0)
          JuMP.add_nonlinear_constraint(Fmodel, expr)
        end
      end
    end
    Feval = NLPEvaluator(Fmodel)
    MOI.initialize(Feval, hessian ? [:Grad, :Jac, :Hess, :HessVec] : [:Grad, :Jac])  # Add :JacVec when available
    Feval.user_output_buffer = ceval.user_output_buffer
  end
  return ceval, Feval, nnlnequ
end
