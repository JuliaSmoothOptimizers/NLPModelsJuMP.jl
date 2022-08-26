using LinearAlgebra, SparseArrays

using NLPModels
import NLPModels.increment!, NLPModels.decrement!

using JuMP, MathOptInterface
const MOI = MathOptInterface

# ScalarAffineFunctions and VectorAffineFunctions
const SAF = MOI.ScalarAffineFunction{Float64}
const VAF = MOI.VectorAffineFunction{Float64}
const AF = Union{SAF, VAF}

# AffLinSets and VecLinSets
const ALS = Union{
  MOI.EqualTo{Float64},
  MOI.GreaterThan{Float64},
  MOI.LessThan{Float64},
  MOI.Interval{Float64},
}
const VLS = Union{MOI.Nonnegatives, MOI.Nonpositives, MOI.Zeros}
const LS = Union{ALS, VLS}

const VI = MOI.VariableIndex
const SQF = MOI.ScalarQuadraticFunction{Float64}
const OBJ = Union{VI, SAF, SQF}

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
    parser_MOI(moimodel)

Parse linear constraints of a `MOI.ModelLike`.
"""
function parser_MOI(moimodel)

  # Variables associated to linear constraints
  nlin = 0
  linrows = Int[]
  lincols = Int[]
  linvals = Float64[]
  lin_lcon = Float64[]
  lin_ucon = Float64[]

  contypes = MOI.get(moimodel, MOI.ListOfConstraintTypesPresent())
  for (F, S) in contypes
    F == VI && continue
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
    parser_NL(jmodel, moimodel)

Parse nonlinear constraints of a `MOI.Nonlinear.Evaluator`.
"""
function parser_NL(jmodel, eval; hessian::Bool = true)

  nnln = num_nonlinear_constraints(jmodel)
  nl_lcon = fill(-Inf, nnln)
  nl_ucon = fill(Inf, nnln)
  for (i, (_, nl_constraint)) in enumerate(jmodel.nlp_model.constraints)
    rhs = nl_constraint.set
    if rhs isa MOI.EqualTo
      nl_lcon[i] = rhs.value
      nl_ucon[i] = rhs.value
    elseif rhs isa MOI.GreaterThan
      nl_lcon[i] = rhs.lower
    elseif rhs isa MOI.LessThan
      nl_ucon[i] = rhs.upper
    elseif rhs isa MOI.Interval
      nl_lcon[i] = rhs.lower
      nl_ucon[i] = rhs.upper
    else
      error("Unexpected constraint type: $(typeof(rhs))")
    end
  end

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
    add_constraint_model(Fmodel, Fi)

Add the nonlinear constraint `Fi == 0` to the model `Fmodel`.
If `Fi` is an Array, then we iterate over each component.
"""
function add_constraint_model(Fmodel, Fi::NonlinearExpression)
  Fmodel.nlp_model.last_constraint_index += 1
  ci = MOI.Nonlinear.ConstraintIndex(Fmodel.nlp_model.last_constraint_index)
  index = Fi.index
  Fmodel.nlp_model.constraints[ci] = MOI.Nonlinear.Constraint(Fmodel.nlp_model.expressions[index], MOI.EqualTo{Float64}(0.0))
  return nothing
end

function add_constraint_model(Fmodel, Fi::GenericAffExpr)
  return nothing
end

function add_constraint_model(Fmodel, Fi::GenericQuadExpr)
  @warn("GenericQuadExpr{Float64, VariableRef} are not supported.")
end

function add_constraint_model(Fmodel, Fi::AbstractArray)
  for Fj in Fi
    add_constraint_model(Fmodel, Fj)
  end
end

"""
    parser_nonlinear_expression(cmodel, nvar, F)

Parse nonlinear expressions of type `NonlinearExpression`.
"""
function parser_nonlinear_expression(cmodel, nvar, F; hessian::Bool = true)

  # Nonlinear least squares model
  F_is_array_of_containers = F isa Array{<:AbstractArray}
  if F_is_array_of_containers
    nnlnequ = sum(sum(isa(Fi, NonlinearExpression) for Fi in FF) for FF in F)
    if nnlnequ > 0
      @NLobjective(
        cmodel,
        Min,
        0.5 * sum(sum(Fi^2 for Fi in FF if isa(Fi, NonlinearExpression)) for FF in F)
      )
    end
  else
    nnlnequ = sum(isa(Fi, NonlinearExpression) for Fi in F)
    if nnlnequ > 0
      @NLobjective(cmodel, Min, 0.5 * sum(Fi^2 for Fi in F if isa(Fi, NonlinearExpression)))
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

  Fjac = Feval ≠ nothing ? MOI.jacobian_structure(Feval) : Tuple{Int,Int}[]
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
