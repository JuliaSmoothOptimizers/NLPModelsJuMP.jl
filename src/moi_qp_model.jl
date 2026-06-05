export MathOptQPModel, qp_model, is_qp_model

import QuadraticModels

MOI.Utilities.@product_of_sets(
  _QPProductOfSets,
  MOI.EqualTo{T},
  MOI.GreaterThan{T},
  MOI.LessThan{T},
  MOI.Interval{T},
)

const QPOptimizerCache = MOI.Utilities.GenericModel{
  Float64,
  MOI.Utilities.ObjectiveContainer{Float64},
  MOI.Utilities.VariablesContainer{Float64},
  MOI.Utilities.MatrixOfConstraints{
    Float64,
    MOI.Utilities.MutableSparseMatrixCSC{Float64, Int, MOI.Utilities.OneBasedIndexing},
    MOI.Utilities.Hyperrectangle{Float64},
    _QPProductOfSets{Float64},
  },
}

"""
    is_qp_model(moimodel::MOI.ModelLike)

Return `true` if `moimodel` contains no `ScalarNonlinearFunction`, no
`VectorNonlinearFunction`, no `VectorNonlinearOracle`, no `NLPBlock`, no
quadratic constraints, no user-defined nonlinear functions and a linear or
quadratic objective. Such a model can be represented as a
`QuadraticModels.QuadraticModel`.
"""
function is_qp_model(model::MOI.ModelLike)
  nlp_block = MOI.get(model, MOI.NLPBlock())
  if nlp_block !== nothing &&
     (length(nlp_block.constraint_bounds) > 0 || nlp_block.has_objective)
    return false
  end
  for attr in MOI.get(model, MOI.ListOfModelAttributesSet())
    if attr isa MOI.UserDefinedFunction
      return false
    end
  end
  F_obj = MOI.get(model, MOI.ObjectiveFunctionType())
  if !(F_obj <: Union{MOI.VariableIndex, SAF, SQF})
    return false
  end
  for (F, S) in MOI.get(model, MOI.ListOfConstraintTypesPresent())
    if F == SNF || F == VNF
      return false
    end
    if F <: Union{SQF, VQF}
      return false
    end
    if F == MOI.VectorOfVariables && S <: MOI.VectorNonlinearOracle{Float64}
      return false
    end
  end
  return true
end

"""
    qp_model(moimodel::MOI.ModelLike; name::String = "Generic")

Build a `QuadraticModels.QuadraticModel` from `moimodel`. The model must
contain only linear constraints and at most a quadratic objective (see
[`is_qp_model`](@ref)). The matrices are extracted by copying `moimodel` into a
[`MOI.Utilities.MatrixOfConstraints`](@ref) cache.

Return `(qp, index_map)`.
"""
function qp_model(moimodel::MOI.ModelLike; name::String = "Generic")
  cache = MOI.Utilities.UniversalFallback(QPOptimizerCache())
  index_map = MOI.copy_to(cache, moimodel)
  return _qp_from_cache(cache, name), index_map
end

function _qp_from_cache(
  cache::MOI.Utilities.UniversalFallback{QPOptimizerCache},
  name::String,
)
  src = cache.model

  for (F, S) in MOI.get(cache, MOI.ListOfConstraintTypesPresent())
    if !MOI.supports_constraint(src, F, S)
      throw(MOI.UnsupportedConstraint{F, S}())
    end
  end

  Ab = src.constraints
  A_csc = convert(SparseArrays.SparseMatrixCSC{Float64, Int}, Ab.coefficients)
  m, nvar = size(A_csc)

  Arows = Int[]
  Acols = Int[]
  Avals = Float64[]
  rowvals = SparseArrays.rowvals(A_csc)
  nzvals = SparseArrays.nonzeros(A_csc)
  for j = 1:nvar
    for k in SparseArrays.nzrange(A_csc, j)
      push!(Arows, rowvals[k])
      push!(Acols, j)
      push!(Avals, nzvals[k])
    end
  end

  lcon = copy(Ab.constants.lower)
  ucon = copy(Ab.constants.upper)

  vc = src.variables
  lvar = copy(vc.lower)
  uvar = copy(vc.upper)

  x0 = zeros(Float64, nvar)
  if MOI.VariablePrimalStart() in MOI.get(src, MOI.ListOfVariableAttributesSet())
    for vi in MOI.get(src, MOI.ListOfVariableIndices())
      val = MOI.get(src, MOI.VariablePrimalStart(), vi)
      if val !== nothing
        x0[vi.value] = val
      end
    end
  end

  c = zeros(Float64, nvar)
  c0 = 0.0
  Hrows = Int[]
  Hcols = Int[]
  Hvals = Float64[]
  sense = MOI.get(src, MOI.ObjectiveSense())
  if sense != MOI.FEASIBILITY_SENSE
    F = MOI.get(src, MOI.ObjectiveFunctionType())
    obj = MOI.get(src, MOI.ObjectiveFunction{F}())
    if F == MOI.VariableIndex
      c[obj.value] = 1.0
    elseif F == SAF
      c0 = obj.constant
      for term in obj.terms
        c[term.variable.value] += term.coefficient
      end
    elseif F == SQF
      c0 = obj.constant
      for term in obj.affine_terms
        c[term.variable.value] += term.coefficient
      end
      for term in obj.quadratic_terms
        i, j = term.variable_1.value, term.variable_2.value
        if i ≥ j
          push!(Hrows, i)
          push!(Hcols, j)
        else
          push!(Hrows, j)
          push!(Hcols, i)
        end
        push!(Hvals, term.coefficient)
      end
    else
      error("Objective function type $F is not supported by qp_model.")
    end
  end

  minimize = sense != MOI.MAX_SENSE

  return QuadraticModels.QuadraticModel(
    c,
    Hrows,
    Hcols,
    Hvals;
    Arows = Arows,
    Acols = Acols,
    Avals = Avals,
    lcon = lcon,
    ucon = ucon,
    lvar = lvar,
    uvar = uvar,
    c0 = c0,
    x0 = x0,
    minimize = minimize,
    name = name,
  )
end

"""
    MathOptQPModel(jmodel::JuMP.Model; name::String = "Generic")
    MathOptQPModel(moimodel::MOI.ModelLike; name::String = "Generic")

Construct a [`QuadraticModels.QuadraticModel`](@extref) from a JuMP or MOI
model containing only linear constraints and at most a quadratic objective.
"""
function MathOptQPModel(jmodel::JuMP.Model; kws...)
  _nlp_sync!(jmodel)
  return MathOptQPModel(backend(jmodel); kws...)
end

function MathOptQPModel(moimodel::MOI.ModelLike; kws...)
  return qp_model(moimodel; kws...)[1]
end
