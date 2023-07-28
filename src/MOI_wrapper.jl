import SolverCore

mutable struct Optimizer <: MOI.AbstractOptimizer
    options::Dict{String,Any}
    silent::Bool
    solver
    nlp::Union{Nothing,MathOptNLPModel}
    stats::SolverCore.GenericExecutionStats{Float64,Vector{Float64},Vector{Float64},Any}
    function Optimizer()
        return new(
            Dict{String,Any}(),
            false,
            nothing,
            nothing,
            SolverCore.GenericExecutionStats{Float64,Vector{Float64},Vector{Float64},Any}(),
        )
    end
end

# FIXME return the name of the underlying NLPModel solver
MOI.get(::Optimizer, ::MOI.SolverName) = "NLPModels"

MOI.is_empty(optimizer::Optimizer) = isnothing(optimizer.solver) && isnothing(optimizer.nlp)

function MOI.empty!(optimizer::Optimizer)
    optimizer.solver = nothing
    optimizer.nlp = nothing
    reset!(stats)
    return
end

###
### MOI.RawOptimizerAttribute
###

function MOI.set(optimizer::Optimizer, param::MOI.RawOptimizerAttribute, value)
    return optimizer.options[param.name] = value
end

function MOI.get(optimizer::Optimizer, param::MOI.RawOptimizerAttribute)
    return optimizer.options[param.name]
end

###
### MOI.Silent
###

MOI.supports(::Optimizer, ::MOI.Silent) = true

function MOI.set(optimizer::Optimizer, ::MOI.Silent, value::Bool)
    optimizer.silent = value
    return
end

MOI.get(optimizer::Optimizer, ::MOI.Silent) = optimizer.silent

###
### MOI.AbstractModelAttribute
###

function MOI.supports(
    ::Optimizer,
    ::Union{
        MOI.ObjectiveSense,
        MOI.ObjectiveFunction{<:OBJ},
        MOI.NLPBlock,
    },
)
    return true
end

###
### MOI.AbstractVariableAttribute
###

function MOI.supports(
    ::Optimizer,
    ::MOI.VariablePrimalStart,
    ::Type{MOI.VariableIndex},
)
    return true
end

###
### `supports_constraint`
###

MOI.supports_constraint(::Optimizer, ::Type{VI}, ::Type{<:ALS}) = true
MOI.supports_constraint(::Optimizer, ::Type{SAF}, ::Type{<:ALS}) = true
MOI.supports_constraint(::Optimizer, ::Type{VAF}, ::Type{<:VLS}) = true

function MOI.copy_to(dest::Optimizer, src::MOI.ModelLike)
    if !haskey(dest.options, "solver")
        error("No solver specified, use for instance `using Percival; JuMP.set_attribute(model, \"solver\", PercivalSolver)`")
    end
    dest.nlp, index_map = nlp_model(src)
    dest.solver = dest.options["solver"](dest.nlp)
    return index_map
end

function MOI.optimize!(model::Optimizer)
    options = Dict{Symbol,Any}(
        Symbol(key) => model.options[key]
        for key in keys(model.options) if key != "solver"
    )
    if model.silent
        options[:verbose] = 0
    else
        options[:verbose] = 1
    end
    SolverCore.solve!(model.solver, model.nlp, model.stats; options...)
    return
end

function MOI.get(optimizer::Optimizer, ::MOI.SolveTimeSec)
    return optimizer.stats.elapsed_time
end

function MOI.get(optimizer::Optimizer, ::MOI.RawStatusString)
    return SolverCore.STATUSES[optimizer.stats.status]
end

struct RawStatus <: MOI.AbstractModelAttribute
    name::Symbol
end

MOI.is_set_by_optimize(::RawStatus) = true

function MOI.get(optimizer::Optimizer, attr::RawStatus)
    return getfield(optimizer.stats, attr.name)
end

const TERMINATION_STATUS = Dict(
  :exception => MOI.INTERRUPTED,
  :first_order => MOI.LOCALLY_SOLVED,
  :acceptable => MOI.ALMOST_LOCALLY_SOLVED,
  :infeasible => MOI.LOCALLY_INFEASIBLE,
  :max_eval => MOI.OTHER_LIMIT,
  :max_iter => MOI.ITERATION_LIMIT,
  :max_time => MOI.TIME_LIMIT,
  :neg_pred => MOI.NUMERICAL_ERROR,
  :not_desc => MOI.NUMERICAL_ERROR,
  :small_residual => MOI.NUMERICAL_ERROR,
  :small_step => MOI.SLOW_PROGRESS,
  :stalled => MOI.SLOW_PROGRESS,
  :unbounded => MOI.NORM_LIMIT,
  :unknown => MOI.OPTIMIZE_NOT_CALLED,
  :user => MOI.INTERRUPTED,
)

function MOI.get(optimizer::Optimizer, ::MOI.TerminationStatus)
    if isnothing(optimizer.stats)
        return MOI.OPTIMIZE_NOT_CALLED
    end
    return TERMINATION_STATUS[optimizer.stats.status]
end

function MOI.get(optimizer::Optimizer, attr::MOI.ObjectiveValue)
    MOI.check_result_index_bounds(optimizer, attr)
    return optimizer.stats.objective
end

function MOI.get(optimizer::Optimizer, attr::MOI.PrimalStatus)
    if attr.result_index > MOI.get(optimizer, MOI.ResultCount())
        return MOI.NO_SOLUTION
    elseif MOI.get(optimizer, MOI.TerminationStatus()) == MOI.LOCALLY_SOLVED
        return MOI.FEASIBLE_POINT
    else
        # TODO
        return MOI.UNKNOWN_RESULT_STATUS
    end
end

function MOI.get(::Optimizer, ::MOI.DualStatus)
    # TODO
    return MOI.NO_SOLUTION
end

function MOI.get(
    optimizer::Optimizer,
    attr::MOI.VariablePrimal,
    vi::MOI.VariableIndex,
)
    MOI.check_result_index_bounds(optimizer, attr)
    return optimizer.stats.solution[vi.value]
end

MOI.get(optimizer::Optimizer, ::MOI.ResultCount) = 1
