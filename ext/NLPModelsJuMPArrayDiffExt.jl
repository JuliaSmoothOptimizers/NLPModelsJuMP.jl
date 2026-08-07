module NLPModelsJuMPArrayDiffExt

import NLPModelsJuMP
import ArrayDiff
import MathOptInterface as MOI
import NLPModels
import LinearAlgebra

NLPModelsJuMP._nonlinear_model(ad::ArrayDiff.Mode) = ArrayDiff.model(ad)

# Detect `(...).^2` (broadcast `:^` with exponent 2) and return the residual `...`.
function NLPModelsJuMP._detect_squared_residual(inner::ArrayDiff.ArrayNonlinearFunction)
    if inner.head !== :^ || !inner.broadcasted
        return nothing
    end
    if length(inner.args) != 2
        return nothing
    end
    exponent = inner.args[2]
    if !(exponent isa Number) || exponent != 2
        return nothing
    end
    return inner.args[1]
end

mutable struct ArrayDiffNLSModel{T, V <: AbstractVector{T}, R} <: NLPModels.AbstractNLSModel{T, V}
    meta::NLPModels.NLPModelMeta{T, V}
    nls_meta::NLPModels.NLSMeta{T, V}
    counters::NLPModels.NLSCounters
    evaluator::ArrayDiff.Evaluator{T, R}
end

function NLPModelsJuMP._build_nls_from_residual(
    moimodel::MOI.ModelLike,
    residual::ArrayDiff.ArrayNonlinearFunction,
    ad::ArrayDiff.Mode{S},
) where {S <: AbstractVector{<:Real}}
    T = eltype(S)
    V = S
    _, nvar, lvar, uvar, x0 = NLPModelsJuMP.parser_variables(moimodel)
    lvar = convert(V, lvar)
    uvar = convert(V, uvar)
    x0 = convert(V, x0)
    model = ArrayDiff.model(ad)
    ArrayDiff.set_residual!(model, residual)
    vars = MOI.get(moimodel, MOI.ListOfVariableIndices())
    evaluator = MOI.Nonlinear.Evaluator(model, ad, vars)
    MOI.initialize(evaluator, [:Grad, :Jac, :JacVec])
    nresid = ArrayDiff.residual_dimension(evaluator)
    meta = NLPModels.NLPModelMeta{T, V}(
        nvar;
        x0 = x0,
        lvar = lvar,
        uvar = uvar,
        minimize = MOI.get(moimodel, MOI.ObjectiveSense()) == MOI.MIN_SENSE,
        islp = false,
        name = "ArrayDiffNLS",
        hprod_available = false,
        hess_available = false,
    )
    nls_meta = NLPModels.NLSMeta{T, V}(
        nresid,
        nvar;
        x0 = x0,
        nnzj = nresid * nvar,
        nnzh = 0,
        jac_residual_available = false,
        hess_residual_available = false,
        jprod_residual_available = true,
        jtprod_residual_available = true,
        hprod_residual_available = false,
    )
    return ArrayDiffNLSModel(meta, nls_meta, NLPModels.NLSCounters(), evaluator)
end

function NLPModels.residual!(
    nls::ArrayDiffNLSModel,
    x::AbstractVector,
    Fx::AbstractVector,
)
    NLPModels.increment!(nls, :neval_residual)
    ArrayDiff.eval_residual!(nls.evaluator, Fx, x)
    return Fx
end

function NLPModels.jprod_residual!(
    nls::ArrayDiffNLSModel,
    x::AbstractVector,
    v::AbstractVector,
    Jv::AbstractVector,
)
    NLPModels.increment!(nls, :neval_jprod_residual)
    ArrayDiff.eval_residual_jprod!(nls.evaluator, Jv, x, v)
    return Jv
end

function NLPModels.jtprod_residual!(
    nls::ArrayDiffNLSModel,
    x::AbstractVector,
    v::AbstractVector,
    Jtv::AbstractVector,
)
    NLPModels.increment!(nls, :neval_jtprod_residual)
    ArrayDiff.eval_residual_jtprod!(nls.evaluator, Jtv, x, v)
    return Jtv
end


# ── Constrained model from vector-function constraints ───────────────────────
#
# `@constraint(model, expr in MOI.Zeros(n))` over ArrayDiff array expressions
# stores whole `ArrayNonlinearFunction`s on the model (no scalarization).
# `_try_array_nlp_model` collects them into an `ArrayDiffNLPModel`:
#
#     min f(x)  s.t.  lcon ≤ c(x) ≤ ucon,  lvar ≤ x ≤ uvar
#
# where each vector constraint is one residual group evaluated with a single
# fused vectorized pass (`eval_residual!` / `eval_residual_jtprod!` /
# `eval_residual_jprod!`). Row bounds come from the set: `Zeros` → [0, 0],
# `Nonnegatives` → [0, ∞), `Nonpositives` → (-∞, 0].

const _ArrayVLS = Union{MOI.Zeros, MOI.Nonnegatives, MOI.Nonpositives}

function MOI.supports_constraint(
    ::NLPModelsJuMP.Optimizer,
    ::Type{<:ArrayDiff.ArrayNonlinearFunction},
    ::Type{<:_ArrayVLS},
)
    return true
end

_row_bounds(::MOI.Zeros, ::Type{T}) where {T} = (zero(T), zero(T))
_row_bounds(::MOI.Nonnegatives, ::Type{T}) where {T} = (zero(T), T(Inf))
_row_bounds(::MOI.Nonpositives, ::Type{T}) where {T} = (T(-Inf), zero(T))

mutable struct ArrayDiffNLPModel{T, V <: AbstractVector{T}, E} <:
               NLPModels.AbstractNLPModel{T, V}
    meta::NLPModels.NLPModelMeta{T, V}
    counters::NLPModels.Counters
    obj::Union{Nothing, E}   # objective evaluator, or `nothing` (feasibility)
    cons::Vector{E}          # one residual evaluator per vector constraint
    offsets::Vector{Int}     # group g occupies rows offsets[g]+1 : offsets[g+1]
    Jtv_tmp::V               # scratch for accumulating J'v across groups
end

_slice(nlp::ArrayDiffNLPModel, g::Int) = (nlp.offsets[g] + 1):nlp.offsets[g + 1]

function NLPModelsJuMP._try_array_nlp_model(
    moimodel::MOI.ModelLike,
    ad::ArrayDiff.Mode{S},
) where {S <: AbstractVector{<:Real}}
    contypes = MOI.get(moimodel, MOI.ListOfConstraintTypesPresent())
    array_types = [
        (Fi, Si) for (Fi, Si) in contypes if
        Fi <: ArrayDiff.ArrayNonlinearFunction && Si <: _ArrayVLS
    ]
    if isempty(array_types)
        return nothing
    end
    T = eltype(S)
    V = S
    index_map, nvar, lvar, uvar, x0 = NLPModelsJuMP.parser_variables(moimodel)
    vars = MOI.get(moimodel, MOI.ListOfVariableIndices())
    # Objective: a scalar reduction of array expressions (SNF), or absent.
    obj_ev = nothing
    if any(a -> a isa MOI.ObjectiveFunction, MOI.get(moimodel, MOI.ListOfModelAttributesSet()))
        F = MOI.get(moimodel, MOI.ObjectiveFunctionType())
        if !(F <: MOI.ScalarNonlinearFunction)
            error(
                "Objective of type $F is not supported together with " *
                "ArrayDiff vector constraints; use a scalar nonlinear " *
                "objective (e.g. `sum(...)` of array expressions).",
            )
        end
        obj_model = ArrayDiff.model(ad)
        MOI.Nonlinear.set_objective(obj_model, MOI.get(moimodel, MOI.ObjectiveFunction{F}()))
        obj_ev = MOI.Nonlinear.Evaluator(obj_model, ad, vars)
        MOI.initialize(obj_ev, Symbol[:Grad])
    end
    # One residual evaluator per vector constraint, in a deterministic order.
    cons = ArrayDiff.Evaluator{T, ArrayDiff.NLPEvaluator{T, V}}[]
    offsets = [0]
    lcon = T[]
    ucon = T[]
    for (F, SetType) in array_types
        for ci in MOI.get(moimodel, MOI.ListOfConstraintIndices{F, SetType}())
            f = MOI.get(moimodel, MOI.ConstraintFunction(), ci)
            set = MOI.get(moimodel, MOI.ConstraintSet(), ci)
            con_model = ArrayDiff.model(ad)
            ArrayDiff.set_residual!(con_model, f)
            ev = MOI.Nonlinear.Evaluator(con_model, ad, vars)
            MOI.initialize(ev, Symbol[:Grad, :Jac, :JacVec])
            dim = ArrayDiff.residual_dimension(ev)
            push!(cons, ev)
            push!(offsets, offsets[end] + dim)
            lo, hi = _row_bounds(set, T)
            append!(lcon, fill(lo, dim))
            append!(ucon, fill(hi, dim))
            index_map[ci] = ci
        end
    end
    ncon = offsets[end]
    # `findall`-based bound analysis scalar-indexes GPU arrays, but it is
    # needed on CPU (e.g. `SlackModel` reads the jlow/jupp/jrng index sets).
    analysis = V <: Array
    meta = NLPModels.NLPModelMeta{T, V}(
        nvar;
        x0 = convert(V, x0),
        lvar = convert(V, lvar),
        uvar = convert(V, uvar),
        ncon = ncon,
        lcon = convert(V, lcon),
        ucon = convert(V, ucon),
        y0 = fill!(V(undef, ncon), zero(T)),
        nnzj = ncon * nvar, # dense Jacobian (see `jac_coord!`)
        nnzh = 0,           # no exact Hessian; quasi-Newton solvers supply it
        minimize = MOI.get(moimodel, MOI.ObjectiveSense()) != MOI.MAX_SENSE,
        islp = false,
        name = "ArrayDiffNLP",
        lin = Int[],
        variable_bounds_analysis = analysis,
        constraint_bounds_analysis = analysis,
        hess_available = false,
        hprod_available = false,
    )
    E = ArrayDiff.Evaluator{T, ArrayDiff.NLPEvaluator{T, V}}
    nlp = ArrayDiffNLPModel{T, V, E}(
        meta,
        NLPModels.Counters(),
        obj_ev,
        cons,
        offsets,
        fill!(V(undef, nvar), zero(T)),
    )
    return nlp, index_map
end

function NLPModels.obj(nlp::ArrayDiffNLPModel, x::AbstractVector)
    NLPModels.increment!(nlp, :neval_obj)
    return nlp.obj === nothing ? zero(eltype(x)) : MOI.eval_objective(nlp.obj, x)
end

function NLPModels.grad!(nlp::ArrayDiffNLPModel, x::AbstractVector, g::AbstractVector)
    NLPModels.increment!(nlp, :neval_grad)
    if nlp.obj === nothing
        fill!(g, zero(eltype(g)))
    else
        MOI.eval_objective_gradient(nlp.obj, g, x)
    end
    return g
end

function NLPModels.cons!(nlp::ArrayDiffNLPModel, x::AbstractVector, c::AbstractVector)
    NLPModels.increment!(nlp, :neval_cons)
    for g in eachindex(nlp.cons)
        ArrayDiff.eval_residual!(nlp.cons[g], view(c, _slice(nlp, g)), x)
    end
    return c
end

# Stacked J = [J_1; …; J_G], so (Jv)_g = J_g v: write each group into its slice.
function NLPModels.jprod!(
    nlp::ArrayDiffNLPModel,
    x::AbstractVector,
    v::AbstractVector,
    Jv::AbstractVector,
)
    NLPModels.increment!(nlp, :neval_jprod)
    for g in eachindex(nlp.cons)
        ArrayDiff.eval_residual_jprod!(nlp.cons[g], view(Jv, _slice(nlp, g)), x, v)
    end
    return Jv
end

# J' v = Σ_g J_g' v_g (v_g the slice of v): accumulate group contributions.
function NLPModels.jtprod!(
    nlp::ArrayDiffNLPModel,
    x::AbstractVector,
    v::AbstractVector,
    Jtv::AbstractVector,
)
    NLPModels.increment!(nlp, :neval_jtprod)
    fill!(Jtv, zero(eltype(Jtv)))
    for g in eachindex(nlp.cons)
        ArrayDiff.eval_residual_jtprod!(nlp.cons[g], nlp.Jtv_tmp, x, view(v, _slice(nlp, g)))
        Jtv .+= nlp.Jtv_tmp
    end
    return Jtv
end

# ── Explicit Jacobian (for solvers that assemble a KKT system, e.g. MadNLP) ──
#
# ArrayDiff is matrix-free (jprod/jtprod); the Jacobian is materialized densely,
# one reverse pass (J' eᵢ) per constraint row, in column-major dense order.

function NLPModels.jac_structure!(
    nlp::ArrayDiffNLPModel,
    rows::AbstractVector{<:Integer},
    cols::AbstractVector{<:Integer},
)
    m, n = nlp.meta.ncon, nlp.meta.nvar
    k = 0
    for j in 1:n, i in 1:m
        k += 1
        rows[k] = i
        cols[k] = j
    end
    return rows, cols
end

function NLPModels.jac_coord!(
    nlp::ArrayDiffNLPModel,
    x::AbstractVector,
    vals::AbstractVector,
)
    NLPModels.increment!(nlp, :neval_jac)
    m, n = nlp.meta.ncon, nlp.meta.nvar
    ei = fill!(similar(x, m), zero(eltype(x)))
    row = similar(x, n)
    valsm = reshape(vals, m, n) # column-major: valsm[i, j] = J[i, j]
    for i in 1:m
        fill!(ei, zero(eltype(x)))
        view(ei, i:i) .= one(eltype(x)) # GPU-safe scalar write
        NLPModels.jtprod!(nlp, x, ei, row) # row = ∇c_i = J[i, :]
        valsm[i, :] .= row
    end
    return vals
end

# No exact Hessian (`nnzh == 0`); quasi-Newton solvers (MadNLP's CompactLBFGS,
# Percival with an LBFGSModel subproblem) build their own approximation.
function NLPModels.hess_structure!(
    nlp::ArrayDiffNLPModel,
    rows::AbstractVector{<:Integer},
    cols::AbstractVector{<:Integer},
)
    return rows, cols
end

function NLPModels.hess_coord!(
    nlp::ArrayDiffNLPModel,
    x::AbstractVector,
    y::AbstractVector,
    vals::AbstractVector;
    obj_weight = one(eltype(x)),
)
    NLPModels.increment!(nlp, :neval_hess)
    return vals
end

end
