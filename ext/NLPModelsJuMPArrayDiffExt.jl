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

mutable struct ArrayDiffNLSModel{R} <: NLPModels.AbstractNLSModel{Float64, Vector{Float64}}
    meta::NLPModels.NLPModelMeta{Float64, Vector{Float64}}
    nls_meta::NLPModels.NLSMeta{Float64, Vector{Float64}}
    counters::NLPModels.NLSCounters
    evaluator::ArrayDiff.Evaluator{Float64, R}
end

function NLPModelsJuMP._build_nls_from_residual(
    moimodel::MOI.ModelLike,
    residual::ArrayDiff.ArrayNonlinearFunction,
    ad::ArrayDiff.Mode,
)
    _, nvar, lvar, uvar, x0 = NLPModelsJuMP.parser_variables(moimodel)
    model = ArrayDiff.model(ad)
    ArrayDiff.set_residual!(model, residual)
    vars = MOI.get(moimodel, MOI.ListOfVariableIndices())
    evaluator = MOI.Nonlinear.Evaluator(model, ad, vars)
    MOI.initialize(evaluator, [:Grad, :Jac, :JacVec])
    nresid = ArrayDiff.residual_dimension(evaluator)
    meta = NLPModels.NLPModelMeta(
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
    nls_meta = NLPModels.NLSMeta{Float64, Vector{Float64}}(
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

end
