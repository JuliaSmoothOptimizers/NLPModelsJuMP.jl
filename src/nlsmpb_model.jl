using MathProgBase

include("mpb_model.jl")

export MathProgNLSModel

mutable struct MathProgNLSModel <: AbstractNLSModel
  meta :: NLPModelMeta
  nls_meta :: NLSMeta
  Fmodel :: MathProgModel
  cmodel :: MathProgModel
  counters :: NLSCounters      # Evaluation counters.

  Fjrows :: Vector{Int}      # Jacobian sparsity pattern.
  Fjcols :: Vector{Int}
  Fjvals :: Vector{Float64}  # Room for the constraints Jacobian.

  Fhrows :: Vector{Int}      # Hessian sparsity pattern.
  Fhcols :: Vector{Int}
  Fhvals :: Vector{Float64}  # Room for the Lagrangian Hessian.

  cjrows :: Vector{Int}      # Jacobian sparsity pattern.
  cjcols :: Vector{Int}
  cjvals :: Vector{Float64}  # Room for the constraints Jacobian.

  chrows :: Vector{Int}      # Hessian sparsity pattern.
  chcols :: Vector{Int}
  chvals :: Vector{Float64}  # Room for the Lagrangian Hessian.
end

"Construct a `MathProgNLSModel` from two `MathProgModel`s."
function MathProgNLSModel(Fmodel :: MathProgModel,
                          cmodel :: MathProgModel;
                          name :: String="Generic")

  nvar = cmodel.numVar
  nequ = Fmodel.numConstr
  lvar = cmodel.lvar
  uvar = cmodel.uvar

  nlin = length(cmodel.eval.m.linconstr)         # Number of linear constraints.
  nquad = length(cmodel.eval.m.quadconstr)       # Number of quadratic constraints.
  nnln = length(cmodel.eval.m.nlpdata.nlconstr)  # Number of nonlinear constraints.
  ncon = cmodel.numConstr                        # Total number of constraints.
  lcon = cmodel.lcon
  ucon = cmodel.ucon

  Fjrows, Fjcols = MathProgBase.jac_structure(Fmodel.eval)
  cjrows, cjcols = MathProgBase.jac_structure(cmodel.eval)
  Fhrows, Fhcols = MathProgBase.hesslag_structure(Fmodel.eval)
  chrows, chcols = MathProgBase.hesslag_structure(cmodel.eval)

  meta = NLPModelMeta(nvar,
                      x0=cmodel.x,
                      lvar=lvar,
                      uvar=uvar,
                      ncon=ncon,
                      y0=zeros(ncon),
                      lcon=lcon,
                      ucon=ucon,
                      nnzj=length(cjrows),
                      nnzh=length(chrows),
                      lin=collect(1:nlin),  # linear constraints appear first in MPB
                      nln=collect(nlin+1:ncon),
                      name=name,
                      )

  return MathProgNLSModel(meta,
                          NLSMeta(nequ, nvar, nnzj=length(Fjrows), nnzh=length(Fhrows)),
                          Fmodel,
                          cmodel,
                          NLSCounters(),
                          Fjrows,
                          Fjcols,
                          zeros(length(Fjrows)),  # Fjvals
                          Fhrows,
                          Fhcols,
                          zeros(length(Fhrows)),  # Fhvals
                          cjrows,
                          cjcols,
                          zeros(length(cjrows)),  # cjvals
                          chrows,
                          chcols,
                          zeros(length(chrows)),  # chvals
                         )
end

function NLPModels.residual!(nls :: MathProgNLSModel, x :: AbstractVector, Fx :: AbstractVector)
  increment!(nls, :neval_residual)
  MathProgBase.eval_g(nls.Fmodel.eval, Fx, x)
  return Fx
end

function NLPModels.jac_residual(nls :: MathProgNLSModel, x :: AbstractVector)
  increment!(nls, :neval_jac_residual)
  m, n = nls.nls_meta.nequ, nls.meta.nvar
  MathProgBase.eval_jac_g(nls.Fmodel.eval, nls.Fjvals, x)
  return sparse(nls.Fjrows, nls.Fjcols, nls.Fjvals, m, n)
end

function NLPModels.jac_structure_residual(nls :: MathProgNLSModel)
  return nls.Fjrows, nls.Fjcols
end

function NLPModels.jac_coord_residual!(nls :: MathProgNLSModel, x :: AbstractVector, rows :: AbstractVector{Int}, cols :: AbstractVector{Int}, vals :: AbstractVector)
  increment!(nls, :neval_jac_residual)
  MathProgBase.eval_jac_g(nls.Fmodel.eval, vals, x)
  return (rows, cols, vals)
end

function NLPModels.jac_coord_residual(nls :: MathProgNLSModel, x :: AbstractVector)
  increment!(nls, :neval_jac_residual)
  MathProgBase.eval_jac_g(nls.Fmodel.eval, nls.Fjvals, x)
  return nls.Fjrows, nls.Fjcols, nls.Fjvals
end

function NLPModels.jprod_residual!(nls :: MathProgNLSModel, x :: AbstractVector, v :: AbstractVector, Jv :: AbstractVector)
  nls.counters.neval_jac_residual -= 1
  increment!(nls, :neval_jprod_residual)
  Jv .= jac_residual(nls, x) * v
  return Jv
end

function NLPModels.jtprod_residual!(nls :: MathProgNLSModel, x :: AbstractVector, v :: AbstractVector, Jtv :: AbstractVector)
  nls.counters.neval_jac_residual -= 1
  increment!(nls, :neval_jtprod_residual)
  Jtv .= jac_residual(nls, x)' * v
  return Jtv
end

function NLPModels.hess_residual(nls :: MathProgNLSModel, x :: AbstractVector, v :: AbstractVector)
  increment!(nls, :neval_hess_residual)
  n = nls.meta.nvar
  MathProgBase.eval_hesslag(nls.Fmodel.eval, nls.Fhvals, x, 0.0, v)
  return sparse(nls.Fhrows, nls.Fhcols, nls.Fhvals, n, n)
end

function NLPModels.hess_structure_residual(nls :: MathProgNLSModel)
  return nls.Fhrows, nls.Fhcols
end

function NLPModels.hess_coord_residual!(nls :: MathProgNLSModel, x :: AbstractVector, v :: AbstractVector, rows :: AbstractVector{Int}, cols :: AbstractVector{Int}, vals :: AbstractVector)
  increment!(nls, :neval_hess_residual)
  MathProgBase.eval_hesslag(nls.Fmodel.eval, vals, x, 0.0, v)
  return (rows, cols, vals)
end

function NLPModels.hess_coord_residual(nls :: MathProgNLSModel, x :: AbstractVector, v :: AbstractVector)
  increment!(nls, :neval_hess_residual)
  MathProgBase.eval_hesslag(nls.Fmodel.eval, nls.Fhvals, x, 0.0, v)
  return nls.Fhrows, nls.Fhcols, nls.Fhvals
end

function NLPModels.jth_hess_residual(nls :: MathProgNLSModel, x :: AbstractVector, i :: Int)
  increment!(nls, :neval_jhess_residual)
  y = [j == i ? 1.0 : 0.0 for j = 1:nls.nls_meta.nequ]
  n = nls.meta.nvar
  MathProgBase.eval_hesslag(nls.Fmodel.eval, nls.Fhvals, x, 0.0, y)
  return sparse(nls.Fhrows, nls.Fhcols, nls.Fhvals, n, n)
end

function NLPModels.hprod_residual!(nls :: MathProgNLSModel, x :: AbstractVector, i :: Int, v :: AbstractVector, Hiv :: AbstractVector)
  increment!(nls, :neval_hprod_residual)
  y = [j == i ? 1.0 : 0.0 for j = 1:nls.nls_meta.nequ]
  MathProgBase.eval_hesslag_prod(nls.Fmodel.eval, Hiv, x, v, 0.0, y)
  return Hiv
end

function NLPModels.obj(nls :: MathProgNLSModel, x :: AbstractVector)
  increment!(nls, :neval_obj)
  return MathProgBase.eval_f(nls.cmodel.eval, x)
end

function NLPModels.grad(nls :: MathProgNLSModel, x :: AbstractVector)
  g = zeros(nls.meta.nvar)
  return grad!(nls, x, g)
end

function NLPModels.grad!(nls :: MathProgNLSModel, x :: AbstractVector, g :: AbstractVector)
  increment!(nls, :neval_grad)
  MathProgBase.eval_grad_f(nls.cmodel.eval, g, x)
  return g
end

function NLPModels.cons(nls :: MathProgNLSModel, x :: AbstractVector)
  c = zeros(nls.meta.ncon)
  return cons!(nls, x, c)
end

function NLPModels.cons!(nls :: MathProgNLSModel, x :: AbstractVector, c :: AbstractVector)
  increment!(nls, :neval_cons)
  MathProgBase.eval_g(nls.cmodel.eval, c, x)
  return c
end

function NLPModels.jac_structure(nls :: MathProgNLSModel)
  return (nls.cjrows, nls.cjcols)
end

function NLPModels.jac_coord!(nls :: MathProgNLSModel, x :: AbstractVector, rows :: AbstractVector{Int}, cols :: AbstractVector{Int}, vals :: AbstractVector)
  increment!(nls, :neval_jac)
  MathProgBase.eval_jac_g(nls.cmodel.eval, vals, x)
  return (rows, cols, vals)
end

function NLPModels.jac_coord(nls :: MathProgNLSModel, x :: AbstractVector)
  increment!(nls, :neval_jac)
  MathProgBase.eval_jac_g(nls.cmodel.eval, nls.cjvals, x)
  return (nls.cjrows, nls.cjcols, nls.cjvals)
end

function NLPModels.jac(nls :: MathProgNLSModel, x :: AbstractVector)
  return sparse(jac_coord(nls, x)..., nls.meta.ncon, nls.meta.nvar)
end

function NLPModels.jprod(nls :: MathProgNLSModel, x :: AbstractVector, v :: AbstractVector)
  Jv = zeros(nls.meta.ncon)
  return jprod!(nls, x, v, Jv)
end

function NLPModels.jprod!(nls :: MathProgNLSModel,
                x :: AbstractVector,
                v :: AbstractVector,
                Jv :: AbstractVector)
  nls.counters.counters.neval_jac -= 1
  increment!(nls, :neval_jprod)
  Jv .= jac(nls, x) * v
  return Jv
end

function NLPModels.jtprod(nls :: MathProgNLSModel, x :: AbstractVector, v :: AbstractVector)
  Jtv = zeros(nls.meta.nvar)
  return jtprod!(nls, x, v, Jtv)
end

function NLPModels.jtprod!(nls :: MathProgNLSModel,
                x :: AbstractVector,
                v :: AbstractVector,
                Jtv :: AbstractVector)
  nls.counters.counters.neval_jac -= 1
  increment!(nls, :neval_jtprod)
  Jtv[1:nls.meta.nvar] .= jac(nls, x)' * v
  return Jtv
end

function NLPModels.hess_structure(nls :: MathProgNLSModel)
  return (nls.chrows, nls.chcols)
end

function NLPModels.hess_coord!(nls :: MathProgNLSModel, x :: AbstractVector, rows :: AbstractVector{Int}, cols :: AbstractVector{Int}, vals :: AbstractVector;
    obj_weight :: Float64=1.0, y :: AbstractVector=zeros(nls.meta.ncon))
  increment!(nls, :neval_hess)
  MathProgBase.eval_hesslag(nls.cmodel.eval, vals, x, obj_weight, y)
  return (rows, cols, vals)
end

function NLPModels.hess_coord(nls :: MathProgNLSModel, x :: AbstractVector;
    obj_weight :: Float64=1.0, y :: AbstractVector=zeros(nls.meta.ncon))
  increment!(nls, :neval_hess)
  MathProgBase.eval_hesslag(nls.cmodel.eval, nls.chvals, x, obj_weight, y)
  return (nls.chrows, nls.chcols, nls.chvals)
end

function NLPModels.hess(nls :: MathProgNLSModel, x :: AbstractVector;
    obj_weight :: Float64=1.0, y :: AbstractVector=zeros(nls.meta.ncon))
  return sparse(hess_coord(nls, x, y=y, obj_weight=obj_weight)...,
                nls.meta.nvar, nls.meta.nvar)
end

function NLPModels.hprod(nls :: MathProgNLSModel, x :: AbstractVector, v :: AbstractVector;
    obj_weight :: Float64=1.0, y :: AbstractVector=zeros(nls.meta.ncon))
  hv = zeros(nls.meta.nvar)
  return hprod!(nls, x, v, hv, obj_weight=obj_weight, y=y)
end

#=
# Removed due to bug https://github.com/JuliaOpt/JuMP.jl/issues/1204
function NLPModels.hprod!(nls :: MathProgNLSModel, x :: AbstractVector, v :: AbstractVector,
    hv :: AbstractVector;
    obj_weight :: Float64=1.0, y :: AbstractVector=zeros(nls.meta.ncon))
  MathProgBase.eval_hesslag_prod(nls.cmodel.eval, hv, x, v, obj_weight, y)
  return hv
end
=#

function NLPModels.hprod!(nls :: MathProgNLSModel, x :: AbstractVector, v :: AbstractVector,
    hv :: AbstractVector;
    obj_weight :: Float64=1.0, y :: AbstractVector=zeros(nls.meta.ncon))
  increment!(nls, :neval_hprod)
  # See bug https://github.com/JuliaOpt/JuMP.jl/issues/1204
  MathProgBase.eval_hesslag_prod(nls.cmodel.eval, hv, x, v, 0.0, y)
  n = nls.meta.nvar
  if obj_weight != 0.0
    Fx = residual(nls, x)
    Jv = jprod_residual(nls, x, v)
    w = jtprod_residual(nls, x, Jv)
    hv[1:n] .+= w
    m = length(Fx)
    for i = 1:m
      hprod_residual!(nls, x, i, v, w)
      @views hv[1:n] .= hv[1:n] .+ Fx[i] * w
    end
    hv[1:n] .*= obj_weight
  end
  return hv
end
