println()
println("Testing MathOptNLSModel")

@printf(
  "%-15s  %4s  %4s  %4s  %10s  %10s  %10s\n",
  "Problem",
  "nequ",
  "nvar",
  "ncon",
  "‖F(x₀)‖²",
  "‖JᵀF‖",
  "‖c(x₀)‖"
)
# Test that every problem can be instantiated.
for prob in Symbol.(lowercase.(nls_problems ∪ extra_nls_problems ∪ ["nlsnohesspb"]))
  prob_fn = eval(prob)
  nls = prob_fn()
  N = nls.nls_meta.nequ
  n = nls.meta.nvar
  m = nls.meta.ncon
  x = nls.meta.x0
  Fx = residual(nls, x)
  Jx = jac_op_residual(nls, x)
  nFx = dot(Fx, Fx)
  JtF = norm(Jx' * Fx)
  ncx = m > 0 ? @sprintf("%10.4e", norm(cons(nls, x))) : "NA"
  @printf("%-15s  %4d  %4d  %4d  %10.4e  %10.4e  %10s\n", prob, N, n, m, nFx, JtF, ncx)
end
println()

println("Testing 2d cat array on NLS")
model = Model()
@variable(model, x[1:2])
@expression(model, F[i = 1:2], x[i] - 1)
@NLexpression(model, G[i = 1:2], x[i]^2 - 1)
@NLexpression(model, H[i = 1:2, j = 1:2], x[i] * x[j] - 1)
@test F isa Array{GenericAffExpr{Float64, VariableRef}}
@test G isa Array{NonlinearExpression}
@test H isa Array{NonlinearExpression}
nls = MathOptNLSModel(model, [[F G]; H])
@test all(residual(nls, ones(2)) .== 0.0)
@test jac_residual(nls, ones(2))' * residual(nls, ones(2)) == [0.0; 0.0]

println("Testing Dense JuMP container on NLS")
# Linear expressions
model = Model()
@variable(model, x[1:2])
@expression(model, F[i = -1:1, j = 1:2], x[j] - i)
@test F isa JuMP.Containers.DenseAxisArray
nls = MathOptNLSModel(model, F)
@test residual(nls, zeros(2)) == [1.0; 0.0; -1.0; 1.0; 0.0; -1.0]
@test jac_residual(nls, zeros(2))' * residual(nls, zeros(2)) == [0.0; 0.0]
# Nonlinear expressions
model = Model()
@variable(model, x[1:2])
@NLexpression(model, F[i = -1:1, j = 1:2], x[j] - i)
@test F isa JuMP.Containers.DenseAxisArray
nls = MathOptNLSModel(model, F)
@test residual(nls, zeros(2)) == [1.0; 0.0; -1.0; 1.0; 0.0; -1.0]
@test jac_residual(nls, zeros(2))' * residual(nls, zeros(2)) == [0.0; 0.0]

println("Testing Sparse JuMP container on NLS")
# Linear expressions
model = Model()
@variable(model, x[1:2])
D = Dict(1 => 2, 2 => 4)
@expression(model, F[i = 1:2, j = 1:D[i]], x[i] - j)
@test F isa JuMP.Containers.SparseAxisArray
nls = MathOptNLSModel(model, F)
@test sort(residual(nls, [1.5; 2.5])) == [-1.5; -0.5; -0.5; 0.5; 0.5; 1.5]
@test jac_residual(nls, [1.5; 2.5])' * residual(nls, [1.5; 2.5]) == [0.0; 0.0]
# Nonlinear expressions
model = Model()
@variable(model, x[1:2])
D = Dict(1 => 2, 2 => 4)
@NLexpression(model, F[i = 1:2, j = 1:D[i]], x[i] - j)
@test F isa JuMP.Containers.SparseAxisArray
nls = MathOptNLSModel(model, F)
@test sort(residual(nls, [1.5; 2.5])) == [-1.5; -0.5; -0.5; 0.5; 0.5; 1.5]
@test jac_residual(nls, [1.5; 2.5])' * residual(nls, [1.5; 2.5]) == [0.0; 0.0]

println("Testing array of JuMP containers on NLS")
# Linear expressions
model = Model()
@variable(model, x[1:4])
D = Dict(1 => 2, 2 => 4)
@expression(model, F[i = 1:2, j = 1:D[i]], x[i] - j)
@expression(model, G[i = -1:1, j = 3:4], x[j] - i)
@test F isa JuMP.Containers.SparseAxisArray
@test G isa JuMP.Containers.DenseAxisArray
@test [F, G] isa Array{<:AbstractArray{GenericAffExpr{Float64, VariableRef}}}
nls = MathOptNLSModel(model, [F, G])
@test sort(residual(nls, [1.5; 2.5; 0.0; 0.0])) ==
      [-1.5; -1.0; -1.0; -0.5; -0.5; 0.0; 0.0; 0.5; 0.5; 1.0; 1.0; 1.5]
@test jac_residual(nls, [1.5; 2.5; 0.0; 0.0])' * residual(nls, [1.5; 2.5; 0.0; 0.0]) ==
      [0.0; 0.0; 0.0; 0.0]
# Nonlinear expressions
model = Model()
@variable(model, x[1:4])
D = Dict(1 => 2, 2 => 4)
@NLexpression(model, F[i = 1:2, j = 1:D[i]], x[i] - j)
@NLexpression(model, G[i = -1:1, j = 3:4], x[j] - i)
@test F isa JuMP.Containers.SparseAxisArray
@test G isa JuMP.Containers.DenseAxisArray
@test [F, G] isa Array{<:AbstractArray{NonlinearExpression}}
nls = MathOptNLSModel(model, [F, G])
@test sort(residual(nls, [1.5; 2.5; 0.0; 0.0])) ==
      [-1.5; -1.0; -1.0; -0.5; -0.5; 0.0; 0.0; 0.5; 0.5; 1.0; 1.0; 1.5]
@test jac_residual(nls, [1.5; 2.5; 0.0; 0.0])' * residual(nls, [1.5; 2.5; 0.0; 0.0]) ==
      [0.0; 0.0; 0.0; 0.0]
