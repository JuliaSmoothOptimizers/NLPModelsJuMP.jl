# Tutorial

NLPModelsJuMP is a combination of NLPModels and JuMP, as the name imply.
Sometimes it may be required to refer to the specific documentation, as we'll present
here the result of combining both.

```@contents
Pages = ["tutorial.md"]
```

## MathProgNLPModel

```@docs
MathProgNLPModel
```

`MathProgNLPModel` is a simple yet efficient model. It uses JuMP to define the problem,
which can then be accessed through the NLPModels API.
Using `ADNLPModel` is simpler, as it comes by default, but `ADNLPModel` doesn't handle
sparse derivatives and `MathProgNLPModel` does.

Let's define the famous Rosenbrock function
```math
f(x) = (x_1 - 1)^2 + 100(x_2 - x_1^2)^2,
```
with starting point ``x^0 = (-1.2,1.0)``.

```@example jumpnlp
using NLPModels, NLPModelsJuMP, JuMP

x0 = [-1.2; 1.0]
model = Model() # No solver is required
@variable(model, x[i=1:2], start=x0[i])
@NLobjective(model, Min, (x[1] - 1)^2 + 100 * (x[2] - x[1]^2)^2)

nlp = MathProgNLPModel(model)
```

This defines the model.
Let's get the objective function value at ``x^0``, using only `nlp`.

```@example jumpnlp
fx = obj(nlp, nlp.meta.x0)
println("fx = $fx")
```

Done.
Let's try the gradient and Hessian.

```@example jumpnlp
gx = grad(nlp, nlp.meta.x0)
Hx = hess(nlp, nlp.meta.x0)
println("gx = $gx")
println("Hx = $Hx")
```

Notice how only the lower triangle of the Hessian is stored, which is the default for
NLPModels.

Let's do something a little more complex here, defining a function to try to
solve this problem through steepest descent method with Armijo search.
Namely, the method

1. Given ``x^0``, ``\varepsilon > 0``, and ``\eta \in (0,1)``. Set ``k = 0``;
2. If ``\Vert \nabla f(x^k) \Vert < \varepsilon`` STOP with ``x^* = x^k``;
3. Compute ``d^k = -\nabla f(x^k)``;
4. Compute ``\alpha_k \in (0,1]`` such that ``f(x^k + \alpha_kd^k) < f(x^k) + \alpha_k\eta \nabla f(x^k)^Td^k``
5. Define ``x^{k+1} = x^k + \alpha_kx^k``
6. Update ``k = k + 1`` and go to step 2.

```@example jumpnlp
using LinearAlgebra

function steepest(nlp; itmax=100000, eta=1e-4, eps=1e-6, sigma=0.66)
  x = nlp.meta.x0
  fx = obj(nlp, x)
  ∇fx = grad(nlp, x)
  slope = dot(∇fx, ∇fx)
  ∇f_norm = sqrt(slope)
  iter = 0
  while ∇f_norm > eps && iter < itmax
    t = 1.0
    x_trial = x - t * ∇fx
    f_trial = obj(nlp, x_trial)
    while f_trial > fx - eta * t * slope
      t *= sigma
      x_trial = x - t * ∇fx
      f_trial = obj(nlp, x_trial)
    end
    x = x_trial
    fx = f_trial
    ∇fx = grad(nlp, x)
    slope = dot(∇fx, ∇fx)
    ∇f_norm = sqrt(slope)
    iter += 1
  end
  optimal = ∇f_norm <= eps
  return x, fx, ∇f_norm, optimal, iter
end

x, fx, ngx, optimal, iter = steepest(nlp)
println("x = $x")
println("fx = $fx")
println("ngx = $ngx")
println("optimal = $optimal")
println("iter = $iter")
```

Maybe this code is too complicated? If you're in a class you just want to show a
Newton step.

```@example jumpnlp
f(x) = obj(nlp, x)
g(x) = grad(nlp, x)
H(x) = Symmetric(hess(nlp, x), :L)
x = nlp.meta.x0
d = -H(x)\g(x)
```

or a few

```@example jumpnlp
for i = 1:5
  global x
  x = x - H(x)\g(x)
  println("x = $x")
end
```

Notice how we can use the method with different NLPModels:

```@example jumpnlp
f(x) = (x[1] - 1.0)^2 + 100 * (x[2] - 1.0)^2

adnlp = ADNLPModel(f, x0)
x, fx, ngx, optimal, iter = steepest(adnlp)
```

### OptimizationProblems

The package
[OptimizationProblems](https://github.com/JuliaSmoothOptimizers/OptimizationProblems.jl)
provides a reasonable amount of problems defined in JuMP format, which can be converted
to `MathProgNLPModel`.

```@example jumpnlp
using OptimizationProblems # Defines a lot of JuMP models

nlp = MathProgNLPModel(woods())
x, fx, ngx, optimal, iter = steepest(nlp)
println("fx = $fx")
println("ngx = $ngx")
println("optimal = $optimal")
println("iter = $iter")
```

Constrained problem can also be converted.

```@example jumpnlp2
using NLPModels, NLPModelsJuMP, JuMP

model = Model()
x0 = [-1.2; 1.0]
@variable(model, x[i=1:2] >= 0.0, start=x0[i])
@NLobjective(model, Min, (x[1] - 1)^2 + 100 * (x[2] - x[1]^2)^2)
@constraint(model, x[1] + x[2] == 3.0)
@NLconstraint(model, x[1] * x[2] >= 1.0)

nlp = MathProgNLPModel(model)

println("cx = $(cons(nlp, nlp.meta.x0))")
println("Jx = $(jac(nlp, nlp.meta.x0))")
```

## MathProgNLSModel Tutorial

```@docs
MathProgNLSModel
```

`MathProgNLSModel` is a model for nonlinear least squares using JuMP.
To use it, we define a JuMP model without the objective, and use `NLexpression`s to
define the residual function.
For instance, the Rosenbrock function in nonlinear least squares format is
```math
F(x) = \begin{bmatrix} x_1 - 1\\ 10(x_2 - x_1^2) \end{bmatrix},
```
which we can implement as

```@example nls
using NLPModels, NLPModelsJuMP, JuMP

model = Model()
x0 = [-1.2; 1.0]
@variable(model, x[i=1:2], start=x0[i])
@NLexpression(model, F1, x[1] - 1)
@NLexpression(model, F2, 10 * (x[2] - x[1]^2))

nls = MathProgNLSModel(model, [F1, F2], name="rosen-nls")

residual(nls, nls.meta.x0)
```

```@example nls
jac_residual(nls, nls.meta.x0)
```

## NLPtoMPB - Convert NLP to MathProgBase

```@docs
NLPtoMPB
```

In addition to creating NLPModels using JuMP, we might want to convert an NLPModel to a
MathProgBase model to use the solvers available. For instance

```@example nlptompb
using Ipopt, NLPModels, NLPModelsJuMP, LinearAlgebra, JuMP, MathProgBase

nlp = ADNLPModel(x -> dot(x, x), ones(2),
                 c=x->[x[1] + 2 * x[2] - 1.0], lcon=[0.0], ucon=[0.0])
model = NLPtoMPB(nlp, IpoptSolver())

MathProgBase.optimize!(model)
```
