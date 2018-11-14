var documenterSearchIndex = {"docs": [

{
    "location": "#",
    "page": "Home",
    "title": "Home",
    "category": "page",
    "text": ""
},

{
    "location": "#NLPModelsJuMP.jl-documentation-1",
    "page": "Home",
    "title": "NLPModelsJuMP.jl documentation",
    "category": "section",
    "text": "This package defines a NLPModels model using MathProgBase and JuMP.jl. This documentation is specific for this model. Please refer to the documentation of NLPModels if in doubt."
},

{
    "location": "#Install-1",
    "page": "Home",
    "title": "Install",
    "category": "section",
    "text": "Install NLPModelsJuMP.jl with the following commands.pkg> add NLPModelsJuMP"
},

{
    "location": "#License-1",
    "page": "Home",
    "title": "License",
    "category": "section",
    "text": "This content is released under the MIT License. (Image: ) "
},

{
    "location": "#Contents-1",
    "page": "Home",
    "title": "Contents",
    "category": "section",
    "text": ""
},

{
    "location": "tutorial/#",
    "page": "Tutorial",
    "title": "Tutorial",
    "category": "page",
    "text": ""
},

{
    "location": "tutorial/#Tutorial-1",
    "page": "Tutorial",
    "title": "Tutorial",
    "category": "section",
    "text": "NLPModelsJuMP is a combination of NLPModels and JuMP, as the name imply. Sometimes it may be required to refer to the specific documentation, as we\'ll present here the result of combining both.Pages = [\"tutorial.md\"]"
},

{
    "location": "tutorial/#NLPModelsJuMP.MathProgNLPModel",
    "page": "Tutorial",
    "title": "NLPModelsJuMP.MathProgNLPModel",
    "category": "type",
    "text": "MathProgNLPModel(model, name=\"Generic\")\n\nConstruct a MathProgNLPModel from a MathProgModel.\n\n\n\n\n\nMathProgNLPModel(model; kwargs...)\n\nConstruct a MathProgNLPModel from a JuMP Model.\n\n\n\n\n\n"
},

{
    "location": "tutorial/#MathProgNLPModel-1",
    "page": "Tutorial",
    "title": "MathProgNLPModel",
    "category": "section",
    "text": "MathProgNLPModelMathProgNLPModel is a simple yet efficient model. It uses JuMP to define the problem, which can then be accessed through the NLPModels API. Using ADNLPModel is simpler, as it comes by default, but ADNLPModel doesn\'t handle sparse derivatives and MathProgNLPModel does.Let\'s define the famous Rosenbrock functionf(x) = (x_1 - 1)^2 + 100(x_2 - x_1^2)^2with starting point x^0 = (-1210).using NLPModels, NLPModelsJuMP, JuMP\n\nx0 = [-1.2; 1.0]\nmodel = Model() # No solver is required\n@variable(model, x[i=1:2], start=x0[i])\n@NLobjective(model, Min, (x[1] - 1)^2 + 100 * (x[2] - x[1]^2)^2)\n\nnlp = MathProgNLPModel(model)This defines the model. Let\'s get the objective function value at x^0, using only nlp.fx = obj(nlp, nlp.meta.x0)\nprintln(\"fx = $fx\")Done. Let\'s try the gradient and Hessian.gx = grad(nlp, nlp.meta.x0)\nHx = hess(nlp, nlp.meta.x0)\nprintln(\"gx = $gx\")\nprintln(\"Hx = $Hx\")Notice how only the lower triangle of the Hessian is stored, which is the default for NLPModels.Let\'s do something a little more complex here, defining a function to try to solve this problem through steepest descent method with Armijo search. Namely, the methodGiven x^0, varepsilon  0, and eta in (01). Set k = 0;\nIf Vert nabla f(x^k) Vert  varepsilon STOP with x^* = x^k;\nCompute d^k = -nabla f(x^k);\nCompute alpha_k in (01 such that f(x^k + alpha_kd^k)  f(x^k) + alpha_keta nabla f(x^k)^Td^k\nDefine x^k+1 = x^k + alpha_kx^k\nUpdate k = k + 1 and go to step 2.using LinearAlgebra\n\nfunction steepest(nlp; itmax=100000, eta=1e-4, eps=1e-6, sigma=0.66)\n  x = nlp.meta.x0\n  fx = obj(nlp, x)\n  ∇fx = grad(nlp, x)\n  slope = dot(∇fx, ∇fx)\n  ∇f_norm = sqrt(slope)\n  iter = 0\n  while ∇f_norm > eps && iter < itmax\n    t = 1.0\n    x_trial = x - t * ∇fx\n    f_trial = obj(nlp, x_trial)\n    while f_trial > fx - eta * t * slope\n      t *= sigma\n      x_trial = x - t * ∇fx\n      f_trial = obj(nlp, x_trial)\n    end\n    x = x_trial\n    fx = f_trial\n    ∇fx = grad(nlp, x)\n    slope = dot(∇fx, ∇fx)\n    ∇f_norm = sqrt(slope)\n    iter += 1\n  end\n  optimal = ∇f_norm <= eps\n  return x, fx, ∇f_norm, optimal, iter\nend\n\nx, fx, ngx, optimal, iter = steepest(nlp)\nprintln(\"x = $x\")\nprintln(\"fx = $fx\")\nprintln(\"ngx = $ngx\")\nprintln(\"optimal = $optimal\")\nprintln(\"iter = $iter\")Maybe this code is too complicated? If you\'re in a class you just want to show a Newton step.f(x) = obj(nlp, x)\ng(x) = grad(nlp, x)\nH(x) = Symmetric(hess(nlp, x), :L)\nx = nlp.meta.x0\nd = -H(x)\\g(x)or a fewfor i = 1:5\n  global x\n  x = x - H(x)\\g(x)\n  println(\"x = $x\")\nendNotice how we can use the method with different NLPModels:f(x) = (x[1] - 1.0)^2 + 100 * (x[2] - 1.0)^2\n\nadnlp = ADNLPModel(f, x0)\nx, fx, ngx, optimal, iter = steepest(adnlp)"
},

{
    "location": "tutorial/#OptimizationProblems-1",
    "page": "Tutorial",
    "title": "OptimizationProblems",
    "category": "section",
    "text": "The package OptimizationProblems provides a reasonable amount of problems defined in JuMP format, which can be converted to MathProgNLPModel.using OptimizationProblems # Defines a lot of JuMP models\n\nnlp = MathProgNLPModel(woods())\nx, fx, ngx, optimal, iter = steepest(nlp)\nprintln(\"fx = $fx\")\nprintln(\"ngx = $ngx\")\nprintln(\"optimal = $optimal\")\nprintln(\"iter = $iter\")Constrained problem can also be converted.using NLPModels, NLPModelsJuMP, JuMP\n\nmodel = Model()\nx0 = [-1.2; 1.0]\n@variable(model, x[i=1:2] >= 0.0, start=x0[i])\n@NLobjective(model, Min, (x[1] - 1)^2 + 100 * (x[2] - x[1]^2)^2)\n@constraint(model, x[1] + x[2] == 3.0)\n@NLconstraint(model, x[1] * x[2] >= 1.0)\n\nnlp = MathProgNLPModel(model)\n\nprintln(\"cx = $(cons(nlp, nlp.meta.x0))\")\nprintln(\"Jx = $(jac(nlp, nlp.meta.x0))\")"
},

{
    "location": "tutorial/#NLPModelsJuMP.MathProgNLSModel",
    "page": "Tutorial",
    "title": "NLPModelsJuMP.MathProgNLSModel",
    "category": "type",
    "text": "Construct a MathProgNLSModel from two MathProgModels.\n\n\n\n\n\nMathProgNLSModel(cmodel, F)\n\nConstruct a MathProgNLSModel from a JuMP Model and a vector of NLexpression.\n\n\n\n\n\n"
},

{
    "location": "tutorial/#MathProgNLSModel-Tutorial-1",
    "page": "Tutorial",
    "title": "MathProgNLSModel Tutorial",
    "category": "section",
    "text": "MathProgNLSModelMathProgNLSModel is a model for nonlinear least squares using JuMP. To use it, we define a JuMP model without the objective, and use NLexpressions to define the residual function. For instance, the Rosenbrock function in nonlinear least squares format isF(x) = beginbmatrix x_1 - 1 10(x_2 - x_1^2) endbmatrixwhich we can implement asusing NLPModels, NLPModelsJuMP, JuMP\n\nmodel = Model()\nx0 = [-1.2; 1.0]\n@variable(model, x[i=1:2], start=x0[i])\n@NLexpression(model, F1, x[1] - 1)\n@NLexpression(model, F2, 10 * (x[2] - x[1]^2))\n\nnls = MathProgNLSModel(model, [F1, F2], name=\"rosen-nls\")\n\nresidual(nls, nls.meta.x0)jac_residual(nls, nls.meta.x0)"
},

{
    "location": "tutorial/#NLPModelsJuMP.NLPtoMPB",
    "page": "Tutorial",
    "title": "NLPModelsJuMP.NLPtoMPB",
    "category": "function",
    "text": "mp = NLPtoMPB(nlp, solver)\n\nReturn a MathProgBase model corresponding to an AbstractNLPModel.\n\nArguments\n\nnlp::AbstractNLPModel\nsolver::AbstractMathProgSolver a solver instance, e.g., IpoptSolver()\n\nCurrently, all models are treated as nonlinear models.\n\nReturn values\n\nThe function returns a MathProgBase model mpbmodel such that it should be possible to call\n\nMathProgBase.optimize!(mpbmodel)\n\n\n\n\n\n"
},

{
    "location": "tutorial/#NLPtoMPB-Convert-NLP-to-MathProgBase-1",
    "page": "Tutorial",
    "title": "NLPtoMPB - Convert NLP to MathProgBase",
    "category": "section",
    "text": "NLPtoMPBIn addition to creating NLPModels using JuMP, we might want to convert an NLPModel to a MathProgBase model to use the solvers available. For instance#using Ipopt, NLPModels, NLPModelsJuMP, LinearAlgebra, JuMP, MathProgBase\n\n#nlp = ADNLPModel(x -> dot(x, x), ones(2),\n#                 c=x->[x[1] + 2 * x[2] - 1.0], lcon=[0.0], ucon=[0.0])\n#model = NLPtoMPB(nlp, IpoptSolver())\n\n#MathProgBase.optimize!(model)"
},

{
    "location": "reference/#",
    "page": "Reference",
    "title": "Reference",
    "category": "page",
    "text": ""
},

{
    "location": "reference/#Reference-1",
    "page": "Reference",
    "title": "Reference",
    "category": "section",
    "text": ""
},

]}
