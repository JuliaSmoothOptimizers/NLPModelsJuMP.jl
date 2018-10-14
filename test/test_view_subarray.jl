using JuMP

function test_view_subarray()
  model = Model()
  @variable(model, x[1:3])
  @NLobjective(model, Min, sum(x[i]^4 for i = 1:3))
  @NLconstraint(model, sum(x[i]^2 for i = 1:3) == 4.0)
  @NLconstraint(model, sum(x[i] for i = 1:3) == 1.0)
  nlp = MathProgNLPModel(model)

  x = rand(5)
  functions = [obj, grad, hess, cons, jac]
  for foo in functions, I = [1:3, 1:2:5, [2;4;5], [4;1;3]]
    @test foo(nlp, x[I]) == foo(nlp, @view x[I])
  end
end

test_view_subarray()
