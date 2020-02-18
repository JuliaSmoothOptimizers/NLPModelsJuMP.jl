using JuMP

include(joinpath(nlpmodels_path, "test_view_subarray.jl"))

function test_view_subarray()
  println()
  @testset "Test view subarrays for MathOpt models" begin
    n = 5
    model = Model()
    @variable(model, x[1:n])
    @NLobjective(model, Min, sum(x[i]^4 for i = 1:n))
    @NLconstraint(model, -1.0 ≤ sum(x[i]^2 for i = 1:n) - 4.0 ≤ 1.0)
    @NLconstraint(model, -1.0 ≤ sum(x[i] for i = 1:n) - 1.0   ≤ 1.0)
    @NLconstraint(model, -1.0 ≤ x[1] * x[2] * x[3]            ≤ 1.0)
    moinlp = MathOptNLPModel(model)

    snlp = SlackModel(moinlp)

    model = Model()
    @variable(model, x[1:n])
    @NLconstraint(model, -1.0 ≤ sum(x[i]^2 for i = 1:n) - 4.0 ≤ 1.0)
    @NLconstraint(model, -1.0 ≤ sum(x[i] for i = 1:n) - 1.0   ≤ 1.0)
    @NLconstraint(model, -1.0 ≤ x[1] * x[2] * x[3]            ≤ 1.0)
    @NLexpression(model, F[p=1:3], sum(x[i]^p for i = 1:n) - p)

    moinls = MathOptNLSModel(model, F)

    snls = SlackNLSModel(moinls)

    for nlp in [moinlp, snlp, moinls, snls]
      test_view_subarray_nlp(nlp)
    end

    for nls in [moinls, snls]
      test_view_subarray_nls(nls)
    end
  end
  println()
end

test_view_subarray()
