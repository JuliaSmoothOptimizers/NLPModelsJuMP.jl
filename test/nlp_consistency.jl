for problem in nlp_problems
  @testset "Problem $problem" begin
    nlp_manual = eval(Symbol(problem))()
    problem_f = eval(Symbol(lowercase(problem)))
    nlp_moi = MathOptNLPModel(problem_f())
    nlps = [nlp_manual; nlp_moi]
    consistent_nlps(nlps, linear_api = true, test_slack = false, exclude = [])
    view_subarray_nlp(nlp_moi)
  end
end
