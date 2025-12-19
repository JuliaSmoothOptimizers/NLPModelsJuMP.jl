for problem in [nlp_problems; extra_nlp_oracle_problems]
  @testset "Problem $problem" begin
    nlp_manual = eval(replace(problem, "_oracle" => "") |> uppercase |> Symbol)()
    problem_f = eval(problem |> lowercase |> Symbol)
    nlp_moi = MathOptNLPModel(problem_f())
    nlps = [nlp_manual; nlp_moi]
    exclude = (problem in extra_nlp_oracle_problems) ? [jprod, jtprod, hprod] : []
    consistent_nlps(
      nlps,
      linear_api = true,
      test_slack = false,
      test_counters = false,
      exclude = exclude,
    )
    coord_memory_nlp(nlp_moi, linear_api = true)
    view_subarray_nlp(nlp_moi)
  end
end
