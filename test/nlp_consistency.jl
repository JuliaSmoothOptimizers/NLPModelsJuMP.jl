for problem in [nlp_problems; nlp_oracle_problems]
  @testset "Problem $problem" begin
    nlp_manual = eval(replace(problem, "_oracle" => "") |> uppercase |> Symbol)()
    problem_f = eval(problem |> lowercase |> Symbol)
    nlp_moi = MathOptNLPModel(problem_f())
    nlps = [nlp_manual; nlp_moi]
    @testset "linear_api = $linear_api" for linear_api in (false, true)
      consistent_nlps(
        nlps,
        linear_api = linear_api,
        test_slack = false,
        test_counters = false,
        exclude = [],
      )
      coord_memory_nlp(nlp_moi, linear_api = linear_api)
    end
    view_subarray_nlp(nlp_moi)
  end
end
