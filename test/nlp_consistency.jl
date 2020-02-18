using NLPModels

include(joinpath(nlpmodels_path, "consistency.jl"))

function test_nlp_consistency()
  println()
  for problem in [:brownden, :hs5, :hs6, :hs10, :hs11, :hs14]
    problem_s = string(problem)
    @printf("Checking NLP problem %-20s", problem_s)
    problem_f = eval(problem)
    nlp_autodiff = eval(Meta.parse("$(problem)_autodiff"))()
    nlp_manual = eval(Meta.parse(uppercase(string(problem))))()
    nlp_moi = MathOptNLPModel(problem_f())
    nlps = [nlp_autodiff; nlp_manual; nlp_moi]

    consistent_nlps(nlps)
  end
  println()
end

test_nlp_consistency()
