using JuMP, NLPModels, NLPModelsJuMP, NLPModelsTest
using LinearAlgebra, SparseArrays
using Test, Printf

nlp_problems = setdiff(NLPModelsTest.nlp_problems, ["MGH01Feas"])
nls_problems = NLPModelsTest.nls_problems

extra_nlp_problems = ["nohesspb", "hs61", "hs100", "hs219", "quadcon"]
extra_nls_problems = ["nlsnohesspb", "HS30", "HS43", "MGH07", "nlsqc"]

for problem in lowercase.(nlp_problems ∪ extra_nlp_problems)
  include(joinpath("nlp_problems", "$problem.jl"))
end

for problem in lowercase.(nls_problems ∪ extra_nls_problems)
  include(joinpath("nls_problems", "$problem.jl"))
end

include("test_moi_nlp_model.jl")
include("test_moi_nls_model.jl")

include("nlp_consistency.jl")
include("nls_consistency.jl")

include("MOI_wrapper.jl")
