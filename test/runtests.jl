using JuMP, NLPModels, NLPModelsJuMP
using LinearAlgebra, SparseArrays
using Test, Printf

nlpmodels_path = joinpath(dirname(pathof(NLPModels)), "..", "test")
nlpmodels_problems_path = joinpath(nlpmodels_path, "problems")

for problem in [:brownden, :hs5, :hs6, :hs10, :hs11, :hs14, :hs28, :hs39, :lincon]
  include(joinpath("nlp_problems", "$problem.jl"))
  if isfile(joinpath(nlpmodels_problems_path, "$problem.jl"))
    include(joinpath(nlpmodels_problems_path, "$problem.jl"))
  end
end

for problem in [:lls, :mgh01, :nlshs20, :hs30, :hs43, :mgh07, :nlslc]
  include(joinpath("nls_problems", "$problem.jl"))
  if isfile(joinpath(nlpmodels_path, "nls_problems", "$problem.jl"))
    include(joinpath(nlpmodels_path, "nls_problems", "$problem.jl"))
  end
end

include("test_moi_nlp_model.jl")
include("test_moi_nls_model.jl")

include("nlp_consistency.jl")
include("nls_consistency.jl")

include("test_view_subarray.jl")
