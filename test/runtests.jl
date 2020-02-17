using JuMP, NLPModels, NLPModelsJuMP
using LinearAlgebra, SparseArrays
using Test, Printf

nlpmodels_path = joinpath(dirname(pathof(NLPModels)), "..", "test")

for problem in [:brownden, :hs5, :hs6, :hs10, :hs11, :hs14]#, :hs30, :hs43, :mgh07, :mgh35]
  include(joinpath("problems", "$problem.jl"))
  if isfile(joinpath(nlpmodels_path, "$problem.jl"))
    include(joinpath(nlpmodels_path, "$problem.jl"))
  end
end

include("test_moi_nlp_model.jl")
# include("test_moi_nls_model.jl")

include("nlp_consistency.jl")
# include("nls_consistency.jl")

# include("test_view_subarray.jl")
