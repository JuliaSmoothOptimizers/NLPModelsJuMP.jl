using JuMP, NLPModels, NLPModelsJuMP
using LinearAlgebra, SparseArrays
using Test, Printf

nlpmodels_path = joinpath(dirname(pathof(NLPModels)), "..", "test")
nlpmodels_problems_path = joinpath(nlpmodels_path, "problems")

for problem in [:brownden, :hs5, :hs6, :hs10, :hs11, :hs14]
  include(joinpath("problems", "$problem.jl"))
  if isfile(joinpath(nlpmodels_problems_path, "$problem.jl"))
    include(joinpath(nlpmodels_problems_path, "$problem.jl"))
  end
end

include("consistency.jl")
