using Test, JuMP, NLPModels, NLPModelsJuMP, LinearAlgebra, SparseArrays, Printf

path = joinpath(dirname(pathof(NLPModels)), "..", "test")

for problem in [:brownden, :hs5, :hs6, :hs10, :hs11, :hs14, :hs30, :hs43, :mgh07, :mgh35]
  include("$problem.jl")
  if isfile(joinpath(path, "$problem.jl"))
    include(joinpath(path, "$problem.jl"))
  end
end

include("test_mpb.jl")
include("consistency.jl")
include("test_mathprognlsmodel.jl")
