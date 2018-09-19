using Base.Test, JuMP, JuMPNLPModels

for problem in [:brownden, :hs5, :hs6, :hs10, :hs11, :hs14, :hs30, :hs43, :mgh07, :mgh35]
  include("$problem.jl")
end

include("test_mpb.jl")
include("consistency.jl")
include("test_mathprognlsmodel.jl")
