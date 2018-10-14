__precompile__()

module NLPModelsJuMP

using Compat

include("nlsmpb_model.jl")
include("nlp_to_mpb.jl")
include("jump_model.jl")

end
