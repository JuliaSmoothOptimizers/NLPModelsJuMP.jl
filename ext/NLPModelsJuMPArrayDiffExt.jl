module NLPModelsJuMPArrayDiffExt

import NLPModelsJuMP
import ArrayDiff

NLPModelsJuMP._nonlinear_model(ad::ArrayDiff.Mode) = ArrayDiff.model(ad)

end
