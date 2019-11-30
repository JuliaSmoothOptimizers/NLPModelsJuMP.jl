# NLPModelsJuMP

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.2574164.svg)](https://doi.org/10.5281/zenodo.2574164)

Cite as

    Abel Soares Siqueira & Dominique Orban. NLPModelsJuMP.jl. Zenodo.
    http://doi.org/10.5281/zenodo.2574164

[![Build
Status](https://travis-ci.org/JuliaSmoothOptimizers/NLPModelsJuMP.jl.svg?branch=master)](https://travis-ci.org/JuliaSmoothOptimizers/NLPModelsJuMP.jl)
[![Build status](https://ci.appveyor.com/api/projects/status/g0ofb4r1aqo56hbo?svg=true)](https://ci.appveyor.com/project/dpo/nlpmodelsjump-jl)
[![Build Status](https://api.cirrus-ci.com/github/JuliaSmoothOptimizers/NLPModelsJuMP.jl.svg)](https://cirrus-ci.com/github/JuliaSmoothOptimizers/NLPModelsJuMP.jl)
[![](https://img.shields.io/badge/docs-stable-3f51b5.svg)](https://JuliaSmoothOptimizers.github.io/NLPModelsJuMP.jl/stable)

This package provides nonlinear programming models as implemented by
[NLPModels](https://github.com/JuliaSmoothOptimizers/NLPModels.jl), through the use of
[JuMP](https://github.com/JuliaOpt/JuMP.jl).

**Disclaimer**: NLPModelsJuMP is *not* developed or maintained by the JuMP developers.

See the documentation on NLPModels for the general description of NLPModels. Here, we
focus on the use of JuMP to create these.

## NLPModels from a JuMP

NLPModelsJuMP provides conversion between MathProgBase models and NLPModels.
