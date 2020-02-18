using Documenter, NLPModelsJuMP

makedocs(
  modules = [NLPModelsJuMP],
  checkdocs = :exports,
  doctest = true,
  strict = true,
  format = Documenter.HTML(
             prettyurls = get(ENV, "CI", nothing) == "true",
             assets = ["assets/style.css"],
            ),
  sitename = "NLPModelsJuMP.jl",
  pages = Any["Home" => "index.md",
              "Tutorial" => "tutorial.md",
              "Reference" => "reference.md"]
)

deploydocs(deps = nothing, make = nothing,
  repo = "github.com/JuliaSmoothOptimizers/NLPModelsJuMP.jl.git",
  target = "build",
  devbranch = "master"
)
