using Documenter, NLPModelsJuMP

makedocs(
  modules = [NLPModelsJuMP],
  checkdocs = :exports,
  doctest = true,
  linkcheck = true,
  format = Documenter.HTML(
    prettyurls = get(ENV, "CI", nothing) == "true",
    ansicolor = true,
    assets = ["assets/style.css"],
  ),
  sitename = "NLPModelsJuMP.jl",
  pages = Any["Home" => "index.md", "Tutorial" => "tutorial.md", "Reference" => "reference.md"],
)

deploydocs(
  repo = "github.com/JuliaSmoothOptimizers/NLPModelsJuMP.jl.git",
  push_preview = true,
  devbranch = "main",
)
