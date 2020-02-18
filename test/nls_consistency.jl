include(joinpath(nlpmodels_path, "nls_consistency.jl"))

for problem in [:lls, :mgh01, :nlshs20]
  @printf("Checking NLS problem %-20s", problem)
  nls_ad = eval(Meta.parse("$(problem)_autodiff"))()
  nls_man = eval(Meta.parse(uppercase(string(problem))))()
  nls_jump = eval(problem)()
  nlss = [nls_ad, nls_man, nls_jump]

  spc = "$(problem)_special"
  if isdefined(Main, Symbol(spc))
    push!(nlss, eval(Meta.parse(spc))())
  end
  consistent_nlss(nlss)
  println("✓")
end
