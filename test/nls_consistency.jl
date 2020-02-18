include(joinpath(nlpmodels_path, "nls_consistency.jl"))

function test_nls_consistency()
  println()
  for problem in [:lls, :mgh01, :nlshs20]
    @printf("Checking NLS problem %-20s", problem)
    nls_autodiff = eval(Meta.parse("$(problem)_autodiff"))()
    nls_manual = eval(Meta.parse(uppercase(string(problem))))()
    nls_moi = eval(problem)()
    nlss = [nls_autodiff, nls_manual, nls_moi]

    spc = "$(problem)_special"
    if isdefined(Main, Symbol(spc))
      push!(nlss, eval(Meta.parse(spc))())
    end
    consistent_nlss(nlss)
    println("✓")
  end
  println()
end

test_nls_consistency()
