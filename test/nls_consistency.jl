for problem in nls_problems
  @testset "Problem $problem" begin
    nls_manual = eval(Symbol(problem))()
    nls_moi = eval(Symbol(lowercase(problem)))()
    nlss = [nls_manual, nls_moi]

    spc = "$(problem)_special"
    if isdefined(Main, Symbol(spc))
      push!(nlss, eval(Meta.parse(spc))())
    end
    consistent_nlss(nlss, linear_api=true, test_slack=false)
    view_subarray_nls(nls_moi)
  end
end
