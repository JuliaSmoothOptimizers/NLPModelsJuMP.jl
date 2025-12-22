for problem in nls_problems
  @testset "Problem $problem" begin
    nls_manual = eval(Symbol(problem))()
    nls_moi = eval(Symbol(lowercase(problem)))()
    nlss = [nls_manual, nls_moi]

    spc = "$(problem)_special"
    if isdefined(Main, Symbol(spc))
      push!(nlss, eval(Meta.parse(spc))())
    end
    @testset "linear_api = $linear_api" for linear_api in (false, true)
      consistent_nlss(nlss, linear_api = linear_api, test_slack = false, test_counters = false)
    end
    view_subarray_nls(nls_moi)
  end
end
