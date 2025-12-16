@testset "Testing Oracle vs Non-Oracle NLP Models: $prob" for prob in extra_nlp_oracle_problems
    prob_no_oracle = replace(prob, "_oracle" => "")
    prob_fn_no_oracle = eval(Symbol(prob_no_oracle))
    prob_fn = eval(Symbol(prob))
    nlp_no_oracle = MathOptNLPModel(prob_fn_no_oracle(), hessian = true)
    nlp_with_oracle = MathOptNLPModel(prob_fn(), hessian = true)
    n = nlp_no_oracle.meta.nvar
    m = nlp_no_oracle.meta.ncon
    x = nlp_no_oracle.meta.x0
    fx_no_oracle = obj(nlp_no_oracle, x)
    fx_with_oracle = obj(nlp_with_oracle, x)
    @test isapprox(fx_no_oracle, fx_with_oracle; atol = 1e-8, rtol = 1e-8)
    ngx_no_oracle = grad(nlp_no_oracle, x)
    ngx_with_oracle = grad(nlp_with_oracle, x)
    @test isapprox(ngx_no_oracle, ngx_with_oracle; atol = 1e-8, rtol = 1e-8)
    if m > 0
        ncx_no_oracle = cons(nlp_no_oracle, x)
        ncx_with_oracle = cons(nlp_with_oracle, x)
        @test isapprox(ncx_no_oracle, ncx_with_oracle; atol = 1e-8, rtol = 1e-8)
    end
end