include(joinpath(nlpmodels_path, "nls_consistency.jl"))

function consistent_nls()
  @testset "Consistency of Linear problem" begin
    m, n = 50, 20
    A = Matrix(1.0I, m, n) .+ 1
    b = collect(1:m)
    lvar = -ones(n)
    uvar = ones(n)
    lls_model = LLSModel(A, b, lvar=lvar, uvar=uvar)
    autodiff_model = ADNLSModel(x->A*x-b, zeros(n), m, lvar=lvar, uvar=uvar)
    nlp = ADNLPModel(x->0, zeros(n), lvar=lvar, uvar=uvar, c=x->A*x-b,
                     lcon=zeros(m), ucon=zeros(m))
    feas_res_model = FeasibilityResidual(nlp)
    model = Model()
    @variable(model, x[1:n], start=0.0)
    @NLexpression(model, F[i=1:m], sum(A[i,j] * x[j] for j = 1:n) - b[i])
    mpnls = MathProgNLSModel(model, F)
    nlss = [lls_model, autodiff_model, feas_res_model, mpnls]
    consistent_nls_counters(nlss)
    consistent_counters(nlss)
    consistent_nls_functions(nlss)
    consistent_nls_counters(nlss)
    consistent_counters(nlss)
    for nls in nlss
      reset!(nls)
    end

    f(x) = begin
      r = A*x - b
      return 0.5*dot(r, r)
    end
    nlps = [nlss; ADNLPModel(f, zeros(n))]
    consistent_functions(nlps, exclude=[hess_coord])
  end

  @testset "Consistency of Linear problem with linear constraints" begin
    m, n = 50, 20
    A = Matrix(1.0I, m, n) .+ 1
    b = collect(1:m)
    lvar = -ones(n)
    uvar = ones(n)
    nc = 10
    C = [ones(nc, n); 2 * ones(nc, n); -ones(nc, n); -Matrix(1.0I, nc, n)]
    lcon = [   zeros(nc); -ones(nc); fill(-Inf,nc); zeros(nc)]
    ucon = [fill(Inf,nc);  ones(nc);     zeros(nc); zeros(nc)]
    K = ((1:4:4nc) .+ (0:3)')[:]
    lcon, ucon = lcon[K], ucon[K]
    lls_model = LLSModel(A, b, lvar=lvar, uvar=uvar, C=C, lcon=lcon,
                         ucon=ucon)
    autodiff_model = ADNLSModel(x->A*x-b, zeros(n), m, lvar=lvar,
                                uvar=uvar, c=x->C*x, lcon=lcon,
                                ucon=ucon)
    model = Model()
    @variable(model, x[1:n], start=0.0)
    @NLexpression(model, F[i=1:m], sum(A[i,j] * x[j] for j = 1:n) - b[i])
    @NLconstraint(model, [i=1:4nc], lcon[i] ≤ sum(C[i,j] * x[j] for j = 1:n) ≤ ucon[i])
    mpnls = MathProgNLSModel(model, F)
    nlss = [lls_model, autodiff_model, mpnls]
    consistent_nls_counters(nlss)
    consistent_counters(nlss)
    consistent_nls_functions(nlss)
    consistent_nls_counters(nlss)
    consistent_counters(nlss)
    consistent_functions(nlss, exclude=[hess_coord])
  end

  @testset "Consistency of Nonlinear problem" begin
    m, n = 10, 2
    lvar = -ones(n)
    uvar =  ones(n)
    F(x) = [2 + 2i - exp(i*x[1]) - exp(i*x[2]) for i = 1:m]
    x0 = [0.3; 0.4]
    autodiff_model = ADNLSModel(F, x0, m, lvar=lvar, uvar=uvar)
    nlp = ADNLPModel(x->0, x0, lvar=lvar, uvar=uvar, c=F, lcon=zeros(m), ucon=zeros(m))
    feas_res_model = FeasibilityResidual(nlp)
    model = Model()
    @variable(model, x[i=1:n], start=x0[i])
    @NLexpression(model, F1[i=1:m], 2 + 2i - exp(i*x[1]) - exp(i*x[2]))
    mpnls = MathProgNLSModel(model, F1)
    nlss = [autodiff_model, feas_res_model, mpnls]
    consistent_nls_counters(nlss)
    consistent_counters(nlss)
    consistent_nls_functions(nlss)
    consistent_nls_counters(nlss)
    consistent_counters(nlss)
    for nls in nlss
      reset!(nls)
    end

    f(x) = begin
      r = F(x)
      return 0.5*dot(r, r)
    end
    nlps = [nlss; ADNLPModel(f, zeros(n))]
    consistent_functions(nlps, exclude=[hess_coord])
  end

  @testset "Consistency of Nonlinear problem with constraints" begin
    m, n = 10, 2
    lvar = -ones(n)
    uvar =  ones(n)
    F(x) = [2 + 2i - exp(i*x[1]) - exp(i*x[2]) for i = 1:m]
    x0 = [0.3; 0.4]
    c(x) = [x[1]^2 - x[2]^2; 2 * x[1] * x[2]; x[1] + x[2]]
    lcon = [0.0; -1.0; -Inf]
    ucon = [Inf;  1.0;  0.0]

    autodiff_model = ADNLSModel(F, x0, m, lvar=lvar, uvar=uvar,
                                lcon=lcon, ucon=ucon, c=c)
    model = Model()
    @variable(model, x[i=1:n], start=x0[i])
    @NLconstraint(model, x[1]^2 - x[2]^2 ≥ 0.0)
    @NLconstraint(model, -1.0 ≤ 2 * x[1] * x[2] ≤ 1.0)
    @NLconstraint(model, x[1] + x[2] ≤ 0.0)
    @NLexpression(model, F1[i=1:m], 2 + 2i - exp(i*x[1]) - exp(i*x[2]))
    mpnls = MathProgNLSModel(model, F1)

    nlss = [autodiff_model, mpnls]
    consistent_nls_counters(nlss)
    consistent_counters(nlss)
    consistent_nls_functions(nlss)
    consistent_nls_counters(nlss)
    consistent_counters(nlss)
    consistent_functions(nlss, exclude=[hess_coord])
  end

  @testset "Consistency of slack variant" begin
    model = Model()
    x0 = [0.3; 0.4; 0.5]
    @variable(model, x[i=1:3], start=x0[i])
    @NLexpression(model, F1, x[1] - 1.0)
    @NLexpression(model, F2, x[2] - x[1]^2)
    @NLexpression(model, F3, sin(x[1] * x[2]) * x[3])
    @NLconstraint(model, x[1]^2 - x[1]^2 ≥ 0.0)
    @NLconstraint(model, x[1] + x[2] ≤ 0.0)
    @NLconstraint(model, cos(x[1]) - x[2] == 0.0)
    nls = MathProgNLSModel(model, [F1; F2; F3])
    nls_auto_slack = SlackNLSModel(nls)

    smodel = Model()
    x0 = [x0; zeros(2)]
    ℓ = [-Inf; -Inf; -Inf; 0.0; -Inf]
    u = [ Inf;  Inf;  Inf; Inf;  0.0]
    @variable(smodel, ℓ[i] ≤ x[i=1:5] ≤ u[i], start=x0[i])
    @NLexpression(smodel, F1s, x[1] - 1.0)
    @NLexpression(smodel, F2s, x[2] - x[1]^2)
    @NLexpression(smodel, F3s, sin(x[1] * x[2]) * x[3])
    @NLconstraint(smodel, x[1]^2 - x[1]^2 - x[4] == 0.0)
    @NLconstraint(smodel, x[1] + x[2] - x[5] == 0.0)
    @NLconstraint(smodel, cos(x[1]) - x[2] == 0.0)
    nls_manual_slack = MathProgNLSModel(smodel, [F1s; F2s; F3s])

    nlss = [nls_manual_slack, nls_auto_slack]
    consistent_nls_counters(nlss)
    consistent_counters(nlss)
    consistent_nls_functions(nlss)
    consistent_nls_counters(nlss)
    consistent_counters(nlss)
    consistent_functions(nlss)
  end
end

consistent_nls()
