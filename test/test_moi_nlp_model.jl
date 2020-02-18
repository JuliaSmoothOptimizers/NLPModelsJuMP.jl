println()
println("Testing MathOptNLPModel")

@printf("%-15s  %4s  %4s  %10s  %10s  %10s\n", "Problem", "nvar", "ncon", "|f(x₀)|", "‖∇f(x₀)‖", "‖c(x₀)‖")
# Test that every problem can be instantiated.
for prob in [:brownden, :hs5, :hs6, :hs10, :hs11, :hs14]
  prob_fn = eval(prob)
  nlp = MathOptNLPModel(prob_fn())
  n   = nlp.meta.nvar
  m   = nlp.meta.ncon
  x   = nlp.meta.x0
  fx  = abs(obj(nlp, x))
  ngx = norm(grad(nlp, x))
  ncx = m > 0 ? @sprintf("%10.4e", norm(cons(nlp, x))) : "NA"
  @printf("%-15s  %4d  %4d  %10.4e  %10.4e  %10s\n", prob, n, m, fx, ngx, ncx)
end
println()
