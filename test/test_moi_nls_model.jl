println()
println("Testing MathOptNLSModel")

@printf("%-15s  %4s  %4s  %4s  %10s  %10s  %10s\n", "Problem", "nequ", "nvar", "ncon", "‖F(x₀)‖²", "‖JᵀF‖", "‖c(x₀)‖")
# Test that every problem can be instantiated.
for prob in [:mgh01, :mgh07, :lls, :hs30, :hs43, :nlshs20]
  prob_fn = eval(prob)
  nls = prob_fn()
  N   = nls.nls_meta.nequ
  n   = nls.meta.nvar
  m   = nls.meta.ncon
  x   = nls.meta.x0
  Fx  = residual(nls, x)
  Jx  = jac_op_residual(nls, x)
  nFx = dot(Fx, Fx)
  JtF = norm(Jx' * Fx)
  ncx = m > 0 ? @sprintf("%10.4e", norm(cons(nls, x))) : "NA"
  @printf("%-15s  %4d  %4d  %4d  %10.4e  %10.4e  %10s\n", prob, N, n, m, nFx, JtF, ncx)
end
println()
