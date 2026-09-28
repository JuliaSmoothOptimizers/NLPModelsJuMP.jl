"Helical valley function"
function mgh07()
  nls = Model()
  x0 = [-1.0, 0.0, 0.0]
  @variable(nls, x[i = 1:3], start = x0[i])

  θ_aux(t) = (t > 0 ? 0.0 : 0.5)

  @operator(nls, θ, 1, θ_aux)

  @expression(nls, F1, x[3])
  @expression(nls, F2, 10 * (x[3] - 10 * (atan(x[2] / x[1]) / (2 * π) + θ(x[1]))))
  @expression(nls, F3, 10 * (sqrt(x[1]^2 + x[2]^2) - 1.0))

  return MathOptNLSModel(nls, [F1, F2, F3], name = "mgh07")
end
