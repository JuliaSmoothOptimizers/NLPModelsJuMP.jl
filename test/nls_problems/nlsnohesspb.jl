function nlsnohesspb()
  nls = Model()

  n = 5
  g(x...) = 10*n + sum(x[i]^2 - 10 * cos(2π * x[i]) for i in 1:n)
  
  x₀ = [1.0, 0.1, 0.2, -0.5, 1.0]
  @variable(nls, x[i=1:n], start = x₀[i])

  register(nls, :g, n, g, autodiff = true)

  @NLexpression(nls, res[i in 1:n], g(x...))

  return MathOptNLSModel(nls, res, hessian = false, name = "nlsnohesspb")
end
