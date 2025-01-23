function nf()
  f(x::Vector{VariableRef}) = sqrt.(x)
  g(x::Vector{VariableRef}) = x.^3
  h(x::Vector{VariableRef}) = sum(x.^4)
  nlp = Model()
  @variable(nlp, x[1:2])
  @constraint(nlp, f(x) .- g(x) in MOI.Nonnegatives(2))
  @objective(nlp, Max, h(x))
  return nlp
end
