function nohesspb()
  nlp = Model()

  @variable(nlp, x[1:2])

  g(x::T, y::T) where {T <: Real} = x * y + 3

  function ∇g(v::AbstractVector{T}, x::T, y::T) where {T <: Real}
    v[1] = y
    v[2] = x
    return v
  end

  register(nlp, :g, 2, g, autodiff = true)
  @NLobjective(nlp, Min, g(x[1], x[2]))

  return nlp
end
