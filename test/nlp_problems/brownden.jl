function brownden()
  nlp = Model()

  @variable(nlp, x[1:4])
  set_start_value.(x, [25.0; 5.0; -5.0; -1.0])

  @NLobjective(
    nlp,
    Min,
    sum(
      ((x[1] + x[2] * i / 5 - exp(i / 5))^2 + (x[3] + x[4] * sin(i / 5) - cos(i / 5))^2)^2 for
      i = 1:20
    )
  )

  return nlp
end
