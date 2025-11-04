using Test
using NLPModelsJuMP

@testset "lencheck macro" begin
    x = [1,2,3]
    @lencheck 3 x
    y = [1,2]
    @test_throws ArgumentError @lencheck 3 y
end
