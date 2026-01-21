using Test
using BayesDensityPitmanYorMixture
using Distributions, Random

const rng = Random.Xoshiro(1)

#include("aqua.jl")

@testset "PitmanYorMixture: Constructor and model object" begin
    x = [1.0, -1.0]

    pym = PitmanYorMixture(x)

    # Test that we have subtyped AbstractBayesDensityModel
    @test typeof(pym) <: AbstractBayesDensityModel{Float64}

    # Check that the hyperparams method can be used without any issues
    @test typeof(hyperparams(pym)) <: NamedTuple

    # Check that the support is returned correctly
    @test support(pym) == (-Inf, Inf)

    # Check that out of bounds hyperparameter values throw errors
    for hyp in (:discount, :strength, :inv_scale_fac, :shape, :rate)
        @eval @test_throws ArgumentError $PitmanYorMixture($x; $hyp = -1)
    end
end

@testset "PitmanYorMixture: pdf and cdf" begin
    x = [1.0, -1.0]

    pym = PitmanYorMixture(x; strength = 1e-10)

    # For α ≈ 0 the new cluster part does not contribute much
    n_rep = 10
    parameters = (μ = [0.0], σ2 = [1.0], cluster_counts = [2])
    parameters_vec = fill(parameters, n_rep)

    # Evaluate at single point:
    @test isapprox(pdf(pym, parameters, 0.0), pdf(Normal(0, 1), 0.0))
    @test isapprox(pdf(pym, parameters_vec, 0.0), fill(pdf(Normal(0, 1), 0.0), (n_rep, 1)))
    @test isapprox(cdf(pym, parameters, 0.0), cdf(Normal(0, 1), 0.0))
    @test isapprox(cdf(pym, parameters_vec, 0.0), fill(cdf(Normal(0, 1), 0.0), (n_rep, 1)))

    # Evaluate at multiple points:
    t = LinRange(-5, 5, 11)
    @test isapprox(pdf(pym, parameters, t), pdf(Normal(0, 1), t))
    @test isapprox(pdf(pym, parameters_vec, t), mapreduce(t->pdf(Normal(0, 1), t)), hcat, fill(0, n_rep))
    @test isapprox(cdf(pym, parameters, t), cdf(Normal(0, 1), t))
    @test isapprox(cdf(pym, parameters_vec, t), mapreduce(t->cdf(Normal(0, 1), t)), hcat, fill(0, n_rep))

end

@testset "PitmanYorMixture: sample" begin
    x = collect(-5:0.1:5)

    pym = PitmanYorMixture(x)
    @test typeof(sample(rng, pym, 100)) <: PosteriorSamples{Float64}
end