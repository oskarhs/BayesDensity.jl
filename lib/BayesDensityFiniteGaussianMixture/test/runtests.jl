using Test
using BayesDensityFiniteGaussianMixture
using Distributions, Random

const rng = Random.Xoshiro(1)

include("aqua.jl")

@testset "FiniteGaussianMixture: Constructor and model object" begin
    x = [1.0, -1.0]

    gm = FiniteGaussianMixture(x, 2)

    # Test equality
    @test gm == FiniteGaussianMixture(x, 2)

    # Test that we have subtyped AbstractBayesDensityModel
    @test gm isa AbstractBayesDensityModel{Float64}

    # Check that the hyperparams method can be used without any issues
    @test hyperparams(gm) isa NamedTuple

    # Check that the support is returned correctly
    @test Distributions.support(gm) == (-Inf, Inf)

    # Check that out of bounds hyperparameter values throw errors
    for hyp in (:prior_strength, :prior_variance, :prior_shape, :hyperprior_rate, :hyperprior_shape)
        @eval @test_throws ArgumentError $FiniteGaussianMixture($x, 2; $hyp = -1)
    end

    # Test show method
    io = IOBuffer() # just checks that we can call the show method
    show(io, gm)
    output = String(take!(io))
    @test output isa String
end

@testset "RandomFiniteGaussianMixture: Constructor and model object" begin
    x = [1.0, -1.0]

    gm = RandomFiniteGaussianMixture(x)

    # Test equality
    @test gm == RandomFiniteGaussianMixture(x)

    # Test that we have subtyped AbstractBayesDensityModel
    @test gm isa AbstractBayesDensityModel{Float64}

    # Check that the hyperparams method can be used without any issues
    @test hyperparams(gm) isa NamedTuple

    # Check that the support is returned correctly
    @test Distributions.support(gm) == (-Inf, Inf)

    # Check that out of bounds hyperparameter values throw errors
    for hyp in (:prior_strength, :prior_variance, :prior_shape, :hyperprior_rate, :hyperprior_shape)
        @eval @test_throws ArgumentError $RandomFiniteGaussianMixture($x; $hyp = -1)
    end

    # Test show method
    io = IOBuffer() # just checks that we can call the show method
    show(io, gm)
    output = String(take!(io))
    @test output isa String
end


@testset "FiniteGaussianMixture: pdf and cdf" begin
    x = [1.0, -1.0]

    for gm in (FiniteGaussianMixture(x, 2; prior_strength = 1), RandomFiniteGaussianMixture(x; prior_strength = 1))

        n_rep = 10
        for parameters in ((μ = [0.0], σ2 = [1.0], w = [1.0]), (μ = [0.0], σ2 = [1.0], w = [1.0], β=[1.0]))
            parameters_vec = fill(parameters, n_rep)
            # Evaluate at single point:
            @test isapprox(pdf(gm, parameters, 0.0), pdf(Normal(0, 1), 0.0))
            @test isapprox(pdf(gm, parameters_vec, 0.0), fill(pdf(Normal(0, 1), 0.0), (1, n_rep)))
            @test isapprox(cdf(gm, parameters, 0.0), cdf(Normal(0, 1), 0.0))
            @test isapprox(cdf(gm, parameters_vec, 0.0), fill(cdf(Normal(0, 1), 0.0), (1, n_rep)))

            # Evaluate at multiple points:
            t = LinRange(-5, 5, 11)
            @test isapprox(pdf(gm, parameters, t), pdf(Normal(0, 1), t))
            @test isapprox(pdf(gm, parameters_vec, t), mapreduce(x->pdf(Normal(0, 1), t), hcat, fill(0, n_rep)))
            @test isapprox(cdf(gm, parameters, t), cdf(Normal(0, 1), t))
            @test isapprox(cdf(gm, parameters_vec, t), mapreduce(x->cdf(Normal(0, 1), t), hcat, fill(0, n_rep)))
        end
    end
end

@testset "FiniteGaussianMixture: MCMC sample" begin
    K = 2
    x = collect(-5:0.1:5)

    fgm = FiniteGaussianMixture(x, 2)
    @test sample(rng, fgm, 20) isa PosteriorSamples{Float64}
end

@testset "FiniteGaussianMixture: varinf" begin
    x = collect(-5:0.1:5)
    fgm = FiniteGaussianMixture(x, 3)

    vip, _ = varinf(fgm)
    @test vip isa AbstractVIPosterior{Float64}
    @test sample(rng, vip, 100) isa PosteriorSamples{Float64}
end

@testset "FiniteGaussianMixture: VIPosterior" begin
    # Make a VI posterior very strongly concentrated on a given mixture.
    # Then the posterior mean should be very close to just evaluating the pdf of the mixture
    x = collect(-5:0.1:5)
    fgm = FiniteGaussianMixture{Float64}(x, 3)
    d_target = MixtureModel([Normal(j, sqrt(j+2)) for j in -1:1], [0.2, 0.6, 0.2])
    vip = FiniteGaussianMixtureVIPosterior{Float64}(
        1e12*[0.2, 0.6, 0.2], # dirichlet_params
        [-1.0, 0.0, 1.0],     # location_params
        fill(1e-12, 3),       # variance_params
        fill(1e12, 3),        # shape_params
        1e12*[1.0, 2.0, 3.0], # rate_params
        1.0,                  # shape hyperparam (irrelevant for pdf)
        1.0,                  # rate hyperparam (irrelevant for pdf)
        fgm
    )

    t = LinRange(-5, 5, 1001)
    @test isapprox(pdf(d_target, t), mean(vip, t); rtol=1e-5)

    @test model(vip) isa FiniteGaussianMixture{Float64}

    io = IOBuffer() # just checks that we can call the show method
    show(io, vip)
    output = String(take!(io))
    @test output isa String
end

@testset "RandomFiniteGaussianMixture: varinf" begin
    x = collect(-5:0.1:5)
    rfgm = RandomFiniteGaussianMixture(x)

    vip = varinf(rfgm)
    @test vip isa AbstractVIPosterior{Float64}
    @test sample(rng, vip, 100) isa PosteriorSamples{Float64}
end

@testset "RandomFiniteGaussianMixture: VIPosterior" begin
    # Make a VI posterior very strongly concentrated on a given mixture.
    # Then the posterior mean should be very close to just evaluating the pdf of the mixture
    x = collect(-5:0.1:5)
    fgm = FiniteGaussianMixture{Float64}(x, 3)
    rfgm = RandomFiniteGaussianMixture{Float64}(x; prior_components = DiscreteNonParametric([3], [1.0]))
    d_target = MixtureModel([Normal(j, sqrt(j+2)) for j in -1:1], [0.2, 0.6, 0.2])
    vip_fgm = FiniteGaussianMixtureVIPosterior{Float64}(
        1e12*[0.2, 0.6, 0.2], # dirichlet_params
        [-1.0, 0.0, 1.0],     # location_params
        fill(1e-12, 3),       # variance_params
        fill(1e12, 3),        # shape_params
        1e12*[1.0, 2.0, 3.0], # rate_params
        1.0,                  # shape hyperparam (irrelevant for pdf)
        1.0,                  # rate hyperparam (irrelevant for pdf)
        fgm
    )
    vip = RandomFiniteGaussianMixtureVIPosterior{Float64}(DiscreteNonParametric([3], [1.0]), [vip_fgm], rfgm)

    t = LinRange(-5, 5, 1001)
    @test isapprox(pdf(d_target, t), mean(vip, t); rtol=1e-5)

    @test maximum_a_posteriori(vip) == vip_fgm
    @test posterior_components(vip) == DiscreteNonParametric([3], [1.0])

    @test model(vip) isa RandomFiniteGaussianMixture{Float64}

    io = IOBuffer() # just checks that we can call the show method
    show(io, vip)
    output = String(take!(io))
    @test output isa String
end