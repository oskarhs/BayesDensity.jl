using Aqua
using BayesDensityLogisticGaussianProcess
using Test

@testset "LogisticGaussianProcess: Aqua.jl" begin
    Aqua.test_all(BayesDensityLogisticGaussianProcess)
end