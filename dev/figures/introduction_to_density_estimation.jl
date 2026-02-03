using BayesDensityHistSmoother
using CairoMakie
using Distributions
using LaTeXStrings
using Random

# Simulate some data:
const rng = Random.Xoshiro(1)
const d_true = MixtureModel([Normal(-0.2, 0.25), Normal(0.5, 0.15)], [0.4, 0.6])
const n = 1_000
const x = rand(rng, d_true, n)

# Evalutation grid
const ts = LinRange(-1, 1, 1001)

# HistSmoother
const hs = HistSmoother(x)
const posterior_samples = sample(rng, hs, 2100; n_burnin=100)

# Kernel example
function kernel_example()
    bandwidths = [1e-2, 5e-2, 2.5e-1]
    ys = Vector{Vector{Float64}}(undef, length(bandwidths))
    y = similar(ts)
    for i in eachindex(bandwidths)
        for j in eachindex(ts)
            y[j] = sum(pdf(Normal(), (ts[j] .- x) / bandwidths[i]) / bandwidths[i]) / n
        end
        ys[i] = copy(y)
    end

    fig = Figure(size=(550, 380))
    ax = Axis(fig[1,1], xlabel=L"x", ylabel="Density", xlabelsize=20, ylabelsize=20)
    lines!(ax, ts, pdf(d_true, ts), label="True density", linewidth=1.2, linestyle=:dash, color=:black)
    for i in eachindex(bandwidths)
        lines!(ax, ts, ys[i], label=L"h = %$(bandwidths[i])", linewidth = 1.0)
    end
    Legend(fig[1,2], ax, framevisible=false)
    save(joinpath("src", "assets", "introduction_to_density_estimation", "kernel.svg"), fig)
end

#kernel_example()

# MCMC animation
function mcmc_animation()
    fs = pdf(hs, posterior_samples.samples, ts)
    running_means = cumsum(fs, dims = 2) ./ (1:size(fs, 2))'

    # Actual animation code:
    anim_path = joinpath("src", "assets", "introduction_to_density_estimation", "mcmc_animation.gif")
    f = Observable(fs[:,1])
    running_mean = Observable(fs[:,1])
    iteration = Observable(0)

    fig = Figure(size=(550, 380))
    ax = Axis(fig[1,1], title=@lift("Iteration $($iteration)"), xlabel=L"x", ylabel="Density", xlabelsize=20, ylabelsize=20)
    lines!(ax, ts, pdf(d_true, ts), label="True density", linewidth=1.5, linestyle=:dash, color=:black, alpha=0.5)
    lines!(ax, ts, f, label="Current sample")
    lines!(ax, ts, running_mean, label="Running mean")
    ylims!(ax, -0.1, 1.8)
    Legend(fig[1,2], ax, framevisible=false)

    record(fig, anim_path, 1:220; framerate=10) do frame
        iter = clamp(frame - 10, 1, 200)
        f[] = fs[:,iter]
        running_mean[] = running_means[:,iter]
        iteration[] = iter
    end
    
end

# mcmc_animation()

function mcmc_estimate()
    fig = Figure(size=(550, 380))
    ax = Axis(fig[1,1], title="MCMC estimate and true density", xlabel=L"x", ylabel="Density", xlabelsize=20, ylabelsize=20)
    lines!(ax, ts, pdf(d_true, ts), label="True density", linewidth=1.5, linestyle=:dash, color=:black, alpha=0.5)
    plot!(ax, posterior_samples, ts, label="MCMC")
    ylims!(ax, -0.1, 1.8)
    Legend(fig[1,2], ax, framevisible=false)
    save(joinpath("src", "assets", "introduction_to_density_estimation", "mcmc_estimate.svg"), fig)
end

mcmc_estimate()

function varinf_estimate()
    viposterior, _ = varinf(hs; max_iter=1_000)
    posterior_samples = sample(rng, hs, 2_000)
    fig = Figure(size=(550, 380))
    ax = Axis(fig[1,1], title="VI estimate and true density", xlabel=L"x", ylabel="Density", xlabelsize=20, ylabelsize=20)
    lines!(ax, ts, pdf(d_true, ts), label="True density", linewidth=1.5, linestyle=:dash, color=:black, alpha=0.5)
    plot!(ax, posterior_samples, ts, label="VI")
    ylims!(ax, -0.1, 1.8)
    Legend(fig[1,2], ax, framevisible=false)
    save(joinpath("src", "assets", "introduction_to_density_estimation", "varinf_estimate.svg"), fig) 
end

varinf_estimate()