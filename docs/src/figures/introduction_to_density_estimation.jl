using CairoMakie
using Distributions
using LaTeXStrings
using Random

# Simulate some data:
rng = Random.Xoshiro(1)
d_true = MixtureModel([Normal(-0.2, 0.25), Normal(0.5, 0.15)], [0.4, 0.6])
n = 1_000
x = rand(rng, d_true, n)

# Evalutation grid
ts = LinRange(-1, 1, 1001)
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
ax = Axis(fig[1,1], xlabel=L"x", ylabel="Density", xlabelsize=18, ylabelsize=18)
lines!(ax, ts, pdf(d_true, ts), label="True density", linewidth=1.2, linestyle=:dash, color=:black)
for i in eachindex(bandwidths)
    lines!(ax, ts, ys[i], label=L"h = %$(bandwidths[i])", linewidth = 1.0)
end
Legend(fig[1,2], ax, framevisible=false)
save(joinpath("src", "assets", "introduction_to_density_estimation", "kernel.svg"), fig)