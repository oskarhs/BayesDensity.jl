# One-time environment setup for the performance benchmark.
#
#   julia perf/setup.jl
#
# Creates a standalone environment in perf/ that `dev`s the in-repo packages (so the benchmark
# always runs against the live source) and adds StableRNGs + Distributions. Run this once; the
# resulting perf/Manifest.toml is machine-specific and git-ignored.

import Pkg

# Default to the local registry cache — locally this avoids hanging on the private PumasRegistry SSH
# fetch. CI (which has only the public General registry and a cold depot) sets JULIA_PKG_OFFLINE=false
# so packages can actually be downloaded.
haskey(ENV, "JULIA_PKG_OFFLINE") || (ENV["JULIA_PKG_OFFLINE"] = "true")

const PERF = @__DIR__
const ROOT = dirname(PERF)

Pkg.activate(PERF)

locals = [
    "BayesDensityCore",
    "BayesDensityRandomBernsteinPoly",
    "BayesDensityBSplineMixture",
    "BayesDensityHistSmoother",
    "BayesDensityFiniteGaussianMixture",
    "BayesDensityPitmanYorMixture",
    "BayesDensity",
]
Pkg.develop([Pkg.PackageSpec(path = joinpath(ROOT, "lib", l)) for l in locals])
Pkg.add(["StableRNGs", "Distributions", "Statistics", "DelimitedFiles", "Printf", "Random"])
Pkg.instantiate()
Pkg.precompile()

println("\nperf environment ready. Run the benchmark with:")
println("  julia -t auto --project=perf perf/benchmark.jl smoke     # quick end-to-end check")
println("  julia -t auto --project=perf perf/benchmark.jl full      # the real (multi-hour) run")
