# Performance + density-stability benchmark for BayesDensity.jl.
#
#   julia -t auto --project=perf perf/benchmark.jl smoke            # quick end-to-end check
#   julia -t auto --project=perf perf/benchmark.jl full [outdir]    # the real (multi-hour) run
#
# For every (Marron–Wand density d = 1..10) × (sample size n ∈ {250,1000,5000}) × (method), it runs
# `n_chains` independent chains (MCMC) — or one variational fit plus `n_chains` Monte-Carlo sample
# sets (VI) — and records, on a fixed per-density grid:
#   * the pooled posterior-mean density  (average of the chains) — the cross-version stability target,
#   * the per-grid-point std across chains — used to derive the Monte-Carlo noise floor,
#   * the wall-clock time and allocations of one representative fit.
#
# Synthetic data and all chains are driven by StableRNGs, so the whole benchmark is reproducible
# across machines and Julia versions.
#
# Outputs (CSV, appended incrementally so a long run keeps partial results):
#   <outdir>/densities.csv   method,kind,regression,density,n,nchains,k,x,pooled_mean,chain_std
#   <outdir>/timings.csv     method,kind,regression,density,n,min_time_s,alloc_MiB,n_samples,n_burnin,n_chains

include(joinpath(@__DIR__, "src", "marron_wand.jl"))
include(joinpath(@__DIR__, "src", "methods.jl"))

using StableRNGs, Statistics, Printf, Dates

Base.@kwdef struct BenchConfig
    n_samples::Int           = 10_000        # non-burn-in MCMC draws (methods may cap below this)
    n_burnin::Int            = 1_000
    n_chains::Int            = 10
    sample_sizes::Vector{Int}= [250, 1000, 5000]
    n_grid::Int              = 200
    n_mc_vi::Int             = 10_000          # Monte-Carlo samples for a VI density estimate
    fgm_K::Int               = 30             # fixed number of components for FiniteGaussianMixture
    density_indices::Vector{Int} = collect(1:10)
    method_filter::Union{Nothing,String} = nothing   # substring filter on method names
    threaded::Bool           = Threads.nthreads() > 1
    data_seed_base::Int      = 20_240_000
    chain_seed_base::Int     = 70_000_000
end

# Seeds are pure functions of (density index, n, chain), so they are identical across runs/versions.
data_seed(cfg, di, n)        = cfg.data_seed_base  + 1_000_003*di + n
chain_seed(cfg, di, n, c)    = cfg.chain_seed_base + 1_000_003*di + 1009*n + c

smoke_config() = BenchConfig(n_samples = 250, n_burnin = 50, n_chains = 3,
                             sample_sizes = [200], n_grid = 100, n_mc_vi = 500,
                             fgm_K = 12, density_indices = [1, 10])

# The default CI configuration: 5 chains, n ∈ {400, 2000}; everything else at full settings.
reduced_config() = BenchConfig(n_chains = 5, sample_sizes = [400, 2000])

function _fill_chains!(dens, crange, f, threaded)
    if threaded
        Threads.@threads for c in collect(crange)
            @inbounds dens[:, c] .= f(c)
        end
    else
        for c in crange
            @inbounds dens[:, c] .= f(c)
        end
    end
end

# Run one (method, density, n) cell: returns (dens::Matrix grid×nchains, rep_time_s, rep_bytes).
function run_config(method::BenchMethod, x, grid, cfg, di, n)
    ng = length(grid)
    dens = Matrix{Float64}(undef, ng, cfg.n_chains)
    niter = cfg.n_burnin + nonburnin(method, cfg)   # total MCMC iterations for this method
    if method.kind == :mcmc
        model = method.prepare(x, cfg)
        # First chain serially + timed (a single full-length run is a stable measurement).
        r1 = @timed method.draw(model, StableRNG(chain_seed(cfg, di, n, 1)), cfg, niter)
        dens[:, 1] .= mean(r1.value, grid)
        _fill_chains!(dens, 2:cfg.n_chains,
                      c -> mean(method.draw(model, StableRNG(chain_seed(cfg, di, n, c)), cfg, niter), grid),
                      cfg.threaded)
        return dens, r1.time, r1.bytes
    else # :vi — the variational fit is the timed cost; sampling from it is cheap.
        r = @timed method.prepare(x, cfg)
        vip = r.value
        _fill_chains!(dens, 1:cfg.n_chains,
                      c -> mean(method.draw(vip, StableRNG(chain_seed(cfg, di, n, c)), cfg, niter), grid),
                      cfg.threaded)
        return dens, r.time, r.bytes
    end
end

_csv(io, xs...) = println(io, join(xs, ","))

# Compile every method once on a tiny problem so the recorded timings exclude JIT compilation.
function _warmup(methods, cfg)
    wcfg = BenchConfig(n_samples = 20, n_burnin = 5, n_mc_vi = 50, fgm_K = cfg.fgm_K)
    mw   = marron_wand_densities()[1]
    x    = rand(StableRNG(1), mw.dist, 60)
    grid = density_grid(mw.dist, 20)
    for m in methods
        try
            obj = m.prepare(x, wcfg)
            mean(m.draw(obj, StableRNG(1), wcfg, wcfg.n_burnin + nonburnin(m, wcfg)), grid)
        catch e
            @warn "warmup failed" method = m.name exception = e
        end
    end
    return nothing
end

function run_benchmark(cfg::BenchConfig; outdir)
    mkpath(outdir)
    dfile = joinpath(outdir, "densities.csv")
    tfile = joinpath(outdir, "timings.csv")
    open(dfile, "w") do io
        _csv(io, "method", "kind", "regression", "density", "n", "nchains", "k", "x", "pooled_mean", "chain_std")
    end
    open(tfile, "w") do io
        _csv(io, "method", "kind", "regression", "density", "n", "min_time_s", "alloc_MiB", "n_samples", "n_burnin", "n_chains")
    end

    mws     = marron_wand_densities()
    methods = benchmark_methods()
    cfg.method_filter !== nothing && (methods = filter(m -> occursin(cfg.method_filter, m.name), methods))

    @info "warming up (compiling methods)…"
    _warmup(methods, cfg)

    ncell = length(cfg.density_indices) * length(cfg.sample_sizes) * length(methods)
    cell  = 0
    t0    = time()
    for di in cfg.density_indices
        mw   = mws[di]
        grid = density_grid(mw.dist, cfg.n_grid)
        for n in cfg.sample_sizes
            x = rand(StableRNG(data_seed(cfg, di, n)), mw.dist, n)
            for method in methods
                cell += 1
                @printf("[%3d/%3d  %5.0fs] %-32s %-20s n=%-5d\n",
                        cell, ncell, time() - t0, method.name, mw.name, n)
                flush(stdout)
                dens, rt, bytes = run_config(method, x, grid, cfg, di, n)
                pooled = vec(mean(dens; dims = 2))
                cstd   = vec(std(dens; dims = 2))
                open(dfile, "a") do io
                    for k in eachindex(grid)
                        _csv(io, method.name, method.kind, method.regression, mw.name, n,
                             cfg.n_chains, k, grid[k], pooled[k], cstd[k])
                    end
                end
                open(tfile, "a") do io
                    _csv(io, method.name, method.kind, method.regression, mw.name, n,
                         rt, bytes / 2^20, nonburnin(method, cfg), cfg.n_burnin, cfg.n_chains)
                end
            end
        end
    end
    @printf("done in %.0fs → %s\n", time() - t0, outdir)
    return outdir
end

# ---- CLI ----
if abspath(PROGRAM_FILE) == @__FILE__
    mode   = get(ARGS, 1, "reduced")
    cfg    = mode == "smoke"   ? smoke_config()   :
             mode == "reduced" ? reduced_config() :
             mode == "full"    ? BenchConfig()    :
             error("unknown mode '$mode' (use: smoke | reduced | full)")
    outdir = get(ARGS, 2, joinpath(@__DIR__, "results", mode))
    @info "starting benchmark" mode nthreads=Threads.nthreads() n_samples=cfg.n_samples n_burnin=cfg.n_burnin n_chains=cfg.n_chains sample_sizes=cfg.sample_sizes
    run_benchmark(cfg; outdir = outdir)
end
