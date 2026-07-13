# Cross-version density-stability check.
#
#   julia --project=perf perf/compare.jl <baseline densities.csv> <new densities.csv> [ratio_threshold]
#
# The idea: MCMC output legitimately changes its *trajectory* whenever an optimization alters the
# RNG stream, so a bit-for-bit or single-chain comparison is meaningless. What should be stable is
# the *pooled posterior-mean density* (averaged over the chains), up to Monte-Carlo error. So for
# each (method, density, n) we compare
#
#     Δ     = max_grid | pooled_mean_new − pooled_mean_baseline |
#
# against the Monte-Carlo noise floor of that difference,
#
#     noise = max_grid sqrt(std_new² + std_baseline²) / sqrt(nchains)
#
# (std across chains → std of the pooled mean is std/√nchains; the difference of two independent
# pooled means has std √(·²+·²)). When the estimates are statistically equivalent, Δ sits at a small
# multiple of `noise` (a max over the grid inflates it a little); a genuine change makes Δ ≫ noise.
# We report ratio = Δ / noise and flag ratio > threshold for REVIEW.
#
# VI is deterministic and driven by the same seeds in both runs, so its Δ collapses to ~machine
# precision unless the variational fit itself changed — the same test handles it.
#
# Methods with regression = false in the *new* file (e.g. RandomBernsteinPoly right after its sampler
# bug fix) are reported as EXCLUDED rather than compared.

using DelimitedFiles, Statistics, Printf

const REL_TOL = 1e-3   # also PASS if Δ is below this fraction of the peak baseline density

function _load(path)
    data, _ = readdlm(path, ','; header = true)
    groups = Dict{Tuple{String,String,Int}, Any}()
    for r in axes(data, 1)
        key = (String(data[r, 1]), String(data[r, 4]), Int(data[r, 5]))
        g = get!(groups, key, (kind = String(data[r, 2]), regression = Bool(data[r, 3]),
                               nchains = Int(data[r, 6]), rows = Tuple{Int,Float64,Float64,Float64}[]))
        push!(g.rows, (Int(data[r, 7]), Float64(data[r, 8]), Float64(data[r, 9]), Float64(data[r, 10])))
    end
    # sort each group by grid index k and split into aligned arrays
    out = Dict{Tuple{String,String,Int}, NamedTuple}()
    for (key, g) in groups
        sort!(g.rows; by = first)
        out[key] = (kind = g.kind, regression = g.regression, nchains = g.nchains,
                    x = getindex.(g.rows, 2), pooled = getindex.(g.rows, 3), std = getindex.(g.rows, 4))
    end
    return out
end

function compare(baseline_path, new_path; threshold = 5.0)
    base = _load(baseline_path)
    new  = _load(new_path)

    results = []   # (key, kind, Δ, noise, ratio, rel, status)
    excluded = String[]
    missing_base = String[]
    for (key, gn) in new
        label = @sprintf("%-32s %-20s n=%-5d", key[1], key[2], key[3])
        if !gn.regression
            push!(excluded, label); continue
        end
        if !haskey(base, key)
            push!(missing_base, label); continue
        end
        gb = base[key]
        Δ     = maximum(abs.(gn.pooled .- gb.pooled))
        noise = maximum(sqrt.(gn.std .^ 2 .+ gb.std .^ 2)) / sqrt(gn.nchains)
        peak  = max(maximum(abs.(gb.pooled)), eps())
        ratio = Δ / max(noise, eps())
        status = (ratio ≤ threshold || Δ ≤ REL_TOL * peak) ? "PASS" : "REVIEW"
        push!(results, (label = label, kind = gn.kind, Δ = Δ, noise = noise, ratio = ratio,
                        rel = Δ / peak, status = status))
    end
    sort!(results; by = r -> -r.ratio)

    println("\n=== density-stability comparison (threshold: ratio ≤ $(threshold)) ===")
    println("baseline : ", baseline_path)
    println("new      : ", new_path, "\n")
    @printf("%-58s %-5s %10s %10s %8s %9s  %s\n", "method / density / n", "kind", "Δ", "noise", "ratio", "rel", "status")
    println("-"^115)
    for r in results
        @printf("%-58s %-5s %10.5f %10.5f %8.2f %9.1e  %s\n",
                r.label, r.kind, r.Δ, r.noise, r.ratio, r.rel, r.status)
    end
    nfail = count(r -> r.status == "REVIEW", results)
    println("-"^115)
    @printf("%d compared · %d PASS · %d REVIEW · %d excluded (regression=false) · %d missing in baseline\n",
            length(results), length(results) - nfail, nfail, length(excluded), length(missing_base))
    if !isempty(excluded)
        println("\nexcluded from the check (regression=false):")
        foreach(l -> println("  ", l), sort(excluded))
    end
    if !isempty(missing_base)
        println("\npresent in new run but absent from baseline (regenerate baseline?):")
        foreach(l -> println("  ", l), sort(missing_base))
    end
    return nfail
end

if abspath(PROGRAM_FILE) == @__FILE__
    length(ARGS) ≥ 2 || error("usage: julia --project=perf perf/compare.jl <baseline densities.csv> <new densities.csv> [ratio_threshold]")
    thr = length(ARGS) ≥ 3 ? parse(Float64, ARGS[3]) : 5.0
    nfail = compare(ARGS[1], ARGS[2]; threshold = thr)
    exit(nfail == 0 ? 0 : 1)
end
