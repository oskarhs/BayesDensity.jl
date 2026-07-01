# Registry of the (model, inference-algorithm) methods exercised by the benchmark.
#
# Each method exposes:
#   prepare(x, cfg)         -> the "expensive prep": for MCMC the model object (cheap); for VI the
#                              fitted variational posterior (the varinf call — this is what is timed).
#   draw(prepared, rng, cfg)-> a PosteriorSamples object. For MCMC this is one Gibbs/telescope chain
#                              (this is what is timed); for VI it draws Monte-Carlo samples from the
#                              fitted variational posterior (cheap).
#
# `regression = false` marks a method whose stored baseline should NOT be treated as a cross-version
# reference — e.g. right after a bug fix that legitimately changed the output. RandomBernsteinPoly is
# flagged now because its telescope sampler was recently corrected.

using BayesDensity   # umbrella package: re-exports every model + sample + varinf + mean

# varinf returns (vip, info) for most models but a bare vip for RandomFiniteGaussianMixture.
_vip(r) = r isa Tuple ? first(r) : r

struct BenchMethod
    name::String
    kind::Symbol        # :mcmc or :vi
    regression::Bool
    max_samples::Union{Nothing,Int}   # cap on non-burn-in MCMC draws (nothing → use cfg.n_samples)
    prepare::Function
    draw::Function                    # (prepared, rng, cfg, niter) -> PosteriorSamples
end

# Effective number of non-burn-in draws for a method under a config.
nonburnin(m::BenchMethod, cfg) = m.max_samples === nothing ? cfg.n_samples : min(cfg.n_samples, m.max_samples)

function benchmark_methods()
    # MCMC draws `niter` total (burn-in + non-burn-in); VI ignores niter and draws cfg.n_mc_vi samples
    # from the fitted variational posterior.
    mcmcdraw = (m, rng, cfg, niter) -> sample(rng, m, niter; n_burnin = cfg.n_burnin)
    vidraw   = (v, rng, cfg, niter) -> sample(rng, v, cfg.n_mc_vi)
    return BenchMethod[
        # RandomBernsteinPoly and HistSmoother are the slow MCMC methods → capped at 5000 non-burn-in.
        BenchMethod("RandomBernsteinPoly",              :mcmc, true, 5_000,
            (x, cfg) -> RandomBernsteinPoly(x),                        mcmcdraw),
        BenchMethod("BSplineMixture-MCMC",              :mcmc, true, nothing,
            (x, cfg) -> BSplineMixture(x),                             mcmcdraw),
        BenchMethod("HistSmoother-MCMC",                :mcmc, true, 5_000,
            (x, cfg) -> HistSmoother(x),                               mcmcdraw),
        BenchMethod("FiniteGaussianMixture-MCMC",       :mcmc, true, nothing,
            (x, cfg) -> FiniteGaussianMixture(x, cfg.fgm_K),           mcmcdraw),
        BenchMethod("RandomFiniteGaussianMixture-MCMC", :mcmc, true, nothing,
            (x, cfg) -> RandomFiniteGaussianMixture(x),                mcmcdraw),
        BenchMethod("PitmanYorMixture-MCMC",            :mcmc, true, nothing,
            (x, cfg) -> PitmanYorMixture(x),                           mcmcdraw),
        BenchMethod("BSplineMixture-VI",                :vi, true, nothing,
            (x, cfg) -> _vip(varinf(BSplineMixture(x))),               vidraw),
        BenchMethod("HistSmoother-VI",                  :vi, true, nothing,
            (x, cfg) -> _vip(varinf(HistSmoother(x))),                 vidraw),
        BenchMethod("FiniteGaussianMixture-VI",         :vi, true, nothing,
            (x, cfg) -> _vip(varinf(FiniteGaussianMixture(x, cfg.fgm_K))), vidraw),
        BenchMethod("RandomFiniteGaussianMixture-VI",   :vi, true, nothing,
            (x, cfg) -> _vip(varinf(RandomFiniteGaussianMixture(x))),  vidraw),
        BenchMethod("PitmanYorMixture-VI",              :vi, true, nothing,
            (x, cfg) -> _vip(varinf(PitmanYorMixture(x))),             vidraw),
    ]
end
