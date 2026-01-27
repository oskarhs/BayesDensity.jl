


function _check_initialparams_mcmc(initial_params::NamedTuple{N, T}) where {N, T}
    (:μ in N && :σ2 in N && :w in N) || throw(ArgumentError("Expected a NamedTuple with fields μ, σ2 and w"))
    (; μ, σ2, w) = initial_params
    (length(μ) == length(σ2) == length(w)) || throw(ArgumentError("Initial μ, σ2 and w dimensions are incompatible."))
end