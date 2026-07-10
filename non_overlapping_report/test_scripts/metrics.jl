# Non-overlap metrics on an inference-layer code tensor.
# code layout: (S, K, 1, N) == (spatial, channels/filters, 1, batch)
#
# Implements the two scores from the report:
#   raw        = mean over sequences of ||CᵀC - diag(CᵀC)||²_F         (magnitude-sensitive)
#   normalized = mean over sequences of mean_{i≠j} cos²(c_i, c_j)      (scale-free, in [0,1])
using LinearAlgebra, Statistics, Random

"""
    nonoverlap_metrics(code) -> NamedTuple

`code`: 4D `(S, K, 1, N)`. Returns raw & normalized non-overlap scores, the zero
fraction, and mean survivors per active channel (should be ≤ ⌈S/p⌉ when the op is on).
"""
function nonoverlap_metrics(code::AbstractArray{<:Real,4})
    S, K, _, N = size(code)
    raw_acc   = 0.0
    norm_acc  = 0.0
    norm_seqs = 0
    survivors = Int[]

    for n in 1:N
        C = Float64.(@view code[:, :, 1, n])        # S × K
        G = C' * C                                   # K × K
        raw_acc += sum(abs2, G) - sum(abs2, @view G[diagind(G)])

        colnorm = vec(sqrt.(sum(abs2, C; dims = 1)))
        active  = findall(>(0.0), colnorm)
        if length(active) ≥ 2
            Ĉ  = C[:, active] ./ colnorm[active]'    # unit-norm active columns
            Ĝ  = Ĉ' * Ĉ                               # cosine similarities (diag = 1)
            Kp = length(active)
            offdiag = sum(abs2, Ĝ) - Kp              # subtract the Kp ones on the diagonal
            norm_acc += offdiag / (Kp * (Kp - 1))
            norm_seqs += 1
        end

        for k in 1:K
            nz = count(!iszero, @view C[:, k])
            nz > 0 && push!(survivors, nz)
        end
    end

    return (
        raw            = raw_acc / N,
        normalized     = norm_seqs > 0 ? norm_acc / norm_seqs : NaN,
        zero_frac      = count(iszero, code) / length(code),
        mean_survivors = isempty(survivors) ? NaN : sum(survivors) / length(survivors),
        max_survivors  = isempty(survivors) ? 0   : maximum(survivors),
    )
end

"""
    position_separation(code; nshuffle=20, seed=0) -> NamedTuple

Directly measures whether *different filters fire at different positions*.

A filter counts as "active" at a position when its activation magnitude exceeds a global
threshold: the `activation_thresh` percentile of the positive activations across the whole
batch. This matches how motif finding decides what is active (BanzhafInference's
`filter_via_magnitude`: `mag > quantile(mags, activation_thresh)`). With `activation_thresh=0`
it falls back to the raw support (any nonzero activation).

For each sequence, let `d_s` = number of filters active at position `s`.
- `mean_occupancy` = mean of `d_s` over occupied positions. 1.0 ⇒ every occupied
  position belongs to exactly one filter (filters perfectly separated); larger ⇒ pile-up.
- `exclusivity` = fraction of occupied positions used by exactly one filter (→1 is best).
- `null_occupancy` = same occupancy under a per-filter position shuffle (chance level).
  `mean_occupancy < null_occupancy` ⇒ filters separated MORE than chance;
  `> null` ⇒ they co-locate MORE than chance.
"""
function position_separation(code::AbstractArray{<:Real,4};
                             activation_thresh::Real=0, nshuffle::Int=20, seed::Int=0)
    S, K, _, N = size(code)
    rng = MersenneTwister(seed)
    # global magnitude threshold = activation_thresh percentile of positive activations
    thr = if activation_thresh > 0
        pos = filter(>(0), vec(code))
        isempty(pos) ? 0.0 : quantile(pos, Float64(activation_thresh))
    else
        0.0
    end
    occ_obs = Float64[]; excl = Float64[]; occ_null = Float64[]
    for n in 1:N
        # active = magnitude at/above threshold (>= so a degenerate code whose values all
        # tie at the max is still counted; activation_thresh==0 falls back to raw >0 support)
        A = activation_thresh > 0 ? ((@view code[:, :, 1, n]) .>= thr) :
                                    ((@view code[:, :, 1, n]) .> 0)
        d = vec(sum(A; dims = 2))                    # filters active per position
        occ = d .> 0
        nocc = count(occ)
        nocc == 0 && continue
        push!(occ_obs, sum(@view d[occ]) / nocc)
        push!(excl, count(==(1), @view d[occ]) / nocc)
        colcount = vec(sum(A; dims = 1))             # #active positions per filter
        for _ in 1:nshuffle
            dn = zeros(Int, S)
            for k in 1:K
                nk = colcount[k]; nk == 0 && continue
                @views dn[randperm(rng, S)[1:nk]] .+= 1
            end
            occn = count(>(0), dn)
            occn > 0 && push!(occ_null, sum(dn) / occn)
        end
    end
    return (
        mean_occupancy = mean(occ_obs),
        exclusivity    = mean(excl),
        null_occupancy = isempty(occ_null) ? NaN : mean(occ_null),
    )
end
