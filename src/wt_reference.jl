# ─────────────────────────────────────────────────────────────────────────────
# Determining the wild-type label value for :zscore_wt normalization.
#
# The model has no bias term, so it predicts exactly 0 for the wild-type
# encoding. Under :zscore a prediction of 0 maps back to the LIBRARY MEAN, which
# asserts that the wild type sits at the mean. It does not: on the one assay in
# this corpus that measured its wild type (Andreasson_2020_ribozyme) the wild
# type sits at the 99.6th percentile, 7.78 sd above the mean.
#
# :zscore_wt fixes that, but needs a value. This file supplies it where it can
# be known WITHOUT guessing.
# ─────────────────────────────────────────────────────────────────────────────

"""
    infer_wt_reference(raw_data; feature=1) -> (value, source) or (nothing, :unknown)

Determine the wild-type label value from dataset metadata.

Returns a `(value, source)` tuple. `source` is one of:

  `:definitional` — the units make the wild type's value a definition, not a
                    measurement. ddG is a change in free energy RELATIVE TO the
                    wild type, so the wild type is 0 by construction. Likewise
                    ln-fitness is 0 and relative fitness is 1.
  `:unknown`      — nothing is known. Returns `nothing`; the caller should fall
                    back to `:zscore` and accept that the origin is the label
                    mean rather than the wild type.

Coverage measured on the 69 processed ProteinGym datasets: 50 resolve via
`kcal/mol`, 2 more via ln/relative fitness, 17 fall through to `:unknown`.

# Why there is no consensus-row rule

An obvious rule is "find the variant whose sequence equals the consensus and
read its label". It was tested on all 69 processed datasets. It fires 3 times
and is WRONG ALL 3 TIMES.

The reason is structural. In a combinatorial library the wild-type residue is a
MINORITY at the saturated positions, so the per-column mode returns a non-wild-
type residue and the "wild-type row" it finds is a multi-mutant. In
SPG1_STRSG_Wu_2016 positions 265/266/267/280 are mutated in ~95% of 149,360
rows, and the row it selects sits at the 55th percentile — entirely plausible,
and entirely wrong. The failures land at percentiles 54.8 / 91.3 / 95.4, so
there is no distributional signature to catch them.

Separately, ProteinGym contains 0 wild-type rows in 2,465,767 protein variants;
the rule has nothing correct to find. It is deliberately not implemented.

# Why there is no shape heuristic

Detecting "0 looks like the reference" from the label distribution was tested.
The best two-feature rule, with thresholds tuned on the evaluation set itself,
reached 93.8% true-positive at 13.1% false-positive. A 13% false-positive rate
means pinning `wt_reference = 0.0` on ~20 functional assays where 0 means
nothing. The units string is already present in `feature_names` for all 69
files, so there is no reason to guess.
"""
function infer_wt_reference(raw_data; feature::Int=1)
    fn = raw_data.feature_names
    (isnothing(fn) || isempty(fn) || feature > length(fn)) && return (nothing, :unknown)
    units = lowercase(fn[feature])

    # ddG: change in folding free energy relative to the wild type => 0 exactly.
    occursin("kcal/mol", units)          && return (0.0, :definitional)
    # ln of fitness relative to wild type => ln(1) = 0.
    occursin("ln fitness", units)        && return (0.0, :definitional)
    # fitness expressed as a ratio to wild type => 1.
    occursin("rel. fitness", units)      && return (1.0, :definitional)
    occursin("relative fitness", units)  && return (1.0, :definitional)

    return (nothing, :unknown)
end

"""
    resolve_wt_reference(raw_data, explicit; feature=1, quiet=false)

Precedence wrapper. An explicitly supplied value always wins and suppresses
inference. Otherwise `infer_wt_reference` is consulted. Returns `nothing` when
nothing is known — never a fabricated `0.0`.
"""
function resolve_wt_reference(raw_data, explicit; feature::Int=1, quiet::Bool=false)
    isnothing(explicit) || return explicit
    value, source = infer_wt_reference(raw_data; feature)
    if isnothing(value)
        quiet || @warn """
            No wild-type reference could be determined for this dataset.
            Falling back to mean-centred :zscore, so the model's structural
            f(wild type) = 0 will land on the LIBRARY MEAN, not the wild type.
            Rankings are unaffected; the sign of a contribution is relative to
            the library mean rather than to the wild type.
            Pass wt_reference=... explicitly to fix this."""
    else
        quiet || @info "Wild-type reference inferred" value source units=raw_data.feature_names[feature]
    end
    return value
end
