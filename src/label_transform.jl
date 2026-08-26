# ─────────────────────────────────────────────────────────────────────────────
# Choosing the label normalization automatically.
#
# Two separate corrections have to be decided per dataset, and both were being
# decided by a fixed default that is wrong for most of the corpus.
#
#   1. Should the labels be log transformed?
#   2. Where is the origin -- the label mean, or the wild-type value?
#
# `resolve_normalization` answers both from the dataset and returns a concrete
# method that AutoComputationalGraphTuning and RealLabelNormalization already
# understand. No new normalization method is introduced.
#
# Why a data-driven rule is acceptable here, when `infer_wt_reference`
# deliberately refuses one: that function has to assert WHERE THE ORIGIN IS, and
# a false positive silently reinterprets every number in the dataset. This one
# only decides the SCALE of the fit, it reads two directly measured properties
# of the labels, and the choice is printed on the rendered page. A reader can
# see it and override it.
#
# Measured selection over the two corpora (33 RNA, 69 protein):
#
#            :log   :zscore_wt   :zscore
#   RNA        14            0        19
#   protein     1           52        16
#
# See benchmark_design/right_skewed.pdf.
# ─────────────────────────────────────────────────────────────────────────────

"""
    label_skew(y) -> Float64

Sample skew of `y`, ignoring `NaN`. Returns `0.0` when fewer than 3 usable
values remain or the spread is zero, so a degenerate column never selects a
transform.
"""
function label_skew(y)
    v = Float64[]
    for x in y
        xf = float(x)
        isfinite(xf) && push!(v, xf)
    end
    length(v) < 3 && return 0.0
    m = sum(v) / length(v)
    s2 = sum(abs2, v .- m) / length(v)
    (isfinite(s2) && s2 > 0) || return 0.0
    s = sqrt(s2)
    return sum(((v .- m) ./ s) .^ 3) / length(v)
end

"""
    already_log_scale(feature_names; feature=1) -> Bool

Whether the assay's units are themselves a logarithm, so a further log would be
a second one.

`kcal/mol` is a free energy, and ΔΔG = -RT ln K. `ln fitness` says so in the
name. Both are already log scale, and both measure as symmetric: the median skew
over the 50 `kcal/mol` datasets is -0.01.
"""
function already_log_scale(feature_names; feature::Int=1)
    (isnothing(feature_names) || isempty(feature_names) ||
        feature > length(feature_names)) && return false
    u = lowercase(String(feature_names[feature]))
    return occursin("kcal/mol", u) || occursin("ln fitness", u)
end

"""
    detect_label_transform(y, feature_names; feature=1, skew_threshold=1.0)
        -> (take_log::Bool, reason::String)

Decide whether `y` should be log transformed, and say why in one sentence.

`take_log` is true only when BOTH hold:

  * every label is strictly positive, so no shift is needed, and
  * the skew exceeds `skew_threshold`.

The positivity test is what keeps this safe. A shifted log,
`log(y - min(y) + eps)`, was measured and can invent structure: on
GCN4_YEAST_Staller_2018 it drove the skew from 1.02 to -11.09 and the bimodality
coefficient from 0.103 to 0.891, because the labels sitting at exactly 0 land on
`log(eps)` and separate into their own mode. Requiring `min(y) > 0` makes that
branch unreachable -- RealLabelNormalization computes the offset as
`min <= 0 ? abs(min) + shift : 0`, so the transform here is always a pure log.
"""
function detect_label_transform(y, feature_names; feature::Int=1,
                                skew_threshold::Real=1.0)
    if already_log_scale(feature_names; feature=feature)
        return (false, "the units are already a log scale, so no further log was taken")
    end
    v = Float64[]
    for x in y
        xf = float(x)
        isfinite(xf) && push!(v, xf)
    end
    isempty(v) && return (false, "no usable labels")
    mn = minimum(v)
    g1 = label_skew(v)
    if mn <= 0
        return (false, string("no log was taken because some labels are not positive (min ",
                              round(mn, sigdigits=3), "), and a shifted log can create a mode ",
                              "the experiment did not measure"))
    end
    if g1 <= skew_threshold
        return (false, string("no log was taken because the labels are not strongly skewed (skew ",
                              round(g1, digits=2), ")"))
    end
    return (true, string("every label is positive (min ", round(mn, sigdigits=3),
                         ") and the labels are right skewed (skew ", round(g1, digits=2), ")"))
end

"""
    resolve_normalization(raw_data; requested=:auto, feature=1, skew_threshold=1.0)
        -> (method::Symbol, wt_reference, note::String)

Pick the normalization for one dataset.

`requested` is passed straight back when it is anything other than `:auto`, so
an explicit choice is never overridden. Only `:auto` consults the data.

The `:auto` decision, in order:

 1. units already log scale  -> `:zscore_wt` when the wild-type value is known,
    else `:zscore`. Never a log.
 2. positive and skewed      -> `:log` (offset 0, a pure log).
 3. wild-type value known    -> `:zscore_wt`.
 4. otherwise                -> `:zscore`, the historical default.

`note` is a human-readable sentence for the rendered page. It is empty when
`requested` was explicit, because then nothing was decided here.
"""
function resolve_normalization(raw_data; requested::Symbol=:auto, feature::Int=1,
                               skew_threshold::Real=1.0)
    if requested !== :auto
        wt, _ = infer_wt_reference(raw_data; feature=feature)
        return (requested, requested === :zscore_wt ? wt : nothing, "")
    end

    y = _labels_for_feature(raw_data, feature)
    fn = raw_data.feature_names
    wt, src = infer_wt_reference(raw_data; feature=feature)
    take_log, why = detect_label_transform(y, fn; feature=feature,
                                           skew_threshold=skew_threshold)

    if take_log
        return (:log, nothing,
                string("log, chosen automatically: ", why,
                       ". Predictions and motif values are on the log scale."))
    end
    if wt !== nothing
        return (:zscore_wt, wt,
                string("z-score centred on the wild type (", wt,
                       "), chosen automatically: ", why, "."))
    end
    return (:zscore, nothing,
            string("z-score centred on the label mean, chosen automatically: ", why,
                   ". The wild-type value is not recoverable from this dataset, so the ",
                   "origin is the library mean and not the wild type."))
end

"""
    _labels_for_feature(raw_data, feature) -> AbstractVector

The label column used for the decision. A matrix of labels is indexed by
`feature`; a vector is returned as-is.
"""
function _labels_for_feature(raw_data, feature::Int)
    y = raw_data.labels
    if y isa AbstractMatrix
        ncols = size(y, 2)
        return feature <= ncols ? view(y, :, feature) : view(y, :, 1)
    end
    return y
end

"""
    resolve_normalization!(trc, data; feature=1, skew_threshold=1.0) -> trc

Replace `trc.normalization_method == :auto` with a concrete method, chosen from
`data`. Sets `trc.wt_reference` and `trc.normalization_note` at the same time.

Idempotent and safe to call more than once: a config whose method is already
concrete is returned untouched, so an explicit choice is never overridden.

Call this before anything reads `trc.normalization_method`. `:auto` must never
reach RealLabelNormalization -- `compute_normalization_stats` does NOT validate
the method symbol, and an unrecognised one falls through to its `:log` branch,
so an unresolved `:auto` would silently log-transform the labels.
"""
function resolve_normalization!(trc, data; feature::Int=1, skew_threshold::Real=1.0)
    trc.normalization_method === :auto || return trc
    method, wt, note = resolve_normalization(data.raw_data;
        requested=:auto, feature=feature, skew_threshold=skew_threshold)
    trc.normalization_method = method
    trc.wt_reference = wt
    trc.normalization_note = note
    @info "label normalization chosen automatically" method wt_reference=wt note
    return trc
end
