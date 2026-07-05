# [`run_method` reference](@id run_method-reference)

`run_method` is the single entry point for the whole pipeline. Whatever you
give it — a CSV path, a `.jld2` path, or in-memory `strings` + `labels` — it
runs the same five steps: **load → (optionally) tune → train → attribute →
render**.

This page lists **every argument** it accepts, grouped by what it controls, with
defaults and plain-English meanings. Jump to [Recipes](@ref) if you just want
something to copy.

## The four ways to call it

```julia
using MotifInference

# 1. A two-column CSV: `sequence, measurement` (header auto-detected)
run_method("data/mydata.csv")

# 2. A .jld2 file containing a SEQ2EXP_Dataset as `raw_data`
run_method("data/mydata.jld2")

# 3. In memory — no file needed
run_method(strings, labels)                 # strings::Vector{String}, labels::Vector

# 4. A registered dataset entry
run_method(load_datasets("ecoli")[1])
```

All four accept the same keyword arguments below.

!!! note "When does tuning happen?"
    Hyperparameter tuning runs **only when `seed === nothing`** (the default).
    Pass `seed=<n>` to reuse a known-good model and skip tuning entirely. The
    `tune_*` arguments have no effect once a `seed` is given.

---

## 1. Data & model

How your sequences are interpreted and which model is built.

| Argument                | Default    | Meaning |
| ----------------------- | ---------- | ------- |
| `seq_type`              | `:dna`     | `:dna`, `:rna`, or `:protein`. |
| `type`                  | `:conv`    | `:conv` = convolutional motif discovery; `:mut` = mutagenesis mode (per-position attribution, e.g. deep-mutational-scan / protein data). |
| `normalization_method`  | `:zscore`  | How labels are normalized before training: `:zscore`, `:identity`, `:log`, … |
| `multioutput`           | `false`    | Set `true` when `labels` has **multiple columns** (DNA/RNA multi-output model). |
| `conv_bottleneck`       | `false`    | DNA/RNA only: use a bottleneck conv layer (squeezes the inference-code conv layer; needs sequences ≳ 25 nt). |
| `GET_CONSENSUS`         | `false`    | Compute a consensus sequence — typically `true` for amino-acid mutagenesis. |
| `feature_names`         | `nothing`  | Column names for multi-output `labels` (used to label per-output result folders). |

!!! warning "Protein currently requires `type=:mut`"
    The model resolver only builds a protein model for `seq_type=:protein,
    type=:mut`. `seq_type=:protein` with `type=:conv` has no model and will
    fail. For DNA/RNA either `type` works.

---

## 2. Reproducibility & tuning

`seed` decides whether tuning runs at all; the `tune_*` knobs control how long
it takes when it does.

| Argument          | Default   | Meaning |
| ----------------- | --------- | ------- |
| `seed`            | `nothing` | RNG seed. `nothing` → run hyperparameter tuning and pick the best seed. An `Int` → skip tuning, train directly with that seed. |
| `tune_n_trials`   | `25`      | **Number of candidate models tried** during tuning. Lower this (e.g. `3`) for a quick run. |
| `tune_max_epochs` | `25`      | Max training epochs per tuning trial. |
| `tune_patience`   | `5`       | Early-stopping patience (epochs with no improvement) during tuning. |

> Want tuning to finish fast? The single biggest lever is `tune_n_trials` — it
> is the "how many models" count. `tune_n_trials=3` tries 3 models instead of 25.

---

## 3. Motif search

What the attribution step looks for.

| Argument           | Default        | Meaning |
| ------------------ | -------------- | ------- |
| `motif_sizes`      | `[2, 3, 4, 5]` | Multi-motif group sizes to search. Always starts at 2 (pairs); `[2,3]` searches pairs and triples, etc. |
| `activation_thresh`| `0.9`          | Percentile threshold for treating a motif as "active" in a sequence. |

---

## 4. Output selection & rendering

Which outputs get processed and how the HTML report is labeled.

| Argument              | Default   | Meaning |
| --------------------- | --------- | ------- |
| `output_indices`      | `nothing` | Which output columns to process. `nothing` → all outputs (when the model predicts all positions), otherwise the first. Pass e.g. `1:5` or `[2]`. |
| `sensitivity_analysis`| `false`   | Run sensitivity analysis in the conv-case rendering. |
| `dataset_name`        | `nothing` | Name shown in the rendered report. |
| `protein_name`        | `nothing` | Protein name shown in the `:mut`-case rendering. |

---

## 5. Where results are saved

| Argument           | Default              | Meaning |
| ------------------ | -------------------- | ------- |
| `save_root`        | `"."`                | Parent directory in which the results folder is created. |
| `save_folder_name` | `nothing`            | Exact results-folder name. Overrides the CSV/dataset name. |
| `name`             | CSV basename / `"inmemory"` | Fallback folder name when `save_folder_name` is unset (mainly for the in-memory call). |

Final results path: `joinpath(save_root, save_folder_name_or_name)/`.

---

## 6. Advanced (trc-only)

A handful of knobs are **not** exposed as `run_method` keyword arguments. To
change them, build a `training_and_rendering_config` yourself and call
`run_method(trc)`:

| Field                 | Default        | Meaning |
| --------------------- | -------------- | ------- |
| `loss_spec`           | `:mse`         | Loss + aggregation; also `:mae`, `:huber`, `:binary_cross_entropy`. |
| `max_training_epochs` | `40`           | Max epochs for final model training. |
| `max_processor_epochs`| `60`           | Max epochs for processor training. |
| `predict_position`    | `:all`         | `:all` or an integer output index. |
| `patience`            | `10`           | Early-stopping patience for final training. |
| `scale_back`          | `true`         | Undo normalization before Banzhaf-index calculations. |
| `top_and_bot_counts`  | `8`            | Number of top/bottom significant motifs to render. |
| `count_threshold`     | `25`           | Minimum count for a motif to be considered. |
| `Q_threshold`         | `1e-25`        | Q-value threshold for significance filtering. |
| `dpi`                 | `60`           | Resolution of rendered plots. |

```julia
trc = MotifInference.training_and_rendering_config(
    "data/mydata.jld2", model_creator, "results/run01", "My title";
    seq_type=:dna, loss_spec=MotifInference.loss_specs[:huber], dpi=120)
run_method(trc)
```

---

## Recipes

```julia
# Quick protein mutagenesis run — only 3 candidate models instead of 25
run_method("lacI_ec50.csv";
    save_folder_name="ec50", seq_type=:protein, type=:mut,
    GET_CONSENSUS=true, tune_n_trials=3)

# RNA with a fixed seed → no tuning at all (fastest)
run_method("data/utr.csv"; seq_type=:rna, seed=42)

# DNA, search only pairs and triples, write results elsewhere
run_method("data/mydata.csv"; motif_sizes=[2, 3], save_root="/scratch/results")

# Multi-output DNA, process just the first five outputs
run_method("data/multi.jld2"; multioutput=true, output_indices=1:5)

# In-memory, no file on disk
run_method(strings, labels; seq_type=:protein, type=:mut, GET_CONSENSUS=true)
```

## Valid symbol values at a glance

| Keyword                | Accepts |
| ---------------------- | ------- |
| `seq_type`             | `:dna`, `:rna`, `:protein` |
| `type`                 | `:conv`, `:mut` |
| `normalization_method` | `:zscore`, `:identity`, `:log`, … |
| `loss_spec` (trc-only) | `:mse`, `:mae`, `:huber`, `:binary_cross_entropy` |
