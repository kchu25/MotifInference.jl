# Experiment 2 — motif-discovery co-occurrence.
#
# Run the real discovery pipeline (train model → process_output) with sparsify OFF vs ON
# and count discovered co-occurring motif groups. Both share ONE tuned architecture (the
# OFF run's trial JSON is reused for ON) so the only difference is the sparsify op.
# No HTML rendering — we stop at process_output and read its dataframes.
#
# Usage: julia --project non_overlapping_report/test_scripts/exp2_motif_cooccurrence.jl [train_epochs] [proc_epochs]
include(joinpath(@__DIR__, "common.jl"))
using DataFrames, Printf

const TRAIN_EPOCHS = length(ARGS) ≥ 1 ? parse(Int, ARGS[1]) : 30
const PROC_EPOCHS  = length(ARGS) ≥ 2 ? parse(Int, ARGS[2]) : 30
const RESULTS = joinpath(@__DIR__, "..", "results")

"Count distinct co-occurring groups per size, and significant interacting groups."
function cooccurrence_counts(dfs, isum, motif_sizes)
    rows = NamedTuple[]
    for (i, dfk) in enumerate(dfs)
        k = motif_sizes[i]
        msyms = [Symbol("m$j") for j in 1:k]
        ngroups = (nrow(dfk) == 0) ? 0 : length(keys(groupby(dfk, msyms)))
        nsig = (i ≤ length(isum)) ? length(isum[i]) : 0
        push!(rows, (size=k, occurrences=nrow(dfk), groups=ngroups, significant=nsig))
    end
    rows
end

"Full discovery for one condition; returns the per-size co-occurrence counts."
function discover(save_root; on::Bool, share_json_from=nothing, seed=nothing)
    trc = MI.make_trc(DATA; seq_type=:protein, type=:mut, seed=seed, motif_sizes=[2,3],
                      save_root=save_root)
    trc.max_training_epochs  = TRAIN_EPOCHS
    trc.max_processor_epochs = PROC_EPOCHS
    trc.patience = 5
    if on
        base = trc.model_creator
        trc.model_creator = (a...; kw...) -> base(a...; use_sparse_unpool=true, kw...)
    end
    data = MI.load_data(trc)

    if share_json_from === nothing
        # generate an architecture (1 short tuning trial) and record its seed
        trc.seed = nothing
        MI.tune_if_needed!(trc, data; tune_n_trials=1, tune_max_epochs=3, tune_patience=2)
    else
        # reuse the shared architecture: copy the trial JSON, force a fresh train
        dstj = joinpath(trc.save_path, "json"); mkpath(dstj)
        for f in readdir(joinpath(share_json_from, "json"); join=false)
            cp(joinpath(share_json_from, "json", f), joinpath(dstj, f); force=true)
        end
        mdir = joinpath(trc.save_path, "models")
        isdir(mdir) && foreach(f -> rm(joinpath(mdir, f); force=true), readdir(mdir))
    end

    m, train_stats, dl_tr, dl_te, splits =
        MI.obtain_trained_model_and_splited_datasets(data, trc)
    println("  [$(on ? "ON" : "OFF")] trained; inference_code_layer=$(m.hp.inference_code_layer), ",
            "use_sparse_unpool=$(m.hp.use_sparse_unpool)")

    cdf, dfs, pa, pt, isum_str, isum, folder =
        MI.process_output(data, m, train_stats, dl_tr, dl_te, trc, 1)

    counts = cooccurrence_counts(dfs, isum, trc.motif_sizes)
    return (trc=trc, counts=counts, n_singletons=nrow(cdf))
end

println("=== OFF discovery ===")
off = discover(joinpath(RESULTS, "_disc_off"); on=false)
println("=== ON (softmax) discovery (shared architecture) ===")
on  = discover(joinpath(RESULTS, "_disc_on"); on=true, seed=off.trc.seed,
               share_json_from=off.trc.save_path)

conds2 = (("OFF", off), ("ON-softmax", on))
println("\n--- co-occurrence counts ---")
for (tag, r) in conds2
    println("$tag: singletons=$(r.n_singletons)")
    for row in r.counts
        @printf("   size %d: %d occurrences, %d distinct groups, %d significant\n",
                row.size, row.occurrences, row.groups, row.significant)
    end
end

# --- LaTeX table ---
open(joinpath(RESULTS, "exp2_table.tex"), "w") do io
    println(io, "\\begin{table}[h]\\centering")
    println(io, "\\caption{Discovered co-occurring motif groups (shared architecture, ",
                "$TRAIN_EPOCHS train / $PROC_EPOCHS processor epochs). ",
                "``Groups'' = distinct filter-index tuples; ``sig.'' = FDR-significant interacting groups.}\\label{tab:exp2}")
    println(io, "\\begin{tabular}{llrrr}\\toprule")
    println(io, "Condition & Motif size & Occurrences & Distinct groups & Significant \\\\ \\midrule")
    for (tag, r) in (("OFF (baseline)", off), ("ON softmax", on))
        for (j, row) in enumerate(r.counts)
            cond = j == 1 ? tag : ""
            @printf(io, "%s & %d & %d & %d & %d \\\\\n",
                    cond, row.size, row.occurrences, row.groups, row.significant)
        end
        println(io, "\\midrule")
    end
    println(io, "\\bottomrule\\end{tabular}\\end{table}")
end
println("\nwrote results/exp2_table.tex")
