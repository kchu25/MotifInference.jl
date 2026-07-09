# Experiment 3 — softmax strength (alpha) vs. separation vs. generalization.
#
# The op's softmax over channels has a strength alpha. alpha=1 is plain softmax; larger
# alpha sharpens toward a per-position winner-take-all across filters, which should push the
# op from "within-filter" toward "between-filter" separation. We sweep alpha and, for each,
# measure (a) how separated the filters are (occupancy) and (b) whether it costs held-out
# accuracy (test Spearman/Pearson on a DMS regression). Same init/schedule throughout.
#
# Usage: julia --project non_overlapping_report/test_scripts/exp3_alpha_sweep.jl [epochs]
include(joinpath(@__DIR__, "common.jl"))
using Printf, JLD2

const EPOCHS  = length(ARGS) ≥ 1 ? parse(Int, ARGS[1]) : 40
const SEED    = 1
const ALPHAS  = [1, 5, 20, 100]
const RESULTS = joinpath(@__DIR__, "..", "results")
const FIGS    = joinpath(@__DIR__, "..", "figures")

Random.seed!(SEED); CUDA.functional() && CUDA.seed!(SEED)
trc, data = load_amfr(joinpath(RESULTS, "_run_sweep"))
X = data.X; y = data.raw_data.labels
sp = train_test_split(length(y); frac=0.8, seed=0)
Xtr, ytr = X[:, :, :, sp.train], y[sp.train]
Xte, yte = X[:, :, :, sp.test],  y[sp.test]
println("N=$(length(y))  train=$(length(sp.train))  test=$(length(sp.test))")

"Train one condition on the train split; measure separation + generalization on test."
function run_condition(name; on::Bool, alpha=1)
    m = build_model(trc, data; on=on, seed=SEED, alpha=alpha)
    train_model!(m, Xtr, ytr; epochs=EPOCHS)
    code = inference_code(m, Xte)
    ps   = position_separation(code)
    gen  = generalization(m, Xte, yte)
    @printf("%-16s occ=%.3f (null %.3f) excl=%.3f | test ρ=%.3f r=%.3f\n",
            name, ps.mean_occupancy, ps.null_occupancy, ps.exclusivity, gen.spearman, gen.pearson)
    return (name=name, alpha=alpha, ps=ps, gen=gen, K=size(code,2), S=size(code,1))
end

rows = Any[run_condition("off (baseline)"; on=false)]
L = nothing; p = nothing
for a in ALPHAS
    r = run_condition("on, α=$a"; on=true, alpha=a)
    push!(rows, r)
end
mref = build_model(trc, data; on=true, seed=SEED, alpha=1)
L = mref.hp.inference_code_layer; p = V.sparse_unpool_window(mref.hp, L)
@printf("\ninference layer L=%d, window p=%d, channels K=%d (bottleneck=%d)\n",
        L, p, rows[1].K, V.BOTTLENECK_FILTERS)

@save joinpath(RESULTS, "exp3_rows.jld2") rows L p

# --- table ---
open(joinpath(RESULTS, "exp3_table.tex"), "w") do io
    println(io, "\\begin{table}[h]\\centering")
    println(io, "\\caption{Softmax strength \$\\alpha\$ vs.\\ separation vs.\\ generalization ",
                "(\$L^\\star=$L\$, \$p=$p\$, \$K=$(rows[1].K)\$ filters, bottleneck $(V.BOTTLENECK_FILTERS); ",
                "$EPOCHS epochs, 80/20 split). Lower occupancy = filters more separated; ",
                "test \$\\rho\$ (Spearman) is held-out DMS accuracy.}\\label{tab:exp3}")
    println(io, "\\begin{tabular}{lrrrrr}\\toprule")
    println(io, "Condition & Occupancy & (null) & Exclusivity & test \$\\rho\$ & test \$r\$ \\\\ \\midrule")
    for r in rows
        @printf(io, "%s & %.3f & %.3f & %.3f & %.3f & %.3f \\\\\n",
                r.name, r.ps.mean_occupancy, r.ps.null_occupancy, r.ps.exclusivity,
                r.gen.spearman, r.gen.pearson)
    end
    println(io, "\\bottomrule\\end{tabular}\\end{table}")
end
println("wrote results/exp3_table.tex")

@save joinpath(RESULTS, "exp3_rows.jld2") rows L p

# --- figure: occupancy and test-ρ vs alpha ---
try
    using CairoMakie
    CairoMakie.activate!()
    on_rows = rows[2:end]
    as  = Float64.(ALPHAS)
    occ = [r.ps.mean_occupancy for r in on_rows]
    rho = [r.gen.spearman for r in on_rows]
    fig = Figure(size=(760, 320))
    ax1 = Axis(fig[1,1], xlabel="softmax strength α", ylabel="occupancy (↓ = separated)",
               title="Separation vs α", xscale=log10)
    lines!(ax1, as, occ); scatter!(ax1, as, occ)
    hlines!(ax1, [rows[1].ps.mean_occupancy]; color=:gray, linestyle=:dash, label="off")
    hlines!(ax1, [on_rows[1].ps.null_occupancy]; color=:red, linestyle=:dot, label="chance")
    axislegend(ax1)
    ax2 = Axis(fig[1,2], xlabel="softmax strength α", ylabel="test Spearman ρ",
               title="Generalization vs α", xscale=log10)
    lines!(ax2, as, rho); scatter!(ax2, as, rho)
    hlines!(ax2, [rows[1].gen.spearman]; color=:gray, linestyle=:dash, label="off")
    axislegend(ax2)
    save(joinpath(FIGS, "exp3_alpha.pdf"), fig)
    println("wrote figures/exp3_alpha.pdf")
catch e
    @warn "figure skipped" exception=e
end
