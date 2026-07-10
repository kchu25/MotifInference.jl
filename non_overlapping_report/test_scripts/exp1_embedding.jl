# Experiment 1 — embedding non-overlap.
#
# Train two models with identical init/schedule (sparsify off vs on) and compare the
# non-overlap of their inference-layer code. A third "op-at-eval" condition applies the
# op post-hoc to the OFF model, isolating the mechanical effect from learned focusing.
#
# Usage:  julia --project non_overlapping_report/test_scripts/exp1_embedding.jl [epochs]
include(joinpath(@__DIR__, "common.jl"))
using Printf

const EPOCHS = length(ARGS) ≥ 1 ? parse(Int, ARGS[1]) : 40
const SEED   = 1
const RESULTS = joinpath(@__DIR__, "..", "results")
const FIGS    = joinpath(@__DIR__, "..", "figures")

Random.seed!(SEED); CUDA.functional() && CUDA.seed!(SEED)
trc, data = load_amfr(joinpath(RESULTS, "_run_off"))
X = data.X; y = data.raw_data.labels
println("dataset: X=$(size(X)), N=$(length(y)) sequences")

# --- train off & on (same init) ---
println("\n=== training OFF ($EPOCHS epochs) ===")
m_off = build_model(trc, data; on=false, seed=SEED)
loss_off = train_model!(m_off, X, y; epochs=EPOCHS)
@printf("  final train MSE (off) = %.4f\n", loss_off[end])

println("\n=== training ON ($EPOCHS epochs) ===")
m_on = build_model(trc, data; on=true, seed=SEED)
loss_on = train_model!(m_on, X, y; epochs=EPOCHS)
@printf("  final train MSE (on)  = %.4f\n", loss_on[end])

# --- inference-layer code for all sequences ---
L = m_on.hp.inference_code_layer
p = V.sparse_unpool_window(m_on.hp, L)
S = size(V.compute_code_at_layer(m_on, gpu(Float32.(X[:,:,:,1:1])), L; training=false), 1)
@printf("\ninference layer L=%d, window p=%d, spatial S=%d, bound ⌈S/p⌉=%d\n", L, p, S, cld(S,p))

code_off = inference_code(m_off, X)                 # baseline
code_on  = inference_code(m_on,  X)                 # sparsify (trained on)
code_eval = V.sparse_max_unpool(code_off, p)        # op-at-eval on the off model

conds = [("Baseline (off)",        code_off),
         ("Op-at-eval (off+op)",   code_eval),
         ("Sparsify (on)",         code_on)]

# save codes so metrics can be recomputed without retraining
using JLD2
@save joinpath(RESULTS, "exp1_codes.jld2") code_off code_eval code_on L p S

mets = [(name, nonoverlap_metrics(c)) for (name, c) in conds]
pos  = [(name, position_separation(c; activation_thresh=trc.activation_thresh)) for (name, c) in conds]
println()
for ((name, mt), (_, ps)) in zip(mets, pos)
    @printf("%-22s | occupancy=%.3f (null %.3f)  exclusivity=%.3f | norm=%.4f raw=%.4f surv=%.2f\n",
            name, ps.mean_occupancy, ps.null_occupancy, ps.exclusivity,
            mt.normalized, mt.raw, mt.mean_survivors)
end

# --- write LaTeX results table (leads with position separation) ---
open(joinpath(RESULTS, "exp1_table.tex"), "w") do io
    println(io, "\\begin{table}[h]\\centering")
    println(io, "\\caption{Do different filters fire at different positions? Inference layer ",
                "\$L^\\star=$L\$, window \$p=$p\$, \$S=$S\$; $(length(y)) sequences, $EPOCHS epochs. ",
                "A filter counts as active where its magnitude exceeds the ",
                "$(trc.activation_thresh) percentile (the same threshold motif finding uses). ",
                "\\emph{Occupancy} = mean active filters per occupied position (1.0 = perfectly separated); ",
                "\\emph{null} = chance level under a per-filter position shuffle; ",
                "\\emph{exclusivity} = fraction of positions used by exactly one filter. ",
                "\$\\NonOv\$ is the scale-free cosine overlap \\eqref{eq:norm}.}\\label{tab:exp1}")
    println(io, "\\begin{tabular}{lrrrrr}\\toprule")
    println(io, "Condition & Occupancy & (null) & Exclusivity & \$\\NonOv\$ & survivors/ch \\\\ \\midrule")
    for ((name, mt), (_, ps)) in zip(mets, pos)
        @printf(io, "%s & %.3f & %.3f & %.3f & %.4f & %.2f \\\\\n",
                name, ps.mean_occupancy, ps.null_occupancy, ps.exclusivity,
                mt.normalized, mt.mean_survivors)
    end
    println(io, "\\bottomrule\\end{tabular}\\end{table}")
end
println("\nwrote results/exp1_table.tex")

# --- figures: training curves + mean Gram heatmaps (best-effort) ---
try
    using CairoMakie
    CairoMakie.activate!()
    # loss curves
    fig1 = Figure(size=(500,320))
    ax1 = Axis(fig1[1,1], xlabel="epoch", ylabel="train MSE", title="Training loss")
    lines!(ax1, 1:EPOCHS, loss_off, label="off"); lines!(ax1, 1:EPOCHS, loss_on, label="on")
    axislegend(ax1); save(joinpath(FIGS, "exp1_loss.pdf"), fig1)

    # mean normalized Gram (cosine) heatmaps, off vs on
    function mean_cos_gram(code)
        S,K,_,N = size(code); acc = zeros(K,K); cnt = 0
        for n in 1:N
            C = Float64.(code[:,:,1,n]); nrm = vec(sqrt.(sum(abs2,C;dims=1)))
            act = findall(>(0.0), nrm); length(act) < 2 && continue
            Ĉ = zeros(S,K); Ĉ[:,act] = C[:,act] ./ nrm[act]'
            acc .+= abs.(Ĉ'Ĉ); cnt += 1
        end
        acc ./ max(cnt,1)
    end
    Goff = mean_cos_gram(code_off); Gon = mean_cos_gram(code_on)
    fig2 = Figure(size=(720,320))
    for (j,(t,Gm)) in enumerate([("off",Goff),("on",Gon)])
        ax = Axis(fig2[1,j], title="mean |cos| Gram — $t", aspect=1)
        heatmap!(ax, Gm, colorrange=(0,1));
    end
    Colorbar(fig2[1,3], colorrange=(0,1))
    save(joinpath(FIGS, "exp1_gram.pdf"), fig2)
    println("wrote figures/exp1_loss.pdf, figures/exp1_gram.pdf")
catch e
    @warn "figure generation skipped" exception=e
end
