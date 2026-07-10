# Regenerate exp3 figure from saved rows (softmax sweep only; no re-training).
using JLD2, CairoMakie
CairoMakie.activate!()
const RESULTS = joinpath(@__DIR__, "..", "results")
const FIGS    = joinpath(@__DIR__, "..", "figures")
const ALPHAS  = [1, 5, 20, 100]

@load joinpath(RESULTS, "exp3_rows.jld2") rows
off = rows[1]
sm  = rows[2:1+length(ALPHAS)]            # softmax rows only (drop any trailing gumbel row)
as  = Float64.(ALPHAS)
occ = [r.ps.mean_occupancy for r in sm]
rho = [r.gen.spearman for r in sm]

fig = Figure(size=(760, 320))
ax1 = Axis(fig[1,1], xlabel="softmax strength α", ylabel="occupancy (↓ = separated)",
           title="Separation vs α (strong activations)", xscale=log10)
lines!(ax1, as, occ); scatter!(ax1, as, occ, label="softmax")
hlines!(ax1, [off.ps.mean_occupancy]; color=:gray, linestyle=:dash, label="off")
hlines!(ax1, [1.0]; color=:green, linestyle=:dashdot, label="ideal (1/pos)")
hlines!(ax1, [sm[1].ps.null_occupancy]; color=:red, linestyle=:dot, label="chance")
axislegend(ax1)
ax2 = Axis(fig[1,2], xlabel="softmax strength α", ylabel="test Spearman ρ",
           title="Generalization vs α", xscale=log10)
lines!(ax2, as, rho); scatter!(ax2, as, rho, label="softmax")
hlines!(ax2, [off.gen.spearman]; color=:gray, linestyle=:dash, label="off")
axislegend(ax2)
save(joinpath(FIGS, "exp3_alpha.pdf"), fig)
println("regenerated figures/exp3_alpha.pdf (softmax-only)")
