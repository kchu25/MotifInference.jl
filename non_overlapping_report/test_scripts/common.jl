# Shared setup for the non-overlapping experiments:
#   load the AMFR protein DMS dataset, build on/off models with identical init,
#   train them with an identical short schedule, and read out the inference-layer code.
#
# Training runs on GPU: the model's channel-dropout path uses CUDA.rand, so a CPU
# training pass with the default dropout would error. Inference/metrics run on CPU.
using MotifInference, Flux, CUDA, Random, Statistics
const MI = MotifInference
const V  = MotifInference.VeryBasicCNN2

include(joinpath(@__DIR__, "metrics.jl"))

const DATA = "/home/kchu25/Desktop/work/data/seq2label/processed/DMS_ProteinGym_substitutions/AMFR_HUMAN_Tsuboyama_2023_4G3O.jld2"

"Load the dataset and a config; `save_root` isolates any pipeline caches."
function load_amfr(save_root)
    trc = MI.make_trc(DATA; seq_type=:protein, type=:mut, seed=1, motif_sizes=[2,3],
                      save_root=save_root)
    data = MI.load_data(trc)
    return trc, data
end

"Build a model via the dataset's own creator; `on` toggles the sparsify op, `alpha`
sets the softmax strength. Same `seed` ⇒ identical architecture and initial weights."
function build_model(trc, data; on::Bool, seed::Int, alpha=1, batch::Int=256)
    trc.model_creator((data.X_dim[1], data.X_dim[2]), data.Y_dim, batch;
        use_cuda=true, rng=MersenneTwister(seed),
        use_sparse_unpool=on, sparse_unpool_alpha=alpha)
end

"Deterministic train/test split of `n` items."
function train_test_split(n; frac=0.8, seed=0)
    idx = randperm(MersenneTwister(seed), n)
    ntr = round(Int, frac * n)
    return (train = idx[1:ntr], test = idx[ntr+1:end])
end

"z-scored regression targets from the DMS labels."
zscore(y) = (y .- mean(y)) ./ (std(y) + eps(eltype(y)))

# Spearman = Pearson on ranks (ties ignored; fine for continuous DMS scores)
_rank(v) = (p = sortperm(v); r = similar(v, Float64); r[p] = 1:length(v); r)
spearman(a, b) = cor(_rank(a), _rank(b))

"Held-out generalization: correlate model predictions with true labels on `X,y`.
Correlations are affine-invariant, so the z-score training target is irrelevant here."
function generalization(model, X, y)
    ŷ  = vec(Array(V.predict_from_sequences(model, gpu(Float32.(X)); training=false)))
    yv = Float64.(y)
    return (pearson = cor(Float64.(ŷ), yv), spearman = spearman(Float64.(ŷ), yv))
end

"Train `model` to regress `y` from `X` (MSE). Returns per-epoch mean loss."
function train_model!(model, X, y; epochs::Int, batch::Int=256, lr=1f-3)
    Xg = gpu(Float32.(X)); yg = gpu(Float32.(zscore(y)))
    opt = Flux.setup(Adam(lr), model)
    V.train!(model)
    losses = Float32[]
    for e in 1:epochs
        tot = 0f0; nb = 0
        for (xb, yb) in Flux.DataLoader((Xg, yg); batchsize=batch, shuffle=true)
            l, gs = Flux.withgradient(model) do m
                Flux.mse(vec(V.predict_from_sequences(m, xb; training=true)), yb)
            end
            Flux.update!(opt, model, gs[1]); tot += l; nb += 1
        end
        push!(losses, tot / nb)
    end
    V.eval!(model)
    return losses
end

"Inference-layer code for all sequences, on CPU as (S, K, 1, N)."
function inference_code(model, X)
    L = model.hp.inference_code_layer
    Array(V.compute_code_at_layer(model, gpu(Float32.(X)), L; training=false))
end
