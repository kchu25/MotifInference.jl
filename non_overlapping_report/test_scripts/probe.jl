using MotifInference
const MI = MotifInference
const V = MotifInference.VeryBasicCNN2

const DATA = "/home/kchu25/Desktop/work/data/seq2label/processed/DMS_ProteinGym_substitutions/AMFR_HUMAN_Tsuboyama_2023_4G3O.jld2"

trc = MI.make_trc(DATA; seq_type=:protein, type=:mut, seed=1, motif_sizes=[2,3],
                  save_root=joinpath(@__DIR__, "..", "results", "_probe"))
data = MI.load_data(trc)

println("model_creator = ", trc.model_creator)
println("X_dim = ", data.X_dim, "   Y_dim = ", data.Y_dim)
X = data.X
println("size(data.X) = ", size(X), "  eltype = ", eltype(X))
println("has consensus (mut path)? ", getfield(data, :raw_data).consensus !== nothing)
for f in propertynames(data)
    try
        v = getproperty(data, f)
        if v isa AbstractArray
            println("  data.$f :: ", typeof(v), " size ", size(v))
        elseif v isa Number || v isa Tuple
            println("  data.$f = ", v)
        end
    catch; end
end
println("--- raw_data fields ---")
rd = getfield(data, :raw_data)
for f in propertynames(rd)
    try
        v = getproperty(rd, f)
        if v isa AbstractArray
            println("  raw.$f :: ", typeof(v), " size ", size(v))
        elseif v isa Number || v isa Symbol || v isa Tuple || v isa AbstractString
            println("  raw.$f = ", v)
        end
    catch; end
end

# Build a model to inspect the architecture (CPU, no training)
using Random
m = trc.model_creator((data.X_dim[1], data.X_dim[2]), data.Y_dim, 64; use_cuda=false, rng=MersenneTwister(1))
if m === nothing
    println("model_creator returned nothing!")
else
    L = m.hp.inference_code_layer
    println("\ninference_code_layer L = ", L)
    println("num_conv_layers = ", m.num_conv_layers)
    println("num_pfms = ", m.hp.num_pfms, "  pfm_len = ", m.hp.pfm_len)
    println("img_fil_heights = ", m.hp.img_fil_heights)
    println("num_img_filters = ", m.hp.num_img_filters)
    p = V.sparse_unpool_window(m.hp, L)
    println("sparse window p at L = ", p)
    code = V.compute_code_at_layer(m, Float32.(X[:, :, :, 1:min(8,size(X,4))]), L; training=false)
    println("code at L shape = ", size(code))
end
