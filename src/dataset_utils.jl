
# ─────────────────────────────────────────────────────────────────────────────
# Dataset utilities — constructors and loaders
# ─────────────────────────────────────────────────────────────────────────────

"""
    resolve_model_creator(; seq_type, type, multioutput)

Automatically select the correct model constructor based on sequence type and output mode.

# Logic
- Protein + mutation → bottleneck model
- Protein otherwise  → standard aminoacid model
- DNA/RNA + multioutput → multioutput nucleotide model
- DNA/RNA otherwise    → standard nucleotide model
"""
function resolve_model_creator(; seq_type::Symbol, type::Symbol, multioutput::Bool=false)
    if seq_type == :protein
        if type == :mut
            return VeryBasicCNN2.create_model_aminoacids_fixed_pool_stride_w_bottleneck
        else
            return VeryBasicCNN2.create_model_aminoacids_fixed_pool_stride
        end
    else  # :dna or :rna
        if multioutput
            return VeryBasicCNN2.create_model_nucleotides_fixed_pool_stride_multioutputs
        else
            return VeryBasicCNN2.create_model_nucleotides_fixed_pool_stride
        end
    end
end


const loss_specs = Dict(
    :mse => (loss=Flux.mse, agg=StatsBase.mean),
    :mae => (loss=Flux.mae, agg=StatsBase.mean),
    :huber => (loss=Flux.huber_loss, agg=StatsBase.mean),
    :binary_cross_entropy => (loss=Flux.binarycrossentropy, agg=StatsBase.mean),
)


"""
    dataset(; name, file, kwargs...) -> NamedTuple

Construct a dataset entry with sensible defaults. 
Only `name` and `file` are required; everything else has defaults.

# Keyword Arguments
- `name::String` — human-readable dataset name (required)
- `file::String` — .jld2 filename (required)
- `seq_type = :dna` — sequence type (:dna, :rna, :protein)
- `type = :conv` — model type (:conv, :mut)
- `normalization = :zscore` — normalization method
- `seed = nothing` — random seed (nothing → triggers tuning)
- `motif_sizes = [2, 3]` — motif group sizes to search
- `activation_thresh = 0.9` — activation threshold
- `multioutput = false` — multiple output columns?

# Examples
    dataset(name="ecoli", file="ecoli.jld2", seed=2)
    dataset(name="utr_human", file="utr_human.jld2", seq_type=:rna, seed=63)
"""
function dataset(; 
        name::String,
        file::String,
        seq_type::Symbol = :dna,
        type::Symbol = :conv,
        normalization_method::Symbol = :zscore,
        seed::Union{Int, Nothing} = nothing,
        motif_sizes::Vector{Int} = [2,3,4,5],
        activation_thresh::Float64 = 0.95,
        multioutput::Bool = false,
        loss_spec = loss_specs[:mse],
        model_creator = nothing
    )
    if isnothing(model_creator)
        model_creator = resolve_model_creator(; seq_type, type, multioutput)
    end
    return (;
        name, file, model_creator, normalization_method, 
        type, seq_type, seed, motif_sizes, activation_thresh, loss_spec
    )
end

"""
    load_datasets(names...; include_mut=false) -> Vector

Load dataset entries by name. With no arguments, returns all active datasets.

Requires `DATASETS` and `DATASETS_MUT` to be defined (see `datasets.jl`).

# Examples
    load_datasets()                          # all standard datasets
    load_datasets("ecoli", "yeast")          # specific ones
    load_datasets(; include_mut=true)        # include mutation datasets
"""
function load_datasets(names::String...; include_mut::Bool=false)
    all = include_mut ? vcat(DATASETS, DATASETS_MUT) : DATASETS
    if isempty(names)
        return all
    else
        selected = filter(d -> d.name in names, all)
        found = Set(d.name for d in selected)
        missing_names = setdiff(Set(names), found)
        !isempty(missing_names) && @warn "Datasets not found: $(collect(missing_names))"
        return selected
    end
end
