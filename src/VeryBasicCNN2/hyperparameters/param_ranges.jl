# ────────────────────────────────────────────────────────────────────────────
# Hyperparameter Range Specifications
# ────────────────────────────────────────────────────────────────────────────

"""
    HyperParamRanges

Specification of valid ranges for random hyperparameter generation.
"""
Base.@kwdef struct HyperParamRanges
    num_img_layers_range = 3:5
    pfm_length_range = 3:9
    num_base_filters_range = 72:12:512
    conv_filter_range = 128:32:512
    conv_filter_height_range = 1:5
    pool_size_range = 1:2
    stride_range = 1:2
    num_no_pool_layers::Int = 0
    batch_size_options = [64, 128, 256]
    final_layer_filters::Int = 48
    base_pool_size::Int = 1
    base_stride::Int = 1
    softmax_alpha = SOFTMAX_ALPHA
    infer_base_layer_code::Bool = true

    # Conv layer that the bottleneck squeeze is applied to (filters→BOTTLENECK_FILTERS,
    # height→25). 0 = use the inference-code layer (legacy behavior, e.g. amino-acid
    # bottleneck squeezes its inference layer). Set >0 to decouple the squeeze from the
    # interpreted layer — used by nucleotides so the interpreted base/first layer keeps
    # its full motif filters and the squeeze sits at a deeper conv layer instead.
    bottleneck_layer::Int = 0

    # MBConv options (default: disabled, use with_mbconv to enable)
    num_mbconv_range = 0:0
    mbconv_expansion_options = [4]
    
    # Final nonlinearity (default: identity)
    final_nonlinearity::Function = identity
end

const DEFAULT_RANGES = HyperParamRanges()

# ────────────────────────────────────────────────────────────────────────────
# Domain-Specific Range Presets
# ────────────────────────────────────────────────────────────────────────────

"""
    nucleotide_ranges(; kwargs...)

Hyperparameter ranges optimized for nucleotide sequences (DNA/RNA).
4-letter alphabet, typical motif lengths 6-12nt.
"""
nucleotide_ranges(; kwargs...) = HyperParamRanges(;
    num_img_layers_range = 3:5,
    pfm_length_range = 6:12,
    num_base_filters_range = 48:24:256,
    conv_filter_range = 64:32:256,
    conv_filter_height_range = 2:4,
    infer_base_layer_code = true,
    kwargs...
)

"""
    amino_acid_ranges(; kwargs...)

Hyperparameter ranges optimized for amino acid sequences (proteins).
20-letter alphabet, larger filters needed.
"""
amino_acid_ranges(; kwargs...) = HyperParamRanges(;
    num_img_layers_range = 3:5,
    pfm_length_range = 5:10,
    num_base_filters_range = 64:32:320,
    conv_filter_range = 96:48:384,
    conv_filter_height_range = 6:12,
    batch_size_options = [32, 64, 128],
    infer_base_layer_code = false,
    kwargs...
)

# Simplified ranges for testing
nucleotide_ranges_simple(; kwargs...) = HyperParamRanges(;
    num_img_layers_range = 2:3,
    pfm_length_range = 7:9,
    num_base_filters_range = 32:64,
    conv_filter_range = 32:64,
    conv_filter_height_range = 2:4,
    kwargs...
)

# Fixed pooling/stride for controlled experiments
nucleotide_ranges_fixed_pool_stride(; kwargs...) = HyperParamRanges(;
    num_img_layers_range = 3:4,
    pfm_length_range = 5:2:7,
    num_base_filters_range = 16:4:32,
    conv_filter_range = 48:4:64,
    
    conv_filter_height_range = 2:4,
    pool_size_range = 2:2, # overrides by num_no_pool_layers in generation
    stride_range = 2:2, # overrides by num_no_pool_layers in generation
    num_no_pool_layers = 0,
    infer_base_layer_code = false,
    kwargs...
)

nucleotide_ranges_fixed_pool_stride_multioutputs(; kwargs...) = HyperParamRanges(;
    num_img_layers_range = 3:4,
    pfm_length_range = 5:2:7,
    num_base_filters_range = 64:8:96,
    conv_filter_range = 48:4:64,
    conv_filter_height_range = 2:3,
    pool_size_range = 2:2, # overrides by num_no_pool_layers in generation
    stride_range = 2:2, # overrides by num_no_pool_layers in generation
    num_no_pool_layers = 0,
    infer_base_layer_code = false,
    kwargs...
)

# Bottleneck variants — squeeze the SECOND conv layer (filters → BOTTLENECK_FILTERS,
# filter height → 25) while leaving interpretation on the base PWM/motif layer, so the
# interpreted first layer keeps its full motif filters. Two overrides vs the non-bottleneck
# preset:
#   • infer_base_layer_code = true  → inference_code_layer = 0 (motifs still read from the
#     base PWM layer, exactly like the non-bottleneck nucleotide model).
#   • num_no_pool_layers = 2        → no pooling through the bottleneck layer, so the
#     height-25 filter doesn't collapse the (already downsampled) sequence — mirrors the
#     amino-acid bottleneck, whose squeeze layer also has no pooling.
# `bottleneck_layer = 2` is clamped to 1:(num_img_layers-1), always valid for the 3:4-layer
# range. NOTE: the height-25 filter spans ~25 input positions, so sequences should be
# ≳ 30nt; shorter reads yield an invalid architecture (create_model returns `nothing`).
nucleotide_ranges_fixed_pool_stride_bottleneck(; kwargs...) =
    nucleotide_ranges_fixed_pool_stride(;
        bottleneck_layer = 2, num_no_pool_layers = 2, infer_base_layer_code = true, kwargs...)

nucleotide_ranges_fixed_pool_stride_multioutputs_bottleneck(; kwargs...) =
    nucleotide_ranges_fixed_pool_stride_multioutputs(;
        bottleneck_layer = 2, num_no_pool_layers = 2, infer_base_layer_code = true, kwargs...)

"""
    nucleotide_ranges_fixed_pool_stride_mut(; kwargs...)

Hyperparameter ranges for **nucleotide mutagenesis** (`seq_type = :dna` / `:rna`
with `type = :mut`) — the DNA/RNA counterpart of
[`amino_acid_ranges_fixed_pool_stride`](@ref), which is what the protein
mutagenesis pipeline runs on.

A mutagenesis model sees the *mutation* encoding (`OnehotSEQ2EXP_Dataset.X_mut`):
a sparse tensor that is 1 only where a sequence differs from the consensus. A
motif is then a *region of co-occurring substitutions*, not a sliding PWM match,
so the architecture departs from the convolution presets on three points:

  • `pfm_length = 1` — the base layer is a per-site letter detector ("which base
    did this position mutate to"), never a width-5-7 sliding PWM.
  • `pool_size = stride = 1` with `num_no_pool_layers = 1` — no downsampling
    before the interpreted layer, so every code position maps back to exactly
    one input site.
  • `infer_base_layer_code = false` — motifs are read at conv layer 1 instead of
    the base layer. Paired with `bottleneck = true` (see
    [`create_model_nucleotides_fixed_pool_stride_mut_w_bottleneck`](@ref)) that
    layer is squeezed to `BOTTLENECK_FILTERS` filters of height
    `BOTTLENECK_HEIGHT`, and its height *is* the mutation-region width: the
    receptive field works out to `pfm_len + BOTTLENECK_HEIGHT - 1` input
    positions (8 nt at the current defaults). Widen the regions with
    the `bottleneck_height` keyword, not with `pfm_length`.

The one substantive departure from the amino-acid preset is `num_base_filters`.
The base layer's filters are near-hard PWMs (`softmax_alpha = SOFTMAX_ALPHA`), so
a width-1 filter over a 4-row alphabet can express only a handful of distinct
detectors where a 20-row alphabet can express many; 48 base filters would be
mostly redundant copies. Sequences should be ≳ `BOTTLENECK_HEIGHT + 2` long, as
with any bottleneck preset.
"""
nucleotide_ranges_fixed_pool_stride_mut(; kwargs...) = HyperParamRanges(;
    num_img_layers_range = 4:4,
    pfm_length_range = 1:1,
    num_base_filters_range = 12:4:24,
    conv_filter_range = 32:8:64,
    conv_filter_height_range = 1:1,
    pool_size_range = 1:1,  # overridden by num_no_pool_layers in generation
    stride_range = 1:1,     # overridden by num_no_pool_layers in generation
    batch_size_options = [32, 64, 128],
    num_no_pool_layers = 1,
    infer_base_layer_code = false,
    kwargs...
)

amino_acid_ranges_fixed_pool_stride(; kwargs...) = HyperParamRanges(;
    # num_img_layers_range = 3:4,
    # pfm_length_range = 3:2:5,
    # num_base_filters_range = 16:8:48,
    # conv_filter_range = 32:8:64,
    # conv_filter_height_range = 2:4,
    # pool_size_range = 2:2, # overrides by num_no_pool_layers in generation
    # stride_range = 2:2, # overrides by num_no_pool_layers in generation
    # batch_size_options = [32, 64, 128],
    # num_no_pool_layers = 1,
    # infer_base_layer_code = false,
    num_img_layers_range = 4:4,
    pfm_length_range = 1:1,
    num_base_filters_range = 48:48,
    conv_filter_range = 32:8:64,
    conv_filter_height_range = 1:1,
    pool_size_range = 1:1, # overrides by num_no_pool_layers in generation
    stride_range = 1:1, # overrides by num_no_pool_layers in generation
    batch_size_options = [32, 64, 128],
    num_no_pool_layers = 1,
    infer_base_layer_code = false,
    kwargs...
)


# ────────────────────────────────────────────────────────────────────────────
# Tanh variants (for models with tanh final nonlinearity)
# ────────────────────────────────────────────────────────────────────────────

nucleotide_ranges_tanh(; kwargs...) = nucleotide_ranges(; final_nonlinearity=tanh, kwargs...)
amino_acid_ranges_tanh(; kwargs...) = amino_acid_ranges(; final_nonlinearity=tanh, kwargs...)
nucleotide_ranges_simple_tanh(; kwargs...) = nucleotide_ranges_simple(; final_nonlinearity=tanh, kwargs...)
nucleotide_ranges_fixed_pool_stride_tanh(; kwargs...) = nucleotide_ranges_fixed_pool_stride(; final_nonlinearity=tanh, kwargs...)
nucleotide_ranges_fixed_pool_stride_multioutputs_tanh(; kwargs...) = nucleotide_ranges_fixed_pool_stride_multioutputs(; final_nonlinearity=tanh, kwargs...)
amino_acid_ranges_fixed_pool_stride_tanh(; kwargs...) = amino_acid_ranges_fixed_pool_stride(; final_nonlinearity=tanh, kwargs...)

# ────────────────────────────────────────────────────────────────────────────
# Sigmoid variants (for models with sigmoid final nonlinearity)
# ────────────────────────────────────────────────────────────────────────────

nucleotide_ranges_sigmoid(; kwargs...) = nucleotide_ranges(; final_nonlinearity=Flux.sigmoid, kwargs...)
amino_acid_ranges_sigmoid(; kwargs...) = amino_acid_ranges(; final_nonlinearity=Flux.sigmoid, kwargs...)
nucleotide_ranges_simple_sigmoid(; kwargs...) = nucleotide_ranges_simple(; final_nonlinearity=Flux.sigmoid, kwargs...)
nucleotide_ranges_fixed_pool_stride_sigmoid(; kwargs...) = nucleotide_ranges_fixed_pool_stride(; final_nonlinearity=Flux.sigmoid, kwargs...)
nucleotide_ranges_fixed_pool_stride_multioutputs_sigmoid(; kwargs...) = nucleotide_ranges_fixed_pool_stride_multioutputs(; final_nonlinearity=Flux.sigmoid, kwargs...)
amino_acid_ranges_fixed_pool_stride_sigmoid(; kwargs...) = amino_acid_ranges_fixed_pool_stride(; final_nonlinearity=Flux.sigmoid, kwargs...)
