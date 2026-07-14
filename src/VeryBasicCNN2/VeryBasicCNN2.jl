module VeryBasicCNN2

# Write your package code here.

using Flux
using CUDA 
using Random
using ChainRulesCore: @ignore_derivatives
using Statistics: mean, var

# ────────────────────────────────────────────────────────────────────────────
# Constants
# ────────────────────────────────────────────────────────────────────────────

const DEFAULT_FLOAT_TYPE = Float32
const SOFTMAX_ALPHA = DEFAULT_FLOAT_TYPE(500)
const BOTTLENECK_FILTERS = 50
const BOTTLENECK_HEIGHT = 8
# ────────────────────────────────────────────────────────────────────────────
# Core Components (Order Matters)
# ────────────────────────────────────────────────────────────────────────────

include("hyperparameters.jl")
include("utils.jl")
include("layers.jl")
include("model.jl")
include("forward.jl")
include("loss.jl")
include("convert.jl")
include("processors/processor.jl")

# ────────────────────────────────────────────────────────────────────────────
# Exports
# ────────────────────────────────────────────────────────────────────────────

# Types
export SeqCNN
export HyperParameters, HyperParamRanges
export LearnedPWMs, LearnedCodeImgFilters, MBConvBlock
export CodeProcessor, CodeProcessorType
export plain, resnet, mbconv, deep_plain

# Hyperparameter functions
export generate_random_hyperparameters
export nucleotide_ranges, amino_acid_ranges
export nucleotide_ranges_simple, nucleotide_ranges_fixed_pool_stride
export amino_acid_ranges_fixed_pool_stride
export receptive_field, with_batch_size, with_mbconv, with_sparse_unpool, num_layers
export efficientnet_mbconv_config, with_efficientnet_mbconv

# Model construction — dispatch tag types
export Nucleotides, AminoAcids, Simple, FixedPoolStride, Multioutputs, UseTanh, UseSigmoid, Bottleneck
export model_ranges
export create_model
# Backward-compatible convenience names (thin wrappers)
export create_model_nucleotides, create_model_nucleotides_simple
export create_model_nucleotides_fixed_pool_stride
export create_model_nucleotides_fixed_pool_stride_bottleneck
export create_model_nucleotides_fixed_pool_stride_multioutputs_bottleneck
export create_model_aminoacids, create_model_aminoacids_fixed_pool_stride

# Forward pass functions
export predict_from_sequences, predict_from_code
export compute_code_at_layer, extract_features

# Loss functions
export compute_training_loss, huber_loss, masked_mse

# Utilities
export model2cpu, model2gpu
export final_conv_embedding_length
export gumbel_softmax_sample, hard_threshold_mask
export verify_device
export eval!, train!

# Code processor
export create_code_processor, process_code, processor2cpu, processor2gpu

# wrapper for processor functions
# TODO: see if these can be elimilated later, using e.g. multiple dispatch
const proc_wrap = (
    create_processor = create_code_processor,
    arch_type = mbconv,
    predict_from_code = predict_from_code,
)

export proc_wrap

end
