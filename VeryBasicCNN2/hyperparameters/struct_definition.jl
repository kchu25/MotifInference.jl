# ============================================================================
# Core HyperParameters Struct Definition
# ============================================================================

"""
    HyperParameters

CNN hyperparameters for biological sequence analysis.

# Architecture Fields
- `pfm_len::Int`: Length of Position Weight Matrix filters (motif length)
- `num_pfms::Int`: Number of PWM filters in base layer
- `num_img_filters::Vector{Int}`: Number of filters per convolutional layer
- `img_fil_widths::Vector{Int}`: Input channels for each conv layer
- `img_fil_heights::Vector{Int}`: Filter heights for each conv layer

# Pooling Fields  
- `pool_base::Int`: Pooling size for base layer
- `stride_base::Int`: Stride for base layer pooling
- `poolsize::Vector{Int}`: Pooling sizes for each conv layer
- `stride::Vector{Int}`: Strides for each conv layer
- `pool_lvl_top::Int`: Highest layer index that uses pooling

# Training Fields
- `softmax_strength_img_fil::Float32`: Softmax strength for filter normalization
- `batch_size::Int`: Training batch size
- `inference_code_layer::Int`: Layer to extract code from (0 = PWM layer)

# Normalization Fields
- `use_layernorm::Bool`: Apply LayerNorm after pooling for layers > inference_code_layer (default: false)
- `use_channel_mask::Bool`: Apply channel masking to PWM and conv layers (default: true)

# MBConv Fields
- `num_mbconv::Int`: Number of MBConv blocks to add (0 = none, default)
- `mbconv_expansion::Int`: MBConv expansion ratio (default: 4)
"""
Base.@kwdef struct HyperParameters
    # Architecture
    pfm_len::Int = 10
    num_pfms::Int = 24
    num_img_filters::Vector{Int} = [65, 98, 128, 128, 76, 5]
    img_fil_widths::Vector{Int} = vcat([num_pfms], num_img_filters[1:(end-1)])
    img_fil_heights::Vector{Int} = [6, 6, 6, 6, 6, 5]
    
    # Pooling
    pool_base::Int = 2
    stride_base::Int = 1
    poolsize::Vector{Int} = [2, 2, 2, 2, 2, 1]
    stride::Vector{Int} = [1, 1, 2, 2, 2, 1]
    pool_lvl_top::Int = 5
    
    # Training
    softmax_strength_img_fil::DEFAULT_FLOAT_TYPE = 500.0
    batch_size::Int = 256
    inference_code_layer::Int = 0
    
    # Normalization
    use_layernorm::Bool = false  # Apply LayerNorm after inference_code_layer
    use_channel_mask::Bool = true  # Apply channel masking to layers
    
    # MBConv (optional EfficientNet-style blocks)
    num_mbconv::Int = 0
    mbconv_expansion::Int = 4
end

# Convenience accessors
num_layers(hp::HyperParameters) = length(hp.num_img_filters)

function Base.show(io::IO, hp::HyperParameters)
    n_layers = num_layers(hp)
    
    println(io, "HyperParameters:")
    println(io, "  Architecture:")
    println(io, "    Base: $(hp.num_pfms) PWMs × $(hp.pfm_len)nt")
    println(io, "    Conv: $(n_layers) layers")
    println(io, "    Batch: $(hp.batch_size)")
    println(io, "\n  Layers:")
    println(io, "    Layer │ Filters │ Height │ Pool │ Stride")
    println(io, "    ──────┼─────────┼────────┼──────┼────────")
    println(io, "    Base  │  $(lpad(hp.num_pfms, 4)) │   N/A  │   $(hp.pool_base)  │    $(hp.stride_base)")
    
    for i in 1:n_layers
        no_pool = i > hp.pool_lvl_top ? " (skip)" : ""
        println(io, "      $(lpad(i, 2))  │  $(lpad(hp.num_img_filters[i], 4)) │    $(lpad(hp.img_fil_heights[i], 2))  │   $(hp.poolsize[i])  │    $(hp.stride[i])$no_pool")
    end
    
    println(io, "\n  Inference code layer: $(hp.inference_code_layer)")
    hp.use_layernorm && print(io, "\n  LayerNorm: enabled (layers > $(hp.inference_code_layer))")
    hp.num_mbconv > 0 && print(io, "\n  MBConv blocks: $(hp.num_mbconv) (expansion: $(hp.mbconv_expansion)x)")
end
