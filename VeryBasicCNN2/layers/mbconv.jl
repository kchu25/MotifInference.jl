# ============================================================================
# MBConv Block (Optional EfficientNet-style refinement)
# ============================================================================

"""
    MBConvBlock

Mobile Inverted Bottleneck Convolution (MBConv) block with SE attention.

# Fields
- `expand_filters`: Expansion filters (1, in_channels, 1, expanded)
- `dw_filters`: Depthwise filters (kernel_size, 1, 1, expanded)
- `se_w1`: SE reduction weights (se_channels, expanded, 1)
- `se_w2`: SE expansion weights (expanded, se_channels, 1)
- `project_filters`: Projection filters (1, expanded, 1, out_channels)
- `use_skip`: Whether to use skip connection
"""
struct MBConvBlock
    expand_filters::Union{AbstractArray{DEFAULT_FLOAT_TYPE, 4}, Nothing}
    dw_filters::AbstractArray{DEFAULT_FLOAT_TYPE, 4}
    se_w1::AbstractArray{DEFAULT_FLOAT_TYPE, 3}
    se_w2::AbstractArray{DEFAULT_FLOAT_TYPE, 3}
    project_filters::AbstractArray{DEFAULT_FLOAT_TYPE, 4}
    use_skip::Bool
    
    function MBConvBlock(;
        in_channels::Int,
        out_channels::Int,
        kernel_size::Int = 2,
        expansion_ratio::Int = 2,
        se_ratio::DEFAULT_FLOAT_TYPE = DEFAULT_FLOAT_TYPE(8),  # SE reduction ratio (higher = fewer params)
        use_se::Bool = true,
        use_cuda::Bool = false,
        rng = Random.GLOBAL_RNG
    )
        expanded = in_channels * expansion_ratio
        init_scale = DEFAULT_FLOAT_TYPE(0.1)
        
        # Expansion filters (1, in_channels, 1, expanded)
        expand_filters = expansion_ratio > 1 ?
            init_scale .* randn(rng, DEFAULT_FLOAT_TYPE, (1, in_channels, 1, expanded)) :
            nothing
        
        # Depthwise filters (kernel_size, 1, 1, expanded)
        dw_filters = init_scale .* randn(rng, DEFAULT_FLOAT_TYPE, 
                                         (kernel_size, 1, 1, expanded))
        
        # Squeeze-Excitation weights
        se_channels = max(1, floor(Int, expanded / se_ratio))
        se_w1 = init_scale .* randn(rng, DEFAULT_FLOAT_TYPE, (se_channels, expanded, 1))
        se_w2 = init_scale .* randn(rng, DEFAULT_FLOAT_TYPE, (expanded, se_channels, 1))
        
        # Projection filters (1, expanded, 1, out_channels)
        project_filters = init_scale .* randn(rng, DEFAULT_FLOAT_TYPE,
                                              (1, expanded, 1, out_channels))
        
        if use_cuda
            expand_filters = isnothing(expand_filters) ? nothing : cu(expand_filters)
            dw_filters = cu(dw_filters)
            se_w1 = cu(se_w1)
            se_w2 = cu(se_w2)
            project_filters = cu(project_filters)
        end
        
        use_skip = (in_channels == out_channels)
        return new(expand_filters, dw_filters, se_w1, se_w2, project_filters, use_skip)
    end
    
    # Direct constructor for GPU/loading
    MBConvBlock(expand_filters, dw_filters, se_w1, se_w2, project_filters, use_skip) = 
        new(expand_filters, dw_filters, se_w1, se_w2, project_filters, use_skip)
end

Flux.@layer MBConvBlock
Flux.trainable(mb::MBConvBlock) = (
    expand_filters = mb.expand_filters,
    dw_filters = mb.dw_filters,
    se_w1 = mb.se_w1,
    se_w2 = mb.se_w2,
    project_filters = mb.project_filters
)

function (mb::MBConvBlock)(x)
    identity_input = x
    l, M, _, n = size(x)  # (spatial, channels, 1, batch)
    
    # Expansion
    if !isnothing(mb.expand_filters)
        x = Flux.conv(x, mb.expand_filters; pad=0, flipped=true)
        x = Flux.swish.(x)
    end
    
    edim = size(x, 3)  # expanded channels
    
    # Reshape for depthwise conv
    x = reshape(x, (l, 1, edim, n))
    
    # Depthwise convolution with groups
    pad_h = (size(mb.dw_filters, 1) - 1) ÷ 2
    x = Flux.conv(x, mb.dw_filters; pad=(pad_h, 0), flipped=true, groups=edim)
    x = Flux.swish.(x)
    
    # Squeeze-Excitation
    # Global average pooling over spatial dimension
    x_pooled = reshape(mean(x; dims=1), (edim, 1, n))
    
    # SE: reduce → swish → expand → sigmoid
    attn = Flux.NNlib.batched_mul(mb.se_w1, x_pooled)
    attn = Flux.swish.(attn)
    attn = Flux.NNlib.batched_mul(mb.se_w2, attn)
    attn = Flux.sigmoid.(attn)
    
    # Apply attention and reshape
    attn = reshape(attn, (1, 1, edim, n))
    x = x .* attn
    x = reshape(x, (l, edim, 1, n))
    
    # Projection back
    x = Flux.conv(x, mb.project_filters; pad=0, flipped=true)
    x = reshape(x, (l, size(mb.project_filters, 4), 1, n))
    
    # Skip connection
    mb.use_skip && return x .+ identity_input
    return x
end
