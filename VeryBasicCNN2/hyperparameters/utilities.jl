# ============================================================================
# Hyperparameter Utility Functions
# ============================================================================

"""
    receptive_field(hp::HyperParameters)

Calculate receptive field size in the input sequence at the inference code layer.
This accounts for all convolutions, pooling, and strides up to that layer.
"""
function receptive_field(hp::HyperParameters)
    layer = hp.inference_code_layer
    layer == 0 && return hp.pfm_len
    
    # Start with base PWM filter
    rf = hp.pfm_len
    jump = 1  # Effective spacing in input coordinates
    
    # Base layer pooling
    rf += (hp.pool_base - 1) * jump
    jump *= hp.stride_base
    
    # Each conv layer
    for i in 1:layer
        rf += (hp.img_fil_heights[i] - 1) * jump
        
        if i ≤ hp.pool_lvl_top
            rf += (hp.poolsize[i] - 1) * jump
            jump *= hp.stride[i]
        end 
    end
    
    return rf
end

"""
    with_batch_size(hp::HyperParameters, new_batch_size::Int)

Create new HyperParameters with different batch size.
"""
function with_batch_size(hp::HyperParameters, new_batch_size::Int)
    HyperParameters(
        pfm_len = hp.pfm_len,
        num_pfms = hp.num_pfms,
        num_img_filters = hp.num_img_filters,
        img_fil_widths = hp.img_fil_widths,
        img_fil_heights = hp.img_fil_heights,
        pool_base = hp.pool_base,
        stride_base = hp.stride_base,
        poolsize = hp.poolsize,
        stride = hp.stride,
        pool_lvl_top = hp.pool_lvl_top,
        softmax_strength_img_fil = hp.softmax_strength_img_fil,
        batch_size = new_batch_size,
        inference_code_layer = hp.inference_code_layer,
        use_layernorm = hp.use_layernorm,
        num_mbconv = hp.num_mbconv,
        mbconv_expansion = hp.mbconv_expansion
    )
end

"""
    with_layernorm(hp::HyperParameters, enabled::Bool=true)

Create new HyperParameters with LayerNorm enabled or disabled.

LayerNorm is applied after pooling for layers > inference_code_layer when enabled.

# Example
```julia
hp = generate_random_hyperparameters()
hp_ln = with_layernorm(hp, true)   # Enable LayerNorm
hp_no_ln = with_layernorm(hp, false)  # Disable LayerNorm
```
"""
function with_layernorm(hp::HyperParameters, enabled::Bool=true)
    HyperParameters(
        pfm_len = hp.pfm_len,
        num_pfms = hp.num_pfms,
        num_img_filters = hp.num_img_filters,
        img_fil_widths = hp.img_fil_widths,
        img_fil_heights = hp.img_fil_heights,
        pool_base = hp.pool_base,
        stride_base = hp.stride_base,
        poolsize = hp.poolsize,
        stride = hp.stride,
        pool_lvl_top = hp.pool_lvl_top,
        softmax_strength_img_fil = hp.softmax_strength_img_fil,
        batch_size = hp.batch_size,
        inference_code_layer = hp.inference_code_layer,
        use_layernorm = enabled,
        num_mbconv = hp.num_mbconv,
        mbconv_expansion = hp.mbconv_expansion
    )
end

"""
    with_mbconv(hp::HyperParameters; num_blocks=2, expansion=4)

Create new HyperParameters with MBConv blocks enabled.

# Example
```julia
hp = generate_random_hyperparameters()
hp_mbconv = with_mbconv(hp; num_blocks=3, expansion=6)
```
"""
function with_mbconv(hp::HyperParameters; num_blocks::Int=2, expansion::Int=4)
    HyperParameters(
        pfm_len = hp.pfm_len,
        num_pfms = hp.num_pfms,
        num_img_filters = hp.num_img_filters,
        img_fil_widths = hp.img_fil_widths,
        img_fil_heights = hp.img_fil_heights,
        pool_base = hp.pool_base,
        stride_base = hp.stride_base,
        poolsize = hp.poolsize,
        stride = hp.stride,
        pool_lvl_top = hp.pool_lvl_top,
        softmax_strength_img_fil = hp.softmax_strength_img_fil,
        batch_size = hp.batch_size,
        inference_code_layer = hp.inference_code_layer,
        use_layernorm = hp.use_layernorm,
        num_mbconv = num_blocks,
        mbconv_expansion = expansion
    )
end

# ============================================================================
# EfficientNet-style MBConv Configuration
# ============================================================================

"""
    efficientnet_mbconv_config(phi::Int=0)

Get EfficientNet-style MBConv configuration based on compound scaling coefficient φ.

# EfficientNet Scaling
- φ=0 (B0): num_blocks=2, expansion=4  (baseline)
- φ=1 (B1): num_blocks=3, expansion=4  (1.2× depth)
- φ=2 (B2): num_blocks=3, expansion=6  (1.4× depth, wider)
- φ=3 (B3): num_blocks=4, expansion=6  (1.8× depth)
- φ=4 (B4): num_blocks=5, expansion=6  (2.2× depth)

# Returns
- `(num_blocks, expansion)`: Configuration tuple

# Example
```julia
hp = generate_random_hyperparameters()
num_blocks, expansion = efficientnet_mbconv_config(2)  # B2 config
hp_b2 = with_mbconv(hp; num_blocks=num_blocks, expansion=expansion)
```
"""
function efficientnet_mbconv_config(phi::Int=0)
    configs = [
        (2, 4),  # B0: baseline
        (3, 4),  # B1: deeper
        (3, 6),  # B2: deeper + wider
        (4, 6),  # B3: much deeper
        (5, 6),  # B4: very deep
        (6, 6),  # B5: extremely deep
        (7, 6),  # B6: ultra deep
        (8, 8),  # B7: maximum depth + width
    ]
    
    @assert 0 ≤ phi ≤ 7 "phi must be between 0 and 7 (B0-B7)"
    return configs[phi + 1]
end

"""
    with_efficientnet_mbconv(hp::HyperParameters, phi::Int=0)

Add EfficientNet-style MBConv blocks with compound scaling coefficient φ.

This uses the standard EfficientNet scaling strategy where φ controls
the depth (number of blocks) and width (expansion ratio) of MBConv layers.

# Arguments
- `hp`: Base hyperparameters
- `phi`: EfficientNet scaling coefficient (0-7 for B0-B7)

# Example
```julia
hp = generate_random_hyperparameters()
hp_b0 = with_efficientnet_mbconv(hp, 0)  # EfficientNet-B0 style
hp_b2 = with_efficientnet_mbconv(hp, 2)  # EfficientNet-B2 style
hp_b4 = with_efficientnet_mbconv(hp, 4)  # EfficientNet-B4 style
```
"""
function with_efficientnet_mbconv(hp::HyperParameters, phi::Int=0)
    num_blocks, expansion = efficientnet_mbconv_config(phi)
    return with_mbconv(hp; num_blocks=num_blocks, expansion=expansion)
end
