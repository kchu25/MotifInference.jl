# ============================================================================
# Channel Masking Layer
# ============================================================================
# Learnable channel-wise masking with Gumbel-Softmax
# - Each channel gets a learned importance score via 1x1 convolution
# - Spatial positions are aggregated (summed) to get channel-level mask
# - Uses Gumbel-Softmax for soft/hard masking during training/inference

"""
    layer_channel_mask

Learnable channel-wise masking layer using Gumbel-Softmax.

# Fields
- `mixing_filter`: 1×C×1×C conv filter for channel mixing
- `temp`: Gumbel-Softmax temperature
- `eta`: Right stretch parameter for hard concrete
- `gamma`: Left stretch parameter for hard concrete

# Process
1. Apply 1×1 conv for channel mixing
2. Sum over spatial dimension to get channel importance
3. Apply Gumbel-Softmax masking (soft in training, hard in test)
"""
struct layer_channel_mask
    mixing_filter::AbstractArray{DEFAULT_FLOAT_TYPE, 4}  # (1, C, 1, C)
    temp::DEFAULT_FLOAT_TYPE
    eta::DEFAULT_FLOAT_TYPE
    gamma::DEFAULT_FLOAT_TYPE

    function layer_channel_mask(input_channels::Int;
                                init_scale::DEFAULT_FLOAT_TYPE = DEFAULT_FLOAT_TYPE(0.01),
                                temp::DEFAULT_FLOAT_TYPE = DEFAULT_FLOAT_TYPE(0.5),
                                eta::DEFAULT_FLOAT_TYPE = DEFAULT_FLOAT_TYPE(1.1),
                                gamma::DEFAULT_FLOAT_TYPE = DEFAULT_FLOAT_TYPE(-0.1),
                                rng = Random.GLOBAL_RNG)
        # Filter: (height=1, width=C, in_channels=1, out_channels=C)
        # This performs channel mixing across the width dimension
        mixing_filter = init_scale .* randn(rng, DEFAULT_FLOAT_TYPE,
                                           (1, input_channels, 1, input_channels))
        return new(mixing_filter, temp, eta, gamma)
    end

    # Direct constructor for GPU/loading
    layer_channel_mask(mixing_filter, temp, eta, gamma) =
        new(mixing_filter, temp, eta, gamma)
end

Flux.@layer layer_channel_mask
Flux.trainable(l::layer_channel_mask) = (mixing_filter = l.mixing_filter,)

"""
    (l::layer_channel_mask)(code; training=true)

Forward pass through channel masking layer.

# Arguments
- `code`: Input features, either (1, length, channels, batch) or (length, channels, 1, batch)
- `training`: Whether in training mode (soft masks) or test mode (hard masks)

# Returns
- Masked features with same shape as input
"""
function (l::layer_channel_mask)(code; training=true)
    # Save original shape for restoration at the end
    original_shape = size(code)
    is_pwm_format = size(code, 1) == 1  # Track if input was (1, length, channels, batch)
    
    # Normalize input to (length, channels, 1, batch) format
    if is_pwm_format
        # PWM-style input: (1, length, channels, batch)
        num_channels = size(code, 3)
        code = reshape(code, (size(code, 2), num_channels, 1, size(code, 4)))
    else
        # Conv-style input: (length, 1, channels, batch)
        num_channels = size(code, 3)
        code = reshape(code, (size(code, 1), num_channels, 1, size(code, 4)))
    end
    
    # Apply channel mixing via 1×1 conv
    # Input: (length, channels, 1, batch)
    # Filter: (1, channels, 1, channels)
    # Output: (length, 1, channels, batch)
    z = Flux.conv(code, l.mixing_filter; pad=0, flipped=true)
    
    # Reshape to standard format: (length, channels, 1, batch)
    z = reshape(z, (size(z, 1), num_channels, 1, size(z, 4)))
    
    # Aggregate over spatial dimension to get channel-level importance
    # (length, channels, 1, batch) -> (1, channels, 1, batch)
    channel_importance = sum(z; dims=1)
    
    # Convert to probabilities
    p = Flux.sigmoid.(channel_importance)
    
    # Apply Gumbel-Softmax masking
    # mask = if training
    #     gumbel_softmax_sample(p, l.temp, l.eta, l.gamma)  # Soft mask
    # else
    #     hard_threshold_mask(p, l.temp, l.eta, l.gamma)    # Hard binary mask
    # end
    
    # Apply mask (broadcasts over spatial dimension)
    # masked_code = code .* mask
    masked_code = code .* p
    
    # Reshape back to original format if needed
    if is_pwm_format
        masked_code = reshape(masked_code, original_shape)
    else
        masked_code = reshape(masked_code, original_shape)
    end
    
    return masked_code
end


