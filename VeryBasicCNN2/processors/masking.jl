# ============================================================================
# Gumbel-Softmax Masking Logic (CodeProcessor-specific)
# ============================================================================
# Note: gumbel_softmax_sample() and hard_threshold_mask() are in utils.jl

"""
    apply_gumbel_mask(cp::CodeProcessor, x; training::Bool=true, step::Union{Nothing, Int}=nothing)

Apply hierarchical Gumbel-Softmax masking (component-wise + channel-wise).

# Arguments
- `cp`: CodeProcessor instance
- `x`: Input features (l, out_channels, 1, n)
- `training`: Whether in training mode (affects Gumbel sampling)
- `step`: Training step for temperature annealing

# Returns
- Masked features with same shape as input
"""
function apply_gumbel_mask(cp::CodeProcessor, x; training::Bool=true, step::Union{Nothing, Int}=nothing)
    # Component-wise mask: Learn mask probabilities from features
    # x shape: (l, out_channels, 1, n)
    mask_logits = Flux.conv(x, cp.mask_proj; pad=0, flipped=true)
    # Output: (l, 1, out_channels, n) -> reshape to (l, out_channels, 1, n)
    mask_logits = reshape(mask_logits, (size(mask_logits, 1), size(mask_logits, 3), 
                                        1, size(mask_logits, 4)))
    
    # Get probabilities: p_c = sigmoid(logits)
    p_c = Flux.sigmoid.(mask_logits)
    
    # Channel-wise mask: Average features across spatial dimension, then project
    # x shape: (l, out_channels, 1, n) -> average to (out_channels, 1, n)
    x_avg = reshape(mean(x; dims=1), (size(x, 2), 1, size(x, 4)))
    # Project: batched_mul(W, x_avg) where W: (out_channels, out_channels, 1)
    channel_logits = Flux.NNlib.batched_mul(cp.channel_mask_proj, x_avg)
    p_ch = Flux.sigmoid.(channel_logits)  # Shape: (out_channels, 1, n)
    
    if training
        # Temperature annealing based on training steps
        # Decay: 0.5 * 0.9995^step → reaches ~0.1 after ~3000 steps
        temp = if isnothing(step)
            cp.mask_temp
        else
            max(DEFAULT_FLOAT_TYPE(0.1), cp.mask_temp * DEFAULT_FLOAT_TYPE(0.9995)^step)
        end
        
        # Component-wise Gumbel-Softmax
        z_c = gumbel_softmax_sample(p_c, temp, cp.mask_eta, cp.mask_gamma)
        
        # Channel-wise Gumbel-Softmax
        z_ch = gumbel_softmax_sample(p_ch, temp, cp.mask_eta, cp.mask_gamma)
    else
        # Test time: deterministic hard masks
        temp = DEFAULT_FLOAT_TYPE(0.1)
        z_c = hard_threshold_mask(p_c, temp, cp.mask_eta, cp.mask_gamma)
        z_ch = hard_threshold_mask(p_ch, temp, cp.mask_eta, cp.mask_gamma)
    end
    
    # Combine masks: channel mask broadcasts over spatial dimension
    z_ch_broadcast = reshape(z_ch, (1, size(z_ch, 1), 1, size(z_ch, 3)))
    combined_mask = z_c .* z_ch_broadcast
    
    # Apply combined mask
    return x .* combined_mask
end
