# ============================================================================
# MBConv Architecture (with optional SE attention)
# ============================================================================

"""
    forward_mbconv!(cp::CodeProcessor, x)

MBConv-style with expansion, depthwise conv, optional SE attention, and projection.
"""
function forward_mbconv!(cp::CodeProcessor, x)
    l, M, _, n = size(x)
    
    # Expansion
    if !isnothing(cp.expand_filters)
        x = Flux.conv(x, cp.expand_filters; pad=0, flipped=true)
        x = Flux.swish.(x)
        # After 1x1 conv, channels are in dim 3: (l, 1, expanded, n)
        # Reshape to standard format: (l, expanded, 1, n)
        l_new = size(x, 1)
        expanded = size(x, 3)
        n_new = size(x, 4)
        x = reshape(x, (l_new, expanded, 1, n_new))
    end
    
    # Get current number of channels after expansion
    current_channels = size(x, 2)
    
    # Reshape for depthwise conv
    x = reshape(x, (l, 1, current_channels, n))
    
    # Depthwise convolution
    pad_h = (size(cp.dw_filters, 1) - 1) ÷ 2
    x = Flux.conv(x, cp.dw_filters; pad=(pad_h, 0), flipped=true, groups=current_channels)
    x = Flux.swish.(x)
    
    # Reshape back for SE/projection
    x = reshape(x, (l, current_channels, 1, n))
    
    # Squeeze-Excitation (if enabled)
    if !isnothing(cp.se_w1)
        x_temp = reshape(x, (l, 1, current_channels, n))
        x_pooled = reshape(mean(x_temp; dims=1), (current_channels, 1, n))
        attn = Flux.NNlib.batched_mul(cp.se_w1, x_pooled)
        attn = Flux.swish.(attn)
        attn = Flux.NNlib.batched_mul(cp.se_w2, attn)
        attn = Flux.sigmoid.(attn)
        # Reshape: (current_channels, 1, n) -> (1, current_channels, 1, n)
        attn = reshape(attn, (1, current_channels, 1, n))
        x = x .* attn
    end
    
    # Projection
    x = Flux.conv(x, cp.project_filters; pad=0, flipped=true)
    x = reshape(x, (l, size(cp.project_filters, 4), 1, n))
    
    return x
end
