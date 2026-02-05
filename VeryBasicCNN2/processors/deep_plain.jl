# ============================================================================
# Deep Plain Architecture (3 stacked layers with lightweight attention)
# ============================================================================

"""
    forward_deep_plain!(cp::CodeProcessor, x)

Deep plain architecture with 3 stacked depthwise+projection layers,
each followed by lightweight channel attention.
"""
function forward_deep_plain!(cp::CodeProcessor, x)
    l, M, _, n = size(x)
    
    # ========== Layer 1 ==========
    current_channels = M
    x = reshape(x, (l, 1, current_channels, n))
    
    # First depthwise conv
    pad_h = (size(cp.dw_filters, 1) - 1) ÷ 2
    x = Flux.conv(x, cp.dw_filters; pad=(pad_h, 0), flipped=true, groups=current_channels)
    x = Flux.swish.(x)
    
    # Reshape and first projection
    x = reshape(x, (l, current_channels, 1, n))
    x = Flux.conv(x, cp.project_filters; pad=0, flipped=true)
    x = reshape(x, (l, size(cp.project_filters, 4), 1, n))
    
    # ========== Layer 2 ==========
    if !isnothing(cp.dw_filters_2)
        current_channels_2 = size(x, 2)
        x = reshape(x, (l, 1, current_channels_2, n))
        
        # Second depthwise conv
        pad_h_2 = (size(cp.dw_filters_2, 1) - 1) ÷ 2
        x = Flux.conv(x, cp.dw_filters_2; pad=(pad_h_2, 0), flipped=true, groups=current_channels_2)
        x = Flux.swish.(x)
        
        # Reshape and second projection
        x = reshape(x, (l, current_channels_2, 1, n))
        x = Flux.conv(x, cp.project_filters_2; pad=0, flipped=true)
        x = reshape(x, (l, size(cp.project_filters_2, 4), 1, n))
        
        # Lightweight channel attention (no extra params!)
        attn = mean(x; dims=1)  # (1, C, 1, n) - global average pooling
        attn = Flux.sigmoid.(attn)  # Soft gating
        x = x .* attn  # Channel reweighting
    end
    
    # ========== Layer 3 ==========
    if !isnothing(cp.dw_filters_3)
        current_channels_3 = size(x, 2)
        x = reshape(x, (l, 1, current_channels_3, n))
        
        # Third depthwise conv
        pad_h_3 = (size(cp.dw_filters_3, 1) - 1) ÷ 2
        x = Flux.conv(x, cp.dw_filters_3; pad=(pad_h_3, 0), flipped=true, groups=current_channels_3)
        x = Flux.swish.(x)
        
        # Reshape and third projection
        x = reshape(x, (l, current_channels_3, 1, n))
        x = Flux.conv(x, cp.project_filters_3; pad=0, flipped=true)
        x = reshape(x, (l, size(cp.project_filters_3, 4), 1, n))
        
        # Lightweight channel attention (no extra params!)
        attn = mean(x; dims=1)  # (1, C, 1, n)
        attn = Flux.sigmoid.(attn)  # Soft gating
        x = x .* attn  # Channel reweighting
    end
    
    return x
end
