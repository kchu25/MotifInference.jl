# ============================================================================
# Plain Architecture
# ============================================================================

"""
    forward_plain!(cp::CodeProcessor, x)

Plain depthwise convolution (no expansion, no SE, optional residual).
"""
function forward_plain!(cp::CodeProcessor, x)
    l, M, _, n = size(x)
    current_channels = M
    
    # Reshape for depthwise conv
    x = reshape(x, (l, 1, current_channels, n))
    
    # Depthwise convolution
    pad_h = (size(cp.dw_filters, 1) - 1) ÷ 2
    x = Flux.conv(x, cp.dw_filters; pad=(pad_h, 0), flipped=true, groups=current_channels)
    x = Flux.swish.(x)
    
    # Reshape back for projection
    x = reshape(x, (l, current_channels, 1, n))
    
    # Projection
    x = Flux.conv(x, cp.project_filters; pad=0, flipped=true)
    x = reshape(x, (l, size(cp.project_filters, 4), 1, n))
    
    return x
end
