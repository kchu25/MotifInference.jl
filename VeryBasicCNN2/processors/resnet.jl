# ============================================================================
# ResNet Architecture
# ============================================================================

"""
    forward_resnet!(cp::CodeProcessor, x)

ResNet-style with depthwise conv + residual connection.
Identical to plain but residual is handled externally.
"""
function forward_resnet!(cp::CodeProcessor, x)
    # Same as plain architecture
    return forward_plain!(cp, x)
end
