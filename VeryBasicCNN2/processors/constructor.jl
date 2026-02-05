# ============================================================================
# CodeProcessor Constructor
# ============================================================================

"""
    CodeProcessor(; kwargs...)

Construct a CodeProcessor with specified architecture.

# Arguments
- `in_channels::Int`: Input channels (code + gradient concatenated)
- `out_channels::Int`: Output channels (same as code)
- `kernel_size::Int=3`: Depthwise conv kernel size
- `expansion_ratio::Int=2`: Channel expansion for mbconv
- `se_ratio::Float=8.0`: SE reduction ratio for mbconv
- `use_se::Bool=true`: Enable SE attention for mbconv
- `use_hard_mask::Bool=false`: Enable Gumbel-Softmax masking
- `mask_temp::Float=0.5`: Initial temperature for Gumbel-Softmax
- `mask_eta::Float=1.0`: Right stretch parameter
- `mask_gamma::Float=0.0`: Left stretch parameter
- `arch_type::CodeProcessorType=mbconv`: Architecture type
- `use_cuda::Bool=false`: Use GPU
- `rng`: Random number generator

# Returns
- CodeProcessor instance
"""
function CodeProcessor(;
    in_channels::Int,
    out_channels::Int,
    kernel_size::Int = 3,
    expansion_ratio::Int = 2,
    se_ratio::DEFAULT_FLOAT_TYPE = DEFAULT_FLOAT_TYPE(8),
    use_se::Bool = true,
    use_hard_mask::Bool = false,
    mask_temp::DEFAULT_FLOAT_TYPE = DEFAULT_FLOAT_TYPE(0.5),
    mask_eta::DEFAULT_FLOAT_TYPE = DEFAULT_FLOAT_TYPE(1.0),
    mask_gamma::DEFAULT_FLOAT_TYPE = DEFAULT_FLOAT_TYPE(0.0),
    arch_type::CodeProcessorType = mbconv,
    use_cuda::Bool = false,
    rng = Random.GLOBAL_RNG
)
    init_scale = DEFAULT_FLOAT_TYPE(0.01)
    
    # Initialize mask projections if requested
    if use_hard_mask
        # Component-wise mask: 1×1 conv
        mask_proj = init_scale .* randn(rng, DEFAULT_FLOAT_TYPE,
                                       (1, out_channels, 1, out_channels))
        # Channel-wise mask: batched matmul
        channel_mask_proj = init_scale .* randn(rng, DEFAULT_FLOAT_TYPE,
                                                (out_channels, out_channels, 1))
    else
        mask_proj = nothing
        channel_mask_proj = nothing
    end
    
    # Architecture-specific initialization
    if arch_type == deep_plain
        result = init_deep_plain(in_channels, out_channels, kernel_size, init_scale, rng)
        expand_filters, dw_filters, se_w1, se_w2, project_filters,
        dw_filters_2, project_filters_2, dw_filters_3, project_filters_3, use_residual = result
    elseif arch_type == mbconv
        result = init_mbconv(in_channels, out_channels, kernel_size, expansion_ratio, se_ratio, use_se, init_scale, rng)
        expand_filters, dw_filters, se_w1, se_w2, project_filters, use_residual = result
        dw_filters_2 = project_filters_2 = dw_filters_3 = project_filters_3 = nothing
    elseif arch_type == resnet
        result = init_resnet(in_channels, out_channels, kernel_size, init_scale, rng)
        expand_filters, dw_filters, se_w1, se_w2, project_filters, use_residual = result
        dw_filters_2 = project_filters_2 = dw_filters_3 = project_filters_3 = nothing
    else  # plain
        result = init_plain(in_channels, out_channels, kernel_size, init_scale, rng)
        expand_filters, dw_filters, se_w1, se_w2, project_filters, use_residual = result
        dw_filters_2 = project_filters_2 = dw_filters_3 = project_filters_3 = nothing
    end
    
    # Move to GPU if requested
    if use_cuda
        expand_filters = isnothing(expand_filters) ? nothing : cu(expand_filters)
        dw_filters = cu(dw_filters)
        se_w1 = isnothing(se_w1) ? nothing : cu(se_w1)
        se_w2 = isnothing(se_w2) ? nothing : cu(se_w2)
        project_filters = cu(project_filters)
        dw_filters_2 = isnothing(dw_filters_2) ? nothing : cu(dw_filters_2)
        project_filters_2 = isnothing(project_filters_2) ? nothing : cu(project_filters_2)
        dw_filters_3 = isnothing(dw_filters_3) ? nothing : cu(dw_filters_3)
        project_filters_3 = isnothing(project_filters_3) ? nothing : cu(project_filters_3)
        mask_proj = isnothing(mask_proj) ? nothing : cu(mask_proj)
        channel_mask_proj = isnothing(channel_mask_proj) ? nothing : cu(channel_mask_proj)
    end
    
    # Count and print parameters
    num_params = count_parameters(expand_filters, dw_filters, se_w1, se_w2, project_filters,
                                  dw_filters_2, project_filters_2, dw_filters_3, project_filters_3,
                                  mask_proj, channel_mask_proj)
    
    println("CodeProcessor ($arch_type): $num_params parameters")
    println("  - in_channels: $in_channels, out_channels: $out_channels")
    if arch_type == mbconv
        println("  - expansion_ratio: $expansion_ratio, se_ratio: $se_ratio")
    end
    
    return CodeProcessor(expand_filters, dw_filters, se_w1, se_w2, project_filters,
                        dw_filters_2, project_filters_2, dw_filters_3, project_filters_3,
                        mask_proj, channel_mask_proj, mask_temp, mask_eta, mask_gamma,
                        use_hard_mask, use_residual, arch_type, Ref(true))
end

# Helper functions for architecture-specific initialization
function init_plain(in_channels, out_channels, kernel_size, init_scale, rng)
    dw_filters = init_scale .* randn(rng, DEFAULT_FLOAT_TYPE,
                                     (kernel_size, 1, 1, in_channels))
    project_filters = init_scale .* randn(rng, DEFAULT_FLOAT_TYPE,
                                         (1, in_channels, 1, out_channels))
    return nothing, dw_filters, nothing, nothing, project_filters, false
end

function init_resnet(in_channels, out_channels, kernel_size, init_scale, rng)
    # Same as plain but with residual
    expand_filters, dw_filters, se_w1, se_w2, project_filters, _ = 
        init_plain(in_channels, out_channels, kernel_size, init_scale, rng)
    return expand_filters, dw_filters, se_w1, se_w2, project_filters, true
end

function init_mbconv(in_channels, out_channels, kernel_size, expansion_ratio, se_ratio, use_se, init_scale, rng)
    expanded = in_channels * expansion_ratio
    
    # Expansion
    expand_filters = init_scale .* randn(rng, DEFAULT_FLOAT_TYPE, 
                                         (1, in_channels, 1, expanded))
    
    # Depthwise
    dw_filters = init_scale .* randn(rng, DEFAULT_FLOAT_TYPE,
                                     (kernel_size, 1, 1, expanded))
    
    # SE (if enabled)
    if use_se
        se_channels = max(1, floor(Int, expanded / se_ratio))
        se_w1 = init_scale .* randn(rng, DEFAULT_FLOAT_TYPE, (se_channels, expanded, 1))
        se_w2 = init_scale .* randn(rng, DEFAULT_FLOAT_TYPE, (expanded, se_channels, 1))
    else
        se_w1 = nothing
        se_w2 = nothing
    end
    
    # Projection
    project_filters = init_scale .* randn(rng, DEFAULT_FLOAT_TYPE,
                                         (1, expanded, 1, out_channels))
    
    return expand_filters, dw_filters, se_w1, se_w2, project_filters, true
end

function init_deep_plain(in_channels, out_channels, kernel_size, init_scale, rng)
    # Layer 1: 2C → C
    dw_filters = init_scale .* randn(rng, DEFAULT_FLOAT_TYPE,
                                     (kernel_size, 1, 1, in_channels))
    project_filters = init_scale .* randn(rng, DEFAULT_FLOAT_TYPE,
                                         (1, in_channels, 1, out_channels))
    
    # Layer 2: C → C
    dw_filters_2 = init_scale .* randn(rng, DEFAULT_FLOAT_TYPE,
                                       (kernel_size, 1, 1, out_channels))
    project_filters_2 = init_scale .* randn(rng, DEFAULT_FLOAT_TYPE,
                                           (1, out_channels, 1, out_channels))
    
    # Layer 3: C → C
    dw_filters_3 = init_scale .* randn(rng, DEFAULT_FLOAT_TYPE,
                                       (kernel_size, 1, 1, out_channels))
    project_filters_3 = init_scale .* randn(rng, DEFAULT_FLOAT_TYPE,
                                           (1, out_channels, 1, out_channels))
    
    return nothing, dw_filters, nothing, nothing, project_filters,
           dw_filters_2, project_filters_2, dw_filters_3, project_filters_3, true
end

function count_parameters(expand_filters, dw_filters, se_w1, se_w2, project_filters,
                         dw_filters_2, project_filters_2, dw_filters_3, project_filters_3,
                         mask_proj, channel_mask_proj)
    num_params = 0
    num_params += isnothing(expand_filters) ? 0 : length(expand_filters)
    num_params += length(dw_filters)
    num_params += isnothing(se_w1) ? 0 : length(se_w1)
    num_params += isnothing(se_w2) ? 0 : length(se_w2)
    num_params += length(project_filters)
    num_params += isnothing(dw_filters_2) ? 0 : length(dw_filters_2)
    num_params += isnothing(project_filters_2) ? 0 : length(project_filters_2)
    num_params += isnothing(dw_filters_3) ? 0 : length(dw_filters_3)
    num_params += isnothing(project_filters_3) ? 0 : length(project_filters_3)
    num_params += isnothing(mask_proj) ? 0 : length(mask_proj)
    num_params += isnothing(channel_mask_proj) ? 0 : length(channel_mask_proj)
    return num_params
end
