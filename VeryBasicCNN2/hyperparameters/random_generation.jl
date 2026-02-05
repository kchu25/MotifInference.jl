# ============================================================================
# Random Hyperparameter Generation
# ============================================================================

"""
    generate_random_hyperparameters(; batch_size=nothing, rng=Random.GLOBAL_RNG, ranges=DEFAULT_RANGES)

Generate randomized hyperparameters for architecture search.

# Arguments
- `batch_size`: Fixed batch size (defaults to random selection from ranges)
- `rng`: Random number generator for reproducibility
- `ranges`: HyperParamRanges defining valid parameter ranges

# Returns
- `HyperParameters` instance with randomized valid configuration
"""
function generate_random_hyperparameters(; 
    batch_size = nothing, 
    rng = Random.GLOBAL_RNG,
    ranges = DEFAULT_RANGES,
    bottleneck = false,
    bottleneck_filters = BOTTLENECK_FILTERS,
)
    n_layers = rand(rng, ranges.num_img_layers_range)
    
    # Sample architecture parameters
    pfm_len = rand(rng, ranges.pfm_length_range)
    n_base = rand(rng, ranges.num_base_filters_range)
    
    # Conv layer filters (last layer fixed for compatibility)
    filters = [rand(rng, ranges.conv_filter_range) for _ in 1:(n_layers-1)]
    push!(filters, ranges.final_layer_filters)
    
    # Input channels track previous layer outputs
    widths = vcat([n_base], filters[1:(end-1)])
    heights = [rand(rng, ranges.conv_filter_height_range) for _ in 1:n_layers]
    
    # Pooling config: respect num_no_pool_layers, always 1 on final layer
    pools = Vector{Int}(undef, n_layers)
    strides = Vector{Int}(undef, n_layers)
    
    for i in 1:n_layers
        if i ≤ ranges.num_no_pool_layers || i == n_layers
            pools[i] = 1
            strides[i] = 1
        else
            pools[i] = rand(rng, ranges.pool_size_range)
            strides[i] = rand(rng, ranges.stride_range)
        end
    end
    
    batch = isnothing(batch_size) ? rand(rng, ranges.batch_size_options) : batch_size
    infer_layer = ranges.infer_base_layer_code ? 0 : ranges.num_no_pool_layers
    
    # Apply bottleneck if specified
    if bottleneck
        filters[infer_layer] = bottleneck_filters;
        heights[infer_layer] = 25;
        widths[infer_layer+1] = bottleneck_filters;
    end

    # MBConv configuration
    n_mbconv = rand(rng, ranges.num_mbconv_range)
    mbconv_exp = rand(rng, ranges.mbconv_expansion_options)

    return HyperParameters(
        pfm_len = pfm_len,
        num_pfms = n_base,
        num_img_filters = filters,
        img_fil_widths = widths,
        img_fil_heights = heights,
        pool_base = ranges.base_pool_size,
        stride_base = ranges.base_stride,
        poolsize = pools,
        stride = strides,
        pool_lvl_top = n_layers - 1,
        softmax_strength_img_fil = ranges.softmax_alpha,
        batch_size = batch,
        inference_code_layer = infer_layer,
        use_layernorm = false,  # Default: no LayerNorm
        num_mbconv = n_mbconv,
        mbconv_expansion = mbconv_exp
    )
end
