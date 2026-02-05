# ============================================================================
# Learnable Convolutional Filters (Intermediate Layers)
# ============================================================================

"""   
    layernorm(x, gamma, beta; eps=DEFAULT_FLOAT_TYPE(1e-5))

Apply Layer Normalization.

# Arguments
- `x`: Input tensor (height, width, channels, batch)
- `gamma`: Scale parameters (per channel)
- `beta`: Shift parameters (per channel)
- `eps`: Numerical stability constant

# Returns
- Normalized and affine-transformed tensor
"""
function layernorm(x, gamma, beta; eps=DEFAULT_FLOAT_TYPE(1e-5))
    # Normalize over spatial and channel dimensions (everything except batch)
    dims = (1, 4) # spatial and batch
    x_mean = mean(x; dims=dims)
    x_var = var(x; dims=dims, corrected=false)
    x_norm = (x .- x_mean) ./ sqrt.(x_var .+ eps)
    
    # Scale and shift with learnable parameters (broadcast over spatial dims)
    gamma_r = reshape(gamma, (1, :, 1, 1))
    beta_r = reshape(beta, (1, :, 1, 1))
    return gamma_r .* x_norm .+ beta_r
end

"""   
    LearnedCodeImgFilters

Learnable convolutional filters for intermediate CNN layers.

# Fields
- `filters`: 4D array (height, width, 1, num_filters)
- `ln_gamma`: LayerNorm scale parameter (optional, per channel)
- `ln_beta`: LayerNorm shift parameter (optional, per channel)
- `mask`: Channel masking layer (optional)
- `dropout_p`: Channel dropout probability (0.0 = no dropout)

# Forward Pass
Applies normalized convolution followed by ReLU activation, optionally with LayerNorm, channel masking, and dropout.
"""
struct LearnedCodeImgFilters
    filters::AbstractArray{DEFAULT_FLOAT_TYPE, 4}
    ln_gamma::Union{Nothing, AbstractArray{DEFAULT_FLOAT_TYPE, 1}}
    ln_beta::Union{Nothing, AbstractArray{DEFAULT_FLOAT_TYPE, 1}}
    mask::Union{Nothing, layer_channel_mask}
    dropout_p::DEFAULT_FLOAT_TYPE
    
    function LearnedCodeImgFilters(;
        input_channels::Int,
        filter_height::Int,
        num_filters::Int,
        init_scale::DEFAULT_FLOAT_TYPE = DEFAULT_FLOAT_TYPE(0.01),
        use_layernorm::Bool = false,
        use_channel_mask::Bool = false,
        dropout_p::DEFAULT_FLOAT_TYPE = DEFAULT_FLOAT_TYPE(0.0),
        use_cuda::Bool = false,
        rng = Random.GLOBAL_RNG
    )
        # keep thhis for now to ensure training stability; full debug later
        init_scale = DEFAULT_FLOAT_TYPE(0.01); 
        
        filters = init_scale .* randn(rng, DEFAULT_FLOAT_TYPE,
                                      (filter_height, input_channels, 1, num_filters))

        # Initialize LayerNorm params if needed
        if use_layernorm
            ln_gamma = init_scale .* ones(DEFAULT_FLOAT_TYPE, num_filters)
            ln_beta = zeros(DEFAULT_FLOAT_TYPE, num_filters)
        else
            ln_gamma = nothing
            ln_beta = nothing
        end
        
        # Initialize channel mask if requested (uses defaults)
        if use_channel_mask
            mask = layer_channel_mask(num_filters; rng=rng)
        else
            mask = nothing
        end

        if use_cuda
            filters = cu(filters)
            ln_gamma = isnothing(ln_gamma) ? nothing : cu(ln_gamma)
            ln_beta = isnothing(ln_beta) ? nothing : cu(ln_beta)
            if !isnothing(mask)
                mask = mask |> gpu
            end
        end
        
        return new(filters, ln_gamma, ln_beta, mask, dropout_p)
    end
    
    # Direct constructors for loading models (backward compatibility)
    LearnedCodeImgFilters(filters) = new(filters, nothing, nothing, nothing, DEFAULT_FLOAT_TYPE(0.0))
    LearnedCodeImgFilters(filters, ln_gamma, ln_beta) = new(filters, ln_gamma, ln_beta, nothing, DEFAULT_FLOAT_TYPE(0.0))
    LearnedCodeImgFilters(filters, ln_gamma, ln_beta, mask) = new(filters, ln_gamma, ln_beta, mask, DEFAULT_FLOAT_TYPE(0.0))
    LearnedCodeImgFilters(filters, ln_gamma, ln_beta, mask, dropout_p) = new(filters, ln_gamma, ln_beta, mask, dropout_p)
end

Flux.@layer LearnedCodeImgFilters
Flux.trainable(l::LearnedCodeImgFilters) = begin
    params = (filters = l.filters,)
    if !isnothing(l.ln_gamma)
        params = merge(params, (ln_gamma = l.ln_gamma, ln_beta = l.ln_beta))
    end
    if !isnothing(l.mask)
        params = merge(params, (mask = l.mask,))
    end
    return params
end

"""
    prepare_conv_params(conv_filters::LearnedCodeImgFilters, hp::HyperParameters; use_sparsity=false)

Prepare convolutional filter parameters.

# Returns
- L2-normalized filters, optionally with sparsity weighting
"""
function prepare_conv_params(conv_filters::LearnedCodeImgFilters, hp::HyperParameters; 
                             use_sparsity=false)
    return normalize_filters_l2(conv_filters.filters; 
                               softmax_alpha=hp.softmax_strength_img_fil,
                               use_sparsity=use_sparsity)
end

"""
    (conv_filters::LearnedCodeImgFilters)(code_input, hp::HyperParameters; use_sparsity=false, training::Bool=true)

Forward pass through convolutional layer.

# Process
1. L2-normalize filters (optionally with sparsity)
2. Convolve with input code
3. Apply ReLU activation
4. Optional channel masking (if enabled)

# Arguments
- `code_input`: Input features
- `hp`: Hyperparameters
- `use_sparsity`: Whether to apply sparsity weighting
- `training`: Whether in training mode (affects masking)

# Returns
- Activated code (4D: height, width, filters, batch)
"""
function (conv_filters::LearnedCodeImgFilters)(code_input, hp::HyperParameters; 
                                                use_sparsity=false,
                                                training::Bool=true)
    normalized_filters = prepare_conv_params(conv_filters, hp; use_sparsity=use_sparsity)
    gradient = conv(code_input, normalized_filters; pad=0, flipped=true)
    code = Flux.NNlib.relu(gradient) # (l-filter_height, 1, num_filters, batch_size)
    
    # Apply channelwise dropout during training
    if training && conv_filters.dropout_p > DEFAULT_FLOAT_TYPE(0.0)
        # Create channel-wise dropout mask: (1, 1, num_filters, batch_size)
        # Each filter is either kept (scaled by 1/(1-p)) or dropped (0) across all positions
        dropout_mask = @ignore_derivatives begin
            keep_prob = DEFAULT_FLOAT_TYPE(1.0) - conv_filters.dropout_p
            num_filters = size(code, 3)
            batch_size = size(code, 4)
            CUDA.rand(DEFAULT_FLOAT_TYPE, 1, 1, num_filters, batch_size) .< keep_prob
        end
        code = code .* dropout_mask ./ (DEFAULT_FLOAT_TYPE(1.0) - conv_filters.dropout_p)
    end
    
    # Apply channel mask if present
    if !isnothing(conv_filters.mask)
        code = conv_filters.mask(code; training=training)
    end
    
    return code
end
