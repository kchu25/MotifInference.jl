# ============================================================================
# Learnable PWM Layer (Base Layer)
# ============================================================================

""" 
    LearnedPWMs

Learnable Position Weight Matrices for the first CNN layer.

# Fields
- `filters`: 4D array (alphabet_size, motif_len, 1, num_filters)
- `mask`: Channel masking layer (optional)
- `dropout_p`: Channel dropout probability (0.0 = no dropout)

# Forward Pass
Applies PWM convolution followed by ReLU activation,
optionally with channel masking and dropout.
"""
struct LearnedPWMs
    filters::AbstractArray{DEFAULT_FLOAT_TYPE, 4}
    mask::Union{Nothing, layer_channel_mask}
    dropout_p::DEFAULT_FLOAT_TYPE
    
    function LearnedPWMs(;
        filter_width::Int,
        filter_height::Int = 4,
        num_filters::Int,
        init_scale::DEFAULT_FLOAT_TYPE = DEFAULT_FLOAT_TYPE(1e-1),
        use_channel_mask::Bool = false,
        dropout_p::DEFAULT_FLOAT_TYPE = DEFAULT_FLOAT_TYPE(0.0),
        use_cuda::Bool = false,
        rng = Random.GLOBAL_RNG
    )
        # Initialize with small random values
        filters = init_scale .* randn(rng, DEFAULT_FLOAT_TYPE, 
                                      (filter_height, filter_width, 1, num_filters))
        
        # Initialize channel mask if requested (uses defaults)
        if use_channel_mask
            mask = layer_channel_mask(num_filters; rng=rng)
        else
            mask = nothing
        end

        if use_cuda
            filters = cu(filters)
            if !isnothing(mask)
                mask = mask |> gpu
            end
        end

        return new(filters, mask, dropout_p)
    end
    
    # Direct constructors for loading models
    LearnedPWMs(filters) = new(filters, nothing, DEFAULT_FLOAT_TYPE(0.0))
    LearnedPWMs(filters, mask) = new(filters, mask, DEFAULT_FLOAT_TYPE(0.0))
    LearnedPWMs(filters, mask, dropout_p) = new(filters, mask, dropout_p)
end

Flux.@layer LearnedPWMs

Flux.trainable(l::LearnedPWMs) = begin
    params = (filters = l.filters,)
    if !isnothing(l.mask)
        params = merge(params, (mask = l.mask,))
    end
    return params
end

"""
    prepare_pwm_params(pwms::LearnedPWMs; reverse_comp=false)

Prepare PWM parameters for forward pass.

# Returns
- `pwm_matrix`: Position weight matrix (log odds ratios)
"""
function prepare_pwm_params(pwms::LearnedPWMs; reverse_comp=false)
    return create_pwm(pwms.filters; reverse_comp=reverse_comp)
end

"""
    (pwms::LearnedPWMs)(sequences; reverse_comp=false, training::Bool=true)

Forward pass through PWM layer.

# Process
1. Create PWM from learned frequencies
2. Convolve with input sequences  
3. Apply ReLU with learned scaling
4. Optional channel masking (if enabled)

# Arguments
- `sequences`: Input sequences
- `reverse_comp`: Whether to use reverse complement
- `training`: Whether in training mode (affects masking)

# Returns
- Code activations (3D: length, filters, batch)
"""
function (pwms::LearnedPWMs)(sequences; reverse_comp=false, training::Bool=true)
    pwm = prepare_pwm_params(pwms; reverse_comp=reverse_comp)
    gradient = conv(sequences, pwm; pad=0, flipped=true)
    code = Flux.NNlib.relu.(gradient) # (1, l-pwm_len, num_filters, batch_size)
    
    # Apply channelwise dropout during training
    if training && pwms.dropout_p > DEFAULT_FLOAT_TYPE(0.0)
        # Create channel-wise dropout mask: (1, num_filters, batch_size)
        # Each filter is either kept (scaled by 1/(1-p)) or dropped (0) across all positions
        dropout_mask = @ignore_derivatives begin
            keep_prob = DEFAULT_FLOAT_TYPE(1.0) - pwms.dropout_p
            num_filters = size(code, 3)
            batch_size = size(code, 4)
            CUDA.rand(DEFAULT_FLOAT_TYPE, 1, 1, num_filters, batch_size) .< keep_prob
        end
        code = @. code * dropout_mask / (DEFAULT_FLOAT_TYPE(1.0) - pwms.dropout_p)  # Scale by 1/(1-p) to maintain expected value
    end
    
    # Apply channel mask if present
    if !isnothing(pwms.mask)
        code = pwms.mask(code; training=training)
    end
    
    return code
end
