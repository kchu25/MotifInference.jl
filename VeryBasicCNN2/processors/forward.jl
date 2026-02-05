# ============================================================================
# CodeProcessor Forward Pass (Dispatcher)
# ============================================================================

"""
    (cp::CodeProcessor)(code, gradient; step::Union{Nothing, Int}=nothing)

Forward pass with separate code and gradient inputs.

# Arguments
- `code`: Code features (spatial, channels, 1, batch)
- `gradient`: Gradient features (spatial, channels, 1, batch)
- `step`: Training step for temperature annealing

# Returns
- Processed features

# Note
Use `train!(processor)` or `eval!(processor)` to set training mode.
"""
function (cp::CodeProcessor)(code, gradient; step::Union{Nothing, Int}=nothing)
    combined = cat(code, gradient; dims=2)
    return cp(combined; step=step)
end

"""
    (cp::CodeProcessor)(x; step::Union{Nothing, Int}=nothing)

Forward pass through code processor - dispatches to architecture-specific implementation.

# Process
1. Extract identity for residual (if needed)
2. Forward through architecture-specific layers
3. Add residual connection (if enabled)
4. Apply hard mask (if enabled)

# Arguments
- `x`: Input tensor (spatial, 2×channels, 1, batch) - pre-concatenated code+gradient
- `step`: Training step for temperature annealing

# Returns
- Processed features

# Note
Use `train!(processor)` or `eval!(processor)` to set training mode.
"""
function (cp::CodeProcessor)(x; step::Union{Nothing, Int}=nothing)
    l, M, _, n = size(x)
    training = cp.training[]  # Use internal training flag
    
    # For residual, extract gradient portion
    if cp.use_residual
        out_channels = size(cp.project_filters, 4)
        identity_input = @view x[:, out_channels+1:end, :, :]
    end
    
    # Dispatch to architecture-specific forward pass
    x = if cp.arch_type == deep_plain
        forward_deep_plain!(cp, x)
    elseif cp.arch_type == mbconv
        forward_mbconv!(cp, x)
    elseif cp.arch_type == resnet
        forward_resnet!(cp, x)
    else  # plain
        forward_plain!(cp, x)
    end
    
    # Residual connection (if enabled)
    if cp.use_residual
        x = x .+ identity_input
    end
    
    # Hard mask (if enabled)
    if cp.use_hard_mask && !isnothing(cp.mask_proj)
        x = apply_gumbel_mask(cp, x; training=training, step=step)
    end
    
    return x
end

"""
    create_code_processor(hp; kwargs...)

Create a CodeProcessor network based on hyperparameters.

# Arguments
- `hp`: HyperParameters (determines input/output dimensions)
- `arch_type::CodeProcessorType=mbconv`: Architecture type
- `kernel_size::Int=3`: Depthwise convolution kernel size
- `expansion_ratio::Int=2`: Channel expansion ratio (mbconv only)
- `use_se::Bool=true`: Enable SE attention (mbconv only)
- `use_hard_mask::Bool=false`: Enable Gumbel-Softmax masking
- `mask_temp::Float=0.5`: Initial temperature
- `mask_eta::Float=1.0`: Right stretch parameter
- `mask_gamma::Float=0.0`: Left stretch parameter
- `use_cuda::Bool=true`: Use GPU
- `rng`: Random number generator

# Returns
- CodeProcessor instance

# Example
```julia
hp = generate_random_hyperparameters()

# Plain depthwise conv
proc_plain = create_code_processor(hp; arch_type=plain)

# ResNet-style with residual
proc_resnet = create_code_processor(hp; arch_type=resnet)

# MBConv-style with SE attention
proc_mbconv = create_code_processor(hp; arch_type=mbconv, expansion_ratio=4)

# Forward pass: concatenate code and gradient
# code_and_grad = cat(code, gradient; dims=2)
# output = proc(code_and_grad)
```
"""
function create_code_processor(hp;
                              arch_type::CodeProcessorType = mbconv,
                              kernel_size::Int = 3,
                              expansion_ratio::Int = 2,
                              use_se::Bool = true,
                              use_hard_mask::Bool = false,
                              mask_temp::DEFAULT_FLOAT_TYPE = DEFAULT_FLOAT_TYPE(0.5),
                              mask_eta::DEFAULT_FLOAT_TYPE = DEFAULT_FLOAT_TYPE(1.0),
                              mask_gamma::DEFAULT_FLOAT_TYPE = DEFAULT_FLOAT_TYPE(0.0),
                              use_cuda::Bool = true,
                              rng = Random.GLOBAL_RNG)
    # Get dimensions at inference code layer
    if hp.inference_code_layer == 0
        code_channels = hp.num_pfms
    else
        code_channels = hp.num_img_filters[hp.inference_code_layer]
    end
    
    # Input: code + gradient (2× channels when concatenated)
    in_channels = 2 * code_channels
    out_channels = code_channels
    
    return CodeProcessor(;
        in_channels = in_channels,
        out_channels = out_channels,
        kernel_size = kernel_size,
        expansion_ratio = expansion_ratio,
        use_se = use_se,
        use_hard_mask = use_hard_mask,
        mask_temp = mask_temp,
        mask_eta = mask_eta,
        mask_gamma = mask_gamma,
        arch_type = arch_type,
        use_cuda = use_cuda,
        rng = rng
    )
end

"""
    process_code(processor::CodeProcessor, code, gradient; training::Bool=true, step::Union{Nothing, Int}=nothing)

Process code and gradient features through the processor network.

# Arguments
- `processor`: CodeProcessor instance
- `code`: Code features at inference code layer (l, C, 1, n)
- `gradient`: Gradient features at same layer (l, C, 1, n)
- `training`: Whether in training mode
- `step`: Current training step (for temperature annealing)

# Returns
- Processed features (l, C, 1, n) - same size as code

# Example
```julia
# In training loop
for (step, batch) in enumerate(dataloader)
    output = process_code(proc, code, grad; step=step)
end
```
"""
function process_code(processor::CodeProcessor, code, gradient; step::Union{Nothing, Int}=nothing)
    # Concatenate along channel dimension
    combined = cat(code, gradient; dims=2)
    
    # Process through network (uses processor's internal training flag)
    return processor(combined; step=step)
end
