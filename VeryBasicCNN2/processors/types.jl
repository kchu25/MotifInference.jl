# ============================================================================
# Code Processor Types and Enums
# ============================================================================

"""
    CodeProcessorType

Type of architecture for code processing.
- `:plain`: Simple depthwise convolution
- `:resnet`: Depthwise conv with residual connection
- `:mbconv`: MBConv-style with expansion and SE attention
- `:deep_plain`: Stacked plain layers (3 layers for more capacity)
"""
@enum CodeProcessorType plain resnet mbconv deep_plain

"""
    CodeProcessor

Network for processing concatenated code and gradient features.

# Fields
- `expand_filters`: Channel expansion (optional, for mbconv)
- `dw_filters`: Depthwise convolution filters
- `se_w1, se_w2`: Squeeze-excitation weights (optional, for mbconv)
- `project_filters`: Channel projection back to output size
- `dw_filters_2, project_filters_2`: Second layer (deep_plain)
- `dw_filters_3, project_filters_3`: Third layer (deep_plain)
- `mask_proj`: Component-wise mask projection
- `channel_mask_proj`: Channel-wise mask projection
- `mask_temp, mask_eta, mask_gamma`: Gumbel-Softmax parameters
- `use_hard_mask`: Whether to use hard mask
- `use_residual`: Whether to use skip connection
- `arch_type`: Architecture type
- `training`: Mutable flag for training mode
"""
struct CodeProcessor
    expand_filters::Union{Nothing, AbstractArray{DEFAULT_FLOAT_TYPE, 4}}
    dw_filters::AbstractArray{DEFAULT_FLOAT_TYPE, 4}
    se_w1::Union{Nothing, AbstractArray{DEFAULT_FLOAT_TYPE, 3}}
    se_w2::Union{Nothing, AbstractArray{DEFAULT_FLOAT_TYPE, 3}}
    project_filters::AbstractArray{DEFAULT_FLOAT_TYPE, 4}
    dw_filters_2::Union{Nothing, AbstractArray{DEFAULT_FLOAT_TYPE, 4}}
    project_filters_2::Union{Nothing, AbstractArray{DEFAULT_FLOAT_TYPE, 4}}
    dw_filters_3::Union{Nothing, AbstractArray{DEFAULT_FLOAT_TYPE, 4}}
    project_filters_3::Union{Nothing, AbstractArray{DEFAULT_FLOAT_TYPE, 4}}
    mask_proj::Union{Nothing, AbstractArray{DEFAULT_FLOAT_TYPE, 4}}
    channel_mask_proj::Union{Nothing, AbstractArray{DEFAULT_FLOAT_TYPE, 3}}
    mask_temp::DEFAULT_FLOAT_TYPE
    mask_eta::DEFAULT_FLOAT_TYPE
    mask_gamma::DEFAULT_FLOAT_TYPE
    use_hard_mask::Bool
    use_residual::Bool
    arch_type::CodeProcessorType
    training::Ref{Bool}
    
    # Positional constructor for Flux/Optimisers
    CodeProcessor(expand_filters, dw_filters, se_w1, se_w2, project_filters, 
                  dw_filters_2, project_filters_2, dw_filters_3, project_filters_3,
                  mask_proj, channel_mask_proj, mask_temp, mask_eta, mask_gamma,
                  use_hard_mask, use_residual, arch_type, training=Ref(true)) = 
        new(expand_filters, dw_filters, se_w1, se_w2, project_filters, 
            dw_filters_2, project_filters_2, dw_filters_3, project_filters_3,
            mask_proj, channel_mask_proj, mask_temp, mask_eta, mask_gamma,
            use_hard_mask, use_residual, arch_type, training)
end

Flux.@layer CodeProcessor
Flux.trainable(cp::CodeProcessor) = NamedTuple(
    k => getfield(cp, k) for k in fieldnames(CodeProcessor)
    if k ∉ (:mask_temp, :mask_eta, :mask_gamma, :use_hard_mask, :use_residual, :arch_type, :training) 
    && !isnothing(getfield(cp, k))
)

# Training mode control
"""
    train!(processor::CodeProcessor)

Set processor to training mode.
"""
train!(cp::CodeProcessor) = (cp.training[] = true; cp)

"""
    eval!(processor::CodeProcessor)

Set processor to evaluation mode.
"""
eval!(cp::CodeProcessor) = (cp.training[] = false; cp)

"""
    is_training(processor::CodeProcessor)

Check if processor is in training mode.
"""
is_training(cp::CodeProcessor) = cp.training[]
