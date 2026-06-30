# ────────────────────────────────────────────────────────────────────────────
# Main SeqCNN Model
# ────────────────────────────────────────────────────────────────────────────

"""
    SeqCNN

Convolutional neural network for biological sequence analysis.

# Architecture
1. **Base Layer**: Learnable PWMs for motif detection
2. **Conv Layers**: Hierarchical feature extraction with pooling
3. **MBConv Blocks** (optional): EfficientNet-style refinement
4. **Output Layer**: Linear transformation to predictions

# Fields
- `hp::HyperParameters`: Architecture configuration
- `pwms::LearnedPWMs`: Base layer PWM filters
- `conv_layers::Vector{LearnedCodeImgFilters}`: Convolutional layers
- `mbconv_blocks::Vector{MBConvBlock}`: Optional MBConv refinement blocks
- `output_weights::Array{Float32,3}`: Final linear layer (output_dim × embed_dim × 1)

# Constructor
    SeqCNN(hp, input_dims, output_dim; init_scale=0.5, use_cuda=true, rng=Random.GLOBAL_RNG)

# Arguments
- `hp`: HyperParameters specifying architecture
- `input_dims`: Tuple (alphabet_size, sequence_length)
- `output_dim`: Number of output targets
- `init_scale`: Weight initialization scale
- `use_cuda`: Whether to place on GPU
- `rng`: Random number generator

# Example
```julia
hp = generate_random_hyperparameters()
model = SeqCNN(hp, (4, 41), 244)  # DNA sequences, 244 outputs
predictions = model(sequences)
```
"""
struct SeqCNN
    hp::HyperParameters
    pwms::LearnedPWMs
    conv_layers::Vector{LearnedCodeImgFilters}
    mbconv_blocks::Vector{MBConvBlock}
    output_weights::AbstractArray{DEFAULT_FLOAT_TYPE, 3}
    final_nonlinearity::Function
    training::Ref{Bool}  # Mutable flag for training mode
    
    function SeqCNN(
        hp::HyperParameters,
        input_dims::Tuple{T, T},
        output_dim::T;
        init_scale = DEFAULT_FLOAT_TYPE(5e-1), 
        pwm_dropout_p = DEFAULT_FLOAT_TYPE(0.1),
        conv_dropout_p = DEFAULT_FLOAT_TYPE(0.0),
        final_nonlinearity = identity,
        use_cuda = true,
        rng = Random.GLOBAL_RNG
    ) where T <: Integer
        
        alphabet_size, seq_len = input_dims
        
        # Base PWM layer
        pwms = LearnedPWMs(;
            filter_width = hp.pfm_len,
            filter_height = alphabet_size,
            num_filters = hp.num_pfms,
            init_scale = init_scale,
            use_channel_mask = hp.use_channel_mask,
            dropout_p = pwm_dropout_p,
            use_cuda = use_cuda,
            rng = rng
        )

        # Convolutional layers (with LayerNorm for layers > inference_code_layer if enabled)
        conv_layers = [
            LearnedCodeImgFilters(; 
                input_channels = hp.img_fil_widths[i],
                filter_height = hp.img_fil_heights[i],
                num_filters = hp.num_img_filters[i],
                # init_scale = init_scale,
                use_layernorm = (hp.use_layernorm && i > hp.inference_code_layer),
                use_channel_mask = hp.use_channel_mask,
                dropout_p = conv_dropout_p,
                use_cuda = use_cuda,
                rng = rng
            ) for i in 1:num_layers(hp)
        ]
        
        # Report LayerNorm usage
        if hp.use_layernorm
            ln_count = sum(i > hp.inference_code_layer for i in 1:num_layers(hp))
            @debug "Using LayerNorm for $ln_count conv layers (layers > $(hp.inference_code_layer))"
        end

        # Optional MBConv blocks (EfficientNet-style refinement)
        mbconv_blocks = MBConvBlock[]
        if hp.num_mbconv > 0
            @debug "Using MBConv refinement: $(hp.num_mbconv) blocks with $(hp.mbconv_expansion)x expansion (EfficientNet-style)"
            final_channels = hp.num_img_filters[end]
            for _ in 1:hp.num_mbconv
                push!(mbconv_blocks, MBConvBlock(;
                    in_channels = final_channels,
                    out_channels = final_channels,
                    kernel_size = 3,
                    expansion_ratio = hp.mbconv_expansion,
                    use_se = true,
                    use_cuda = use_cuda,
                    rng = rng
                ))
            end
        end

        # Output layer
        final_len = final_conv_embedding_length(hp, seq_len)
        embed_dim = final_len * hp.num_img_filters[end]

        output_weights = randn(rng, DEFAULT_FLOAT_TYPE, (output_dim, embed_dim, 1))
        if use_cuda && (output_dim * embed_dim > 1)
            output_weights = cu(output_weights)
        end
        # Create model and report parameter count
        model = new(hp, pwms, conv_layers, mbconv_blocks, output_weights, final_nonlinearity, Ref(true))
        @debug "Model created with $(count_parameters(model)) trainable parameters"

        return model
    end
    
    # Direct constructors for model loading/conversion (backward compatibility)
    SeqCNN(hp, pwms, conv_layers, mbconv_blocks, output_weights, final_nonlinearity=identity) = 
        new(hp, pwms, conv_layers, mbconv_blocks, output_weights, final_nonlinearity, Ref(true))
    
    SeqCNN(hp, pwms, conv_layers, mbconv_blocks, output_weights, final_nonlinearity, training) = 
        new(hp, pwms, conv_layers, mbconv_blocks, output_weights, final_nonlinearity, training)
end

Flux.@layer SeqCNN

# Specify trainable parameters
Flux.trainable(m::SeqCNN) = (
    pwms = m.pwms, 
    conv_layers = m.conv_layers,
    mbconv_blocks = m.mbconv_blocks,
    output_weights = m.output_weights
)

# Training mode control
"""
    train!(model::SeqCNN)

Set model to training mode (uses soft Gumbel-Softmax masks).
"""
train!(m::SeqCNN) = (m.training[] = true; m)

"""
    eval!(model::SeqCNN)

Set model to evaluation mode (uses hard binary masks).
"""
eval!(m::SeqCNN) = (m.training[] = false; m)

"""
    is_training(model::SeqCNN)

Check if model is in training mode.
"""
is_training(m::SeqCNN) = m.training[]

# ────────────────────────────────────────────────────────────────────────────
# Model Parameter Counting
# ────────────────────────────────────────────────────────────────────────────

"""
    count_parameters(model::SeqCNN)

Count total trainable parameters in the model.

# Returns
- Total number of trainable parameters

# Example
```julia
model = SeqCNN(hp, (4, 41), 244)
n_params = count_parameters(model)
println("Model has \$n_params trainable parameters")
```
"""
function count_parameters(model::SeqCNN)
    total = 0
    
    # PWM parameters
    total += length(model.pwms.filters)
    if !isnothing(model.pwms.mask)
        total += length(model.pwms.mask.mixing_filter)
    end
    
    # Conv layer parameters
    for layer in model.conv_layers
        total += length(layer.filters)
        if !isnothing(layer.ln_gamma)
            total += length(layer.ln_gamma) + length(layer.ln_beta)
        end
        if !isnothing(layer.mask)
            total += length(layer.mask.mixing_filter)
        end
    end
    
    # MBConv parameters
    for block in model.mbconv_blocks
        total += length(block.expand_filters)
        total += length(block.dw_filters)
        total += length(block.se_w1) + length(block.se_w2)
        total += length(block.project_filters)
    end
    
    # Output layer
    total += length(model.output_weights)
    
    return total
end

"""
    print_model_summary(model::SeqCNN)

Print a summary of model architecture and parameter count.

# Example
```julia
model = SeqCNN(hp, (4, 41), 244)
print_model_summary(model)
```
"""
function print_model_summary(model::SeqCNN)
    println("="^60)
    println("SeqCNN Model Summary")
    println("="^60)
    
    # Architecture
    println("\n📐 Architecture:")
    println("  Base Layer: $(model.hp.num_pfms) PWMs × $(model.hp.pfm_len)nt")
    println("  Conv Layers: $(model.num_conv_layers)")
    println("  MBConv Blocks: $(length(model.mbconv_blocks))")
    
    # Parameter counts by component
    println("\n🔢 Parameters:")
    
    # PWM
    pwm_params = length(model.pwms.filters)
    if !isnothing(model.pwms.mask)
        pwm_params += length(model.pwms.mask.mixing_filter)
    end
    println("  PWM Layer: $(pwm_params)")
    
    # Conv layers
    conv_params = sum(length(layer.filters) for layer in model.conv_layers)
    ln_count = 0
    for layer in model.conv_layers
        if !isnothing(layer.ln_gamma)
            conv_params += length(layer.ln_gamma) + length(layer.ln_beta)
            ln_count += 1
        end
        if !isnothing(layer.mask)
            conv_params += length(layer.mask.mixing_filter)
        end
    end
    println("  Conv Layers: $(conv_params)")
    if ln_count > 0
        println("    ($(ln_count) layer$(ln_count > 1 ? "s" : "") with LayerNorm)")
    end
    
    # MBConv
    mbconv_params = 0
    for block in model.mbconv_blocks
        mbconv_params += length(block.expand_filters) + length(block.dw_filters)
        mbconv_params += length(block.se_w1) + length(block.se_w2)
        mbconv_params += length(block.project_filters)
    end
    if mbconv_params > 0
        println("  MBConv Blocks: $(mbconv_params)")
    end
    
    # Output
    output_params = length(model.output_weights)
    println("  Output Layer: $(output_params)")
    
    total = count_parameters(model)
    println("\n  ✓ Total: $(total) parameters")
    
    # Training mode
    mode = model.training[] ? "Training" : "Evaluation"
    println("\n🎯 Mode: $(mode)")
    
    # Optional features
    features = String[]
    if model.hp.use_layernorm
        push!(features, "LayerNorm")
    end
    if model.hp.use_channel_mask
        push!(features, "Channel Masking")
    end
    if length(model.mbconv_blocks) > 0
        push!(features, "MBConv")
    end
    if !isempty(features)
        println("⚡ Features: ", join(features, ", "))
    end
    
    println("="^60)
end

# ────────────────────────────────────────────────────────────────────────────
# Model Properties (via getproperty)
# ────────────────────────────────────────────────────────────────────────────

"""
Custom property access for convenient model introspection and utilities.

# Virtual Properties
- `model.num_conv_layers`: Number of convolutional layers
- `model.receptive_field`: Receptive field at inference layer
- `model.code`: Function to extract code at inference layer
- `model.first_layer_code`: Function to extract base PWM code
- `model.linear_sum`: Function for linear output sum (optimization)
"""
function Base.getproperty(m::SeqCNN, sym::Symbol)
    if sym === :num_conv_layers
        return num_layers(m.hp)
    elseif sym === :receptive_field
        return receptive_field(m.hp)
    elseif sym === :code
        # Extract code at inference layer
        return x -> compute_code_at_layer(m, x, m.hp.inference_code_layer)
    elseif sym === :first_layer_code
        # Extract base PWM code
        return x -> m.pwms(x)
    elseif sym === :linear_sum
        # Linear sum for optimization (no nonlinearity)
        return (x; predict_position) -> sum(
            predict_from_sequences(m, x; 
                predict_position=predict_position, 
                apply_nonlinearity=false)
        )
    elseif sym === :predict_up_to_final_nonlinearity
        return  (x; kwargs...) -> VeryBasicCNN2.predict_from_code(m, x; 
            layer = m.hp.inference_code_layer, 
            apply_nonlinearity=false, kwargs...)
    else
        return getfield(m, sym)
    end
end

function predict_up_to_final_nonlinearity(m, code; predict_position=nothing)
    VeryBasicCNN2.predict_from_code(m, code; 
        layer=m.hp.inference_code_layer,
        apply_nonlinearity=false,
        predict_position=predict_position)
end

# ────────────────────────────────────────────────────────────────────────────
# Model Construction Helpers
# ────────────────────────────────────────────────────────────────────────────

"""
    create_model(input_dims, output_dim, batch_size; rng=Random.GLOBAL_RNG, use_cuda=true, ranges=DEFAULT_RANGES)

Create a SeqCNN with randomly generated hyperparameters.

# Arguments
- `input_dims`: Tuple (alphabet_size, sequence_length)
- `output_dim`: Number of output targets
- `batch_size`: Training batch size
- `rng`: Random number generator
- `use_cuda`: Whether to use GPU
- `ranges`: HyperParamRanges for architecture sampling

# Returns
- SeqCNN instance, or `nothing` if architecture is invalid

# Examples
```julia
# DNA sequences
model = create_model((4, 41), 244, 128; ranges=nucleotide_ranges())

# Protein sequences
model = create_model((20, 100), 1, 64; ranges=amino_acid_ranges())
```
"""
function create_model(input_dims, output_dim, batch_size::Int; 
                     rng = Random.GLOBAL_RNG, 
                     use_cuda::Bool = true,
                     ranges = DEFAULT_RANGES, 
                     bottleneck = false)
    hp = generate_random_hyperparameters(; 
        batch_size=batch_size, rng=rng, ranges=ranges, bottleneck=bottleneck)
    
    # Enable LayerNorm and MBConv by default 
    # disable these two if needed for specific experiments
    # hp = with_layernorm(hp, true)
    hp = with_mbconv(hp; num_blocks=3, expansion=3)
    
    # Validate architecture
    if final_conv_embedding_length(hp, input_dims[2]) < 1
        @warn "Invalid architecture: final embedding length < 1"
        return nothing
    end
    
    return SeqCNN(hp, input_dims, output_dim; 
                  final_nonlinearity=ranges.final_nonlinearity,
                  use_cuda=use_cuda, 
                  rng=rng)
end

# ────────────────────────────────────────────────────────────────────────────
# Domain / Variant Tag Types for Multiple Dispatch
# ────────────────────────────────────────────────────────────────────────────

"""
Lightweight tag types used with `create_model` for domain-specific construction.
Combine them to select architecture variants:

```julia
create_model(Nucleotides(), input_dims, output_dim, batch_size)
create_model(Nucleotides(), FixedPoolStride(), input_dims, output_dim, batch_size)
create_model(AminoAcids(), FixedPoolStride(), Bottleneck(), input_dims, output_dim, bs)
```
"""
struct Nucleotides    end
struct AminoAcids     end
struct Simple         end
struct FixedPoolStride end
struct Multioutputs   end
struct UseTanh        end
struct UseSigmoid     end
struct Bottleneck     end

# ---- model_ranges: map tag combos → HyperParamRanges ---

# Base domains
model_ranges(::Nucleotides)                                  = nucleotide_ranges()
model_ranges(::AminoAcids)                                   = amino_acid_ranges()

# Variants
model_ranges(::Nucleotides, ::Simple)                        = nucleotide_ranges_simple()
model_ranges(::Nucleotides, ::FixedPoolStride)               = nucleotide_ranges_fixed_pool_stride()
model_ranges(::Nucleotides, ::FixedPoolStride, ::Multioutputs) = nucleotide_ranges_fixed_pool_stride_multioutputs()
model_ranges(::AminoAcids,  ::FixedPoolStride)               = amino_acid_ranges_fixed_pool_stride()

# Tanh variants
model_ranges(::Nucleotides, ::UseTanh)                       = nucleotide_ranges_tanh()
model_ranges(::Nucleotides, ::Simple, ::UseTanh)             = nucleotide_ranges_simple_tanh()
model_ranges(::Nucleotides, ::FixedPoolStride, ::UseTanh)    = nucleotide_ranges_fixed_pool_stride_tanh()
model_ranges(::Nucleotides, ::FixedPoolStride, ::Multioutputs, ::UseTanh) = nucleotide_ranges_fixed_pool_stride_multioutputs_tanh()
model_ranges(::AminoAcids,  ::UseTanh)                       = amino_acid_ranges_tanh()
model_ranges(::AminoAcids,  ::FixedPoolStride, ::UseTanh)    = amino_acid_ranges_fixed_pool_stride_tanh()

# Sigmoid variants
model_ranges(::Nucleotides, ::UseSigmoid)                       = nucleotide_ranges_sigmoid()
model_ranges(::Nucleotides, ::Simple, ::UseSigmoid)             = nucleotide_ranges_simple_sigmoid()
model_ranges(::Nucleotides, ::FixedPoolStride, ::UseSigmoid)    = nucleotide_ranges_fixed_pool_stride_sigmoid()
model_ranges(::Nucleotides, ::FixedPoolStride, ::Multioutputs, ::UseSigmoid) = nucleotide_ranges_fixed_pool_stride_multioutputs_sigmoid()
model_ranges(::AminoAcids,  ::UseSigmoid)                       = amino_acid_ranges_sigmoid()
model_ranges(::AminoAcids,  ::FixedPoolStride, ::UseSigmoid)    = amino_acid_ranges_fixed_pool_stride_sigmoid()

# ---- Dispatched create_model with tags ---

"""
    create_model(tag, input_dims, output_dim, batch_size; kwargs...)
    create_model(tag1, tag2, input_dims, output_dim, batch_size; kwargs...)
    create_model(tag1, tag2, tag3, input_dims, output_dim, batch_size; kwargs...)

Create a `SeqCNN` by dispatching on domain/variant tag types.

# Examples
```julia
create_model(Nucleotides(), (4, 41), 244, 128)
create_model(AminoAcids(), FixedPoolStride(), (20, 100), 1, 64)
create_model(AminoAcids(), FixedPoolStride(), Bottleneck(), (20, 100), 1, 64)
```
"""
# Helper to extract bottleneck flag and dispatch model_ranges
function _create_model_with_tags(tags::Tuple, input_dims, output_dim, batch_size::Int; kwargs...)
    use_bottleneck = any(t -> t isa Bottleneck, tags)
    range_tags     = filter(t -> !(t isa Bottleneck), tags)
    ranges         = model_ranges(range_tags...)
    return create_model(input_dims, output_dim, batch_size;
                        ranges=ranges, bottleneck=use_bottleneck, kwargs...)
end

# Dispatch for 1-4 tags (covers all current use cases)
const TagType = Union{Nucleotides,AminoAcids,Simple,FixedPoolStride,Multioutputs,UseTanh,UseSigmoid,Bottleneck}

create_model(t1::TagType, input_dims, output_dim, batch_size::Int; kwargs...) = 
    _create_model_with_tags((t1,), input_dims, output_dim, batch_size; kwargs...)

create_model(t1::TagType, t2::TagType, input_dims, output_dim, batch_size::Int; kwargs...) = 
    _create_model_with_tags((t1, t2), input_dims, output_dim, batch_size; kwargs...)

create_model(t1::TagType, t2::TagType, t3::TagType, input_dims, output_dim, batch_size::Int; kwargs...) = 
    _create_model_with_tags((t1, t2, t3), input_dims, output_dim, batch_size; kwargs...)

create_model(t1::TagType, t2::TagType, t3::TagType, t4::TagType, input_dims, output_dim, batch_size::Int; kwargs...) = 
    _create_model_with_tags((t1, t2, t3, t4), input_dims, output_dim, batch_size; kwargs...)

# ────────────────────────────────────────────────────────────────────────────
# Backward-Compatible Convenience Constructors
# (Thin wrappers around the dispatched create_model — safe to call from
#  existing code; new code should prefer the tag-based API above.)
# ────────────────────────────────────────────────────────────────────────────

create_model_nucleotides(args...; kw...) =
    create_model(Nucleotides(), args...; kw...)

create_model_nucleotides_simple(args...; kw...) =
    create_model(Nucleotides(), Simple(), args...; kw...)

create_model_nucleotides_fixed_pool_stride(args...; kw...) =
    create_model(Nucleotides(), FixedPoolStride(), args...; kw...)

create_model_aminoacids(args...; kw...) =
    create_model(AminoAcids(), args...; kw...)

create_model_aminoacids_fixed_pool_stride(args...; kw...) =
    create_model(AminoAcids(), FixedPoolStride(), args...; kw...)

create_model_aminoacids_fixed_pool_stride_w_bottleneck(args...; kw...) =
    create_model(AminoAcids(), FixedPoolStride(), Bottleneck(), args...; kw...)

# Tanh variants
create_model_nucleotides_tanh(args...; kw...) =
    create_model(Nucleotides(), UseTanh(), args...; kw...)

create_model_nucleotides_simple_tanh(args...; kw...) =
    create_model(Nucleotides(), Simple(), UseTanh(), args...; kw...)

create_model_nucleotides_fixed_pool_stride_tanh(args...; kw...) =
    create_model(Nucleotides(), FixedPoolStride(), UseTanh(), args...; kw...)

create_model_nucleotides_fixed_pool_stride_multioutputs(args...; kw...) =
    create_model(Nucleotides(), FixedPoolStride(), Multioutputs(), args...; kw...)

create_model_nucleotides_fixed_pool_stride_multioutputs_tanh(args...; kw...) =
    create_model(Nucleotides(), FixedPoolStride(), Multioutputs(), UseTanh(), args...; kw...)

create_model_aminoacids_tanh(args...; kw...) =
    create_model(AminoAcids(), UseTanh(), args...; kw...)

create_model_aminoacids_fixed_pool_stride_tanh(args...; kw...) =
    create_model(AminoAcids(), FixedPoolStride(), UseTanh(), args...; kw...)

# Sigmoid variants
create_model_nucleotides_sigmoid(args...; kw...) =
    create_model(Nucleotides(), UseSigmoid(), args...; kw...)

create_model_nucleotides_simple_sigmoid(args...; kw...) =
    create_model(Nucleotides(), Simple(), UseSigmoid(), args...; kw...)

create_model_nucleotides_fixed_pool_stride_sigmoid(args...; kw...) =
    create_model(Nucleotides(), FixedPoolStride(), UseSigmoid(), args...; kw...)

create_model_nucleotides_fixed_pool_stride_multioutputs_sigmoid(args...; kw...) =
    create_model(Nucleotides(), FixedPoolStride(), Multioutputs(), UseSigmoid(), args...; kw...)

create_model_aminoacids_sigmoid(args...; kw...) =
    create_model(AminoAcids(), UseSigmoid(), args...; kw...)

create_model_aminoacids_fixed_pool_stride_sigmoid(args...; kw...) =
    create_model(AminoAcids(), FixedPoolStride(), UseSigmoid(), args...; kw...)
