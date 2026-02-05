# ============================================================================
# Forward Pass Implementation
# ============================================================================

"""
    compute_code_at_layer(model::SeqCNN, sequences, layer; use_sparsity=false)

Compute code representation at specified layer depth.

# Arguments
- `model`: SeqCNN instance
- `sequences`: Input sequences (4D tensor)
- `layer`: Target layer (0 = base PWM, 1+ = conv layers)
- `use_sparsity`: Apply sparsity-inducing normalization

# Returns
- Code tensor at specified layer

# Examples
```julia
code_pwm = compute_code_at_layer(model, seqs, 0)      # Base PWM code
code_l1 = compute_code_at_layer(model, seqs, 1)       # After 1st conv
code_final = compute_code_at_layer(model, seqs, 3)    # After 3rd conv
```
"""
function compute_code_at_layer(model::SeqCNN, sequences, layer::Int; use_sparsity=false, training::Union{Nothing,Bool}=nothing)
    @assert 0 ≤ layer ≤ model.num_conv_layers "Layer must be 0 to $(model.num_conv_layers)"
    
    # Use model's training flag if not explicitly provided
    train_mode = isnothing(training) ? model.training[] : training

    code = model.pwms(sequences; training=train_mode)
    code = reshape_to_4d(code; is_base_layer=true)

    # Base layer only
    layer == 0 && return code
    
    # Start with base layer + pooling
    code = pool_code(code, 
        (model.hp.pool_base, 1), 
        (model.hp.stride_base, 1); skip_pooling=false)
    
    # Process conv layers recursively up to target
    return forward_conv_recursive(model, code, 1, layer; use_sparsity=use_sparsity, training=train_mode)
end

"""
    forward_conv_recursive(model, code, current_layer, target_layer; use_sparsity=false, training=true)

Recursively process convolutional layers.

# Arguments
- `model`: SeqCNN instance
- `code`: Current code representation
- `current_layer`: Current layer index (1-indexed)
- `target_layer`: Stop at this layer (nothing = process all)
- `use_sparsity`: Apply sparsity-inducing normalization
- `training`: Whether in training mode (affects channel masking)

# Returns
- Code after processing layers up to target_layer
"""
function forward_conv_recursive(model::SeqCNN, code, current_layer::Int, 
                               target_layer=nothing; use_sparsity=false, training::Bool=true)
    max_layer = isnothing(target_layer) ? model.num_conv_layers : target_layer
    
    # Base case: processed all requested layers
    current_layer > max_layer && return code
    
    # Apply convolution
    code = model.conv_layers[current_layer](code, model.hp; use_sparsity=use_sparsity, training=training)
    
    # Apply pooling (or skip if beyond pool_lvl_top)
    skip_pool = current_layer > model.hp.pool_lvl_top
    code = reshape_to_4d(code)
    code = pool_code(code,
                    (model.hp.poolsize[current_layer], 1),
                    (model.hp.stride[current_layer], 1);
                    skip_pooling=skip_pool)
    
    # Apply LayerNorm after pooling if past inference layer
    if current_layer > model.hp.inference_code_layer
        conv_layer = model.conv_layers[current_layer]
        if !isnothing(conv_layer.ln_gamma)
            code = layernorm(code, conv_layer.ln_gamma, conv_layer.ln_beta)
        end
    end
    
    # Recurse to next layer
    return forward_conv_recursive(model, code, current_layer + 1, target_layer; 
                                 use_sparsity=use_sparsity, training=training)
end

"""
    extract_features(model::SeqCNN, sequences; use_sparsity=false)

Extract CNN features from sequences (full forward pass through all conv layers).

# Process
1. Base PWM layer → pool
2. All conv layers → pool
3. Optional MBConv refinement blocks
4. Flatten to embedding vector

# Returns
- Feature embedding (embed_dim, 1, batch_size)
"""
function extract_features(model::SeqCNN, sequences; use_sparsity=false, training::Union{Nothing,Bool}=nothing)
    # Use model's training flag if not explicitly provided
    train_mode = isnothing(training) ? model.training[] : training
    
    # Base layer
    code = model.pwms(sequences; training=train_mode)
    
    # Base pooling
    code = reshape_to_4d(code; is_base_layer=true)
    code = pool_code(code, 
        (model.hp.pool_base, 1), 
        (model.hp.stride_base, 1); skip_pooling=false)

    # code = pool_code(code,
    #                 (model.hp.pool_base, 1),
    #                 (model.hp.stride_base, 1);
    #                 is_base_layer=true)
    
    # All conv layers
    code = forward_conv_recursive(model, code, 1; use_sparsity=use_sparsity, training=train_mode)
    
    # Optional MBConv refinement
    for mbconv in model.mbconv_blocks
        code = mbconv(code)
    end
    
    # Flatten to embedding
    spatial_len = size(code, 1)
    n_channels = size(code, 2)
    batch_size = size(code, 4)
    embed_dim = spatial_len * n_channels
    
    return reshape(code, (embed_dim, 1, batch_size))
end

"""
    select_output_weights(model::SeqCNN; predict_position=nothing)

Select output weights for prediction.

# Arguments
- `model`: SeqCNN instance
- `predict_position`: Specific output index (nothing = all outputs)

# Returns
- Output weight matrix or view
"""
function select_output_weights(model::SeqCNN; predict_position=nothing)
    isnothing(predict_position) && return model.output_weights
    
    @assert 1 ≤ predict_position ≤ size(model.output_weights, 1) "Invalid position"
    return @view model.output_weights[predict_position:predict_position, :, :]
end

"""
    format_predictions(linear_output)

Format linear output to appropriate prediction shape.

# Returns
- 1D vector for single output
- 2D matrix for multi-output (output_dim, batch_size)
"""
function format_predictions(linear_output)
    output_dim = size(linear_output, 1)
    batch_size = size(linear_output, 3)
    
    return output_dim == 1 ? 
        reshape(linear_output, (batch_size,)) : 
        reshape(linear_output, (output_dim, batch_size))
end

"""
    predict_from_sequences(model::SeqCNN, sequences; 
                          use_sparsity=false, 
                          predict_position=nothing,
                          apply_nonlinearity=true)

Complete forward pass from sequences to predictions.

# Process
1. Extract CNN features
2. Linear transformation (output layer)
3. Optional nonlinearity (identity by default)

# Arguments
- `model`: SeqCNN instance
- `sequences`: Input sequences
- `use_sparsity`: Apply sparsity-inducing normalization
- `predict_position`: Predict specific output only
- `apply_nonlinearity`: Apply final activation (currently identity)

# Returns
- Predictions (output_dim, batch_size) or (batch_size,) for single output
"""
function predict_from_sequences(model::SeqCNN, sequences; 
                               use_sparsity=false,
                               predict_position=nothing,
                               apply_nonlinearity=true,
                               training::Union{Nothing,Bool}=nothing)
    # Extract features (will use model.training if training is nothing)
    features = extract_features(model, sequences; use_sparsity=use_sparsity, training=training)
    
    # Select output weights
    weights = select_output_weights(model; predict_position=predict_position)
    
    # Linear transformation
    linear_out = batched_mul(weights, features)
    
    # Format predictions
    preds = format_predictions(linear_out)
    
    # Apply nonlinearity (currently identity)
    return apply_nonlinearity ? identity.(preds) : preds
end

"""
    predict_from_code(model::SeqCNN, code; 
                     layer=0,
                     use_sparsity=false,
                     predict_position=nothing,
                     apply_nonlinearity=true)

Make predictions starting from code at any layer.

# Arguments
- `model`: SeqCNN instance
- `code`: Code representation at specified layer
- `layer`: Which layer this code comes from (0 = PWM, 1+ = conv)
- `use_sparsity`: Apply sparsity to remaining layers
- `predict_position`: Predict specific output
- `apply_nonlinearity`: Apply final activation

# Returns
- Predictions

# Examples
```julia
# Get code and predict from it
code = compute_code_at_layer(model, seqs, 2)
preds = predict_from_code(model, code; layer=2)
```
"""
function predict_from_code(model::SeqCNN, code; 
                          layer=0,
                          use_sparsity=false,
                          predict_position=nothing,
                          apply_nonlinearity=true,
                          training::Union{Nothing,Bool}=nothing)
    @assert 0 ≤ layer ≤ model.num_conv_layers "Invalid layer"
    
    # Use model's training flag if not explicitly provided
    train_mode = isnothing(training) ? model.training[] : training
    
    # Process remaining layers
    if layer == 0
        # From PWM layer: apply base pooling then all conv layers
        # code = reshape_to_4d(code; is_base_layer=true)
        code = pool_code(code,
                        (model.hp.pool_base, 1),
                        (model.hp.stride_base, 1);
                        skip_pooling=false)
        code = forward_conv_recursive(model, code, 1; use_sparsity=use_sparsity, training=train_mode)
    else
        # From intermediate layer: continue from next layer
        code = forward_conv_recursive(model, code, layer + 1; use_sparsity=use_sparsity, training=train_mode)
    end
    
    # Apply MBConv blocks
    for mbconv in model.mbconv_blocks
        code = mbconv(code)
    end
    
    # Flatten to features
    spatial_len = size(code, 1)
    n_channels = size(code, 2)
    batch_size = size(code, 4)
    embed_dim = spatial_len * n_channels
    features = reshape(code, (embed_dim, 1, batch_size))
    
    # Linear layer
    weights = select_output_weights(model; predict_position=predict_position)
    linear_out = batched_mul(weights, features)
    preds = format_predictions(linear_out)
    
    return apply_nonlinearity ? identity.(preds) : preds
end

"""
    (model::SeqCNN)(sequences; use_sparsity=false, linear_sum=false, predict_position=nothing)

Callable interface for SeqCNN forward pass.

# Arguments
- `sequences`: Input sequences
- `use_sparsity`: Apply sparsity-inducing normalization
- `linear_sum`: Return sum of linear outputs (for optimization)
- `predict_position`: Predict specific output position

# Returns
- Model predictions

# Examples
```julia
# Standard prediction
preds = model(sequences)

# Linear sum for gradient-based optimization
loss_term = model(sequences; linear_sum=true, predict_position=1)

# Sparse filters
preds_sparse = model(sequences; use_sparsity=true)
```
"""
function (model::SeqCNN)(sequences; 
                        use_sparsity=false,
                        linear_sum=false,
                        predict_position=nothing,
                        training::Union{Nothing,Bool}=nothing)
    if linear_sum
        @assert !isnothing(predict_position) "linear_sum requires predict_position"
        # Return linear sum without nonlinearity
        return sum(predict_from_sequences(model, sequences;
                                         use_sparsity=use_sparsity,
                                         predict_position=predict_position,
                                         apply_nonlinearity=false,
                                         training=training))
    end
    
    # Standard prediction with nonlinearity
    return predict_from_sequences(model, sequences;
                                 use_sparsity=use_sparsity,
                                 predict_position=predict_position,
                                 apply_nonlinearity=true,
                                 training=training)
end
