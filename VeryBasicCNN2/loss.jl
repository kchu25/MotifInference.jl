# ============================================================================
# Loss Functions
# ============================================================================

"""
    huber_loss(predictions, targets; delta=0.85, quadratic_weight=0.5)

Compute Huber loss with automatic NaN handling.

The Huber loss combines quadratic loss for small errors and linear loss 
for large errors, providing robustness to outliers.

# Formula
- For |error| < δ: L = α * error²
- For |error| ≥ δ: L = δ * (|error| - α * δ)

# Arguments
- `predictions`: Model predictions
- `targets`: Ground truth values
- `delta`: Threshold between quadratic and linear regions
- `quadratic_weight`: Weight factor α for quadratic term

# Returns
- Mean Huber loss over valid (non-NaN) entries

# Notes
- NaN values in targets are automatically excluded
- Differentiable for gradient-based optimization
"""
function huber_loss(predictions, targets; 
                   delta=DEFAULT_FLOAT_TYPE(0.85),
                   quadratic_weight=DEFAULT_FLOAT_TYPE(0.5))
    # Precompute threshold
    linear_threshold = quadratic_weight * delta
    
    # Filter valid entries
    valid_mask = @ignore_derivatives .!isnan.(targets)
    num_valid = @ignore_derivatives sum(valid_mask) |> DEFAULT_FLOAT_TYPE
    
    # Compute absolute errors
    abs_errors = abs.(predictions - targets)[valid_mask]
    
    # Determine loss regime
    use_quadratic = @ignore_derivatives abs_errors .< delta
    use_linear = @ignore_derivatives .!use_quadratic
    
    # Compute loss components
    quadratic_part = @. quadratic_weight * (abs_errors^2) * use_quadratic
    linear_part = @. delta * (abs_errors - linear_threshold) * use_linear
    
    return sum(quadratic_part + linear_part) / num_valid
end

"""
    masked_mse(predictions, targets, mask)

Compute MSE only on masked (valid) entries.

# Arguments
- `predictions`: Model predictions
- `targets`: Ground truth values
- `mask`: Boolean mask indicating valid entries

# Returns
- Mean squared error over masked entries
"""
function masked_mse(predictions, targets, mask)
    valid_preds = @view predictions[mask]
    valid_targets = @view targets[mask]
    return Flux.mse(valid_preds, valid_targets)
end

"""
    compute_training_loss(model::SeqCNN, sequences, targets; use_sparsity=false, verbose=true)

Compute training loss for the CNN model.

# Arguments
- `model`: SeqCNN instance
- `sequences`: Input biological sequences
- `targets`: Ground truth target values
- `use_sparsity`: Apply sparsity-inducing normalization
- `verbose`: Print loss value

# Returns
- Scalar loss value for optimization
"""
function compute_training_loss(model::SeqCNN, sequences, targets;
                              use_sparsity=false,
                              verbose=true)
    # Forward pass
    predictions = predict_from_sequences(model, sequences; use_sparsity=use_sparsity)
    
    # Compute loss
    loss = huber_loss(predictions, targets)
    
    verbose && println("Loss: $loss")
    
    return loss
end
