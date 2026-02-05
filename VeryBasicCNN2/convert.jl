# ============================================================================
# CPU/GPU Conversion Utilities
# ============================================================================

"""
    model2cpu(model::SeqCNN)

Convert SeqCNN model from GPU to CPU.

# Arguments
- `model`: SeqCNN instance (potentially on GPU)

# Returns
- SeqCNN instance with all arrays moved to CPU

# Example
```julia
model_cpu = model2cpu(model)
```
"""
function model2cpu(model::SeqCNN)
    # Convert conv layers - preserve LayerNorm and masks if present
    conv_cpu = [
        let mask_cpu = isnothing(layer.mask) ? nothing : 
                layer_channel_mask(
                    layer.mask.mixing_filter |> Array,
                    layer.mask.temp,
                    layer.mask.eta,
                    layer.mask.gamma
                )
            if isnothing(layer.ln_gamma)
                LearnedCodeImgFilters(layer.filters |> Array, nothing, nothing, mask_cpu)
            else
                LearnedCodeImgFilters(
                    layer.filters |> Array,
                    layer.ln_gamma |> Array,
                    layer.ln_beta |> Array,
                    mask_cpu
                )
            end
        end
        for layer in model.conv_layers
    ]
    
    # Convert PWM layer
    pwm_mask_cpu = isnothing(model.pwms.mask) ? nothing : (
        layer_channel_mask(
            model.pwms.mask.mixing_filter |> Array,
            model.pwms.mask.temp,
            model.pwms.mask.eta,
            model.pwms.mask.gamma
        )
    )
    pwms_cpu = LearnedPWMs(
        model.pwms.filters |> Array,
        pwm_mask_cpu
    )
    
    # Convert MBConv blocks (explicit conversion for all fields)
    mbconv_cpu = [
        MBConvBlock(
            isnothing(block.expand_filters) ? nothing : block.expand_filters |> Array,
            block.dw_filters |> Array,
            block.se_w1 |> Array,
            block.se_w2 |> Array,
            block.project_filters |> Array,
            block.use_skip
        )
        for block in model.mbconv_blocks
    ]
    
    # Convert output weights
    weights_cpu = model.output_weights |> Array
    
    return SeqCNN(deepcopy(model.hp), pwms_cpu, conv_cpu, mbconv_cpu, weights_cpu, model.final_nonlinearity, Ref(model.training[]))
end

"""
    model2gpu(model::SeqCNN)

Convert SeqCNN model from CPU to GPU.

# Arguments
- `model`: SeqCNN instance (potentially on CPU)

# Returns
- SeqCNN instance with all arrays moved to GPU

# Example
```julia
model_gpu = model2gpu(model)
```
"""
function model2gpu(model::SeqCNN)
    # Convert conv layers - preserve LayerNorm and masks if present
    conv_gpu = [
        let mask_gpu = isnothing(layer.mask) ? nothing : 
                layer_channel_mask(
                    layer.mask.mixing_filter |> cu,
                    layer.mask.temp,
                    layer.mask.eta,
                    layer.mask.gamma
                )
            if isnothing(layer.ln_gamma)
                LearnedCodeImgFilters(layer.filters |> cu, nothing, nothing, mask_gpu)
            else
                LearnedCodeImgFilters(
                    layer.filters |> cu,
                    layer.ln_gamma |> cu,
                    layer.ln_beta |> cu,
                    mask_gpu
                )
            end
        end
        for layer in model.conv_layers
    ]
    
    # Convert PWM layer
    pwm_mask_gpu = isnothing(model.pwms.mask) ? nothing : (
        layer_channel_mask(
            model.pwms.mask.mixing_filter |> cu,
            model.pwms.mask.temp,
            model.pwms.mask.eta,
            model.pwms.mask.gamma
        )
    )
    pwms_gpu = LearnedPWMs(
        model.pwms.filters |> cu,
        pwm_mask_gpu
    )
    
    # Convert MBConv blocks (explicit conversion for all fields)
    mbconv_gpu = [
        MBConvBlock(
            isnothing(block.expand_filters) ? nothing : block.expand_filters |> cu,
            block.dw_filters |> cu,
            block.se_w1 |> cu,
            block.se_w2 |> cu,
            block.project_filters |> cu,
            block.use_skip
        )
        for block in model.mbconv_blocks
    ]
    
    # Convert output weights
    weights_gpu = model.output_weights |> cu
    
    return SeqCNN(deepcopy(model.hp), pwms_gpu, conv_gpu, mbconv_gpu, weights_gpu, model.final_nonlinearity, Ref(model.training[]))
end
