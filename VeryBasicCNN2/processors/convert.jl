# ============================================================================
# CPU/GPU Conversion Utilities for CodeProcessor
# ============================================================================

"""
    processor2cpu(proc::CodeProcessor)

Convert CodeProcessor from GPU to CPU.

# Arguments
- `proc`: CodeProcessor instance (potentially on GPU)

# Returns
- CodeProcessor instance with all arrays moved to CPU

# Example
```julia
proc_cpu = processor2cpu(proc)
```
"""
function processor2cpu(proc::CodeProcessor)
    CodeProcessor(
        isnothing(proc.expand_filters) ? nothing : proc.expand_filters |> Array,
        proc.dw_filters |> Array,
        isnothing(proc.se_w1) ? nothing : proc.se_w1 |> Array,
        isnothing(proc.se_w2) ? nothing : proc.se_w2 |> Array,
        proc.project_filters |> Array,
        isnothing(proc.dw_filters_2) ? nothing : proc.dw_filters_2 |> Array,
        isnothing(proc.project_filters_2) ? nothing : proc.project_filters_2 |> Array,
        isnothing(proc.dw_filters_3) ? nothing : proc.dw_filters_3 |> Array,
        isnothing(proc.project_filters_3) ? nothing : proc.project_filters_3 |> Array,
        isnothing(proc.mask_proj) ? nothing : proc.mask_proj |> Array,
        isnothing(proc.channel_mask_proj) ? nothing : proc.channel_mask_proj |> Array,
        proc.mask_temp,
        proc.mask_eta,
        proc.mask_gamma,
        proc.use_hard_mask,
        proc.use_residual,
        proc.arch_type,
        Ref(proc.training[])  # Create new Ref to avoid sharing
    )
end

"""
    processor2gpu(proc::CodeProcessor)

Convert CodeProcessor from CPU to GPU.

# Arguments
- `proc`: CodeProcessor instance (potentially on CPU)

# Returns
- CodeProcessor instance with all arrays moved to GPU

# Example
```julia
proc_gpu = processor2gpu(proc)
```
"""
function processor2gpu(proc::CodeProcessor)
    CodeProcessor(
        isnothing(proc.expand_filters) ? nothing : proc.expand_filters |> cu,
        proc.dw_filters |> cu,
        isnothing(proc.se_w1) ? nothing : proc.se_w1 |> cu,
        isnothing(proc.se_w2) ? nothing : proc.se_w2 |> cu,
        proc.project_filters |> cu,
        isnothing(proc.dw_filters_2) ? nothing : proc.dw_filters_2 |> cu,
        isnothing(proc.project_filters_2) ? nothing : proc.project_filters_2 |> cu,
        isnothing(proc.dw_filters_3) ? nothing : proc.dw_filters_3 |> cu,
        isnothing(proc.project_filters_3) ? nothing : proc.project_filters_3 |> cu,
        isnothing(proc.mask_proj) ? nothing : proc.mask_proj |> cu,
        isnothing(proc.channel_mask_proj) ? nothing : proc.channel_mask_proj |> cu,
        proc.mask_temp,
        proc.mask_eta,
        proc.mask_gamma,
        proc.use_hard_mask,
        proc.use_residual,
        proc.arch_type,
        Ref(proc.training[])  # Create new Ref to avoid sharing
    )
end
