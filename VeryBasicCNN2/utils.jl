# ============================================================================
# Gumbel-Softmax Masking Utilities
# ============================================================================

"""
    gumbel_softmax_sample(p, temp, eta, gamma)

Sample from Gumbel-Softmax distribution for soft masking.

# Arguments
- `p`: Probabilities (any array type, CPU or GPU)
- `temp`: Temperature (lower = sharper, higher = softer)
- `eta`: Right stretch parameter for hard threshold
- `gamma`: Left stretch parameter for hard threshold

# Returns
- Soft mask values in [0, 1]

# Example
```julia
p = sigmoid.(randn(32, 1, 1))  # Channel probabilities
z = gumbel_softmax_sample(p, 0.5, 1.0, 0.0)  # Soft masks
```
"""
function gumbel_softmax_sample(p, temp, eta, gamma)
    gumbel = -log.(-log.(rand(DEFAULT_FLOAT_TYPE, size(p)...)))
    if p isa CuArray
        gumbel = cu(gumbel)
    end
    
    logit_p = log.(p .+ DEFAULT_FLOAT_TYPE(1e-8)) .- 
             log.(1 .- p .+ DEFAULT_FLOAT_TYPE(1e-8))
    s = Flux.sigmoid.((logit_p .+ gumbel) ./ temp)
    
    return min.(DEFAULT_FLOAT_TYPE(1), max.(DEFAULT_FLOAT_TYPE(0), 
                s .* (eta - gamma) .+ gamma))
end

"""
    hard_threshold_mask(p, temp, eta, gamma)

Generate hard binary mask from probabilities (test time, no Gumbel noise).

# Arguments
- `p`: Probabilities
- `temp`: Temperature for sharpening (typically 0.1 at test time)
- `eta`: Right stretch parameter
- `gamma`: Left stretch parameter

# Returns
- Hard binary mask (0.0 or 1.0)

# Example
```julia
p = sigmoid.(randn(32, 1, 1))
z = hard_threshold_mask(p, 0.1, 1.0, 0.0)  # Binary: 0 or 1
```
"""
function hard_threshold_mask(p, temp, eta, gamma)
    logit_p = log.(p .+ DEFAULT_FLOAT_TYPE(1e-8)) .- 
             log.(1 .- p .+ DEFAULT_FLOAT_TYPE(1e-8))
    s = Flux.sigmoid.(logit_p ./ temp)
    
    z_soft = min.(DEFAULT_FLOAT_TYPE(1), max.(DEFAULT_FLOAT_TYPE(0), 
                  s .* (eta - gamma) .+ gamma))
    
    return DEFAULT_FLOAT_TYPE.(z_soft .> DEFAULT_FLOAT_TYPE(0.5))
end

# ============================================================================
# Dimension Calculation Utilities
# ============================================================================

"""
    conv_output_length(input_len, filter_len)

Output length after 1D convolution: `input_len - filter_len + 1`
"""
conv_output_length(input_len, filter_len) = input_len - filter_len + 1

"""
    pool_output_length(input_len, pool_size, stride)

Output length after pooling: `(input_len - pool_size) ÷ stride + 1`
"""
pool_output_length(input_len, pool_size, stride) = 
    (input_len - pool_size) ÷ stride + 1

"""
    conv_pool_output_length(input_len, filter_len, pool_size, stride)

Output length after convolution followed by pooling.
"""
conv_pool_output_length(input_len, filter_len, pool_size, stride) =
    pool_output_length(conv_output_length(input_len, filter_len), pool_size, stride)

"""
    final_conv_embedding_length(hp::HyperParameters, seq_len::Int)

Calculate the spatial dimension after all conv/pool layers.
This simulates the full forward pass dimensionality.

# Process
1. Base layer: conv with PWM → pool
2. Each layer ≤ pool_lvl_top: conv → pool
3. Remaining layers: conv only (no pool)

Returns 0 if any dimension becomes invalid (≤ 0).
"""
function final_conv_embedding_length(hp::HyperParameters, seq_len::Int)
    # Base layer
    len = conv_output_length(seq_len, hp.pfm_len)
    len = pool_output_length(len, hp.pool_base, hp.stride_base)
    len ≤ 0 && return 0
    
    # Conv layers
    for i in 1:num_layers(hp)
        len = conv_output_length(len, hp.img_fil_heights[i])
        len ≤ 0 && return 0
        
        if i ≤ hp.pool_lvl_top
            len = pool_output_length(len, hp.poolsize[i], hp.stride[i])
            len ≤ 0 && return 0
        end
    end
    
    return len
end

# ============================================================================
# Pooling Operations
# ============================================================================

"""
    maxpool(x; pool_size=(2,1), stride=(1,1))

Apply 2D max pooling to 4D tensor (height, width, channels, batch).
"""
function maxpool(x; pool_size=(2,1), stride=(1,1))
    Flux.NNlib.maxpool(x, pool_size; pad=0, stride=stride)
end

"""
    reshape_to_4d(code; is_base_layer=false)

Reshape code tensor to 4D format (spatial, channels, 1, batch).

# Arguments
- `code`: Input tensor (3D or 4D)
- `is_base_layer`: Whether this is the base PWM layer (different indexing)

# Returns
- 4D tensor (spatial, channels, 1, batch)
"""
function reshape_to_4d(code; is_base_layer=false)
    if is_base_layer
        len, channels, batch = size(code, 2), size(code, 3), size(code, 4)
    else
        len, channels, batch = size(code, 1), size(code, 3), size(code, 4)
    end
    return reshape(code, len, channels, 1, batch)
end

"""
    pool_code(code_4d, pool_size, stride; skip_pooling=false)

Apply pooling to 4D CNN code tensor.

# Arguments
- `code_4d`: Input 4D tensor (spatial, channels, 1, batch)
- `pool_size`: Size as (height, width) tuple
- `stride`: Stride as (height, width) tuple  
- `skip_pooling`: If true, return input without pooling (identity operation)

# Returns
- Pooled 4D tensor (spatial, channels, 1, batch)

# Note
Input must already be in 4D format. Use `reshape_to_4d` if needed.
"""
function pool_code(code_4d, pool_size, stride; skip_pooling=false)
    @assert ndims(code_4d) == 4 "Input must be 4D (spatial, channels, 1, batch), got $(size(code_4d))"
    @assert size(code_4d, 3) == 1 "Third dimension must be 1, got $(size(code_4d, 3))"
    
    # Skip pooling for identity layers
    skip_pooling && return code_4d
    
    # Apply max pooling
    pooled = maxpool(code_4d; pool_size=pool_size, stride=stride)
    
    # Calculate output dimensions
    len = size(code_4d, 1)
    new_len = @ignore_derivatives pool_output_length(len, pool_size[1], stride[1])
    channels = size(code_4d, 2)
    batch = size(code_4d, 4)
    
    return reshape(pooled, new_len, channels, 1, batch)
end

# ============================================================================
# Filter Normalization & PWM Construction
# ============================================================================

"""
    normalize_squared(matrix; ϵ=1e-5, reverse_comp=false)

Normalize matrix by squaring elements and normalizing columns.
Optionally concatenates reverse complement.

# Process
1. Square all elements and add ϵ
2. Normalize by column sums (creates probability distribution)
3. Optionally create reverse complement and concatenate
"""
function normalize_squared(matrix; ϵ=DEFAULT_FLOAT_TYPE(1e-5), reverse_comp=false)
    # Fused operations for efficiency
    squared = @. matrix^2 + ϵ
    col_sums = @ignore_derivatives sum(squared; dims=1)
    normalized = @. squared / col_sums
    
    reverse_comp || return normalized
    
    # Reverse complement: reverse both dimensions 1 and 2
    rev_comp = reverse(normalized; dims=(1,2))
    return cat(normalized, rev_comp; dims=4)
end

# Background probabilities for nucleotides and amino acids
const NUCLEOTIDE_BG = DEFAULT_FLOAT_TYPE(0.25)
const AMINO_ACID_BG = DEFAULT_FLOAT_TYPE(0.05)
const BACKGROUND = Dict(4 => NUCLEOTIDE_BG, 20 => AMINO_ACID_BG)

"""
    create_pwm(frequencies; reverse_comp=false)

Create Position Weight Matrix from frequency matrix.
Converts to log2 odds ratios relative to background.

PWM[i,j] = log2(freq[i,j] / background[i])
"""
function create_pwm(frequencies; reverse_comp=false)
    alphabet_size = size(frequencies, 1)
    bg = @ignore_derivatives get(BACKGROUND, alphabet_size, DEFAULT_FLOAT_TYPE(1.0/alphabet_size))
    
    # Normalize frequencies and compute log odds
    probs = normalize_squared(frequencies; reverse_comp=reverse_comp)
    return @. log2(probs / bg)
end

"""
    normalize_filters_l2(filters; softmax_alpha=SOFTMAX_ALPHA, use_sparsity=false)

L2-normalize convolutional filters with optional sparsity-inducing weighting.

# Arguments
- `filters`: 4D filter tensor
- `softmax_alpha`: Strength of sparsity (higher = more sparse)
- `use_sparsity`: Whether to apply softmax sparsity weighting

# Returns
- L2-normalized filters
"""
function normalize_filters_l2(filters; softmax_alpha=SOFTMAX_ALPHA, use_sparsity=false)
    if use_sparsity
        # Sparsity-inducing softmax weighting
        abs_filters = @ignore_derivatives abs.(filters)
        weights = softmax(softmax_alpha .* abs_filters; dims=2)
        weighted = filters .* weights
        
        # L2 normalize
        norms = @ignore_derivatives sqrt.(sum(weighted .^ 2; dims=(1,2)))
        return @. weighted / norms
    else
        # Standard L2 normalization
        norms = @ignore_derivatives sqrt.(sum(filters .^ 2; dims=(1,2)))
        return @. filters / norms
    end
end

"""
    clamp_positive(x; upper=25)

ReLU with upper bound: `min(upper, max(0, x))`
"""
clamp_positive(x; upper=DEFAULT_FLOAT_TYPE(25)) = 
    @. min(upper, max(0, x))

"""
    square_clamp(x)

Square and clamp to [0, 0.5] range.
"""
square_clamp(x) = clamp_positive(x .^ 2; upper=DEFAULT_FLOAT_TYPE(0.5))

# ============================================================================
# Batch Matrix Operations
# ============================================================================

"""
    batched_mul(A, B)

Batched matrix multiplication wrapper for Flux.NNlib.
"""
batched_mul(A, B) = Flux.NNlib.batched_mul(A, B)

"""
    conv(x, w; pad=0, flipped=true)

Convolution operation wrapper.
"""
conv(x, w; pad=0, flipped=true) = Flux.NNlib.conv(x, w; pad=pad, flipped=flipped)
