# ============================================================================
# Code Processor Module
# ============================================================================
# Networks that process concatenated code and gradient features
# Input: code ⊕ gradient (concatenated along channel dimension)
# Output: Same dimension as code at inference_code_layer

# Load module components
include("types.jl")
include("convert.jl")
include("masking.jl")
include("plain.jl")
include("resnet.jl")
include("mbconv.jl")
include("deep_plain.jl")
include("constructor.jl")
include("forward.jl")


# for processor can I also have the flag like training::Ref{Bool}  