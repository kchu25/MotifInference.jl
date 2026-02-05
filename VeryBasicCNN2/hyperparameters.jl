# ============================================================================
# Hyperparameters Module
# ============================================================================
# Modular organization of CNN hyperparameter definitions and utilities
# 
# Structure:
#   - struct_definition.jl:  Core HyperParameters struct, show(), num_layers()
#   - param_ranges.jl:       HyperParamRanges, nucleotide/amino_acid presets
#   - random_generation.jl:  generate_random_hyperparameters()
#   - utilities.jl:          receptive_field(), with_*, efficientnet_mbconv_config()
# ============================================================================

include("hyperparameters/struct_definition.jl")
include("hyperparameters/param_ranges.jl")
include("hyperparameters/random_generation.jl")
include("hyperparameters/utilities.jl")

