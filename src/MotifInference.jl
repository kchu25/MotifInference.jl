module MotifInference

using CairoMakie
using JLD2
using CUDA

include(joinpath(@__DIR__, "VeryBasicCNN2", "VeryBasicCNN2.jl"))
using .VeryBasicCNN2
using AutoComputationalGraphTuning
using Flux
using SEQ2EXPdata
using GlyphEctoplasm
using BanzhafInference
using StatsBase
using Arrow

include(joinpath(@__DIR__, "struct_def.jl"))
include(joinpath(@__DIR__, "tuning_and_train_final.jl"))
include(joinpath(@__DIR__, "plotting.jl"))
include(joinpath(@__DIR__, "run_thru.jl"))

include(joinpath(@__DIR__, "dataset_utils.jl"))
include(joinpath(@__DIR__, "datasets.jl"))
include(joinpath(@__DIR__, "pipeline.jl"))

export VeryBasicCNN2, DATASETS, DATASETS_MUT, DATASETS_DEBUG

end
