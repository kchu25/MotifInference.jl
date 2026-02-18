
module MotifInference

using CairoMakie
using JLD2
using CUDA

const cnn2_path = joinpath("VeryBasicCNN2", "VeryBasicCNN2.jl")
include(cnn2_path)
using .VeryBasicCNN2
using AutoComputationalGraphTuning
using Flux
using SEQ2EXPdata
using GlyphEctoplasm
using BanzhafInference

include("struct_def.jl")
include("tuning_and_train_final.jl")
include("plotting.jl")
include("run_thru.jl")

include("dataset_utils.jl")
include("datasets.jl")
include("pipeline.jl")

export VeryBasicCNN2, DATASETS, DATASETS_MUT, DATASETS_DEBUG

end