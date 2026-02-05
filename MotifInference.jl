
module MotifInference

using CairoMakie
using JLD2
using CUDA

const cnn2_path = joinpath("VeryBasicCNN2", "VeryBasicCNN2.jl")
include(cnn2_path)
using .VeryBasicCNN2
# using EfficientNetSeq2label
using AutoComputationalGraphTuning
using Flux
using SEQ2EXPdata
using GlyphEctoplasm
using BanzhafInference




include("struct_def.jl")
include("tuning_and_train_final.jl")
include("plotting.jl")
include("run_thru.jl")

export VeryBasicCNN2
# export EfficientNetSeq2label

end