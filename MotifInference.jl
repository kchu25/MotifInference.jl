
module MotifInference

using CairoMakie
using JLD2
using CUDA
using VeryBasicCNN2
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

end