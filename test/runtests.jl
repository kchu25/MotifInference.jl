using MotifInference
using Test

@testset "MotifInference.jl" begin

    @testset "package loads and exports are defined" begin
        @test isdefined(MotifInference, :VeryBasicCNN2)
        @test isdefined(MotifInference, :DATASETS)
        @test isdefined(MotifInference, :DATASETS_MUT)
        @test isdefined(MotifInference, :DATASETS_DEBUG)
    end

    # CPU-only tests for the CSV / in-memory entry points (no training, no GPU).
    include("csv_and_inmemory_tests.jl")

    # CPU-only forward-pass shape & round-trip tests (no training, no GPU).
    include("forward_shapes_tests.jl")

    # The integration tests require a real .jld2 dataset on disk and a GPU.
    # Point MOTIFINFERENCE_TEST_DATA at a dataset to run them:
    #     MOTIFINFERENCE_TEST_DATA=/path/to/data.jld2 julia --project -e 'using Pkg; Pkg.test()'
    if haskey(ENV, "MOTIFINFERENCE_TEST_DATA")
        include("integration_tests.jl")
    else
        @info "Skipping integration tests (set MOTIFINFERENCE_TEST_DATA to enable)."
    end

end
