# CPU-only forward-pass tests (no GPU, no dataset).
#
# These guard the reshape/pool plumbing in VeryBasicCNN2/forward.jl — the shared
# `reshape_and_pool` transition and the uniform definition of "code at layer L"
# (post-pool for every layer, including the base PWM layer). They assert output
# shapes and the round-trip invariant tying `compute_code_at_layer` to
# `predict_from_code`.

using MotifInference
using Random
using Test

const V = MotifInference.VeryBasicCNN2

@testset "forward pass shapes & round-trip (CPU)" begin
    # Build a deterministic CPU model; retry seeds until the random architecture
    # yields a valid (non-nothing) model.
    model = nothing
    for s in 1:200
        m = V.create_model((4, 41), 1, 8; use_cuda=false, rng=MersenneTwister(s))
        if m !== nothing
            model = m; break
        end
    end
    @test model !== nothing
    V.eval!(model)   # eval mode: no dropout (avoids the GPU-only CUDA.rand path)

    X = rand(MersenneTwister(12345), Float32, 4, 41, 1, 8)
    batch = size(X, 4)

    # --- shape assertions: every layer is (spatial, num_filters, 1, batch) ---
    for L in 0:model.num_conv_layers
        code = V.compute_code_at_layer(model, X, L; training=false)
        @test ndims(code) == 4
        @test size(code, 3) == 1        # singleton channel-of-image axis
        @test size(code, 4) == batch
        @test size(code, 1) ≥ 1         # spatial length stays positive
    end

    # base layer channel count == number of PWMs
    code0 = V.compute_code_at_layer(model, X, 0; training=false)
    @test size(code0, 2) == model.hp.num_pfms

    # --- extract_features flattens to (embed_dim, 1, batch) ---
    feats = V.extract_features(model, X; training=false)
    @test size(feats, 2) == 1
    @test size(feats, 3) == batch

    # --- round-trip invariant: predicting from a mid-layer checkpoint equals the
    #     full forward pass, for EVERY layer including base layer 0. This is what
    #     pins the uniform post-pool definition of "code at layer L". ---
    full = V.predict_from_sequences(model, X; training=false)
    for L in 0:model.num_conv_layers
        code = V.compute_code_at_layer(model, X, L; training=false)
        from_code = V.predict_from_code(model, code; layer=L, training=false)
        @test size(from_code) == size(full)
        @test from_code ≈ full
    end
end
