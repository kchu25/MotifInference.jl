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

@testset "sparse_max_unpool op (non-overlapping sparsify)" begin
    # --- op-level: hand-checkable divisible case (S=6, C=2, p=3) ---
    A = zeros(Float32, 6, 2, 1, 1)
    A[:, 1, 1, 1] = Float32[0.1, 0.9, 0.3,  0.2, 0.5, 0.4]   # ch1 wins pos2, pos5
    A[:, 2, 1, 1] = Float32[0.7, 0.2, 0.6,  0.8, 0.1, 0.3]   # ch2 wins pos1, pos4
    out = V.sparse_max_unpool(A, 3)
    @test size(out) == size(A)
    nz = [(s, c) for s in 1:6, c in 1:2 if out[s, c, 1, 1] != 0]
    @test Set(nz) == Set([(2,1),(5,1),(1,2),(4,2)])          # exactly the winners
    @test out[2,1,1,1] ≈ 0.5498f0 atol=1e-3                   # softmax([0.9,0.7])[1]
    @test out[1,2,1,1] ≈ 0.4502f0 atol=1e-3                   # softmax([0.9,0.7])[2]

    # --- non-divisible S trims cleanly back to S ---
    B = rand(MersenneTwister(1), Float32, 7, 3, 1, 2)
    @test size(V.sparse_max_unpool(B, 3)) == (7, 3, 1, 2)

    # --- dead window (all-zero) stays zero (no invented signal) ---
    Z = zeros(Float32, 6, 2, 1, 1)
    @test all(iszero, V.sparse_max_unpool(Z, 3))

    # --- p ≤ 1 is a no-op ---
    @test V.sparse_max_unpool(A, 1) == A

    # --- model-level: enabling the op sparsifies the inference-layer code and
    #     keeps receptive fields non-overlapping (≤ ⌈S/p⌉ survivors per channel),
    #     while the round-trip invariant still holds. ---
    seed = 0
    for s in 1:200
        (V.create_model((4,41), 1, 8; use_cuda=false, rng=MersenneTwister(s)) !== nothing) &&
            (seed = s; break)
    end
    on = V.create_model((4,41), 1, 8; use_cuda=false, rng=MersenneTwister(seed),
                        use_sparse_unpool=true)
    V.eval!(on)

    # one-hot sequences (realistic input → active, sparse codes)
    rng = MersenneTwister(7)
    Xoh = zeros(Float32, 4, 41, 1, 8)
    for n in 1:8, j in 1:41; Xoh[rand(rng, 1:4), j, 1, n] = 1f0; end

    L = on.hp.inference_code_layer
    p = V.sparse_unpool_window(on.hp, L)
    code = V.compute_code_at_layer(on, Xoh, L; training=false)
    S, C = size(code, 1), size(code, 2)
    maxsurv = cld(S, p)
    for c in 1:C, n in 1:size(code, 4)
        @test count(!iszero, @view code[:, c, 1, n]) ≤ maxsurv
    end

    full_on = V.predict_from_sequences(on, Xoh; training=false)
    for l in 0:on.num_conv_layers
        c = V.compute_code_at_layer(on, Xoh, l; training=false)
        @test V.predict_from_code(on, c; layer=l, training=false) ≈ full_on
    end
end

# ────────────────────────────────────────────────────────────────────────────
# Nucleotide mutagenesis model (seq_type = :dna/:rna, type = :mut)
#
# Guards the two halves of the DNA/RNA mut path: that `resolve_model_creator`
# actually selects the mutagenesis architecture (before this existed, a :mut run
# on nucleotides silently trained a convolution model while the motif extraction
# and rendering went down the mutagenesis path), and that the architecture it
# builds has the properties the mutagenesis pipeline relies on — a width-1 base
# layer, no downsampling, and a receptive field equal to the squeeze height,
# which is what the renderer reports as the mutation-region width.
# ────────────────────────────────────────────────────────────────────────────
@testset "nucleotide mutagenesis model (CPU)" begin
    @testset "model selection" begin
        for st in (:dna, :rna), mo in (false, true), cb in (false, true)
            @test MotifInference.resolve_model_creator(; seq_type=st, type=:mut,
                      multioutput=mo, conv_bottleneck=cb) ===
                  V.create_model_nucleotides_fixed_pool_stride_mut_w_bottleneck
            @test MotifInference.model_uses_bottleneck(; seq_type=st, type=:mut, conv_bottleneck=cb)
        end
        # protein mut and the nucleotide convolution paths are untouched
        @test MotifInference.resolve_model_creator(; seq_type=:protein, type=:mut) ===
              V.create_model_aminoacids_fixed_pool_stride_w_bottleneck
        @test MotifInference.resolve_model_creator(; seq_type=:dna, type=:conv) ===
              V.create_model_nucleotides_fixed_pool_stride
        @test MotifInference.resolve_model_creator(; seq_type=:dna, type=:conv, conv_bottleneck=true) ===
              V.create_model_nucleotides_fixed_pool_stride_bottleneck
        @test !MotifInference.model_uses_bottleneck(; seq_type=:dna, type=:conv)
        @test MotifInference.model_uses_bottleneck(; seq_type=:dna, type=:conv, conv_bottleneck=true)
    end

    @testset "architecture" begin
        m = V.create_model_nucleotides_fixed_pool_stride_mut_w_bottleneck(
                (4, 60), 1, 8; use_cuda=false, rng=MersenneTwister(1))
        @test m !== nothing
        hp = m.hp
        @test hp.pfm_len == 1                                   # per-site letter detector
        @test hp.inference_code_layer == 1                      # motifs read at conv layer 1
        @test all(isone, hp.poolsize)                           # no downsampling anywhere
        @test all(isone, hp.stride)
        @test hp.num_img_filters[1] == V.BOTTLENECK_FILTERS     # squeeze on the interpreted layer
        @test hp.img_fil_heights[1] == V.BOTTLENECK_HEIGHT
        # Region width == squeeze height: pfm_len is 1 and nothing pools before layer 1.
        @test m.receptive_field == V.BOTTLENECK_HEIGHT
        @test V.receptive_field(hp) == V.BOTTLENECK_HEIGHT

        # bottleneck_height is the knob for widening mutation regions
        wide = V.create_model_nucleotides_fixed_pool_stride_mut_w_bottleneck(
                   (4, 60), 1, 8; use_cuda=false, rng=MersenneTwister(1), bottleneck_height=12)
        @test wide.receptive_field == 12
    end

    @testset "forward pass on a mutation encoding" begin
        m = V.create_model_nucleotides_fixed_pool_stride_mut_w_bottleneck(
                (4, 60), 1, 8; use_cuda=false, rng=MersenneTwister(1))
        V.eval!(m)

        # Mutation encoding: mostly zeros, a 1 only where a sequence differs from
        # the consensus (what OnehotSEQ2EXP_Dataset.X_mut looks like for DNA/RNA).
        rng = MersenneTwister(99)
        Xmut = zeros(Float32, 4, 60, 1, 8)
        for n in 1:8, _ in 1:3
            Xmut[rand(rng, 1:4), rand(rng, 1:60), 1, n] = 1f0
        end

        full = V.predict_from_sequences(m, Xmut; training=false)
        @test size(full, ndims(full)) == 8
        @test all(isfinite, full)
        for L in 0:m.num_conv_layers
            code = V.compute_code_at_layer(m, Xmut, L; training=false)
            @test size(code, 4) == 8
            @test V.predict_from_code(m, code; layer=L, training=false) ≈ full
        end
    end
end
