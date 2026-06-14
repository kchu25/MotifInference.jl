# CPU-only tests for the in-memory / CSV entry points (no training, no GPU).
# These exercise data parsing and dataset construction, not run_method itself.

using MotifInference
using SEQ2EXPdata
using Test

@testset "read_seq_label_csv" begin
    # --- with a header row (auto-detected and skipped) ---
    p = tempname() * ".csv"
    write(p, "sequence,label\nACGTACGT,1.5\nTTGGCCAA,2.0\nGGGGCCCC,-0.5\n")
    strings, labels = MotifInference.read_seq_label_csv(p)
    @test strings == ["ACGTACGT", "TTGGCCAA", "GGGGCCCC"]
    @test labels == [1.5, 2.0, -0.5]
    @test eltype(labels) == Float64
    rm(p)

    # --- without a header, plus a blank line and stray whitespace ---
    p = tempname() * ".csv"
    write(p, "ACGTACGT, 1.0\n\n  TTGGCCAA ,2.5\n")
    strings, labels = MotifInference.read_seq_label_csv(p)
    @test strings == ["ACGTACGT", "TTGGCCAA"]   # whitespace stripped
    @test labels == [1.0, 2.5]
    rm(p)

    # --- error: a non-header line with a non-numeric label ---
    p = tempname() * ".csv"
    write(p, "ACGTACGT,1.0\nTTGGCCAA,not_a_number\n")
    @test_throws ArgumentError MotifInference.read_seq_label_csv(p)
    rm(p)

    # --- error: missing file ---
    @test_throws ArgumentError MotifInference.read_seq_label_csv(tempname() * ".csv")

    # --- error: empty file ---
    p = tempname() * ".csv"
    write(p, "")
    @test_throws ArgumentError MotifInference.read_seq_label_csv(p)
    rm(p)
end

@testset "in-memory dataset construction" begin
    strings = ["ACGTACGT", "TTGGCCAA", "GGGGCCCC", "AAAATTTT"]
    labels = [1.0, 2.0, 3.0, 4.0]

    raw = SEQ2EXP_Dataset(strings, labels)
    @test raw.strings == strings
    @test raw.labels == labels

    # one-hot wrapper used internally by run_method(strings, labels; ...)
    data = OnehotSEQ2EXP_Dataset(raw)
    @test data.raw_data === raw
    @test data.Y_dim == 1                        # scalar labels → single output
    @test data.X_dim[1] == 4                     # DNA alphabet (A,C,G,T)
    @test data.X_dim[2] == 8                     # sequence length
    @test size(data.X, 4) == length(strings)     # sequences along the last tensor dim
end

@testset "CSV → dataset round-trip (no training)" begin
    p = tempname() * ".csv"
    write(p, "seq,y\nACGTACGT,1.0\nTTGGCCAA,2.0\nGGGGCCCC,3.0\n")
    strings, labels = MotifInference.read_seq_label_csv(p)
    data = OnehotSEQ2EXP_Dataset(SEQ2EXP_Dataset(strings, labels))
    @test data.Y_dim == 1
    @test size(data.X, 4) == 3                   # 3 sequences
    rm(p)
end
